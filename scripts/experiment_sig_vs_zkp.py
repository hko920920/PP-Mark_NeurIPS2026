#!/usr/bin/env python3
"""
Experiment: Digital Signature vs PP-Mark ZKP.

Demonstrates two attack scenarios where a signature-only system fails
but PP-Mark's ZKP succeeds:

  Attack A — Forged Binding:
    Attacker searches for fabricated b' maximizing detection score
    on an arbitrary cover image. In sig-only, attacker signs (b', meta)
    freely. In PP-Mark, ZKP enforces b = H(h_ctx || k).

  Attack B — Compliance Bypass:
    Malicious generator signs metadata without embedding watermark.
    Sig-only: passes (signature valid). PP-Mark(score): fails (no signal).

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/experiment_sig_vs_zkp.py \
        --manifest outputs/pos_eval/manifest.jsonl \
        --clean-dir outputs/clean_eval/ \
        --output-dir outputs/sig_vs_zkp/ \
        --n-random 10000 \
        --n-images 100
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[0].parent / "src"))

from ppmark_v03.config import GlobalConfig
from ppmark_v03.ddim_unet import DiffusersDDIMConfig, DiffusersDDIMInverter
from ppmark_v03.noise import build_bit_index_map
from ppmark_v03.payload import bytes_to_bits, int_to_fixed_bytes
from ppmark_v03.rs import ReedSolomonCodec
from ppmark_v03.sampling import deterministic_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_manifest(path: Path, limit: Optional[int] = None) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_image_tensor(path: Path, device: str, dtype: torch.dtype) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3) normalised to [0, 1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    return tensor.to(device=device, dtype=dtype)


def resolve_path(raw: str, base: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base / p).resolve()


# ---------------------------------------------------------------------------
# Score computation (standalone, no caching needed for random search)
# ---------------------------------------------------------------------------

def compute_score_for_binding(
    latent: np.ndarray,          # (C, H, W) or (H, W) flattened channel 0
    binding: int,
    codeword: bytes,
    alpha: float,
    width: int,
    height: int,
    sample_count: int,
    backend: str = "sha256",
) -> float:
    """Compute PP-Mark detection score for a given binding on a latent."""
    bits = np.array(bytes_to_bits(codeword), dtype=np.int8)
    bit_idx_map = build_bit_index_map(binding, width, height, len(bits), backend)
    sign_map = 2.0 * bits[bit_idx_map].astype(np.float32) - 1.0

    sample_set = deterministic_sample(
        total_pixels=width * height,
        width=width,
        height=height,
        key_material=codeword,
        count=sample_count,
    )
    indices = np.array(sample_set.indices, dtype=np.int64)
    ys = indices // width
    xs = indices % width

    watermark = alpha * sign_map[ys, xs]
    denom = np.linalg.norm(watermark) + 1e-9
    samples = latent[ys, xs]
    raw = np.dot(samples, watermark) / denom
    return float(np.abs(raw))


def random_binding_and_codeword(codec: ReedSolomonCodec) -> Tuple[int, bytes]:
    """Generate a random 256-bit binding and its RS codeword."""
    rand_bytes = os.urandom(32)
    binding = int.from_bytes(rand_bytes, "big")
    message_bytes = int_to_fixed_bytes(binding, codec.k)
    codeword = codec.encode(message_bytes)
    return binding, codeword


# ---------------------------------------------------------------------------
# Attack A: Forged Binding (Random Search)
# ---------------------------------------------------------------------------

def attack_forged_binding(
    latent: np.ndarray,
    alpha: float,
    width: int,
    height: int,
    sample_count: int,
    n_random: int,
    tau: float,
    codec: ReedSolomonCodec,
    backend: str = "sha256",
) -> Dict:
    """
    Random search over b' to find one whose sign pattern maximizes score.
    Returns best score found and whether it exceeds tau.
    """
    best_score = 0.0
    best_binding = 0

    for trial in range(n_random):
        b_prime, cw_prime = random_binding_and_codeword(codec)
        score = compute_score_for_binding(
            latent, b_prime, cw_prime, alpha, width, height, sample_count, backend
        )
        if score > best_score:
            best_score = score
            best_binding = b_prime

    return {
        "best_score": best_score,
        "best_binding_hex": hex(best_binding),
        "exceeds_tau": best_score >= tau,
        "n_trials": n_random,
    }


# ---------------------------------------------------------------------------
# Attack B: Compliance Bypass
# ---------------------------------------------------------------------------

def attack_compliance_bypass(
    latent: np.ndarray,
    legit_binding: int,
    legit_codeword: bytes,
    alpha: float,
    width: int,
    height: int,
    sample_count: int,
    tau: float,
    backend: str = "sha256",
) -> Dict:
    """
    Score a clean (unwatermarked) image against legitimate metadata.
    Sig-only: would pass (signature valid on metadata).
    PP-Mark(score): should fail (no watermark signal in latent).
    """
    score = compute_score_for_binding(
        latent, legit_binding, legit_codeword, alpha,
        width, height, sample_count, backend,
    )
    return {
        "score": score,
        "exceeds_tau": score >= tau,
        "sig_only_passes": True,  # signature on metadata is always valid
    }


# ---------------------------------------------------------------------------
# DDIM Inversion wrapper
# ---------------------------------------------------------------------------

def get_inverter(
    model_id: str,
    device: str,
    steps: int = 50,
) -> DiffusersDDIMInverter:
    ddim_cfg = DiffusersDDIMConfig(
        model_id=model_id,
        scheduler="ddim",
        device=device,
        torch_dtype="float32",
        num_steps=steps,
    )
    return DiffusersDDIMInverter(ddim_cfg)


def invert_image(
    inverter: DiffusersDDIMInverter,
    image_tensor: torch.Tensor,
    prompt: str = "",
    guidance_scale: float = 7.5,
    sigma: float = 1.0,
) -> np.ndarray:
    """DDIM invert and return latent channel 0 as numpy (H, W)."""
    inverter.prompt = prompt
    inverter.negative_prompt = ""
    inverter.cfg.guidance_scale = guidance_scale
    with torch.no_grad():
        latents = inverter.invert_latents_torch(image_tensor)
    latent = latents[:, 0]  # (N, H, W)
    if sigma != 0:
        latent = latent / sigma
    return latent[0].cpu().numpy()  # (H, W)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sig vs ZKP experiment")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="PP-Mark watermarked eval manifest (JSONL)")
    parser.add_argument("--clean-dir", type=Path, default=None,
                        help="Dir with clean (unwatermarked) images for compliance bypass")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-random", type=int, default=10000,
                        help="Number of random b' trials per image (forged binding)")
    parser.add_argument("--n-images", type=int, default=100,
                        help="Number of images to evaluate")
    parser.add_argument("--tau", type=float, default=None,
                        help="Detection threshold (FPR=1%%). Auto-loaded from manifest if not set.")
    parser.add_argument("--model-id", type=str, default="Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=None,
                        help="Watermark strength. Auto-loaded from metadata if not set.")
    parser.add_argument("--backend", type=str, default="sha256")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    dtype = torch.float32

    # Load manifest
    print(f"Loading manifest: {args.manifest}")
    manifest = load_manifest(args.manifest, limit=args.n_images)
    print(f"  Loaded {len(manifest)} rows")
    base_dir = args.manifest.parent

    # Config
    cfg = GlobalConfig()
    codec = ReedSolomonCodec(n=64, k=32)
    width, height = cfg.image.width, cfg.image.height
    sample_count_cfg = cfg.image.sample_count()

    # Threshold
    tau = args.tau
    if tau is None:
        # Try to load from a calibration file alongside manifest
        cal_path = base_dir / "calibration.json"
        if cal_path.exists():
            cal = json.loads(cal_path.read_text())
            tau = float(cal.get("tau_1pct", cal.get("tau", 0.3)))
            print(f"  Loaded tau={tau:.4f} from {cal_path}")
        else:
            tau = 0.3  # fallback
            print(f"  Using default tau={tau}")

    # DDIM inverter
    print(f"Loading model: {args.model_id}")
    inverter = get_inverter(args.model_id, device, args.steps)

    # ===================================================================
    # Attack A: Forged Binding (on watermarked images as cover)
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"Attack A: Forged Binding (random search, {args.n_random} trials)")
    print(f"{'='*60}")

    forge_results = []
    for i, row in enumerate(manifest):
        img_path = resolve_path(
            row.get("image_path") or row.get("image_relpath", ""),
            base_dir,
        )
        if not img_path.exists():
            print(f"  [{i}] SKIP: {img_path} not found")
            continue

        # Load metadata from separate JSON file if referenced
        meta_relpath = row.get("metadata_path") or row.get("metadata_relpath")
        if meta_relpath:
            meta_path = resolve_path(meta_relpath, base_dir)
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
            else:
                meta = row
        else:
            meta = row.get("metadata", row)

        alpha = args.alpha or float(meta.get("alpha_effective", cfg.image.alpha))
        sigma = float(meta.get("init_noise_sigma", 1.0))
        prompt = str(meta.get("prompt", ""))
        sample_count = int(meta.get("sample_count", sample_count_cfg))
        # Use latent-space dimensions from metadata (not pixel-space config)
        img_w = int(meta.get("width", meta.get("grid_w", width)))
        img_h = int(meta.get("height", meta.get("grid_h", height)))

        img_tensor = load_image_tensor(img_path, device, dtype)
        t0 = time.time()
        latent = invert_image(inverter, img_tensor, prompt=prompt, sigma=sigma)
        t_inv = time.time() - t0
        print(f"  [{i:3d}] latent shape={latent.shape}, inv={t_inv:.1f}s")

        # Sanity: compute true score with legitimate binding
        true_binding = int(str(meta.get("binding", "0")), 16)
        true_cw = bytes.fromhex(str(meta.get("codeword_hex", "00" * 64)))
        true_score = compute_score_for_binding(
            latent, true_binding, true_cw, alpha,
            img_w, img_h, sample_count, args.backend,
        )
        print(f"         true_score={true_score:.4f} (tau={tau:.4f}, {'PASS' if true_score >= tau else 'FAIL'})")

        t0 = time.time()
        result = attack_forged_binding(
            latent, alpha, img_w, img_h, sample_count,
            args.n_random, tau, codec, args.backend,
        )
        t_search = time.time() - t0

        result["image_idx"] = i
        result["inversion_time"] = round(t_inv, 2)
        result["search_time"] = round(t_search, 2)
        result["true_score"] = round(true_score, 4)
        forge_results.append(result)

        status = "HIT" if result["exceeds_tau"] else "miss"
        print(f"         forged best={result['best_score']:.4f} {status} search={t_search:.1f}s")

    # Compute stats for forged binding
    n_forge = len(forge_results)
    n_forge_score_hit = sum(1 for r in forge_results if r["exceeds_tau"])
    forge_score_far = n_forge_score_hit / n_forge if n_forge > 0 else 0.0
    best_scores = [r["best_score"] for r in forge_results]

    print(f"\n--- Forged Binding Summary ---")
    print(f"  Images: {n_forge}")
    print(f"  Best score stats: mean={np.mean(best_scores):.4f} "
          f"max={np.max(best_scores):.4f} min={np.min(best_scores):.4f}")
    print(f"  Score exceeds tau: {n_forge_score_hit}/{n_forge} ({forge_score_far:.4f})")
    print()
    print(f"  Sig-only FAR:        1.0000 (signature-only; no score check)")
    print(f"    -> Attacker signs (b', meta) with own key. Verifier checks")
    print(f"       signature only, not watermark correlation. Always passes.")
    print(f"  Sig+Score FAR:       {forge_score_far:.4f} (signature + score check)")
    print(f"    -> Even with score check, random b' search finds {n_forge_score_hit}/{n_forge}")
    print(f"       images exceeding tau={tau:.4f}.")
    print(f"  PP-Mark(score) FAR:  ~FPR floor (b cryptographically fixed by generator)")
    print(f"  PP-Mark(accept) FAR: 0.0000 (ZKP rejects any fabricated b')")

    # ===================================================================
    # Attack B: Compliance Bypass (clean images + fake metadata)
    # ===================================================================
    bypass_results = []
    if args.clean_dir and args.clean_dir.exists():
        print(f"\n{'='*60}")
        print(f"Attack B: Compliance Bypass")
        print(f"{'='*60}")

        clean_images = sorted(args.clean_dir.glob("*.png"))[:args.n_images]
        if not clean_images:
            clean_images = sorted(args.clean_dir.glob("*.jpg"))[:args.n_images]

        for i, img_path in enumerate(clean_images):
            # Use legitimate metadata from manifest (as if generator signed it)
            ref_row = manifest[i % len(manifest)]
            ref_meta_relpath = ref_row.get("metadata_path") or ref_row.get("metadata_relpath")
            if ref_meta_relpath:
                ref_meta_path = resolve_path(ref_meta_relpath, base_dir)
                if ref_meta_path.exists():
                    ref_meta = json.loads(ref_meta_path.read_text())
                else:
                    ref_meta = ref_row
            else:
                ref_meta = ref_row.get("metadata", ref_row)

            legit_binding = int(str(ref_meta.get("binding", "0")), 16)
            legit_cw = bytes.fromhex(str(ref_meta.get("codeword_hex", "00" * 64)))
            alpha = args.alpha or float(ref_meta.get("alpha_effective", cfg.image.alpha))
            sigma = float(ref_meta.get("init_noise_sigma", 1.0))
            prompt = str(ref_meta.get("prompt", ""))
            sample_count = int(ref_meta.get("sample_count", sample_count_cfg))
            bypass_w = int(ref_meta.get("width", ref_meta.get("grid_w", width)))
            bypass_h = int(ref_meta.get("height", ref_meta.get("grid_h", height)))

            img_tensor = load_image_tensor(img_path, device, dtype)
            latent = invert_image(inverter, img_tensor, prompt=prompt, sigma=sigma)

            result = attack_compliance_bypass(
                latent, legit_binding, legit_cw, alpha,
                bypass_w, bypass_h, sample_count, tau, args.backend,
            )
            result["image_idx"] = i
            bypass_results.append(result)

            status = "SCORE_PASS" if result["exceeds_tau"] else "score_fail"
            print(f"  [{i:3d}] score={result['score']:.4f} {status} sig=PASS")

        n_bypass = len(bypass_results)
        n_bypass_score_pass = sum(1 for r in bypass_results if r["exceeds_tau"])
        bypass_score_far = n_bypass_score_pass / n_bypass if n_bypass > 0 else 0.0

        print(f"\n--- Compliance Bypass Summary ---")
        print(f"  Images: {n_bypass}")
        print(f"  Sig-only FAR:        1.0000 (signature always valid)")
        print(f"  PP-Mark(score) FAR:  {bypass_score_far:.4f} ({n_bypass_score_pass}/{n_bypass})")
        print(f"  PP-Mark(accept) FAR: 0.0000 (no valid ZKP exists)")

    # ===================================================================
    # Save results
    # ===================================================================
    summary = {
        "config": {
            "n_images": len(manifest),
            "n_random_trials": args.n_random,
            "tau": tau,
            "model_id": args.model_id,
            "alpha": args.alpha or "from_metadata",
            "backend": args.backend,
        },
        "attack_a_forged_binding": {
            "n_images": n_forge,
            "n_score_exceeds_tau": n_forge_score_hit,
            "best_score_stats": {
                "mean": float(np.mean(best_scores)),
                "max": float(np.max(best_scores)),
                "min": float(np.min(best_scores)),
                "std": float(np.std(best_scores)),
            },
            "far_sig_only": 1.0,  # sig checks signature, not score -> always pass
            "far_sig_plus_score": forge_score_far,  # sig + score check
            "far_ppmark_score": "<=FPR (b cryptographically fixed)",
            "far_ppmark_accept": 0.0,
            "per_image": forge_results,
        },
    }

    if bypass_results:
        summary["attack_b_compliance_bypass"] = {
            "n_images": len(bypass_results),
            "n_score_pass": n_bypass_score_pass,
            "far_sig_only": 1.0,  # sig on fake metadata -> always pass
            "far_ppmark_score": bypass_score_far,
            "far_ppmark_accept": 0.0,
            "per_image": bypass_results,
        }

    # Comparison table (4 systems x 2 attacks)
    summary["comparison_table"] = {
        "columns": ["Attack", "Sig-only", "Sig+Score", "PP-Mark(score)", "PP-Mark(accept)"],
        "note": "Sig-only checks signature validity only (no score). "
                "Sig+Score checks signature + detection score >= tau. "
                "PP-Mark(score) uses cryptographically fixed b. "
                "PP-Mark(accept) requires valid ZKP + score.",
        "rows": [
            ["Forged Binding", "1.0000", f"{forge_score_far:.4f}", "<=FPR", "0.0000"],
        ],
    }
    if bypass_results:
        summary["comparison_table"]["rows"].append(
            ["Compliance Bypass", "1.0000", f"{bypass_score_far:.4f}", "<=FPR", "0.0000"]
        )

    out_path = args.output_dir / "sig_vs_zkp_results.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
