#!/usr/bin/env python3
"""
Empirical verification of Assumption 3.3 (assum:noise) in PP-Mark paper.

Verifies two properties of DDIM inversion residual ε_i = ẑ_i - z_i:
  1. E[ε_i] ≈ 0  (zero-mean)
  2. Corr(ε_i, s_i) ≈ 0  (weakly correlated with watermark signal)

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src python3 scripts/experiment_assumption_verify.py \
    --manifest <manifest.jsonl> --output-dir outputs/assumption_verify \
    --model-id Manojb/stable-diffusion-2-1-base --device cuda
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from scipy import stats as scipy_stats

from ppmark_v03.noise import build_bit_index_map
from ppmark_v03.payload import bytes_to_bits
from ppmark_v03.ddim_unet import DiffusersDDIMConfig, DiffusersDDIMInverter


# ---------------------------------------------------------------------------
# Helpers (reused from experiment_sig_vs_zkp.py)
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
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=dtype)


def resolve_path(raw: str, base: Path) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def get_inverter(model_id: str, device: str, steps: int = 50) -> DiffusersDDIMInverter:
    ddim_cfg = DiffusersDDIMConfig(
        model_id=model_id,
        scheduler="ddim",
        device=device,
        torch_dtype="float32",
        num_steps=steps,
    )
    return DiffusersDDIMInverter(ddim_cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Assumption 3.3 empirical verification")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/assumption_verify"))
    parser.add_argument("--model-id", type=str, default="Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-images", type=int, default=None)
    parser.add_argument("--steps", type=int, default=50)
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

    # Load model
    print(f"Loading model: {args.model_id}")
    inverter = get_inverter(args.model_id, device, args.steps)

    # Per-image results
    per_image_mean = []
    per_image_std = []
    per_image_corr = []
    per_image_corr_pval = []
    all_residuals = []  # pooled for global histogram (subsample to manage memory)

    print(f"\n{'='*60}")
    print(f"Computing inversion residuals for {len(manifest)} images")
    print(f"{'='*60}")

    for i, row in enumerate(manifest):
        # Resolve image path
        img_path = resolve_path(
            row.get("image_path") or row.get("image_relpath", ""),
            base_dir,
        )
        if not img_path.exists():
            print(f"  [{i}] SKIP: image not found")
            continue

        # Load metadata
        meta_relpath = row.get("metadata_path") or row.get("metadata_relpath")
        if meta_relpath:
            meta_path = resolve_path(meta_relpath, base_dir)
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else row
        else:
            meta = row.get("metadata", row)

        # Load z0 (watermarked latent, saved during generation)
        z0_relpath = meta.get("z0_path", "")
        z0_path = resolve_path(z0_relpath, base_dir) if z0_relpath else None
        if z0_path is None or not z0_path.exists():
            print(f"  [{i}] SKIP: z0_latents not found")
            continue
        z0 = np.load(z0_path)  # (4, 64, 64)
        z0_ch0 = z0[0]  # (64, 64) — watermarked latent channel 0

        # DDIM invert watermarked image → ẑ (raw, no sigma division)
        prompt = str(meta.get("prompt", ""))
        guidance_scale = float(meta.get("guidance_scale", 7.5))
        img_tensor = load_image_tensor(img_path, device, dtype)

        inverter.prompt = prompt
        inverter.negative_prompt = ""
        inverter.cfg.guidance_scale = guidance_scale

        t0 = time.time()
        with torch.no_grad():
            z_hat = inverter.invert_latents_torch(img_tensor)
        t_inv = time.time() - t0
        z_hat_ch0 = z_hat[0, 0].cpu().numpy()  # (64, 64) — raw inverted latent ch0

        # Compute residual ε = ẑ - z0 (raw latent space)
        epsilon = z_hat_ch0 - z0_ch0  # (64, 64)

        # Build sign pattern s from metadata
        binding = int(str(meta.get("binding", "0")), 16)
        cw_hex = str(meta.get("codeword_hex", "00" * 64))
        cw = bytes.fromhex(cw_hex)
        w = int(meta.get("width", meta.get("grid_w", 64)))
        h = int(meta.get("height", meta.get("grid_h", 64)))

        bits = np.array(bytes_to_bits(cw), dtype=np.int8)
        bit_idx_map = build_bit_index_map(binding, w, h, len(bits), args.backend)
        sign_map = 2.0 * bits[bit_idx_map].astype(np.float32) - 1.0  # (64, 64)

        # Statistics
        eps_flat = epsilon.flatten()
        sign_flat = sign_map.flatten()

        eps_mean = float(np.mean(eps_flat))
        eps_std = float(np.std(eps_flat))
        r, p = scipy_stats.pearsonr(eps_flat, sign_flat)

        per_image_mean.append(eps_mean)
        per_image_std.append(eps_std)
        per_image_corr.append(float(r))
        per_image_corr_pval.append(float(p))

        # Subsample for global histogram (1000 random pixels per image)
        rng = np.random.RandomState(i)
        idx = rng.choice(len(eps_flat), size=min(1000, len(eps_flat)), replace=False)
        all_residuals.extend(eps_flat[idx].tolist())

        print(f"  [{i:3d}] mean={eps_mean:+.6f} std={eps_std:.4f} "
              f"corr={r:+.4f} (p={p:.4f}) inv={t_inv:.1f}s")

    # ===================================================================
    # Aggregate statistics
    # ===================================================================
    n = len(per_image_mean)
    print(f"\n{'='*60}")
    print(f"Aggregate results (n={n} images)")
    print(f"{'='*60}")

    means = np.array(per_image_mean)
    stds = np.array(per_image_std)
    corrs = np.array(per_image_corr)

    # Zero-mean verification
    grand_mean = float(np.mean(means))
    grand_se = float(np.std(means, ddof=1) / np.sqrt(n))
    ci_lo = grand_mean - 1.96 * grand_se
    ci_hi = grand_mean + 1.96 * grand_se
    print(f"\n--- E[ε] ≈ 0 verification ---")
    print(f"  Per-image mean of ε: grand mean = {grand_mean:+.6f}")
    print(f"  95% CI: [{ci_lo:+.6f}, {ci_hi:+.6f}]")
    print(f"  Contains 0: {'YES' if ci_lo <= 0 <= ci_hi else 'NO'}")

    # Weak correlation verification
    abs_corrs = np.abs(corrs)
    print(f"\n--- Corr(ε, s) ≈ 0 verification ---")
    print(f"  |Corr| stats: mean={np.mean(abs_corrs):.4f} "
          f"max={np.max(abs_corrs):.4f} median={np.median(abs_corrs):.4f}")
    n_sig = sum(1 for p in per_image_corr_pval if p < 0.05)
    print(f"  Significant at α=0.05: {n_sig}/{n} "
          f"(expected by chance: ~{0.05*n:.0f})")

    # ===================================================================
    # Save results
    # ===================================================================
    results = {
        "n_images": n,
        "zero_mean": {
            "grand_mean": grand_mean,
            "grand_se": grand_se,
            "ci_95_lo": ci_lo,
            "ci_95_hi": ci_hi,
            "contains_zero": bool(ci_lo <= 0 <= ci_hi),
            "per_image_mean": per_image_mean,
        },
        "weak_correlation": {
            "abs_corr_mean": float(np.mean(abs_corrs)),
            "abs_corr_max": float(np.max(abs_corrs)),
            "abs_corr_median": float(np.median(abs_corrs)),
            "n_significant_005": n_sig,
            "expected_by_chance": round(0.05 * n),
            "per_image_corr": per_image_corr,
            "per_image_pval": per_image_corr_pval,
        },
        "residual_stats": {
            "per_image_std_mean": float(np.mean(stds)),
            "per_image_std_min": float(np.min(stds)),
            "per_image_std_max": float(np.max(stds)),
        },
    }

    out_json = args.output_dir / "results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # ===================================================================
    # Generate figures
    # ===================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Figure 1: Global residual histogram
        fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
        residuals_arr = np.array(all_residuals)
        ax.hist(residuals_arr, bins=100, density=True, alpha=0.7,
                color="steelblue", edgecolor="none")
        # Overlay fitted normal
        mu, sigma_fit = float(np.mean(residuals_arr)), float(np.std(residuals_arr))
        x_range = np.linspace(mu - 4 * sigma_fit, mu + 4 * sigma_fit, 200)
        ax.plot(x_range, scipy_stats.norm.pdf(x_range, mu, sigma_fit),
                "r-", linewidth=1.5, label=f"$\\mathcal{{N}}({mu:.3f}, {sigma_fit:.3f}^2)$")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Inversion residual $\\epsilon_i$")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.set_title(f"DDIM inversion residual ($n={n}$ images, subsampled)")
        fig.tight_layout()
        fig_hist = args.output_dir / "fig_residual_hist.pdf"
        fig.savefig(fig_hist, dpi=150)
        plt.close(fig)
        print(f"Histogram saved to {fig_hist}")

        # Figure 2: Per-image |correlation| distribution
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

        # Left: per-image mean(ε) distribution
        axes[0].hist(means, bins=30, alpha=0.7, color="steelblue", edgecolor="none")
        axes[0].axvline(0, color="red", linestyle="--", linewidth=1.2)
        axes[0].set_xlabel("Per-image $\\mathbb{E}[\\epsilon_i]$")
        axes[0].set_ylabel("Count")
        axes[0].set_title("(a) Residual mean distribution")

        # Right: per-image |corr(ε, s)| distribution
        axes[1].hist(abs_corrs, bins=30, alpha=0.7, color="coral", edgecolor="none")
        axes[1].axvline(0.05, color="red", linestyle="--", linewidth=1.2,
                        label="$|r|=0.05$")
        axes[1].set_xlabel("$|\\mathrm{Corr}(\\epsilon, s)|$")
        axes[1].set_ylabel("Count")
        axes[1].set_title("(b) Residual-signal correlation")
        axes[1].legend(fontsize=9)

        fig.tight_layout()
        fig_corr = args.output_dir / "fig_assumption_verify.pdf"
        fig.savefig(fig_corr, dpi=150)
        plt.close(fig)
        print(f"Assumption figure saved to {fig_corr}")

    except ImportError:
        print("matplotlib not available, skipping figures")


if __name__ == "__main__":
    main()
