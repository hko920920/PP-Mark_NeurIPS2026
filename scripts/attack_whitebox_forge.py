#!/usr/bin/env python3
"""White-box PGD forgery attack against the PP-Mark Score detector.

The adversary is given the detector and its gradients, the cover image and the
public metadata of the claim being forged, and maximises the Score statistic
under an l-infinity budget. Forgery acceptance is ``score >= tau`` with ``tau``
calibrated at the target FPR. Following the reported protocol, acceptance is
evaluated under the any-step criterion: an image counts as accepted if the
score reaches ``tau`` at any optimization step, and the reported PSNR for an
accepted image is measured at its first-breach step.

The attack targets Score only. It does not and cannot produce an SP1 receipt,
so a forged image fails Accept verification regardless of its Score; run
``cli.py verifier --mode attest`` on the output to confirm this.

Reported configuration: ``--eps 2/255 --steps 150``, 100 images, with the
threshold file for the matching model. Gradients flow through the DDIM
inversion, so a run is dominated by inversion cost; on SDXL use
``--ddim-dtype bfloat16`` (float16 silently collapses the recovered latent to
zero) together with ``--ddim-grad-checkpointing``.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from ppmark_v03.config import GlobalConfig
from ppmark_v03.ddim_unet import DiffusersDDIMConfig, DiffusersDDIMInverter
from ppmark_v03.noise import build_bit_index_map
from ppmark_v03.payload import bytes_to_bits
from ppmark_v03.sampling import deterministic_sample

GRAD_LOSS_SCALE = 1e4
"""Loss scaling prevents low-precision gradient underflow; sign() is
scale-invariant, so positive scaling does not change the PGD direction."""


def _iter_manifest(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _resolve_path(raw: str, base: Path) -> Path:
    cand = Path(raw)
    if cand.is_absolute():
        return cand
    return (base / cand).resolve()


def _select_image_field(row: dict) -> Optional[str]:
    for key in ("cover_image", "image_path", "image_relpath", "image"):
        value = row.get(key)
        if value:
            return str(value)
    return None


def _select_metadata_field(row: dict) -> Optional[str]:
    for key in ("ref_metadata", "metadata_path", "metadata"):
        value = row.get(key)
        if value:
            return str(value)
    meta = row.get("meta")
    if isinstance(meta, dict):
        for key in ("metadata_path", "ref_metadata"):
            value = meta.get(key)
            if value:
                return str(value)
    return None


def _resolve_image_path(row: dict, manifest_dir: Path, image_root: Optional[Path]) -> Path:
    raw = _select_image_field(row)
    if not raw:
        raise RuntimeError("manifest row missing image path")
    return _resolve_path(raw, image_root if image_root else manifest_dir)


def _resolve_metadata_path(
    row: dict, manifest_dir: Path, metadata_root: Optional[Path]
) -> Optional[Path]:
    raw = _select_metadata_field(row)
    if not raw:
        return None
    return _resolve_path(raw, metadata_root if metadata_root else manifest_dir)


def _infer_relpath(path: Path, root: Optional[Path]) -> Path:
    if root and path.is_absolute():
        try:
            return path.relative_to(root)
        except ValueError:
            pass
    parts = path.parts
    for marker in ("images", "covers", "forged", "clean"):
        if marker in parts:
            idx = parts.index(marker)
            return Path(*parts[idx + 1 :])
    return Path(path.name)


def _load_image_array(path: Path, image_size: int) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if image_size and img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def _load_image_tensor(path: Path, image_size: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    arr = _load_image_array(path, image_size)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).to(device=device, dtype=dtype)
    return tensor.unsqueeze(0)


def _save_image_tensor(tensor: torch.Tensor, path: Path) -> None:
    img = tensor.detach().clamp(0.0, 1.0)[0].permute(1, 2, 0).float().cpu().numpy()
    img_u8 = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img_u8).save(path)


def _compute_psnr(original: torch.Tensor, forged: torch.Tensor) -> float:
    mse = torch.mean((original - forged) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return -10.0 * math.log10(mse)


def _compute_linf(original: torch.Tensor, forged: torch.Tensor) -> float:
    return float(torch.max(torch.abs(original - forged)).item())


def _write_csv(path: Path, rows: List[dict], header: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def _load_threshold_value(payload: dict, keys: List[str]) -> float:
    for key in keys:
        if key in payload:
            return float(payload[key])
    raise ValueError(f"threshold value not found in payload keys={keys}")


def _get_inverter(
    cache: Dict[tuple, DiffusersDDIMInverter],
    model_id: str,
    scheduler: str,
    device: str,
    dtype: str,
    steps: int,
    attn_slicing: bool,
    vae_slicing: bool,
    grad_checkpointing: bool,
    xformers: bool,
) -> DiffusersDDIMInverter:
    key = (model_id, scheduler, device, dtype, steps, attn_slicing, vae_slicing, grad_checkpointing, xformers)
    inverter = cache.get(key)
    if inverter is not None:
        return inverter
    cfg = DiffusersDDIMConfig(
        model_id=model_id,
        device=device,
        torch_dtype=dtype,
        num_steps=steps,
        scheduler=scheduler,
        attn_slicing=attn_slicing,
        vae_slicing=vae_slicing,
        grad_checkpointing=grad_checkpointing,
        xformers=xformers,
    )
    inverter = DiffusersDDIMInverter(cfg)
    inverter.pipe.vae.requires_grad_(False)
    inverter.pipe.unet.requires_grad_(False)
    inverter.pipe.vae.eval()
    inverter.pipe.unet.eval()
    cache[key] = inverter
    return inverter


@dataclass
class PPMarkSampleCache:
    sign_map: torch.Tensor
    ys: torch.Tensor
    xs: torch.Tensor


def _ppmark_score(
    context: dict,
    image: torch.Tensor,
    meta: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable Score statistic for one image against one public claim."""
    cfg = context["cfg"]
    binding_hex = str(meta.get("binding", ""))
    codeword_hex = str(meta.get("codeword_hex", ""))
    if not binding_hex or not codeword_hex:
        raise RuntimeError("PP-Mark metadata missing binding or codeword_hex")
    binding = int(binding_hex, 16)
    codeword = bytes.fromhex(codeword_hex)
    alpha_effective = float(meta.get("alpha_effective", cfg.image.alpha))
    sigma_meta = float(meta.get("init_noise_sigma", 1.0))

    model_id = context["model_id_override"] or meta.get("model_id")
    if not model_id:
        raise RuntimeError("Missing model_id for PP-Mark DDIM inversion")
    scheduler = str(meta.get("scheduler", "ddim"))
    steps = context["ddim_steps_override"] or int(meta.get("num_inference_steps", 50))
    inverter = _get_inverter(
        context["inverter_cache"],
        str(model_id),
        scheduler,
        context["ddim_device"],
        context["ddim_dtype"],
        steps,
        bool(context.get("ddim_attn_slicing", False)),
        bool(context.get("ddim_vae_slicing", False)),
        bool(context.get("ddim_grad_checkpointing", False)),
        bool(context.get("ddim_xformers", False)),
    )
    inverter.prompt = str(meta.get("prompt", ""))
    inverter.negative_prompt = str(meta.get("negative_prompt", ""))
    inverter.cfg.guidance_scale = float(meta.get("guidance_scale", 7.5))

    latents = inverter.invert_latents_torch(image)
    latent = latents[:, 0]
    if sigma_meta != 0:
        latent = latent / sigma_meta

    cache_key = (binding, codeword_hex)
    cached = context["sample_cache"].get(cache_key)
    if cached is None:
        bits = np.array(bytes_to_bits(codeword), dtype=np.int8)
        bit_idx_map = build_bit_index_map(
            binding,
            cfg.image.width,
            cfg.image.height,
            len(bits),
            backend=cfg.zk.sample_backend,
        )
        sign_map = 2.0 * bits[bit_idx_map].astype(np.float32) - 1.0
        sample_count = int(meta.get("sample_count") or cfg.image.sample_count())
        sample_set = deterministic_sample(
            total_pixels=cfg.image.total_pixels,
            width=cfg.image.width,
            height=cfg.image.height,
            key_material=codeword,
            count=sample_count,
        )
        sample_indices = np.array(sample_set.indices, dtype=np.int64)
        ys = sample_indices // cfg.image.width
        xs = sample_indices % cfg.image.width
        cached = PPMarkSampleCache(
            sign_map=torch.from_numpy(sign_map).to(device=latent.device, dtype=latent.dtype),
            ys=torch.from_numpy(ys).to(device=latent.device),
            xs=torch.from_numpy(xs).to(device=latent.device),
        )
        context["sample_cache"][cache_key] = cached
    sign_map = cached.sign_map.to(device=latent.device, dtype=latent.dtype)
    ys = cached.ys.to(device=latent.device)
    xs = cached.xs.to(device=latent.device)
    watermark = alpha_effective * sign_map[ys, xs]
    denom = torch.linalg.norm(watermark) + 1e-9
    samples = latent[:, ys, xs]
    raw = (samples * watermark).sum(dim=1) / denom
    objective = raw.mean()
    score = raw.abs()
    return objective, score, raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="White-box PGD forgery attack against the PP-Mark Score detector."
    )
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest of cover images.")
    parser.add_argument("--output-root", required=True, help="Directory for forged images and metrics.")
    parser.add_argument("--config", required=True, help="PP-Mark configuration JSON.")
    parser.add_argument(
        "--threshold-file",
        required=True,
        help="Threshold JSON providing tau_score for the matching model.",
    )
    parser.add_argument("--image-root", default="", help="Base path for relative image paths.")
    parser.add_argument("--metadata-root", default="", help="Base path for relative metadata paths.")
    parser.add_argument(
        "--opening-metadata",
        action="store_true",
        help="Prefer metadata_opening.json beside the metadata file when present.",
    )
    parser.add_argument("--steps", type=int, default=150, help="PGD steps (reported: 150).")
    parser.add_argument(
        "--eps", type=float, default=2.0 / 255.0, help="l-infinity budget (reported: 2/255)."
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.0,
        help="PGD step size; defaults to eps / steps * 2 when not given.",
    )
    parser.add_argument(
        "--early-stop",
        action="store_true",
        help="Stop an image as soon as it reaches tau. Acceptance is any-step either way.",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ddim-model-id", default="", help="Overrides the model id in metadata.")
    parser.add_argument("--ddim-device", default="cuda")
    parser.add_argument(
        "--ddim-dtype", default="float32", help="Use bfloat16 on SDXL; float16 is unusable."
    )
    parser.add_argument("--ddim-steps", type=int, default=0, help="Overrides the metadata step count.")
    parser.add_argument("--ddim-attn-slicing", action="store_true")
    parser.add_argument("--ddim-vae-slicing", action="store_true")
    parser.add_argument("--ddim-grad-checkpointing", action="store_true")
    parser.add_argument("--ddim-xformers", action="store_true")
    parser.add_argument("--metrics-out", default="", help="CSV metrics output path.")
    parser.add_argument("--summary-out", default="", help="JSON summary output path.")
    parser.add_argument("--manifest-out", default="", help="JSONL manifest of forged images.")
    parser.add_argument("--trace-path", default="", help="CSV per-step trace output path.")
    parser.add_argument("--trace-all", action="store_true", help="Record a trace for every image.")
    parser.add_argument("--trace-idx", type=int, default=-1, help="Record a trace for one image only.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest_dir = manifest_path.parent
    image_root = Path(args.image_root).resolve() if args.image_root else None
    metadata_root = Path(args.metadata_root).resolve() if args.metadata_root else None
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    steps = int(args.steps)
    if steps <= 0:
        raise SystemExit("--steps must be positive")
    step_size = args.step_size if args.step_size > 0 else float(args.eps / max(1, steps) * 2.0)

    threshold_payload = json.loads(Path(args.threshold_file).read_text(encoding="utf-8"))
    tau = _load_threshold_value(threshold_payload, ["tau_score", "tau", "threshold", "threshold_score"])

    cfg = GlobalConfig.load(Path(args.config))
    context: dict = {
        "cfg": cfg,
        "ddim_device": args.ddim_device,
        "ddim_dtype": args.ddim_dtype,
        "ddim_steps_override": int(args.ddim_steps) if args.ddim_steps else 0,
        "model_id_override": args.ddim_model_id,
        "ddim_attn_slicing": args.ddim_attn_slicing,
        "ddim_vae_slicing": args.ddim_vae_slicing,
        "ddim_grad_checkpointing": args.ddim_grad_checkpointing,
        "ddim_xformers": args.ddim_xformers,
        "inverter_cache": {},
        "sample_cache": {},
    }
    image_size = int(getattr(cfg.image, "pixel_width", 512))

    rows = list(_iter_manifest(manifest_path))
    if args.start_index:
        rows = rows[args.start_index :]
    if args.max_images:
        rows = rows[: args.max_images]
    if not rows:
        raise SystemExit("Manifest has no rows to process.")

    print(
        f"[whitebox] images={len(rows)} steps={steps} eps={args.eps:.6f} "
        f"step_size={step_size:.6f} tau={tau:.4f}",
        flush=True,
    )

    metrics_rows: List[dict] = []
    trace_rows: List[dict] = []
    manifest_rows: List[dict] = []

    for local_idx, row in enumerate(rows):
        image_path = _resolve_image_path(row, manifest_dir, image_root)
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        meta_path = _resolve_metadata_path(row, manifest_dir, metadata_root)
        if args.opening_metadata and meta_path:
            opening_path = Path(meta_path).with_name("metadata_opening.json")
            if opening_path.exists():
                meta_path = opening_path
        if meta_path is None:
            raise RuntimeError("PP-Mark requires per-image metadata")
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))

        rel_path = _infer_relpath(image_path, image_root)
        output_path = output_root / rel_path
        if output_path.exists() and not args.overwrite:
            continue

        # Keep the cover and the iterate in fp32 for PGD accumulation precision:
        # bf16 ULP at pixel values >= 0.5 is 0.00390625, so a step of 1/255 would
        # round to zero. The inversion pipe casts internally and the gradient
        # still flows through that differentiable cast.
        original = _load_image_tensor(image_path, image_size, args.device, torch.float32)
        forged = original.clone()

        score_max = float("-inf")
        breach_step: Optional[int] = None
        psnr_at_breach: Optional[float] = None
        linf_at_breach: Optional[float] = None
        score_scalar = float("nan")
        objective = None
        raw_score = None

        for step in range(steps + 1):
            forged.requires_grad_(True)
            objective, det_score, raw_score = _ppmark_score(context, forged, meta)
            score_scalar = float(det_score.detach().mean().item())

            if score_scalar > score_max:
                score_max = score_scalar
            if breach_step is None and score_scalar >= tau:
                breach_step = step
                psnr_at_breach = _compute_psnr(original, forged)
                linf_at_breach = _compute_linf(original, forged)

            if args.trace_all or local_idx == args.trace_idx:
                trace_rows.append(
                    {
                        "idx": int(row.get("idx", local_idx)),
                        "step": step,
                        "objective_score": float(objective.detach().mean().item()),
                        "score": score_scalar,
                        "psnr": _compute_psnr(original, forged),
                        "linf": _compute_linf(original, forged),
                    }
                )

            if step % 10 == 0 or step == steps:
                print(
                    f"  [step {step:3d}/{steps}] score={score_scalar:.4f} | "
                    f"{'BREACH' if score_scalar >= tau else 'no-breach'} | "
                    f"psnr={_compute_psnr(original, forged):.1f}",
                    flush=True,
                )

            if args.early_stop and breach_step is not None:
                forged = forged.detach()
                break
            if step == steps:
                break

            grad = torch.autograd.grad(
                objective * GRAD_LOSS_SCALE, forged, retain_graph=False, create_graph=False
            )[0]
            if grad is None:
                break
            forged = forged + step_size * torch.sign(grad)
            delta = torch.clamp(forged - original, min=-args.eps, max=args.eps)
            forged = torch.clamp(original + delta, 0.0, 1.0).detach()

        _save_image_tensor(forged, output_path)
        final_psnr = _compute_psnr(original, forged)
        final_linf = _compute_linf(original, forged)
        detected_any = breach_step is not None
        # Reported PSNR is taken at the submitted-image step: the first-breach
        # step for an acceptance, the final step otherwise.
        reported_psnr = psnr_at_breach if detected_any else final_psnr
        reported_linf = linf_at_breach if detected_any else final_linf

        metrics_rows.append(
            {
                "idx": int(row.get("idx", local_idx)),
                "image_path": str(rel_path),
                "source_path": str(image_path.resolve()),
                "steps": steps,
                "breach_step": "" if breach_step is None else int(breach_step),
                "score_final": score_scalar,
                "score_max": score_max,
                "objective_score": float(raw_score.detach().mean().item()) if raw_score is not None else "",
                "detected_any": int(detected_any),
                "psnr": reported_psnr,
                "linf": reported_linf,
                "psnr_final": final_psnr,
                "linf_final": final_linf,
            }
        )

        if args.manifest_out:
            out_row = {
                "idx": int(row.get("idx", local_idx)),
                "attack": "whitebox_ppmark",
                "label": str(row.get("label", "forged")),
                "image_path": str(rel_path),
                "source_path": str(image_path.resolve()),
                "metadata_path": str(Path(meta_path).resolve()),
                "meta": {"steps": steps, "eps": float(args.eps), "step_size": float(step_size)},
            }
            manifest_rows.append(out_row)

        print(
            f"[whitebox] {local_idx + 1}/{len(rows)} | score_max={score_max:.4f} | "
            f"breach={'-' if breach_step is None else breach_step}/{steps} | "
            f"{'ACCEPTED' if detected_any else 'rejected'} | psnr={reported_psnr:.1f}",
            flush=True,
        )

        metrics_header = [
            "idx",
            "image_path",
            "source_path",
            "steps",
            "breach_step",
            "score_final",
            "score_max",
            "objective_score",
            "detected_any",
            "psnr",
            "linf",
            "psnr_final",
            "linf_final",
        ]
        # Incremental flush so a long run stays inspectable.
        _write_csv(output_root / "metrics_live.csv", metrics_rows, metrics_header)

        del original, forged, det_score
        try:
            del objective, raw_score, grad
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics_header = [
        "idx",
        "image_path",
        "source_path",
        "steps",
        "breach_step",
        "score_final",
        "score_max",
        "objective_score",
        "detected_any",
        "psnr",
        "linf",
        "psnr_final",
        "linf_final",
    ]
    metrics_path = Path(args.metrics_out) if args.metrics_out else output_root / "metrics.csv"
    _write_csv(metrics_path, metrics_rows, metrics_header)

    if trace_rows:
        trace_path = Path(args.trace_path) if args.trace_path else output_root / "trace_scores.csv"
        _write_csv(
            trace_path, trace_rows, ["idx", "step", "objective_score", "score", "psnr", "linf"]
        )

    if manifest_rows:
        manifest_out = Path(args.manifest_out)
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        with manifest_out.open("w", encoding="utf-8") as handle:
            for entry in manifest_rows:
                handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    accepted = [row for row in metrics_rows if int(row["detected_any"]) == 1]
    steps_to_fa = [int(row["breach_step"]) for row in accepted if row["breach_step"] != ""]
    psnrs = [float(row["psnr"]) for row in metrics_rows if np.isfinite(float(row["psnr"]))]
    summary = {
        "method": "ppmark",
        "count": len(metrics_rows),
        "tau": tau,
        "threshold_file": str(Path(args.threshold_file).resolve()),
        "steps": steps,
        "eps": float(args.eps),
        "step_size": float(step_size),
        "acceptance_criterion": "any-step (score >= tau at any optimization step)",
        "far_any_step": float(len(accepted) / len(metrics_rows)) if metrics_rows else 0.0,
        "avg_steps_to_fa": float(np.mean(steps_to_fa)) if steps_to_fa else None,
        "score_max_mean": float(np.mean([float(r["score_max"]) for r in metrics_rows]))
        if metrics_rows
        else 0.0,
        "psnr_mean": float(np.mean(psnrs)) if psnrs else 0.0,
        "psnr_median": float(np.median(psnrs)) if psnrs else 0.0,
        "note": (
            "Score-only attack. A forged image carries no valid SP1 receipt and fails "
            "Accept verification; confirm with cli.py verifier --mode attest."
        ),
    }
    summary_path = Path(args.summary_out) if args.summary_out else output_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(
        f"[whitebox] FAR(any-step)={summary['far_any_step']:.4f} "
        f"avg_steps_to_FA={summary['avg_steps_to_fa']} psnr_mean={summary['psnr_mean']:.2f}"
    )
    print(f"[whitebox] wrote {metrics_path} and {summary_path}")


if __name__ == "__main__":
    main()
