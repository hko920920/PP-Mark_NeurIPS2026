#!/usr/bin/env python3
"""
SP1 sample_count sweet-spot sweep with a fixed attack set.

Runs the prover for each k, applies the configured attacks, then measures:
- RS decode success (exact message match)
- correlation score distribution (mean, p01, p05)
- sp1_prover_sec and receipt_verify_sec
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
import os
import subprocess
import time
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmark_v03.config import GlobalConfig  # noqa: E402
from ppmark_v03.attestation import producer_key_commitment  # noqa: E402
from ppmark_v03.ddim_unet import DiffusersDDIMConfig, DiffusersDDIMInverter  # noqa: E402
from ppmark_v03.payload import bytes_to_bits, bits_to_bytes  # noqa: E402
from ppmark_v03.rs import ReedSolomonCodec  # noqa: E402
from ppmark_v03.sampling import deterministic_sample  # noqa: E402
from ppmark_v03.sp1_runner import SP1Paths, verify_sp1_receipt  # noqa: E402
from ppmark_v03.noise import hash_uniforms, build_bit_index_map  # noqa: E402
from ppmark_v03.tables import InverseCDFTable  # noqa: E402


@dataclass(slots=True)
class SyncResult:
    dx: int
    dy: int
    theta: float
    score: float
    raw_score: float


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_config(base_path: Path, out_path: Path, sample_count: int) -> None:
    payload = _load_json(base_path)
    payload.setdefault("image", {})
    payload["image"]["sample_count"] = sample_count
    payload.setdefault("tables", {})
    lut = Path(payload["tables"].get("inverse_cdf_path", "tables/invcdf_gaussian.bin"))
    if not lut.is_absolute():
        lut = (base_path.parent / lut).resolve()
    payload["tables"]["inverse_cdf_path"] = str(lut)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _run(cmd: List[str], env: dict | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def _should_skip_prover(run_dir: Path, prompt: str, seed_hex: str | None, secret_hex: str | None, sample_count: int) -> bool:
    meta_path = run_dir / "metadata.json"
    sp1_receipt = run_dir / "sp1" / "attest_receipt.bin"
    image_png = run_dir / "watermarked.png"
    if not (meta_path.exists() and sp1_receipt.exists() and image_png.exists()):
        return False
    try:
        meta = _load_json(meta_path)
    except Exception:
        return False
    if meta.get("prompt") != prompt:
        return False
    if meta.get("sample_count") != sample_count:
        return False
    if seed_hex and meta.get("seed_hex", "").lower() != seed_hex.lower():
        return False
    if secret_hex:
        commitment = meta.get("producer_key_commitment")
        if not commitment:
            return False
        try:
            expected = "0x" + producer_key_commitment(
                bytes.fromhex(secret_hex.lower().removeprefix("0x").zfill(64))
            ).hex()
        except ValueError:
            return False
        if str(commitment).lower() != expected:
            return False
    return True


def _parse_counts(raw: str) -> List[int]:
    parts = [p for p in raw.replace(",", " ").split(" ") if p]
    return [int(p) for p in parts]


def _shift_indices(indices: np.ndarray, width: int, height: int, dx: int, dy: int) -> np.ndarray:
    xs = indices % width
    ys = indices // width
    xs_shift = xs + dx
    ys_shift = ys + dy
    mask = (xs_shift >= 0) & (xs_shift < width) & (ys_shift >= 0) & (ys_shift < height)
    shifted = np.full_like(indices, -1)
    shifted[mask] = ys_shift[mask] * width + xs_shift[mask]
    return shifted


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("cv2 is required for rotation/shift sync search") from exc
    return cv2


def _rotate_latent(latent: np.ndarray, degrees: float, cv2_module) -> np.ndarray:
    if abs(degrees) <= 1e-6:
        return latent
    height, width = latent.shape
    center = (width / 2.0, height / 2.0)
    rot_mat = cv2_module.getRotationMatrix2D(center, degrees, 1.0)
    return cv2_module.warpAffine(latent, rot_mat, (width, height), flags=cv2_module.INTER_LINEAR, borderMode=cv2_module.BORDER_REFLECT)


def _translate_latent(latent: np.ndarray, dx: int, dy: int, cv2_module) -> np.ndarray:
    if dx == 0 and dy == 0:
        return latent
    height, width = latent.shape
    mat = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    return cv2_module.warpAffine(latent, mat, (width, height), flags=cv2_module.INTER_LINEAR, borderMode=cv2_module.BORDER_REFLECT)


def _rotation_sweep(max_angle: float, step: float) -> List[float]:
    if max_angle <= 0 or step <= 0:
        return [0.0]
    angles: List[float] = []
    angle = -max_angle
    while angle <= max_angle + 1e-6:
        angles.append(float(round(angle, 3)))
        angle += step
    if 0.0 not in angles:
        angles.append(0.0)
    return sorted(set(angles))


def _rotation_window(center: float, span: float, step: float, max_abs: float) -> List[float]:
    if span <= 0 or step <= 0:
        return [float(round(center, 3))]
    start = center - span
    end = center + span
    angles: List[float] = []
    angle = start
    while angle <= end + 1e-6:
        clipped = max(-max_abs, min(max_abs, angle))
        angles.append(float(round(clipped, 3)))
        angle += step
    if center not in angles:
        angles.append(float(round(center, 3)))
    return sorted(set(angles))


def _top_candidates(candidates: List[dict], limit: int = 3) -> List[dict]:
    return sorted(candidates, key=lambda item: item["score"], reverse=True)[:limit]


def _sync_search_for_angles(
    latent: np.ndarray,
    sample_indices: np.ndarray,
    sign_map: np.ndarray,
    alpha_effective: float,
    angles: List[float],
    max_shift: int,
) -> tuple[SyncResult, np.ndarray, List[dict]]:
    height, width = latent.shape
    shift_range = [0] if max_shift <= 0 else list(range(-max_shift, max_shift + 1))
    cv2 = _require_cv2() if any(abs(a) > 1e-6 for a in angles) else None
    best = SyncResult(dx=0, dy=0, theta=0.0, score=-1e30, raw_score=-1e30)
    best_view = latent
    candidates: List[dict] = []
    for ang in angles:
        view = _rotate_latent(latent, ang, cv2) if cv2 is not None else latent
        best_raw = -1e30
        best_score = -1e30
        best_dx = 0
        best_dy = 0
        for dy in shift_range:
            for dx in shift_range:
                shifted = _shift_indices(sample_indices, width, height, dx, dy)
                if np.any(shifted < 0):
                    continue
                ys = shifted // width
                xs = shifted % width
                samples = view[ys, xs]
                watermark = alpha_effective * sign_map[ys, xs]
                denom = float(np.linalg.norm(watermark) + 1e-9)
                raw = float(np.dot(samples.astype(np.float64), watermark.astype(np.float64)) / denom) if denom > 0 else -1e30
                score = abs(raw)
                if score > best_score:
                    best_score = score
                    best_raw = raw
                    best_dx = dx
                    best_dy = dy
        candidates.append(
            {
                "angle": ang,
                "score": best_score,
                "raw_score": best_raw,
                "dx": best_dx,
                "dy": best_dy,
            }
        )
        if best_score > best.score:
            best = SyncResult(dx=best_dx, dy=best_dy, theta=ang, score=best_score, raw_score=best_raw)
            best_view = view
    return best, best_view, candidates


def _build_bit_index_map(binding: int, width: int, height: int, bit_len: int, backend: str) -> np.ndarray:
    return build_bit_index_map(binding, width, height, bit_len, backend=backend)


def _decode_bits_hashed(
    latent: np.ndarray,
    binding: int,
    config: GlobalConfig,
    inverse_cdf: InverseCDFTable,
    codec: ReedSolomonCodec,
    bit_idx_map: np.ndarray,
    gaussian: np.ndarray,
) -> tuple[bytes | None, bytes, np.ndarray, dict]:
    latent2d = np.asarray(latent, dtype=np.float32)
    if latent2d.shape != (config.image.height, config.image.width):
        raise ValueError("latent shape does not match config resolution")
    flat = latent2d.reshape(-1)[: config.image.total_pixels]
    alpha = float(config.image.alpha)
    if alpha == 0.0:
        raise ValueError("alpha must be non-zero for decoding")
    residual = (flat - gaussian) / alpha
    indices = bit_idx_map.reshape(-1)[: config.image.total_pixels]
    bit_len = int(codec.n * 8)
    weights = np.bincount(indices, weights=residual, minlength=bit_len)
    counts = np.bincount(indices, minlength=bit_len)
    mean_residual = weights / np.maximum(counts, 1)
    threshold = float(np.median(mean_residual))
    bits = (mean_residual >= threshold).astype(np.uint8)
    codeword = bits_to_bytes(bits.tolist())
    message = None
    bit_flip = False
    try:
        message = codec.decode(codeword)
    except Exception:
        message = None
    if message is None:
        flipped = (1 - bits).astype(np.uint8)
        flipped_codeword = bits_to_bytes(flipped.tolist())
        try:
            message = codec.decode(flipped_codeword)
            bits = flipped
            codeword = flipped_codeword
            bit_flip = True
        except Exception:
            message = None
    stats = {
        "residual_min": float(np.min(residual)),
        "residual_max": float(np.max(residual)),
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "mean_residual_min": float(np.min(mean_residual)),
        "mean_residual_max": float(np.max(mean_residual)),
        "mean_residual_mean": float(np.mean(mean_residual)),
        "mean_residual_std": float(np.std(mean_residual)),
        "threshold": threshold,
        "alpha_used": alpha,
        "decode_uses_abs": False,
        "bit_flip_applied": bit_flip,
    }
    return message, codeword, bits, stats


def _dump_bit_artifacts(
    out_dir: Path,
    pred_bits: np.ndarray,
    gt_bits: np.ndarray,
    pred_bytes: bytes,
    gt_bytes: bytes,
    bit_order: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "pred_bits.npy", pred_bits.astype(np.uint8))
    np.save(out_dir / "gt_bits.npy", gt_bits.astype(np.uint8))
    (out_dir / "pred_bytes.bin").write_bytes(pred_bytes)
    (out_dir / "gt_bytes.bin").write_bytes(gt_bytes)
    meta = {
        "bit_order": bit_order,
        "pred_bits_len": int(len(pred_bits)),
        "gt_bits_len": int(len(gt_bits)),
        "pred_bytes_len": int(len(pred_bytes)),
        "gt_bytes_len": int(len(gt_bytes)),
        "notes": "bits_to_bytes uses MSB-first packing",
    }
    (out_dir / "dump_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _sync_search_watermark(
    latent: np.ndarray,
    sample_indices: np.ndarray,
    sign_map: np.ndarray,
    alpha_effective: float,
    max_shift: int,
    max_rotation: float,
    rotation_step: float,
    rotation_strategy: str = "auto",
) -> Tuple[SyncResult, np.ndarray, dict]:
    search_stats: dict = {}
    if rotation_strategy not in {"auto", "grid", "coarse_to_fine"}:
        raise ValueError(f"Unsupported rotation_strategy: {rotation_strategy}")
    if rotation_strategy == "auto":
        rotation_strategy = "coarse_to_fine" if max_rotation > 10 else "grid"
    if max_rotation <= 0:
        rotation_strategy = "grid"

    if rotation_strategy == "coarse_to_fine":
        coarse_step = 15.0
        coarse_max = max(90.0, float(max_rotation))
        coarse_angles = _rotation_sweep(coarse_max, coarse_step)
        coarse_best, _, coarse_candidates = _sync_search_for_angles(
            latent=latent,
            sample_indices=sample_indices,
            sign_map=sign_map,
            alpha_effective=alpha_effective,
            angles=coarse_angles,
            max_shift=0,
        )
        fine_span = 7.0
        fine_step = 1.0
        fine_angles = _rotation_window(coarse_best.theta, fine_span, fine_step, coarse_max)
        fine_best, best_view, fine_candidates = _sync_search_for_angles(
            latent=latent,
            sample_indices=sample_indices,
            sign_map=sign_map,
            alpha_effective=alpha_effective,
            angles=fine_angles,
            max_shift=max_shift,
        )
        best = fine_best
        search_stats = {
            "rotation_search_strategy": "coarse_to_fine",
            "best_rotation_deg": best.theta,
            "coarse_candidates": len(coarse_angles),
            "fine_candidates": len(fine_angles),
            "rotation_top3_scores": _top_candidates(fine_candidates),
            "rotation_coarse_top3_scores": _top_candidates(coarse_candidates),
        }
        return best, best_view, search_stats

    angles = [0.0]
    if max_rotation > 0 and rotation_step > 0:
        angle = rotation_step
        while angle <= max_rotation + 1e-6:
            angles.extend([angle, -angle])
            angle += rotation_step
    best, best_view, candidates = _sync_search_for_angles(
        latent=latent,
        sample_indices=sample_indices,
        sign_map=sign_map,
        alpha_effective=alpha_effective,
        angles=sorted(set(angles)),
        max_shift=max_shift,
    )
    search_stats = {
        "rotation_search_strategy": "grid",
        "best_rotation_deg": best.theta,
        "coarse_candidates": len(angles),
        "fine_candidates": 0,
        "rotation_top3_scores": _top_candidates(candidates),
    }
    return best, best_view, search_stats


def _attack_resize(image: Image.Image, scale: float) -> Image.Image:
    w, h = image.size
    down = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    resized = image.resize(down, resample=Image.BICUBIC)
    return resized.resize((w, h), resample=Image.BICUBIC)


def _attack_jpeg(image: Image.Image, quality: int) -> Image.Image:
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"ppmark_q{quality}.jpg"
    image.save(tmp, format="JPEG", quality=quality, subsampling=2)
    with Image.open(tmp) as img:
        out = img.convert("RGB")
    tmp.unlink(missing_ok=True)
    return out


def _attack_rotate(image: Image.Image, degrees: float) -> Image.Image:
    return image.rotate(degrees, resample=Image.BICUBIC, expand=False)


def _attack_blur(image: Image.Image, kernel: int) -> Image.Image:
    radius = max(1, int(round(kernel / 2)))
    return image.filter(ImageFilter.BoxBlur(radius))


def _attack_noise(image: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    arr = np.array(image, dtype=np.float32) / 255.0
    noise = rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def _attack_brightness_jitter(image: Image.Image, jitter: float, rng: np.random.Generator) -> Image.Image:
    factor = 1.0 + float(rng.uniform(-jitter, jitter))
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)


def _random_crop_box(width: int, height: int, keep_ratio: float, rng: np.random.Generator) -> Tuple[int, int, int, int]:
    scale = math.sqrt(keep_ratio)
    crop_w = max(1, int(round(width * scale)))
    crop_h = max(1, int(round(height * scale)))
    max_x = max(0, width - crop_w)
    max_y = max(0, height - crop_h)
    x0 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
    y0 = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    return (x0, y0, x0 + crop_w, y0 + crop_h)


def _attack_crop_scale(image: Image.Image, keep_ratio: float, rng: np.random.Generator) -> Image.Image:
    width, height = image.size
    box = _random_crop_box(width, height, keep_ratio, rng)
    cropped = image.crop(box)
    return cropped.resize((width, height), resample=Image.BICUBIC)


def _crop_box_grid(width: int, height: int, keep_ratio: float, grid: int = 3) -> List[Tuple[int, int, int, int]]:
    scale = math.sqrt(keep_ratio)
    crop_w = max(1, int(round(width * scale)))
    crop_h = max(1, int(round(height * scale)))
    max_x = max(0, width - crop_w)
    max_y = max(0, height - crop_h)
    if grid <= 1:
        xs = [max_x // 2]
        ys = [max_y // 2]
    else:
        xs = [int(round(i * max_x / (grid - 1))) for i in range(grid)]
        ys = [int(round(i * max_y / (grid - 1))) for i in range(grid)]
    boxes = []
    for y0 in ys:
        for x0 in xs:
            boxes.append((x0, y0, x0 + crop_w, y0 + crop_h))
    return list(dict.fromkeys(boxes))


def _crop_box_refine(
    base_box: Tuple[int, int, int, int],
    width: int,
    height: int,
    keep_ratio: float,
    step_frac: float = 0.05,
    grid: int = 5,
) -> List[Tuple[int, int, int, int]]:
    scale = math.sqrt(keep_ratio)
    crop_w = max(1, int(round(width * scale)))
    crop_h = max(1, int(round(height * scale)))
    max_x = max(0, width - crop_w)
    max_y = max(0, height - crop_h)
    base_x0, base_y0, _, _ = base_box
    step_x = int(round(width * step_frac))
    step_y = int(round(height * step_frac))
    if grid <= 1:
        offsets_x = [0]
        offsets_y = [0]
    else:
        offsets = np.linspace(-step_x, step_x, grid, dtype=np.int64)
        offsets_x = [int(v) for v in offsets]
        offsets = np.linspace(-step_y, step_y, grid, dtype=np.int64)
        offsets_y = [int(v) for v in offsets]
    boxes = []
    for dy in offsets_y:
        for dx in offsets_x:
            x0 = min(max(base_x0 + dx, 0), max_x)
            y0 = min(max(base_y0 + dy, 0), max_y)
            boxes.append((x0, y0, x0 + crop_w, y0 + crop_h))
    return list(dict.fromkeys(boxes))


def _apply_crop_and_resize(image: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    width, height = image.size
    cropped = image.crop(box)
    return cropped.resize((width, height), resample=Image.BICUBIC)


def _geom_variants(image: Image.Image) -> List[Tuple[str, Image.Image]]:
    return [
        ("shift_x+2", ImageChops.offset(image, 2, 0)),
        ("shift_x-2", ImageChops.offset(image, -2, 0)),
        ("shift_y+2", ImageChops.offset(image, 0, 2)),
        ("shift_y-2", ImageChops.offset(image, 0, -2)),
        ("rot_+1", image.rotate(1.0, resample=Image.BICUBIC, expand=False)),
        ("rot_-1", image.rotate(-1.0, resample=Image.BICUBIC, expand=False)),
    ]


def _crop_search_best(
    image: Image.Image,
    encode_fn,
    sample_indices: np.ndarray,
    sign_map: np.ndarray,
    alpha_effective: float,
    max_shift: int,
    max_rotation: float,
    rotation_step: float,
    rotation_strategy: str,
    sigma_meta: float,
    keep_ratio: float = 0.75,
    refine: bool = False,
    grid: int = 3,
    refine_grid: int = 5,
    refine_step: float = 0.05,
) -> tuple[SyncResult, np.ndarray, dict]:
    width, height = image.size
    coarse_boxes = _crop_box_grid(width, height, keep_ratio, grid=grid)
    candidate_scores: List[dict] = []
    latent_cache: Dict[Tuple[int, int, int, int], np.ndarray] = {}

    def _encode_crop(box: Tuple[int, int, int, int]) -> np.ndarray:
        if box in latent_cache:
            return latent_cache[box]
        cropped = _apply_crop_and_resize(image, box)
        latent = encode_fn(np.array(cropped, dtype=np.float32))
        if np.isnan(latent).any() or np.isinf(latent).any():
            raise RuntimeError("DDIM inversion produced NaN/Inf latents during crop search")
        if sigma_meta != 0:
            latent = latent / sigma_meta
        latent_cache[box] = latent
        return latent

    for box in coarse_boxes:
        latent = _encode_crop(box)
        sync_res, _, _ = _sync_search_watermark(
            latent=latent,
            sample_indices=sample_indices,
            sign_map=sign_map,
            alpha_effective=alpha_effective,
            max_shift=0,
            max_rotation=0.0,
            rotation_step=1.0,
        )
        candidate_scores.append({"box": box, "score": sync_res.score})

    best_box = max(candidate_scores, key=lambda item: item["score"])["box"]
    refine_boxes: List[Tuple[int, int, int, int]] = []
    if refine:
        refine_boxes = _crop_box_refine(best_box, width, height, keep_ratio, step_frac=refine_step, grid=refine_grid)
        for box in refine_boxes:
            latent = _encode_crop(box)
            sync_res, _, _ = _sync_search_watermark(
                latent=latent,
                sample_indices=sample_indices,
                sign_map=sign_map,
                alpha_effective=alpha_effective,
                max_shift=0,
                max_rotation=0.0,
                rotation_step=1.0,
            )
            candidate_scores.append({"box": box, "score": sync_res.score})
        best_box = max(candidate_scores, key=lambda item: item["score"])["box"]

    best_latent = _encode_crop(best_box)
    sync_res, best_view, rot_stats = _sync_search_watermark(
        latent=best_latent,
        sample_indices=sample_indices,
        sign_map=sign_map,
        alpha_effective=alpha_effective,
        max_shift=max_shift,
        max_rotation=max_rotation,
        rotation_step=rotation_step,
        rotation_strategy=rotation_strategy,
    )

    crop_stats = {
        "crop_keep_ratio": keep_ratio,
        "best_crop_box": best_box,
        "crop_candidates_count": len(candidate_scores),
        "crop_coarse_candidates": len(coarse_boxes),
        "crop_refine_candidates": len(refine_boxes),
        "crop_top3_scores": _top_candidates(candidate_scores),
    }
    crop_stats.update(rot_stats)
    return sync_res, best_view, crop_stats


def _build_attack_groups(base_img: Image.Image, crop_keep_ratio: float) -> Dict[str, List[Tuple[str, Image.Image]]]:
    rng_crop = np.random.default_rng(0)
    rng_noise = np.random.default_rng(1)
    rng_bright = np.random.default_rng(2)
    rng_chain = np.random.default_rng(3)

    jpeg_q75 = _attack_jpeg(base_img, 75)
    resize_img = _attack_resize(base_img, 0.75)
    chained_legacy = _attack_resize(jpeg_q75, 0.75)

    rot_75 = _attack_rotate(base_img, 75.0)
    jpeg_q25 = _attack_jpeg(base_img, 25)
    crop_scale = _attack_crop_scale(base_img, crop_keep_ratio, rng_crop)
    blur_8x8 = _attack_blur(base_img, 8)
    noise_sigma = _attack_noise(base_img, 0.1, rng_noise)
    bright_jitter = _attack_brightness_jitter(base_img, 0.6, rng_bright)

    chained = _attack_jpeg(base_img, 25)
    chained = _attack_crop_scale(chained, crop_keep_ratio, rng_chain)
    chained = _attack_rotate(chained, 75.0)
    chained = _attack_blur(chained, 8)
    chained = _attack_noise(chained, 0.1, rng_chain)
    chained = _attack_brightness_jitter(chained, 0.6, rng_chain)

    groups: Dict[str, List[Tuple[str, Image.Image]]] = {
        "clean": [("clean", base_img)],
        "jpeg_q75": [("jpeg_q75", jpeg_q75)],
        "resize_0.75": [("resize_0.75", resize_img)],
        "geom_sync": _geom_variants(base_img),
        "chained_legacy": _geom_variants(chained_legacy),
        "rot_75": [("rot_75", rot_75)],
        "jpeg_q25": [("jpeg_q25", jpeg_q25)],
        "crop_scale_0.75": [("crop_scale_0.75", crop_scale)],
        "blur_8x8": [("blur_8x8", blur_8x8)],
        "noise_sigma_0.1": [("noise_sigma_0.1", noise_sigma)],
        "brightness_jitter_0_6": [("brightness_jitter_0_6", bright_jitter)],
        "chained": [("chained", chained)],
    }
    return groups


def _load_inverter(
    model_id: str,
    device: str,
    dtype: str,
    steps: int,
    prompt: str,
    guidance_scale: float,
    negative_prompt: str,
    scheduler: str,
):
    cfg = DiffusersDDIMConfig(
        model_id=model_id,
        device=device,
        torch_dtype=dtype,
        num_steps=steps,
        scheduler=scheduler,
        prompt=prompt,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
    )
    return DiffusersDDIMInverter(cfg)


class DiffusersVAEEncoder:
    def __init__(self, model_id: str, device: str, torch_dtype: str):
        try:
            import torch  # type: ignore
            from diffusers import StableDiffusionXLPipeline  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("diffusers and torch are required for VAE encoding") from exc
        dtype = getattr(torch, torch_dtype, torch.float16)
        pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=dtype, variant="fp16")
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

    def encode(self, image: np.ndarray) -> np.ndarray:
        import torch  # type: ignore

        pipe = self.pipe
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        with torch.inference_mode():
            pil = pipe.image_processor.numpy_to_pil(image.astype(np.float32))
            vae_inputs = pipe.image_processor.preprocess(pil).to(device=pipe.device, dtype=pipe.dtype)
            latent_dist = pipe.vae.encode(vae_inputs).latent_dist
            latents = latent_dist.mode()
            latents = latents * pipe.vae.config.scaling_factor
        return latents[0, 0].detach().cpu().numpy().astype(np.float32)


def _ensure_sp1_vk(cfg: GlobalConfig, model_id: str | None) -> None:
    if cfg.zk.backend.lower() != "sp1":
        return
    if cfg.zk.vk_path:
        os.environ["SP1_VK_PATH"] = str(Path(cfg.zk.vk_path))
    if model_id and cfg.zk.vk_registry and model_id in cfg.zk.vk_registry:
        os.environ["SP1_VK_PATH"] = str(Path(cfg.zk.vk_registry[model_id]))


def main() -> None:
    parser = argparse.ArgumentParser(description="SP1 sweet-spot sweep (sample_count)")
    parser.add_argument("--config", default="config.json", help="Base config path")
    parser.add_argument("--counts", default="600,1000,1500", help="Comma/space separated sample_count list")
    parser.add_argument("--prompt", required=True, help="Prompt used for prover")
    parser.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--seed-hex")
    parser.add_argument("--secret-hex")
    parser.add_argument("--output", default="outputs/sp1_sweetspot")
    parser.add_argument("--out", dest="output", help="Alias for --output")
    parser.add_argument("--ddim-model-id", help="Diffusers model id for encoding/inversion (defaults to --model-id)")
    parser.add_argument("--encode-mode", choices=["vae", "ddim"], default="vae", help="Latent extraction: VAE encode or DDIM inversion")
    parser.add_argument(
        "--attack-groups",
        default="all",
        help="Comma-separated attack groups to run (e.g., clean,jpeg_q75); default=all",
    )
    parser.add_argument("--ddim-device", default="cuda")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--ddim-dtype", default="float32")
    parser.add_argument("--max-shift", type=int, default=2)
    parser.add_argument("--max-rotation", type=float, default=1.0)
    parser.add_argument("--rotation-step", type=float, default=1.0)
    parser.add_argument(
        "--rotation-strategy",
        choices=["auto", "grid", "coarse_to_fine"],
        default="auto",
        help="Rotation search strategy: auto/grid/coarse_to_fine.",
    )
    parser.add_argument("--sync-mode", choices=["on", "off"], default="on", help="Enable or disable sync search.")
    parser.add_argument("--crop-keep-ratio", type=float, default=0.75, help="Crop keep ratio for crop/scale attacks.")
    parser.add_argument("--crop-grid", type=int, default=3, help="Crop grid size for coarse search.")
    parser.add_argument("--crop-refine-grid", type=int, default=5, help="Crop refine grid size.")
    parser.add_argument("--crop-refine-step", type=float, default=0.05, help="Crop refine step as fraction of size.")
    parser.add_argument("--crop-refine", action="store_true", help="Enable local crop refinement search for crop/scale attacks.")
    parser.add_argument("--dump-bits", action="store_true", help="Dump pred/gt bits + bytes for clean case")
    parser.add_argument("--dump-bits-dir", help="Override dump directory (default: <output>/k<k>/bit_dump)")
    args = parser.parse_args()

    counts = _parse_counts(args.counts)
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    base_cfg = Path(args.config)
    base_cfg = base_cfg if base_cfg.is_absolute() else (ROOT / base_cfg).resolve()

    results: dict = {
        "counts": counts,
        "params": {
            "prompt": args.prompt,
            "model_id": args.model_id,
            "ddim_model_id": args.ddim_model_id or args.model_id,
            "encode_mode": args.encode_mode,
            "ddim_device": args.ddim_device,
            "ddim_steps": args.ddim_steps,
            "ddim_dtype": args.ddim_dtype,
            "max_shift": args.max_shift,
            "max_rotation": args.max_rotation,
            "rotation_step": args.rotation_step,
            "rotation_strategy": args.rotation_strategy,
            "sync_mode": args.sync_mode,
            "crop_keep_ratio": args.crop_keep_ratio,
            "crop_grid": args.crop_grid,
            "crop_refine": args.crop_refine,
            "crop_refine_grid": args.crop_refine_grid,
            "crop_refine_step": args.crop_refine_step,
            "dump_bits": args.dump_bits,
            "dump_bits_dir": args.dump_bits_dir,
        },
        "runs": [],
    }

    encoder = None
    encoder_model = None
    attack_cache: dict | None = None

    report_path = out_root / "sweetspot_results.json"

    for k in counts:
        run_dir = out_root / f"k{k}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cfg_path = run_dir / "config_used.json"
        _write_config(base_cfg, cfg_path, k)

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{ROOT / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
        env["SP1_PROVER"] = "cuda"
        env["SP1_PROOF_MODE"] = "core"
        env["SP1_SKIP_EXECUTE"] = "false"
        env["SP1_SKIP_VERIFY"] = "false"

        prover_cmd = [
            "python",
            "-m",
            "ppmark_v03.cli",
            "prover",
            "--config",
            str(cfg_path),
            "--prompt",
            args.prompt,
            "--output",
            str(run_dir),
            "--model-id",
            args.model_id,
            "--device-backend",
            "cuda",
        ]
        if args.seed_hex:
            prover_cmd.extend(["--seed-hex", args.seed_hex])
        if args.secret_hex:
            prover_cmd.extend(["--secret-hex", args.secret_hex])

        if _should_skip_prover(run_dir, args.prompt, args.seed_hex, args.secret_hex, k):
            print(f"[sweetspot] reuse existing prover output for k={k}")
        else:
            _run(prover_cmd, env=env)

        metadata = _load_json(run_dir / "metadata.json")
        cfg = GlobalConfig.load(cfg_path)
        _ensure_sp1_vk(cfg, args.model_id)
        inverse_cdf = InverseCDFTable.from_file(cfg.tables.inverse_cdf_path)
        codec = ReedSolomonCodec(n=cfg.watermark.rs_n, k=cfg.watermark.rs_k)

        codeword = bytes.fromhex(metadata["codeword_hex"])
        bits = np.array(bytes_to_bits(codeword), dtype=np.int8)
        binding = int(metadata["binding"], 16)
        expected_message = bytes.fromhex(metadata["message_hex"])
        alpha_effective = float(metadata.get("alpha_effective", cfg.image.alpha))
        scale_factor = float(metadata.get("scale_factor", 1.0))

        sample_set = deterministic_sample(
            total_pixels=cfg.image.total_pixels,
            width=cfg.image.width,
            height=cfg.image.height,
            key_material=codeword,
            count=cfg.image.sample_count(),
        )
        sample_indices = np.array(sample_set.indices, dtype=np.int64)

        bit_len = int(len(bits))
        bit_idx_map = _build_bit_index_map(binding, cfg.image.width, cfg.image.height, bit_len, cfg.zk.sample_backend)
        sign_map = (2.0 * bits[bit_idx_map].astype(np.float32) - 1.0)
        spread_stats = {
            "spread_min": float(np.min(sign_map)),
            "spread_max": float(np.max(sign_map)),
            "spread_mean": float(np.mean(sign_map)),
        }
        gaussian = inverse_cdf.batch_lookup(hash_uniforms(binding, cfg.image.total_pixels, backend=cfg.zk.sample_backend))

        image_path = Path(metadata.get("image_png", run_dir / "watermarked.png"))
        if not image_path.is_absolute():
            image_path = (run_dir / image_path).resolve()
        base_img = Image.open(image_path).convert("RGB")
        crop_keep_ratio = float(args.crop_keep_ratio) if args.crop_keep_ratio > 0 else 0.75
        attack_groups = _build_attack_groups(base_img, crop_keep_ratio)
        if args.attack_groups != "all":
            alias = {
                "jpeg": "jpeg_q25",
                "resize": "resize_0.75",
                "geom": "geom_sync",
                "rot": "rot_75",
                "crop": "crop_scale_0.75",
            }
            allow = {alias.get(g.strip(), g.strip()) for g in args.attack_groups.split(",") if g.strip()}
            attack_groups = {k: v for k, v in attack_groups.items() if k in allow}
        attack_cases: List[Tuple[str, str, str, Image.Image, float | None]] = []
        for group, cases in attack_groups.items():
            for name, img in cases:
                key = f"{group}:{name}"
                crop_ratio = crop_keep_ratio if group in {"crop_scale_0.75", "chained"} else None
                attack_cases.append((group, name, key, img, crop_ratio))

        ddim_model_id = args.ddim_model_id or args.model_id
        if args.encode_mode == "ddim":
            if encoder is None or encoder_model != ddim_model_id:
                encoder = _load_inverter(
                    model_id=ddim_model_id,
                    device=args.ddim_device,
                    dtype=args.ddim_dtype,
                    steps=args.ddim_steps,
                    scheduler=str(metadata.get("scheduler", "ddim")),
                    prompt=str(metadata.get("prompt", args.prompt)),
                    guidance_scale=float(metadata.get("guidance_scale", 7.5)),
                    negative_prompt=str(metadata.get("negative_prompt", "")),
                )
                encoder_model = ddim_model_id
            else:
                encoder.prompt = str(metadata.get("prompt", args.prompt))
                encoder.negative_prompt = str(metadata.get("negative_prompt", ""))
            encode_fn = encoder.invert
        else:
            if encoder is None or encoder_model != ddim_model_id:
                encoder = DiffusersVAEEncoder(
                    model_id=ddim_model_id,
                    device=args.ddim_device,
                    torch_dtype=args.ddim_dtype,
                )
                encoder_model = ddim_model_id
            encode_fn = encoder.encode

        sp1_meta = metadata.get("sp1", {})
        public_path = Path(sp1_meta.get("public", run_dir / "sp1/attestation_statement.json"))
        receipt_path = Path(sp1_meta.get("receipt", run_dir / "sp1/attest_receipt.bin"))
        if not public_path.is_absolute():
            public_path = (run_dir / public_path).resolve()
        if not receipt_path.is_absolute():
            receipt_path = (run_dir / receipt_path).resolve()
        sp1_paths = SP1Paths(
            public=public_path,
            receipt=receipt_path,
        )
        receipt_verify_sec = verify_sp1_receipt(sp1_paths, cfg.zk.verifier_cmd)
        timings = metadata.get("timings", {})
        sp1_prover_sec = timings.get("sp1_attest_prover_sec", timings.get("sp1_prover_sec"))

        image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if attack_cache is None or attack_cache.get("image_hash") != image_hash:
            cached_latents: dict[str, np.ndarray] = {}
            sigma_meta = float(metadata.get("init_noise_sigma", 1.0))
            for _, _, key, img, crop_ratio in attack_cases:
                if crop_ratio is not None:
                    continue
                img_arr = np.array(img, dtype=np.float32)
                latent = encode_fn(img_arr)
                if np.isnan(latent).any() or np.isinf(latent).any():
                    raise RuntimeError(
                        "VAE encoding produced NaN/Inf latents; try --ddim-dtype float32 or --ddim-device cpu"
                    )
                if sigma_meta != 0:
                    latent = latent / sigma_meta
                cached_latents[key] = latent
            attack_cache = {"image_hash": image_hash, "latents": cached_latents}
        cached_latents = attack_cache["latents"]

        group_summaries: Dict[str, dict] = {}
        overall_scores: List[float] = []
        overall_decoded: List[bool] = []
        dumped_bits = False
        for group, cases in attack_groups.items():
            case_results = []
            for name, img in cases:
                key = f"{group}:{name}"
                crop_ratio = 0.75 if group in {"crop_scale_0.75", "chained"} else None
                case_max_shift = 0 if args.sync_mode == "off" else args.max_shift
                case_max_rotation = 0.0 if args.sync_mode == "off" else args.max_rotation

                start = time.time()
                if crop_ratio is not None:
                    sync_res, best_view, search_stats = _crop_search_best(
                        image=img,
                        encode_fn=encode_fn,
                        sample_indices=sample_indices,
                        sign_map=sign_map,
                        alpha_effective=alpha_effective,
                        max_shift=case_max_shift,
                        max_rotation=case_max_rotation,
                        rotation_step=args.rotation_step,
                        rotation_strategy=args.rotation_strategy,
                        sigma_meta=float(metadata.get("init_noise_sigma", 1.0)),
                        keep_ratio=crop_ratio,
                        refine=args.crop_refine,
                        grid=args.crop_grid,
                        refine_grid=args.crop_refine_grid,
                        refine_step=args.crop_refine_step,
                    )
                else:
                    latent = cached_latents[key]
                    sync_res, best_view, search_stats = _sync_search_watermark(
                        latent=latent,
                        sample_indices=sample_indices,
                        sign_map=sign_map,
                        alpha_effective=alpha_effective,
                        max_shift=case_max_shift,
                        max_rotation=case_max_rotation,
                        rotation_step=args.rotation_step,
                        rotation_strategy=args.rotation_strategy,
                    )
                cv2 = _require_cv2()
                aligned = _translate_latent(best_view, -sync_res.dx, -sync_res.dy, cv2)
                latent_unscaled = aligned / scale_factor if scale_factor else aligned

                decoded = False
                ber = None
                ber_flip = None
                ber_shift_min = None
                ber_shift_best = None
                ones_ratio = None
                zeros_ratio = None
                codeword_prefix = None
                residual_stats = None
                bit_flip_applied = None
                try:
                    message, codeword_est, est_bits, residual_stats = _decode_bits_hashed(
                        latent=latent_unscaled,
                        binding=binding,
                        config=cfg,
                        inverse_cdf=inverse_cdf,
                        codec=codec,
                        bit_idx_map=bit_idx_map,
                        gaussian=gaussian,
                    )
                    decoded = message == expected_message
                    if args.dump_bits and not dumped_bits and group == "clean" and name == "clean":
                        dump_dir = Path(args.dump_bits_dir) if args.dump_bits_dir else (run_dir / "bit_dump")
                        _dump_bit_artifacts(
                            out_dir=dump_dir,
                            pred_bits=est_bits,
                            gt_bits=bits,
                            pred_bytes=codeword_est,
                            gt_bytes=codeword,
                            bit_order="msb-first",
                        )
                        dumped_bits = True
                    if est_bits.size == bits.size:
                        ber = float(np.mean(est_bits != bits))
                        ber_flip = float(np.mean((1 - est_bits) != bits))
                        max_shift = 32
                        best = (1.0, 0)
                        for s in range(-max_shift, max_shift + 1):
                            shifted = np.roll(est_bits, s)
                            b = float(np.mean(shifted != bits))
                            if b < best[0]:
                                best = (b, s)
                        ber_shift_min = best[0]
                        ber_shift_best = best[1]
                        ones_ratio = float(np.mean(est_bits))
                        zeros_ratio = 1.0 - ones_ratio
                        codeword_prefix = codeword_est[:16].hex()
                    if residual_stats is not None:
                        residual_stats.update(spread_stats)
                        bit_flip_applied = residual_stats.get("bit_flip_applied")
                except Exception:
                    decoded = False
                    codeword_prefix = None

                elapsed = time.time() - start
                case_results.append(
                    {
                        "name": name,
                        "score": sync_res.score,
                        "raw_score": sync_res.raw_score,
                        "decoded": decoded,
                        "ber": ber,
                        "ber_flip": ber_flip,
                        "ber_shift_min": ber_shift_min,
                        "ber_shift_best": ber_shift_best,
                        "bit_ones_ratio": ones_ratio,
                        "bit_zeros_ratio": zeros_ratio,
                        "bit_flip_applied": bit_flip_applied,
                        "codeword_hex_prefix": codeword_prefix,
                        "residual_stats": residual_stats,
                        "best_rotation_deg": search_stats.get("best_rotation_deg"),
                        "rotation_search_strategy": search_stats.get("rotation_search_strategy"),
                        "rotation_coarse_candidates": search_stats.get("coarse_candidates"),
                        "rotation_fine_candidates": search_stats.get("fine_candidates"),
                        "rotation_top3_scores": search_stats.get("rotation_top3_scores"),
                        "rotation_coarse_top3_scores": search_stats.get("rotation_coarse_top3_scores"),
                        "crop_keep_ratio": search_stats.get("crop_keep_ratio"),
                        "best_crop_box": search_stats.get("best_crop_box"),
                        "crop_candidates_count": search_stats.get("crop_candidates_count"),
                        "crop_coarse_candidates": search_stats.get("crop_coarse_candidates"),
                        "crop_refine_candidates": search_stats.get("crop_refine_candidates"),
                        "crop_top3_scores": search_stats.get("crop_top3_scores"),
                        "dx": sync_res.dx,
                        "dy": sync_res.dy,
                        "theta": sync_res.theta,
                        "elapsed_sec": elapsed,
                    }
                )
                overall_scores.append(sync_res.score)
                overall_decoded.append(decoded)
                print(
                    f"[attack] {group}/{name} score={sync_res.score:.4f} "
                    f"decoded={decoded} ber={ber} ber_flip={ber_flip} "
                    f"ber_shift_min={ber_shift_min} shift={ber_shift_best} "
                    f"ones={ones_ratio} zeros={zeros_ratio} flip={bit_flip_applied} "
                    f"elapsed={elapsed:.1f}s"
                )

            scores = [r["score"] for r in case_results]
            decoded_rate = float(np.mean([1.0 if r["decoded"] else 0.0 for r in case_results])) if case_results else 0.0
            score_mean = float(np.mean(scores)) if scores else 0.0
            score_p01 = float(np.quantile(scores, 0.01)) if scores else 0.0
            score_p05 = float(np.quantile(scores, 0.05)) if scores else 0.0
            group_summaries[group] = {
                "decode_success_rate": decoded_rate,
                "score_mean": score_mean,
                "score_p01": score_p01,
                "score_p05": score_p05,
                "cases": case_results,
            }
            report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(
                f"[group={group}] decode={decoded_rate:.3f} score_mean={score_mean:.3f} "
                f"p01={score_p01:.3f}"
            )

        overall_decoded_rate = float(np.mean([1.0 if d else 0.0 for d in overall_decoded])) if overall_decoded else 0.0
        overall_mean = float(np.mean(overall_scores)) if overall_scores else 0.0
        overall_p01 = float(np.quantile(overall_scores, 0.01)) if overall_scores else 0.0
        overall_p05 = float(np.quantile(overall_scores, 0.05)) if overall_scores else 0.0

        results["runs"].append(
            {
                "sample_count": k,
                "sp1_prover_sec": sp1_prover_sec,
                "receipt_verify_sec": receipt_verify_sec,
                "overall": {
                    "decode_success_rate": overall_decoded_rate,
                    "score_mean": overall_mean,
                    "score_p01": overall_p01,
                    "score_p05": overall_p05,
                },
                "groups": group_summaries,
            }
        )
        report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(
            f"[k={k}] overall decode={overall_decoded_rate:.3f} score_mean={overall_mean:.3f} "
            f"p01={overall_p01:.3f} sp1_prover_sec={sp1_prover_sec} receipt_verify_sec={receipt_verify_sec:.3f}"
        )

    # Selection rule
    chosen = None
    by_k = {r["sample_count"]: r for r in results["runs"]}
    for k in [600, 1000, 1500]:
        if k not in by_k:
            continue
        run = by_k[k]
        groups = run.get("groups", {}).values()
        if groups and all(g["decode_success_rate"] >= 0.99 and g["score_p01"] >= 0.5 for g in groups):
            chosen = k
            break
    if chosen is None:
        chosen = 1500
    results["selection"] = {
        "rule": "min k s.t. all attack groups have decode>=0.99 and score_p01>=0.5; fallback 1500",
        "chosen_k": chosen,
    }

    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[sweetspot] report written to {report_path}")

    for run in results["runs"]:
        overall = run.get("overall", {})
        print(
            f"[k={run['sample_count']}] decode={overall.get('decode_success_rate', 0.0):.3f} "
            f"score_mean={overall.get('score_mean', 0.0):.3f} p01={overall.get('score_p01', 0.0):.3f} "
            f"sp1_prover_sec={run['sp1_prover_sec']} receipt_verify_sec={run['receipt_verify_sec']:.3f}"
        )
    print(f"[sweetspot] chosen_k={chosen}")


if __name__ == "__main__":
    main()
