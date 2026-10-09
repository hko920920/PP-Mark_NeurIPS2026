#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageOps

from ppmark_v03.config import GlobalConfig
from ppmark_v03.ddim_unet import DiffusersDDIMConfig, DiffusersDDIMInverter
from ppmark_v03.halo2_interface import FIXED_SCALE
from ppmark_v03.halo2_runner import Halo2Paths, verify_halo2_proof
from ppmark_v03.noise import build_bit_index_map, hash_uniforms
from ppmark_v03.payload import bits_to_bytes, bytes_to_bits
from ppmark_v03.risc0_runner import Risc0Paths, verify_risc0_receipt
from ppmark_v03.rs import ReedSolomonCodec
from ppmark_v03.sampling import deterministic_sample
from ppmark_v03.sp1_runner import SP1Paths, verify_sp1_receipt
from ppmark_v03.sync import recompute_sample_root
from ppmark_v03.tables import InverseCDFTable

HASH_SPEC = {
    "version": "norm_v1",
    "color_space": "sRGB",
    "resize": [512, 512],
    "interpolation": "bicubic",
    "rounding": "round",
    "bytes": "uint8_rgb",
    "exif_transpose": True,
}
OPENING_DOMAIN = b"ppmark_opening_v1"
OPENING_VERSION = "opening_v1"


def _bicubic_resample() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC


def _normalize_image_for_hash(image: Image.Image, size: Tuple[int, int]) -> np.ndarray:
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(size, resample=_bicubic_resample())
    arr = np.asarray(image, dtype=np.float32)
    arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    return arr


def _hash_normalized_image(image: Image.Image, size: Tuple[int, int]) -> bytes:
    normalized = _normalize_image_for_hash(image, size)
    return hashlib.sha256(normalized.tobytes()).digest()


def _opening_digest(entries: List[dict]) -> bytes:
    hasher = hashlib.sha256()
    for entry in entries:
        hasher.update(int(entry["pos"]).to_bytes(4, "little", signed=False))
        hasher.update(int(entry["index"]).to_bytes(4, "little", signed=False))
        hasher.update(int(entry["gaussian_fixed"]).to_bytes(8, "little", signed=True))
        hasher.update(int(entry["combined_fixed"]).to_bytes(8, "little", signed=True))
        hasher.update(int(entry["bit"]).to_bytes(1, "little", signed=False))
    return hasher.digest()


def _opening_eps_ok(expected: int, observed: int, eps_fixed: int) -> bool:
    return abs(expected - observed) <= eps_fixed


def _iter_manifest(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            yield json.loads(line)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalize_hex(value: str) -> str:
    text = str(value).lower()
    if text.startswith("0x"):
        text = text[2:]
    return text


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _resolve(base: Path, target: str) -> Path:
    candidate = Path(target)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate


def _require_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("cv2 is required for rotation/shift sync search") from exc
    return cv2


def _shift_indices(indices: np.ndarray, width: int, height: int, dx: int, dy: int) -> np.ndarray:
    xs = indices % width
    ys = indices // width
    xs_shift = xs + dx
    ys_shift = ys + dy
    mask = (xs_shift >= 0) & (xs_shift < width) & (ys_shift >= 0) & (ys_shift < height)
    shifted = np.full_like(indices, -1)
    shifted[mask] = ys_shift[mask] * width + xs_shift[mask]
    return shifted


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


class SyncResult:
    def __init__(self, dx: int, dy: int, theta: float, score: float, raw_score: float):
        self.dx = dx
        self.dy = dy
        self.theta = theta
        self.score = score
        self.raw_score = raw_score


def _rotate_latent(latent: np.ndarray, degrees: float, cv2_module) -> np.ndarray:
    if abs(degrees) <= 1e-6:
        return latent
    height, width = latent.shape
    center = (width / 2.0, height / 2.0)
    rot_mat = cv2_module.getRotationMatrix2D(center, degrees, 1.0)
    return cv2_module.warpAffine(latent, rot_mat, (width, height), flags=cv2_module.INTER_LANCZOS4, borderMode=cv2_module.BORDER_REFLECT)


def _translate_latent(latent: np.ndarray, dx: int, dy: int, cv2_module) -> np.ndarray:
    if dx == 0 and dy == 0:
        return latent
    height, width = latent.shape
    mat = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    return cv2_module.warpAffine(latent, mat, (width, height), flags=cv2_module.INTER_LINEAR, borderMode=cv2_module.BORDER_REFLECT)


def _sync_search_for_angles(
    latent: np.ndarray,
    sample_indices: np.ndarray,
    sign_map: np.ndarray,
    alpha_effective: float,
    angles: List[float],
    max_shift: int,
) -> Tuple[SyncResult, np.ndarray, List[dict]]:
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
        best, best_view, fine_candidates = _sync_search_for_angles(
            latent=latent,
            sample_indices=sample_indices,
            sign_map=sign_map,
            alpha_effective=alpha_effective,
            angles=fine_angles,
            max_shift=max_shift,
        )
        search_stats = {
            "rotation_search_strategy": "coarse_to_fine",
            "best_rotation_deg": best.theta,
            "coarse_candidates": len(coarse_angles),
            "fine_candidates": len(fine_angles),
            "rotation_top3_scores": _top_candidates(fine_candidates),
            "rotation_coarse_top3_scores": _top_candidates(coarse_candidates),
        }
    else:
        angles = _rotation_sweep(float(max_rotation), float(rotation_step))
        best, best_view, candidates = _sync_search_for_angles(
            latent=latent,
            sample_indices=sample_indices,
            sign_map=sign_map,
            alpha_effective=alpha_effective,
            angles=angles,
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
    width: int, height: int, keep_ratio: float, x0: int, y0: int, step_frac: float = 0.05, grid: int = 5
) -> List[Tuple[int, int, int, int]]:
    scale = math.sqrt(keep_ratio)
    crop_w = max(1, int(round(width * scale)))
    crop_h = max(1, int(round(height * scale)))
    max_x = max(0, width - crop_w)
    max_y = max(0, height - crop_h)
    base_x0 = min(max(x0, 0), max_x)
    base_y0 = min(max(y0, 0), max_y)
    step_x = max(1, int(round(step_frac * crop_w)))
    step_y = max(1, int(round(step_frac * crop_h)))
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
            nx0 = min(max(base_x0 + dx, 0), max_x)
            ny0 = min(max(base_y0 + dy, 0), max_y)
            boxes.append((nx0, ny0, nx0 + crop_w, ny0 + crop_h))
    return list(dict.fromkeys(boxes))


def _apply_crop_and_resize(image: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    width, height = image.size
    cropped = image.crop(box)
    return cropped.resize((width, height), resample=Image.BICUBIC)


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
) -> Tuple[SyncResult, np.ndarray, dict]:
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
            max_shift=max_shift,
            max_rotation=max_rotation,
            rotation_step=rotation_step,
            rotation_strategy=rotation_strategy,
        )
        candidate_scores.append({"box": box, "score": sync_res.score})

    best_box = max(candidate_scores, key=lambda item: item["score"])["box"]
    refine_boxes: List[Tuple[int, int, int, int]] = []
    if refine:
        refine_boxes = _crop_box_refine(width, height, keep_ratio, best_box[0], best_box[1], refine_step, refine_grid)
        for box in refine_boxes:
            latent = _encode_crop(box)
            sync_res, _, _ = _sync_search_watermark(
                latent=latent,
                sample_indices=sample_indices,
                sign_map=sign_map,
                alpha_effective=alpha_effective,
                max_shift=max_shift,
                max_rotation=max_rotation,
                rotation_step=rotation_step,
                rotation_strategy=rotation_strategy,
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


def _decode_bits_hashed(
    latent: np.ndarray,
    binding: int,
    config: GlobalConfig,
    inverse_cdf: InverseCDFTable,
    codec: ReedSolomonCodec,
    bit_idx_map: np.ndarray,
    gaussian: np.ndarray,
    alpha: float | None = None,
) -> Tuple[bytes | None, bytes, np.ndarray, dict]:
    latent2d = np.asarray(latent, dtype=np.float32)
    actual_h, actual_w = latent2d.shape
    flat = latent2d.reshape(-1)[: actual_h * actual_w]
    alpha_value = float(alpha) if alpha is not None else float(config.image.alpha)
    if alpha_value == 0.0:
        raise ValueError("alpha must be non-zero for decoding")
    residual = (flat - gaussian) / alpha_value
    indices = bit_idx_map.reshape(-1)[: actual_h * actual_w]
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
        "threshold": threshold,
        "alpha_used": alpha_value,
        "bit_flip_applied": bit_flip,
    }
    return message, codeword, bits, stats


def _apply_vk_env(cfg: GlobalConfig, model_id: str | None) -> None:
    if cfg.zk.backend.lower() != "sp1":
        return
    if cfg.zk.vk_path:
        os.environ["SP1_VK_PATH"] = str(Path(cfg.zk.vk_path))
    if model_id and cfg.zk.vk_registry and model_id in cfg.zk.vk_registry:
        os.environ["SP1_VK_PATH"] = str(Path(cfg.zk.vk_registry[model_id]))


def _get_inverter(
    cache: Dict[Tuple[str, str, str, str, int], DiffusersDDIMInverter],
    model_id: str,
    scheduler: str,
    device: str,
    dtype: str,
    steps: int,
) -> DiffusersDDIMInverter:
    key = (model_id, scheduler, device, dtype, steps)
    if key in cache:
        return cache[key]
    cfg = DiffusersDDIMConfig(
        model_id=model_id,
        device=device,
        torch_dtype=dtype,
        num_steps=steps,
        scheduler=scheduler,
        prompt="",
        guidance_scale=7.5,
        negative_prompt="",
    )
    cache[key] = DiffusersDDIMInverter(cfg)
    return cache[key]


def _get_expected_message(meta: dict) -> bytes:
    msg_hex = meta.get("message_hex")
    if msg_hex:
        return bytes.fromhex(msg_hex)
    binding = str(meta["binding"])
    if binding.startswith("0x"):
        binding = binding[2:]
    return bytes.fromhex(binding)


def _get_expected_payload_hash(meta: dict) -> str:
    payload_hash = meta.get("payload_hash")
    if payload_hash:
        return _normalize_hex(payload_hash)
    message = _get_expected_message(meta)
    return _sha256_hex(message)


def _proof_meta_missing(meta: dict, backend: str) -> bool:
    if backend == "sp1":
        sp1_meta = meta.get("sp1")
        if not sp1_meta:
            return True
        return any(key not in sp1_meta for key in ("public", "receipt"))
    if backend == "risc0":
        risc0_meta = meta.get("risc0")
        if not risc0_meta:
            return True
        return any(key not in risc0_meta for key in ("public", "witness", "receipt"))
    halo2_meta = meta.get("halo2")
    if not halo2_meta:
        return True
    return any(key not in halo2_meta for key in ("public", "witness", "proof"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate attack results from a manifest.")
    parser.add_argument("--manifest", required=True, help="JSONL manifest with image_path/metadata_path.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--metadata-path", help="Default metadata path if row lacks metadata_path.")
    parser.add_argument("--image-root", default="", help="Override root for relative image_path.")
    parser.add_argument("--metadata-root", default="", help="Override root for relative metadata_path.")
    parser.add_argument("--tau", type=float, default=None, help="Soft-detect threshold (score >= tau).")
    parser.add_argument("--tau-file", default="", help="JSON file with tau_score.")
    parser.add_argument("--scores-out", default="attack_scores.csv")
    parser.add_argument("--summary-out", default="attack_summary.json")
    parser.add_argument("--ddim-model-id", default="")
    parser.add_argument("--ddim-device", default="cuda")
    parser.add_argument("--ddim-steps", type=int, default=50)
    parser.add_argument("--ddim-dtype", default="float32")
    parser.add_argument("--max-shift", type=int, default=0)
    parser.add_argument("--max-rotation", type=float, default=0.0)
    parser.add_argument("--rotation-step", type=float, default=1.0)
    parser.add_argument("--rotation-strategy", choices=["auto", "grid", "coarse_to_fine"], default="auto")
    parser.add_argument("--crop-keep-ratio", type=float, default=0.0)
    parser.add_argument("--crop-grid", type=int, default=3)
    parser.add_argument("--crop-refine", action="store_true")
    parser.add_argument("--crop-refine-grid", type=int, default=5)
    parser.add_argument("--crop-refine-step", type=float, default=0.05)
    parser.add_argument("--skip-decode", action="store_true")
    parser.add_argument("--skip-consistency", action="store_true")
    parser.add_argument("--verify-proof", action="store_true")
    parser.add_argument(
        "--opening-mode",
        default="strict",
        choices=["strict", "robust"],
        help="Opening verification mode (strict exact match or robust tolerance).",
    )
    parser.add_argument(
        "--opening-eps",
        type=float,
        default=0.0,
        help="Tolerance for robust opening checks (latent units).",
    )
    args = parser.parse_args()

    cfg = GlobalConfig.load(Path(args.config))
    manifest_path = Path(args.manifest)
    manifest_dir = manifest_path.parent.resolve()
    image_root = Path(args.image_root).resolve() if args.image_root else None
    metadata_root = Path(args.metadata_root).resolve() if args.metadata_root else None
    default_meta = Path(args.metadata_path).resolve() if args.metadata_path else None

    tau = args.tau
    if tau is None and args.tau_file:
        payload = _load_json(Path(args.tau_file))
        tau = float(payload.get("tau_score"))
    tau = float(tau) if tau is not None else float("nan")

    inverse_cdf = InverseCDFTable.from_file(cfg.tables.inverse_cdf_path)
    codec = ReedSolomonCodec(n=cfg.watermark.rs_n, k=cfg.watermark.rs_k)

    meta_cache: Dict[Path, dict] = {}
    inverter_cache: Dict[Tuple[str, str, str, str, int], DiffusersDDIMInverter] = {}
    gaussian_cache: Dict[int, np.ndarray] = {}
    proof_cache: Dict[Path, Tuple[bool, str | None]] = {}

    rows = list(_iter_manifest(manifest_path))
    results: List[dict] = []
    group_stats: Dict[str, dict] = {}

    def _resolve_manifest_path(raw: str, root_override: Path | None) -> Path:
        p = Path(raw)
        if not p.is_absolute():
            base = root_override if root_override else manifest_dir
            return (base / p).resolve()
        return p.resolve()

    for idx, row in enumerate(rows):
        print(f"[eval] [{idx+1}/{len(rows)}] start", flush=True)
        image_relpath = row.get("image_relpath")
        image_raw = row.get("image_path")
        if image_relpath:
            image_path = _resolve_manifest_path(str(image_relpath), image_root)
        elif image_raw:
            image_path = _resolve_manifest_path(str(image_raw), image_root)
        else:
            raise RuntimeError(f"Missing image path for row {idx}")

        meta_relpath = row.get("metadata_relpath") or row.get("ref_metadata_relpath")
        meta_raw = row.get("metadata_path") or row.get("ref_metadata_path")
        if meta_relpath:
            meta_path = _resolve_manifest_path(str(meta_relpath), metadata_root)
        elif meta_raw:
            meta_path = _resolve_manifest_path(str(meta_raw), metadata_root)
        elif default_meta:
            meta_path = default_meta
        else:
            raise RuntimeError(f"Missing metadata path for row {idx}")

        meta_path = Path(meta_path)
        meta = meta_cache.get(meta_path)
        if meta is None:
            meta = _load_json(meta_path)
            meta_cache[meta_path] = meta

        model_id = args.ddim_model_id or meta.get("model_id")
        if not model_id:
            raise RuntimeError(f"Missing model_id for row {idx}")
        scheduler = str(meta.get("scheduler", "ddim"))
        inverter = _get_inverter(inverter_cache, str(model_id), scheduler, args.ddim_device, args.ddim_dtype, args.ddim_steps)
        inverter.prompt = str(meta.get("prompt", ""))
        inverter.negative_prompt = str(meta.get("negative_prompt", ""))
        inverter.cfg.guidance_scale = float(meta.get("guidance_scale", 7.5))

        start = time.time()
        with Image.open(image_path) as img:
            img_arr = np.array(img.convert("RGB"), dtype=np.float32)
        latent = inverter.invert(img_arr)
        if np.isnan(latent).any() or np.isinf(latent).any():
            raise RuntimeError(f"DDIM inversion produced NaN/Inf latents: {image_path}")
        sigma_meta = float(meta.get("init_noise_sigma", 1.0))
        if sigma_meta != 0:
            latent = latent / sigma_meta

        codeword = bytes.fromhex(str(meta["codeword_hex"]))
        bits = np.array(bytes_to_bits(codeword), dtype=np.int8)
        binding = int(str(meta["binding"]), 16)
        alpha_effective = float(meta.get("alpha_effective", cfg.image.alpha))
        scale_factor = float(meta.get("scale_factor", 1.0))

        # Override latent dimensions from metadata (SDXL=128×128 vs SD2.1=64×64)
        img_w = int(meta.get("width", cfg.image.width))
        img_h = int(meta.get("height", cfg.image.height))
        img_total = img_w * img_h
        img_sample_count = int(meta.get("sample_count", cfg.image.sample_count()))

        sample_set = deterministic_sample(
            total_pixels=img_total,
            width=img_w,
            height=img_h,
            key_material=codeword,
            count=img_sample_count,
        )
        sample_indices = np.array(sample_set.indices, dtype=np.int64)
        bit_idx_map = build_bit_index_map(binding, img_w, img_h, len(bits), backend=cfg.zk.sample_backend)
        sign_map = (2.0 * bits[bit_idx_map].astype(np.float32) - 1.0)
        bits_for_samples = bits[bit_idx_map[sample_indices // img_w, sample_indices % img_w]]

        if args.crop_keep_ratio and args.crop_keep_ratio > 0:
            sync_res, best_view, _ = _crop_search_best(
                image=Image.fromarray(img_arr.astype(np.uint8)),
                encode_fn=inverter.invert,
                sample_indices=sample_indices,
                sign_map=sign_map,
                alpha_effective=alpha_effective,
                max_shift=args.max_shift,
                max_rotation=args.max_rotation,
                rotation_step=args.rotation_step,
                rotation_strategy=args.rotation_strategy,
                sigma_meta=sigma_meta,
                keep_ratio=float(args.crop_keep_ratio),
                refine=bool(args.crop_refine),
                grid=int(args.crop_grid),
                refine_grid=int(args.crop_refine_grid),
                refine_step=float(args.crop_refine_step),
            )
        else:
            sync_res, best_view, _ = _sync_search_watermark(
                latent=latent,
                sample_indices=sample_indices,
                sign_map=sign_map,
                alpha_effective=alpha_effective,
                max_shift=args.max_shift,
                max_rotation=args.max_rotation,
                rotation_step=args.rotation_step,
                rotation_strategy=args.rotation_strategy,
            )

        score = float(sync_res.score)
        raw_score = float(sync_res.raw_score)
        score_pass = bool(np.isfinite(tau) and score >= tau)
        print(f"[eval] [{idx+1}/{len(rows)}] score={score:.4f} pass={score_pass} t={time.time()-start:.1f}s", flush=True)

        rs_ok = None
        payload_match = None
        decoded_match = None
        bit_flip = None
        if not args.skip_decode:
            cv2 = _require_cv2() if args.max_shift or args.max_rotation else None
            aligned = _translate_latent(best_view, -sync_res.dx, -sync_res.dy, cv2) if cv2 is not None else best_view
            latent_unscaled = aligned / scale_factor if scale_factor else aligned
            gaussian = gaussian_cache.get(binding)
            if gaussian is None:
                gaussian = inverse_cdf.batch_lookup(hash_uniforms(binding, img_total, backend=cfg.zk.sample_backend))
                gaussian_cache[binding] = gaussian
            message, _, _, stats = _decode_bits_hashed(
                latent=latent_unscaled,
                binding=binding,
                config=cfg,
                inverse_cdf=inverse_cdf,
                codec=codec,
                bit_idx_map=bit_idx_map,
                gaussian=gaussian,
                alpha=alpha_effective,
            )
            rs_ok = message is not None
            expected_hash = _get_expected_payload_hash(meta)
            decoded_hash = _sha256_hex(message) if message is not None else None
            payload_match = bool(decoded_hash == expected_hash) if decoded_hash is not None else False
            decoded_match = bool(rs_ok and payload_match)
            bit_flip = bool(stats.get("bit_flip_applied"))

        consistency = None
        if not args.skip_consistency and "sample_root" in meta:
            backend = "sha" if cfg.zk.backend.lower() in ("risc0", "sp1") else "poseidon"
            alpha_fixed = int(round(alpha_effective * FIXED_SCALE))
            root, _ = recompute_sample_root(
                latent=best_view,
                binding=binding,
                config=cfg,
                sample_indices=sample_indices,
                bits=bits,
                alpha_fixed=alpha_fixed,
                backend=backend,
                dx=sync_res.dx,
                dy=sync_res.dy,
                alpha_effective=alpha_effective,
            )
            consistency = bool(root.hex() == str(meta["sample_root"]))

        opening_ok = None
        opening_status = None
        opening_mode = getattr(args, "opening_mode", "strict")
        opening_eps = float(getattr(args, "opening_eps", 0.0) or 0.0)
        if "opening" in meta:
            opening = meta["opening"]
            hash_spec = meta.get("hash_spec", {})
            if hash_spec != HASH_SPEC:
                opening_ok = False
                opening_status = "hash_spec_mismatch"
            else:
                if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tiff"}:
                    opening_ok = False
                    opening_status = "image_required"
                else:
                    with Image.open(image_path) as img:
                        image_hash = _hash_normalized_image(
                            img,
                            (int(HASH_SPEC["resize"][0]), int(HASH_SPEC["resize"][1])),
                        )
                    ctx_hash_hex = meta.get("ctx_hash")
                    if not ctx_hash_hex:
                        opening_ok = False
                        opening_status = "ctx_hash_missing"
                    else:
                        ctx_hash = int(ctx_hash_hex, 16).to_bytes(32, "big")
                        sample_root_hex = meta.get("sample_root")
                        if not sample_root_hex:
                            opening_ok = False
                            opening_status = "sample_root_missing"
                        else:
                            challenge_seed = hashlib.sha256(
                                OPENING_DOMAIN + image_hash + ctx_hash + bytes.fromhex(sample_root_hex)
                            ).digest()
                            if opening.get("version") != OPENING_VERSION:
                                opening_ok = False
                                opening_status = "opening_version_mismatch"
                            elif opening.get("challenge_seed") != "0x" + challenge_seed.hex():
                                opening_ok = False
                                opening_status = "opening_challenge_mismatch"
                            else:
                                openings_path = _resolve(meta_path.parent, opening["path"])
                                openings_doc = _load_json(openings_path)
                                entries = openings_doc.get("entries", [])
                                positions = openings_doc.get("positions", [])
                                if openings_doc.get("version") != OPENING_VERSION:
                                    opening_ok = False
                                    opening_status = "openings_version_mismatch"
                                elif openings_doc.get("k") != opening.get("k"):
                                    opening_ok = False
                                    opening_status = "openings_k_mismatch"
                                elif openings_doc.get("digest") != opening.get("digest"):
                                    opening_ok = False
                                    opening_status = "openings_digest_mismatch"
                                elif len(entries) != int(opening.get("k", 0)):
                                    opening_ok = False
                                    opening_status = "openings_entries_mismatch"
                                elif positions and [int(entry["pos"]) for entry in entries] != list(positions):
                                    opening_ok = False
                                    opening_status = "openings_positions_mismatch"
                                else:
                                    opening_digest = _opening_digest(entries)
                                    if "0x" + opening_digest.hex() != opening.get("digest"):
                                        opening_ok = False
                                        opening_status = "openings_digest_mismatch"
                                    else:
                                        alpha = float(alpha_effective)
                                        eps_fixed = int(round(opening_eps * FIXED_SCALE))
                                        opening_ok = True
                                        for entry in entries:
                                            pos = int(entry["pos"])
                                            if pos < 0 or pos >= len(sample_indices):
                                                opening_ok = False
                                                opening_status = "opening_pos_oob"
                                                break
                                            idx = int(sample_indices[pos])
                                            if int(entry["index"]) != idx:
                                                opening_ok = False
                                                opening_status = "opening_index_mismatch"
                                                break
                                            y = idx // img_w
                                            x = idx % img_w
                                            y_shift = y + sync_res.dy
                                            x_shift = x + sync_res.dx
                                            if y_shift < 0 or y_shift >= img_h or x_shift < 0 or x_shift >= img_w:
                                                opening_ok = False
                                                opening_status = "opening_shift_oob"
                                                break
                                            combined_val = float(best_view[y_shift, x_shift])
                                            combined_fixed_expected = int(round(combined_val * FIXED_SCALE))
                                            bit_expected = int(bits_for_samples[pos])
                                            if int(entry["bit"]) != bit_expected:
                                                opening_ok = False
                                                opening_status = "opening_bit_mismatch"
                                                break
                                            gaussian_fixed_expected = int(
                                                round((combined_val - alpha * (2 * bit_expected - 1)) * FIXED_SCALE)
                                            )
                                            if opening_mode == "strict":
                                                if int(entry["combined_fixed"]) != combined_fixed_expected:
                                                    opening_ok = False
                                                    opening_status = "opening_combined_mismatch"
                                                    break
                                                if int(entry["gaussian_fixed"]) != gaussian_fixed_expected:
                                                    opening_ok = False
                                                    opening_status = "opening_gaussian_mismatch"
                                                    break
                                            else:
                                                if not _opening_eps_ok(int(entry["combined_fixed"]), combined_fixed_expected, eps_fixed):
                                                    opening_ok = False
                                                    opening_status = "opening_combined_eps"
                                                    break
                                                if not _opening_eps_ok(int(entry["gaussian_fixed"]), gaussian_fixed_expected, eps_fixed):
                                                    opening_ok = False
                                                    opening_status = "opening_gaussian_eps"
                                                    break
                                        if opening_ok and opening_status is None:
                                            opening_status = "ok"

        proof_ok = None
        proof_status = None
        if args.verify_proof and "sample_root" in meta:
            if meta_path in proof_cache:
                proof_ok, proof_status = proof_cache[meta_path]
            else:
                backend = cfg.zk.backend.lower()
                if _proof_meta_missing(meta, backend):
                    proof_ok = False
                    proof_status = "no_proof"
                else:
                    try:
                        if backend == "sp1":
                            sp1_meta = meta.get("sp1")
                            _apply_vk_env(cfg, str(meta.get("model_id")))
                            sp1_paths = SP1Paths(
                                public=_resolve(meta_path.parent, sp1_meta["public"]),
                                receipt=_resolve(meta_path.parent, sp1_meta["receipt"]),
                            )
                            verify_sp1_receipt(sp1_paths, cfg.zk.verifier_cmd)
                        elif backend == "risc0":
                            risc0_meta = meta.get("risc0")
                            risc0_paths = Risc0Paths(
                                public=_resolve(meta_path.parent, risc0_meta["public"]),
                                witness=_resolve(meta_path.parent, risc0_meta["witness"]),
                                receipt=_resolve(meta_path.parent, risc0_meta["receipt"]),
                            )
                            verify_risc0_receipt(risc0_paths, cfg.zk.verifier_cmd)
                        else:
                            halo2_meta = meta.get("halo2")
                            halo2_paths = Halo2Paths(
                                public=_resolve(meta_path.parent, halo2_meta["public"]),
                                witness=_resolve(meta_path.parent, halo2_meta["witness"]),
                                proof=_resolve(meta_path.parent, halo2_meta["proof"]),
                            )
                            verify_halo2_proof(halo2_paths, cfg.zk)
                        proof_ok = True
                        proof_status = "ok"
                    except Exception:
                        proof_ok = False
                        proof_status = "proof_invalid"
                proof_cache[meta_path] = (proof_ok, proof_status)

        if proof_ok is True and opening_ok is False:
            proof_ok = False
            proof_status = "opening_mismatch"

        runtime = time.time() - start

        attack = row.get("attack", "")
        strength = row.get("strength", "")
        label = row.get("label", "")
        group_key = f"{attack}::{strength}::{label}"
        stats = group_stats.setdefault(
            group_key,
            {
                "count": 0,
                "soft_pass": 0,
                "rs_pass": 0,
                "payload_match_pass": 0,
                "decoded_match_pass": 0,
                "consistency_pass": 0,
                "opening_pass": 0,
                "proof_pass": 0,
                "hard_pass": 0,
                "hard_total": 0,
                "accept_pass": 0,
                "accept_total": 0,
                "fail_reason_counts": {
                    "no_proof": 0,
                    "proof_invalid": 0,
                    "opening_mismatch": 0,
                    "score_low": 0,
                    "decode_fail": 0,
                    "payload_mismatch": 0,
                },
            },
        )
        stats["count"] += 1
        stats["soft_pass"] += 1 if score_pass else 0
        stats["rs_pass"] += 1 if rs_ok else 0
        stats["payload_match_pass"] += 1 if payload_match else 0
        stats["decoded_match_pass"] += 1 if decoded_match else 0
        stats["consistency_pass"] += 1 if consistency else 0
        stats["opening_pass"] += 1 if opening_ok else 0
        stats["proof_pass"] += 1 if proof_ok else 0
        hard_pass = None
        if proof_ok is not None:
            hard_pass = bool(proof_ok)
            stats["hard_total"] += 1
            stats["hard_pass"] += 1 if hard_pass else 0

        accept_pass = None
        if proof_ok is not None:
            accept_pass = bool(proof_ok and score_pass)
            stats["accept_total"] += 1
            stats["accept_pass"] += 1 if accept_pass else 0

        fail_reason = None
        if proof_ok is False:
            if proof_status == "opening_mismatch":
                fail_reason = "opening_mismatch"
            else:
                fail_reason = "no_proof" if proof_status == "no_proof" else "proof_invalid"
        elif proof_ok is True:
            if not score_pass:
                fail_reason = "score_low"
            elif rs_ok is False:
                fail_reason = "decode_fail"
            elif payload_match is False:
                fail_reason = "payload_mismatch"
        if fail_reason:
            stats["fail_reason_counts"][fail_reason] += 1

        results.append(
            {
                "idx": row.get("idx", idx),
                "label": label,
                "attack": attack,
                "strength": strength,
                "image_path": str(image_path),
                "metadata_path": str(meta_path),
                "score": score,
                "raw_score": raw_score,
                "score_pass": score_pass,
                "rs_ok": rs_ok,
                "payload_match": payload_match,
                "decoded_match": decoded_match,
                "bit_flip_applied": bit_flip,
                "consistency": consistency,
                "opening_ok": opening_ok,
                "opening_status": opening_status,
                "proof_ok": proof_ok,
                "proof_status": proof_status,
                "hard_pass": hard_pass,
                "accept_pass": accept_pass,
                "fail_reason": fail_reason,
                "runtime_sec": runtime,
            }
        )

    scores_path = Path(args.scores_out)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    with scores_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "idx",
                "label",
                "attack",
                "strength",
                "image_path",
                "metadata_path",
                "score",
                "raw_score",
                "score_pass",
                "rs_ok",
                "payload_match",
                "decoded_match",
                "bit_flip_applied",
                "consistency",
                "opening_ok",
                "opening_status",
                "proof_ok",
                "proof_status",
                "hard_pass",
                "accept_pass",
                "fail_reason",
                "runtime_sec",
            ],
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    summary = {
        "tau": None if np.isnan(tau) else float(tau),
        "count": len(results),
        "hard_pass_definition": "proof_ok (receipt_ok AND opening_ok)",
        "accept_pass_definition": "proof_ok AND score_pass",
        "decoded_match_definition": "rs_ok AND payload_match",
        "payload_match_definition": "sha256(m_hat) == metadata.payload_hash",
        "groups": {},
    }
    for key, stats in group_stats.items():
        count = stats["count"]
        hard_total = stats.get("hard_total", 0)
        accept_total = stats.get("accept_total", 0)
        summary["groups"][key] = {
            "count": count,
            "soft_pass_rate": stats["soft_pass"] / count if count else 0.0,
            "rs_success_rate": stats["rs_pass"] / count if count else 0.0,
            "payload_match_rate": stats["payload_match_pass"] / count if count else 0.0,
            "decoded_match_rate": stats["decoded_match_pass"] / count if count else 0.0,
            "decode_success_rate": stats["decoded_match_pass"] / count if count else 0.0,
            "consistency_pass_rate": stats["consistency_pass"] / count if count else 0.0,
            "opening_pass_rate": stats["opening_pass"] / count if count else 0.0,
            "proof_pass_rate": stats["proof_pass"] / count if count else 0.0,
            "hard_pass_rate": stats["hard_pass"] / hard_total if hard_total else None,
            "hard_pass_total": hard_total,
            "accept_pass_rate": stats["accept_pass"] / accept_total if accept_total else None,
            "accept_pass_total": accept_total,
            "fail_reason_counts": stats["fail_reason_counts"],
        }

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[eval] wrote scores: {scores_path}")
    print(f"[eval] wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
