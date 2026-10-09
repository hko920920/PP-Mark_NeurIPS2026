"""Blind sync + DDIM extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

from .config import GlobalConfig
from .noise import hash_uniforms, build_bit_index_map
from .payload import bits_to_bytes
from .rs import ReedSolomonCodec
from .tables import InverseCDFTable
from .sampling import SampleObservation


@dataclass(slots=True)
class DDIMSchedule:
    timesteps: np.ndarray
    alpha_cumprod: np.ndarray


def make_ddim_schedule(num_steps: int = 10, beta_start: float = 1e-4, beta_end: float = 5e-3) -> DDIMSchedule:
    """Generate a lightweight linear beta schedule for DDIM-style inversion."""
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    betas = np.linspace(beta_start, beta_end, num_steps, dtype=np.float32)
    alphas = 1.0 - betas
    alpha_cumprod = np.cumprod(alphas)
    return DDIMSchedule(timesteps=np.arange(num_steps, dtype=np.int32), alpha_cumprod=alpha_cumprod)


@dataclass(slots=True)
class ExtractionResult:
    bits: np.ndarray
    codeword: bytes
    message: bytes | None
    diagnostics: Dict[str, Any]


def _ensure_2d(image: np.ndarray) -> np.ndarray:
    """Convert HWC images to a single 2D channel if needed."""
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        return image.mean(axis=2)
    raise ValueError("Expected 2D or HWC image input")


def _resize_nearest(image: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    """Lightweight resize fallback when OpenCV is unavailable."""
    tgt_h, tgt_w = target_hw
    y_idx = np.linspace(0, image.shape[0] - 1, tgt_h).astype(np.int64)
    x_idx = np.linspace(0, image.shape[1] - 1, tgt_w).astype(np.int64)
    return image[np.ix_(y_idx, x_idx)]


def coarse_align(image: np.ndarray, target_resolution: Tuple[int, int] | None = None) -> Tuple[np.ndarray, dict]:
    """
    Coarse rotation/scale alignment placeholder.

    For now this just ensures the spatial resolution matches the configured width/height.
    """
    base = _ensure_2d(np.asarray(image, dtype=np.float32))
    if target_resolution is None:
        return base, {"method": "identity", "from": (base.shape[1], base.shape[0])}
    target_w, target_h = target_resolution
    if base.shape == (target_h, target_w):
        return base, {"method": "identity", "from": (base.shape[1], base.shape[0]), "to": target_resolution}
    if cv2 is not None:
        interp = cv2.INTER_AREA if (target_w < base.shape[1] or target_h < base.shape[0]) else cv2.INTER_CUBIC
        resized = cv2.resize(base, (target_w, target_h), interpolation=interp)
        backend = "cv2"
    else:  # pragma: no cover - rarely triggered when cv2 is installed
        resized = _resize_nearest(base, (target_h, target_w))
        backend = "numpy"
    return resized.astype(np.float32, copy=False), {
        "method": "resize",
        "from": (base.shape[1], base.shape[0]),
        "to": target_resolution,
        "backend": backend,
    }


def ddim_invert(
    image: np.ndarray,
    target_resolution: Tuple[int, int] | None = None,
    normalize: bool = False,
    external_inverter: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """
    Deterministic DDIM-style inversion placeholder.

    If no model_fn is supplied, this returns a normalized 2D latent (identity path)
    to preserve compatibility with spread-spectrum demodulation. When a model_fn is
    provided, a minimal DDIM encode pass (forward to noisy latent) is executed using
    the supplied epsilon predictor.
    """
    if external_inverter is not None:
        return external_inverter(image)
    aligned, _ = coarse_align(image, target_resolution)
    latent = np.asarray(aligned, dtype=np.float32)
    if normalize:
        mean = float(latent.mean())
        std = float(latent.std())
        if std > 1e-6:
            latent = (latent - mean) / std
        else:
            latent = latent - mean
    return latent


def ddim_encode(
    image: np.ndarray,
    model_fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
    schedule: DDIMSchedule | None = None,
    target_resolution: Tuple[int, int] | None = None,
    normalize: bool = True,
    noise_scale: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """
    Lightweight DDIM encode (image → noisy latent).

    Args:
        image: Input image/latent (HWC or 2D).
        model_fn: Optional epsilon predictor: fn(x_t, t_idx) -> epsilon array.
        schedule: Optional precomputed DDIMSchedule; defaults to a short linear beta schedule.
        target_resolution: Optional resize target (W, H).
        normalize: Whether to zero-mean/unit-var normalize before encoding.
        noise_scale: Deterministic noise amplitude when model_fn is absent (0 keeps identity-ish path).
        seed: RNG seed used when model_fn is None and noise_scale > 0.
    """
    aligned, _ = coarse_align(image, target_resolution)
    x0 = np.asarray(aligned, dtype=np.float32)
    if normalize:
        mean = float(x0.mean())
        std = float(x0.std())
        if std > 1e-6:
            x0 = (x0 - mean) / std
        else:
            x0 = x0 - mean

    sched = schedule or make_ddim_schedule()
    rng = np.random.default_rng(seed)

    def _predict_eps(x_t: np.ndarray, step: int) -> np.ndarray:
        if model_fn is not None:
            eps = np.asarray(model_fn(x_t, step), dtype=np.float32)
            if eps.shape != x_t.shape:
                raise ValueError("model_fn must return the same shape as input x_t")
            return eps
        if noise_scale > 0.0:
            return noise_scale * rng.standard_normal(x_t.shape, dtype=np.float32)
        return np.zeros_like(x_t, dtype=np.float32)

    x_t = x0
    for idx, a_bar in enumerate(sched.alpha_cumprod):
        eps_t = _predict_eps(x_t, int(sched.timesteps[idx]))
        sqrt_a_bar = float(np.sqrt(max(a_bar, 1e-8)))
        sqrt_one = float(np.sqrt(max(1.0 - a_bar, 0.0)))
        x_t = sqrt_a_bar * x0 + sqrt_one * eps_t
    return x_t


def recover_bits(
    aligned_image: np.ndarray,
    binding: int,
    config: GlobalConfig,
    inverse_cdf: InverseCDFTable,
    codec: ReedSolomonCodec | None = None,
    bit_length: int | None = None,
) -> ExtractionResult:
    """
    Demodulate spread-spectrum bits from an aligned latent grid.

    Args:
        aligned_image: 2D array after DDIM inversion / coarse alignment.
        binding: Poseidon/SHA binding used during embedding (rebuilds uniforms).
        config: Global configuration (image/alpha/watermark sizes).
        inverse_cdf: Same lookup table used during embedding.
        codec: Optional RS decoder for payload recovery.
        bit_length: Override bit-length; defaults to RS(n)*8 or config watermark length.
    """
    latent = _ensure_2d(np.asarray(aligned_image, dtype=np.float32))
    target = (config.image.height, config.image.width)
    align_diag: Dict[str, Any]
    if latent.shape != target:
        latent, align_diag = coarse_align(latent, (config.image.width, config.image.height))
    else:
        align_diag = {"method": "identity", "from": (latent.shape[1], latent.shape[0]), "to": (config.image.width, config.image.height)}

    flat = latent.reshape(-1)
    total_pixels = config.image.total_pixels
    if flat.size < total_pixels:
        raise ValueError("Aligned image has fewer pixels than expected.")
    flat = flat[:total_pixels]

    uniforms = hash_uniforms(binding, total_pixels, backend=config.zk.sample_backend)
    gaussian = inverse_cdf.batch_lookup(uniforms)
    alpha = float(alpha_fixed) / FIXED_SCALE if alpha_fixed is not None else float(config.image.alpha)
    if alpha == 0.0:
        raise ValueError("Alpha must be non-zero for demodulation.")
    residual = (flat - gaussian) / alpha

    bit_len = bit_length or ((codec.n * 8) if codec else config.watermark.rs_n * 8)
    idx_map = build_bit_index_map(binding, config.image.width, config.image.height, bit_len, backend=config.zk.sample_backend)
    indices = idx_map.reshape(-1)[:total_pixels]
    weights = np.bincount(indices, weights=residual, minlength=bit_len)
    counts = np.bincount(indices, minlength=bit_len)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_residual = weights / np.maximum(counts, 1)
    bits = (mean_residual >= 0).astype(np.uint8)
    codeword = bits_to_bytes(bits.tolist())

    decoded = False
    message: bytes | None = None
    bit_flip = False
    if codec is not None:
        try:
            message = codec.decode(codeword)
            decoded = True
        except Exception:
            decoded = False
        if not decoded:
            flipped = (1 - bits).astype(np.uint8)
            flipped_codeword = bits_to_bytes(flipped.tolist())
            try:
                message = codec.decode(flipped_codeword)
                decoded = True
                bits = flipped
                codeword = flipped_codeword
                bit_flip = True
            except Exception:
                decoded = False

    sq_weights = np.bincount(indices, weights=residual * residual, minlength=bit_len)
    variance = sq_weights / np.maximum(counts, 1) - mean_residual * mean_residual
    std_residual = np.sqrt(np.clip(variance, a_min=0.0, a_max=None))
    confidence = np.divide(mean_residual, std_residual + 1e-6)

    diagnostics = {
        "bit_length": bit_len,
        "alignment": align_diag,
        "residual_mean": mean_residual.tolist(),
        "residual_std": std_residual.tolist(),
        "confidence": confidence.tolist(),
        "decoded": decoded,
        "bit_flip_applied": bit_flip,
    }
    return ExtractionResult(bits=bits, codeword=codeword, message=message, diagnostics=diagnostics)


def extract_watermark(
    image: np.ndarray,
    binding: int,
    config: GlobalConfig,
    inverse_cdf: InverseCDFTable,
    codec: ReedSolomonCodec,
    normalize: bool = False,
    model_fn: Callable[[np.ndarray, int], np.ndarray] | None = None,
    schedule: DDIMSchedule | None = None,
    noise_scale: float = 0.0,
    seed: int = 0,
) -> ExtractionResult:
    """
    Convenience wrapper: coarse align → DDIM inversion → demodulation.
    """
    aligned, align_diag = coarse_align(image, (config.image.width, config.image.height))
    latent = ddim_encode(
        aligned,
        model_fn=model_fn,
        schedule=schedule,
        target_resolution=None,
        normalize=normalize,
        noise_scale=noise_scale,
        seed=seed,
    )
    result = recover_bits(latent, binding, config, inverse_cdf, codec=codec)
    # Preserve outer alignment diagnostics.
    result.diagnostics["alignment"] = align_diag
    return result


@dataclass(slots=True)
class SyncResult:
    dx: int
    dy: int
    score: float
    theta: float = 0.0
    meta: Dict[str, Any] | None = None


def _shift_indices(indices: np.ndarray, width: int, height: int, dx: int, dy: int) -> np.ndarray:
    """Shift flat indices by (dx, dy); returns -1 for out-of-bounds entries."""
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


def sync_search_translation(
    latent: np.ndarray,
    binding: int,
    config: GlobalConfig,
    inverse_cdf: InverseCDFTable,
    sample_indices: np.ndarray,
    bits: np.ndarray,
    max_shift: int = 0,
    rotation_deg: float = 0.0,
) -> SyncResult:
    """
    Search over integer translations (optionally on a rotated view) to maximize correlation between observed samples and expected combined values.

    Returns best (dx, dy) and correlation score. Rotation is not applied here.
    """
    width, height = config.image.width, config.image.height
    total_pixels = width * height
    if latent.shape[0] != height or latent.shape[1] != width:
        raise ValueError("latent shape does not match config resolution for sync search")
    view = latent
    if abs(rotation_deg) > 1e-6:
        if cv2 is None:
            raise RuntimeError("Rotation requested but OpenCV is unavailable")
        center = (width / 2.0, height / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
        view = cv2.warpAffine(latent, rot_mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    uniforms = hash_uniforms(binding, total_pixels, backend=config.zk.sample_backend)
    gaussian = inverse_cdf.batch_lookup(uniforms).reshape(height, width)
    bit_len = int(len(bits))
    bit_idx_map = build_bit_index_map(binding, width, height, bit_len, backend=config.zk.sample_backend)
    alpha = float(config.image.alpha)
    best = SyncResult(dx=0, dy=0, score=-1e30, theta=rotation_deg)
    shift_range = [0] if max_shift <= 0 else list(range(-max_shift, max_shift + 1))
    for dy in shift_range:
        for dx in shift_range:
            shifted = _shift_indices(sample_indices, width, height, dx, dy)
            if np.any(shifted < 0):
                continue
            ys = shifted // width
            xs = shifted % width
            obs = view[ys, xs]
            exp_gauss = gaussian[ys, xs]
            exp_bits = bits[bit_idx_map[ys, xs]]
            exp_combined = exp_gauss + alpha * (2 * exp_bits - 1)
            num = float(np.sum(obs * exp_combined))
            den = float(np.sqrt(np.sum(obs * obs) + 1e-9) * np.sqrt(np.sum(exp_combined * exp_combined) + 1e-9))
            corr = num / den if den > 0 else -1e30
            if corr > best.score:
                best = SyncResult(dx=dx, dy=dy, score=corr, theta=rotation_deg)
    return best


def sync_search(
    latent: np.ndarray,
    binding: int,
    config: GlobalConfig,
    inverse_cdf: InverseCDFTable,
    sample_indices: np.ndarray,
    bits: np.ndarray,
    max_shift: int = 0,
    max_rotation: float = 0.0,
    rotation_step: float = 1.0,
    rotation_strategy: str = "auto",
) -> tuple[SyncResult, np.ndarray]:
    """Combined rotation + translation search (coarse or coarse-to-fine). Returns best SyncResult and aligned latent view."""
    if rotation_strategy not in {"auto", "grid", "coarse_to_fine"}:
        raise ValueError(f"Unsupported rotation_strategy: {rotation_strategy}")
    if rotation_strategy == "auto":
        rotation_strategy = "coarse_to_fine" if max_rotation > 10 else "grid"

    width, height = config.image.width, config.image.height
    total_pixels = width * height
    if latent.shape[0] != height or latent.shape[1] != width:
        raise ValueError("latent shape does not match config resolution for sync search")
    bits = np.asarray(bits, dtype=np.int8)
    uniforms = hash_uniforms(binding, total_pixels, backend=config.zk.sample_backend)
    gaussian = inverse_cdf.batch_lookup(uniforms).reshape(height, width)
    bit_len = int(len(bits))
    bit_idx_map = build_bit_index_map(binding, width, height, bit_len, backend=config.zk.sample_backend)
    alpha = float(config.image.alpha)

    def _search_angles(angles: List[float], shift_limit: int) -> tuple[SyncResult, np.ndarray, List[dict]]:
        candidates: List[dict] = []
        best: SyncResult | None = None
        best_view: np.ndarray | None = None
        shift_range = [0] if shift_limit <= 0 else list(range(-shift_limit, shift_limit + 1))
        for ang in angles:
            if abs(ang) > 1e-6:
                if cv2 is None:
                    continue
                center = (width / 2.0, height / 2.0)
                rot_mat = cv2.getRotationMatrix2D(center, ang, 1.0)
                view = cv2.warpAffine(latent, rot_mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            else:
                view = latent
            best_angle = SyncResult(dx=0, dy=0, score=-1e30, theta=ang)
            for dy in shift_range:
                for dx in shift_range:
                    shifted = _shift_indices(sample_indices, width, height, dx, dy)
                    if np.any(shifted < 0):
                        continue
                    ys = shifted // width
                    xs = shifted % width
                    obs = view[ys, xs]
                    exp_gauss = gaussian[ys, xs]
                    exp_bits = bits[bit_idx_map[ys, xs]]
                    exp_combined = exp_gauss + alpha * (2 * exp_bits - 1)
                    num = float(np.sum(obs * exp_combined))
                    den = float(np.sqrt(np.sum(obs * obs) + 1e-9) * np.sqrt(np.sum(exp_combined * exp_combined) + 1e-9))
                    corr = num / den if den > 0 else -1e30
                    if corr > best_angle.score:
                        best_angle = SyncResult(dx=dx, dy=dy, score=corr, theta=ang)
            candidates.append(
                {
                    "angle": ang,
                    "score": best_angle.score,
                    "dx": best_angle.dx,
                    "dy": best_angle.dy,
                }
            )
            if best is None or best_angle.score > best.score:
                best = best_angle
                best_view = view
        if best is None or best_view is None:
            best = SyncResult(dx=0, dy=0, score=0.0, theta=0.0)
            best_view = latent
        return best, best_view, candidates

    search_stats: Dict[str, Any] = {}
    if rotation_strategy == "coarse_to_fine":
        coarse_step = 15.0
        coarse_max = max(90.0, float(max_rotation))
        coarse_angles = _rotation_sweep(coarse_max, coarse_step)
        coarse_best, _, coarse_candidates = _search_angles(coarse_angles, 0)
        fine_span = 7.0
        fine_step = 1.0
        fine_angles = _rotation_window(coarse_best.theta, fine_span, fine_step, coarse_max)
        best, best_view, fine_candidates = _search_angles(fine_angles, max_shift)
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
        best, best_view, candidates = _search_angles(angles, max_shift)
        search_stats = {
            "rotation_search_strategy": "grid",
            "best_rotation_deg": best.theta,
            "coarse_candidates": len(angles),
            "fine_candidates": 0,
            "rotation_top3_scores": _top_candidates(candidates),
        }

    best.meta = search_stats
    return best, best_view


def recompute_sample_root(
    latent: np.ndarray,
    binding: int,
    config: GlobalConfig,
    sample_indices: np.ndarray,
    bits: np.ndarray,
    alpha_fixed: int,
    backend: str = "sha",
    dx: int = 0,
    dy: int = 0,
    alpha_effective: float | None = None,
):
    """
    Recompute sample Merkle root from an aligned latent and provided sample/binding info.
    """
    from .risc0_runner import build_sample_merkle_sha
    from .halo2_interface import build_sample_merkle, FIXED_SCALE

    width, height = config.image.width, config.image.height
    if latent.shape[0] != height or latent.shape[1] != width:
        raise ValueError("latent shape does not match config resolution for root recomputation")
    shifted = _shift_indices(sample_indices, width, height, dx, dy)
    if np.any(shifted < 0):
        raise ValueError("Shifted sample indices fall outside image bounds")

    ys = shifted // width
    xs = shifted % width
    obs = latent[ys, xs]
    bit_len = int(len(bits))
    bit_idx_map = build_bit_index_map(binding=binding, width=width, height=height, bit_len=bit_len, backend=config.zk.sample_backend)
    bits_for_samples = [int(b) for b in bits[bit_idx_map[ys, xs]]]
    if alpha_effective is None:
        alpha = float(alpha_fixed) / FIXED_SCALE
    else:
        alpha = float(alpha_effective)
    gaussian_fixed = [int(round((float(c) - alpha * (2 * b - 1)) * FIXED_SCALE)) for c, b in zip(obs, bits_for_samples)]
    combined_fixed = [int(round(float(c) * FIXED_SCALE)) for c in obs]
    observations = []
    for idx, c, g in zip(sample_indices, obs, gaussian_fixed):
        observations.append(SampleObservation(index=int(idx), uniform=0.0, z_expected=float(g) / FIXED_SCALE, z_observed=float(c)))
    if backend == "sha":
        root, paths = build_sample_merkle_sha(
            observations,
            gaussian_fixed=gaussian_fixed,
            combined_fixed=combined_fixed,
            bits=bits_for_samples,
        )
    else:
        root, paths = build_sample_merkle(
            observations,
            gaussian_fixed=gaussian_fixed,
            combined_fixed=combined_fixed,
            bits=bits_for_samples,
        )
    return root, paths
