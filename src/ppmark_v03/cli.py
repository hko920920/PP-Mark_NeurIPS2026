"""PP-Mark v0.3 CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import secrets
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageOps

from .attestation import (
    STREAMING_PROTOCOL_VERSION,
    StreamingAttestationStatement,
    alpha_fixed_and_scale_q30,
    canonicalize_scaled_channel,
    derive_streaming_challenge_seed,
    derive_opening_positions,
    expected_sample_relation,
    hash_lut_file,
    hash_sample_indices,
    hash_trace_commitment,
    opening_digest,
    parse_hex32,
    producer_key_commitment,
)
from .config import GlobalConfig
from .cuda import DeviceConfig, get_watermark_kernel
from .halo2_interface import (
    FIXED_SCALE,
    Halo2Package,
    PublicInputs,
    Witness,
    build_sample_merkle,
)
from .halo2_runner import Halo2Paths, prepare_prover_package, run_halo2_prover, verify_halo2_proof
from .keys import KeyPair, derive_public_point, generate_keypair
from .payload import (
    build_payload_bn254,
    build_payload_pallas,
    build_payload_risc0,
    bits_to_bytes,
    bytes_to_bits,
    make_ctx_hash,
)
from .rs import ReedSolomonCodec
from .sampling import SampleSet, SampleTrace, deterministic_sample
from .tables import InverseCDFTable
from .risc0_runner import (
    Risc0Paths,
    build_sample_merkle_sha,
    prepare_risc0_package,
    run_risc0_prover,
    verify_risc0_receipt,
)
from .sp1_runner import (
    SP1Paths,
    SP1SampleWitness,
    SP1Witness,
    prepare_sp1_package,
    run_sp1_prover,
    verify_sp1_receipt,
)
from .sync import SyncResult, recompute_sample_root, sync_search, ddim_invert
from .ddim_unet import DiffusersDDIMInverter, DiffusersDDIMConfig
from .noise import build_bit_index_map, hash_uniforms

PPMARK_VERSION = "0.3"
ATTEST_METADATA_VERSION = "ppmark_attest_v2"
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
OPENING_K_DEFAULT = 32


def _bicubic_resample() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC


def _normalize_image_for_hash(image: Image.Image, size: tuple[int, int]) -> np.ndarray:
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    image = image.resize(size, resample=_bicubic_resample())
    arr = np.asarray(image, dtype=np.float32)
    arr = np.clip(np.round(arr), 0, 255).astype(np.uint8)
    return arr


def _hash_normalized_image(image: Image.Image, size: tuple[int, int]) -> bytes:
    normalized = _normalize_image_for_hash(image, size)
    return hashlib.sha256(normalized.tobytes()).digest()


def _opening_digest(entries: list[dict[str, int]]) -> bytes:
    hasher = hashlib.sha256()
    for entry in entries:
        hasher.update(int(entry["pos"]).to_bytes(4, "little", signed=False))
        hasher.update(int(entry["index"]).to_bytes(4, "little", signed=False))
        hasher.update(int(entry["gaussian_fixed"]).to_bytes(8, "little", signed=True))
        hasher.update(int(entry["combined_fixed"]).to_bytes(8, "little", signed=True))
        hasher.update(int(entry["bit"]).to_bytes(1, "little", signed=False))
    return hasher.digest()


def _load_secret(secret_hex: str | None, secret_file: Path | None) -> KeyPair:
    if secret_hex:
        value = int(secret_hex, 16)
        public = derive_public_point(value)
        return KeyPair(secret=value, public=public)
    if secret_file and secret_file.exists():
        value = int(secret_file.read_text().strip(), 16)
        public = derive_public_point(value)
        return KeyPair(secret=value, public=public)
    return generate_keypair()


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve(base: Path, target: str) -> Path:
    candidate = Path(target)
    if not candidate.is_absolute():
        candidate = (base / candidate).resolve()
    return candidate


def _get_expected_message(metadata: Dict[str, Any]) -> bytes:
    msg_hex = metadata.get("message_hex")
    if msg_hex:
        return bytes.fromhex(str(msg_hex))
    binding = str(metadata["binding"])
    if binding.startswith("0x"):
        binding = binding[2:]
    return bytes.fromhex(binding)


def _normalize_hex(value: str) -> str:
    text = str(value).lower()
    if text.startswith("0x"):
        text = text[2:]
    return text


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_tau_file(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "tau_score" in payload:
        return float(payload["tau_score"])
    if "tau" in payload:
        return float(payload["tau"])
    if "score_threshold" in payload:
        return float(payload["score_threshold"])
    raise ValueError(f"tau_score missing in {path}")


def _resolve_score_threshold(args: argparse.Namespace, metadata: Dict[str, Any]) -> float | None:
    if getattr(args, "score_threshold", None) is not None:
        return float(args.score_threshold)
    if getattr(args, "tau", None) is not None:
        return float(args.tau)
    tau_file = getattr(args, "tau_file", "")
    if tau_file:
        return _load_tau_file(Path(tau_file))
    for key in ("tau_score", "score_threshold", "tau"):
        if key in metadata:
            return float(metadata[key])
    return None


def _ensure_sp1_backend(cfg: GlobalConfig, allow_legacy: bool) -> None:
    backend = cfg.zk.backend.lower()
    if backend == "sp1":
        return
    if not allow_legacy:
        raise RuntimeError(f"Legacy backend '{cfg.zk.backend}' is disabled; use SP1 or pass --allow-legacy-backend")
    print(f"[legacy] backend={cfg.zk.backend} requested; SP1 is the current default")


def _get_expected_payload_hash(metadata: Dict[str, Any]) -> str:
    payload_hash = metadata.get("payload_hash")
    if payload_hash:
        return _normalize_hex(payload_hash)
    message = _get_expected_message(metadata)
    return _sha256_hex(message)


def _decode_payload_from_latent(
    latent: np.ndarray,
    binding: int,
    config: GlobalConfig,
    inverse_cdf: InverseCDFTable,
    codec: ReedSolomonCodec,
    bit_idx_map: np.ndarray,
    alpha: float,
) -> tuple[bytes | None, bool, float]:
    latent2d = np.asarray(latent, dtype=np.float32)
    if latent2d.shape != (config.image.height, config.image.width):
        raise ValueError("latent shape does not match config resolution for decoding")
    flat = latent2d.reshape(-1)[: config.image.total_pixels]
    if alpha == 0.0:
        raise ValueError("alpha must be non-zero for decoding")
    uniforms = hash_uniforms(binding, config.image.total_pixels, backend=config.zk.sample_backend)
    gaussian = inverse_cdf.batch_lookup(uniforms)
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
            bit_flip = True
        except Exception:
            message = None
    return message, bit_flip, threshold


def _crop_box_grid(width: int, height: int, keep_ratio: float, grid: int) -> list[tuple[int, int, int, int]]:
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
    base_box: tuple[int, int, int, int],
    width: int,
    height: int,
    keep_ratio: float,
    step_frac: float,
    grid: int,
) -> list[tuple[int, int, int, int]]:
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


def _apply_crop_and_resize(image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
    width, height = image.size
    cropped = image.crop(box)
    return cropped.resize((width, height), resample=Image.BICUBIC)


def _crop_search_latent(
    base_img: Image.Image,
    invert_fn,
    binding: int,
    config: GlobalConfig,
    inverse_cdf: InverseCDFTable,
    sample_indices: np.ndarray,
    bits: np.ndarray,
    keep_ratio: float,
    grid: int,
    refine: bool,
    refine_grid: int,
    refine_step: float,
    sigma_meta: float,
) -> tuple[np.ndarray, dict]:
    width, height = base_img.size
    coarse_boxes = _crop_box_grid(width, height, keep_ratio, grid=grid)
    candidate_scores: list[dict] = []
    latent_cache: dict[tuple[int, int, int, int], np.ndarray] = {}

    def _encode_box(box: tuple[int, int, int, int]) -> np.ndarray:
        if box in latent_cache:
            return latent_cache[box]
        cropped = _apply_crop_and_resize(base_img, box)
        latent = invert_fn(np.array(cropped, dtype=np.float32))
        if np.isnan(latent).any() or np.isinf(latent).any():
            raise RuntimeError("DDIM inversion produced NaN/Inf latents during crop search")
        if sigma_meta != 0:
            latent = latent / sigma_meta
        latent_cache[box] = latent
        return latent

    for box in coarse_boxes:
        latent = _encode_box(box)
        sync_res, _ = sync_search(
            latent=latent,
            binding=binding,
            config=config,
            inverse_cdf=inverse_cdf,
            sample_indices=sample_indices,
            bits=bits,
            max_shift=0,
            max_rotation=0.0,
            rotation_step=1.0,
            rotation_strategy="grid",
        )
        candidate_scores.append({"box": box, "score": sync_res.score})

    best_box = max(candidate_scores, key=lambda item: item["score"])["box"]
    refine_boxes: list[tuple[int, int, int, int]] = []
    if refine:
        refine_boxes = _crop_box_refine(best_box, width, height, keep_ratio, step_frac=refine_step, grid=refine_grid)
        for box in refine_boxes:
            latent = _encode_box(box)
            sync_res, _ = sync_search(
                latent=latent,
                binding=binding,
                config=config,
                inverse_cdf=inverse_cdf,
                sample_indices=sample_indices,
                bits=bits,
                max_shift=0,
                max_rotation=0.0,
                rotation_step=1.0,
                rotation_strategy="grid",
            )
            candidate_scores.append({"box": box, "score": sync_res.score})
        best_box = max(candidate_scores, key=lambda item: item["score"])["box"]

    stats = {
        "crop_keep_ratio": keep_ratio,
        "best_crop_box": best_box,
        "crop_candidates_count": len(candidate_scores),
        "crop_coarse_candidates": len(coarse_boxes),
        "crop_refine_candidates": len(refine_boxes),
        "crop_top3_scores": sorted(candidate_scores, key=lambda item: item["score"], reverse=True)[:3],
    }
    return _encode_box(best_box), stats


def _device_config_from_args(args: argparse.Namespace) -> DeviceConfig:
    backend = getattr(args, "device_backend", "cpu")
    uniform_backend = getattr(args, "uniform_backend", "auto")
    return DeviceConfig(backend=backend, uniform_backend=uniform_backend)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8192):
            h.update(chunk)
    return "0x" + h.hexdigest()


def _hex_to_int(value: str) -> int:
    s = value.strip()
    if s.lower().startswith("0x"):
        s = s[2:]
    return int(s, 16)


def _load_vk_from_env(cfg: GlobalConfig) -> str | None:
    """Return verifier command, with SP1 VK env injected if provided in config."""
    env_cmd = cfg.zk.verifier_cmd
    if cfg.zk.backend.lower() == "sp1":
        import os

        if cfg.zk.vk_path:
            os.environ["SP1_VK_PATH"] = str(Path(cfg.zk.vk_path))
        if cfg.zk.prover_cmd and cfg.zk.prover_cmd.endswith("sp1-genguard-host"):
            return env_cmd
    return env_cmd


def _resolve_vk_path(cfg: GlobalConfig, model_id: str | None) -> None:
    """Set SP1_VK_PATH from registry if available and model_id is provided."""
    if cfg.zk.backend.lower() != "sp1":
        return
    if model_id and cfg.zk.vk_registry and model_id in cfg.zk.vk_registry:
        import os

        os.environ["SP1_VK_PATH"] = str(Path(cfg.zk.vk_registry[model_id]))


def _is_sdxl_model(model_id: str) -> bool:
    tag = model_id.lower()
    return "sdxl" in tag or "stable-diffusion-xl" in tag


def _relative_artifact(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_secret_output(secret_key: bytes, target: str | None, public_root: Path) -> None:
    if not target:
        print(
            "[prover] producer secret was not exported; pass --secret-output outside the "
            "public output directory if this key must be reused"
        )
        return
    path = Path(target).resolve()
    try:
        path.relative_to(public_root.resolve())
    except ValueError:
        pass
    else:
        raise RuntimeError("--secret-output must be outside the public artifact directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(secret_key.hex() + "\n", encoding="ascii")
    print(f"[prover] producer secret exported separately to {path}")


def _run_sp1_attest_prover(args: argparse.Namespace, cfg: GlobalConfig) -> None:
    """Generate a canonical Detect/Attest artifact for the SP1 backend."""
    if cfg.zk.sample_backend != "sha256":
        raise RuntimeError("SP1 Attest requires zk.sample_backend='sha256'")
    if (cfg.watermark.rs_n, cfg.watermark.rs_k) != (64, 32):
        raise RuntimeError("SP1 Attest currently requires RS(64,32)")
    if getattr(args, "use_random_latents", False):
        raise RuntimeError("--use-random-latents is incompatible with an Attest proof")

    output_dir = Path(args.output).resolve()
    _ensure_dir(output_dir)
    retain_private = bool(
        getattr(args, "retain_private_witness", False)
        or getattr(args, "skip_proof", False)
    )
    private_dir = output_dir / "private"
    if retain_private:
        _ensure_dir(private_dir)

    codec = ReedSolomonCodec(n=cfg.watermark.rs_n, k=cfg.watermark.rs_k)
    keypair = _load_secret(
        args.secret_hex,
        Path(args.secret_file) if args.secret_file else None,
    )
    secret_key = keypair.secret_bytes()
    if len(secret_key) != 32:
        raise RuntimeError("producer secret must serialize to exactly 32 bytes")
    _write_secret_output(secret_key, getattr(args, "secret_output", None), output_dir)
    seed = secrets.token_bytes(32) if args.seed_hex is None else bytes.fromhex(args.seed_hex)
    if len(seed) != 32:
        raise ValueError("--seed-hex must encode exactly 32 bytes")

    model_id = args.model_id or cfg.model.base
    prompt_hash_bytes = hashlib.sha256(args.prompt.encode("utf-8")).digest()
    prompt_hash = int.from_bytes(prompt_hash_bytes, "big")
    ctx_hash_int = make_ctx_hash(
        prompt_hash=prompt_hash,
        model_id=model_id,
        seed=seed,
        width=cfg.image.width,
        height=cfg.image.height,
        usecase_tag=getattr(args, "usecase_tag", None),
        user_tag=getattr(args, "user_tag", None),
    )
    ctx_hash = ctx_hash_int.to_bytes(32, "big")
    binding_bytes = hashlib.sha256(b"ctx_hash" + ctx_hash + secret_key).digest()
    binding = int.from_bytes(binding_bytes, "big")
    codeword = codec.encode(binding_bytes)
    codeword_bits = bytes_to_bits(codeword)
    sample_set = deterministic_sample(
        total_pixels=cfg.image.total_pixels,
        width=cfg.image.width,
        height=cfg.image.height,
        key_material=codeword,
        count=cfg.image.sample_count(),
    )

    inverse_cdf = InverseCDFTable.from_file(cfg.tables.inverse_cdf_path)
    lut_hash = hash_lut_file(cfg.tables.inverse_cdf_path)
    if inverse_cdf.size != 65_536:
        raise RuntimeError("SP1 Attest requires the committed 65,536-entry inverse-CDF LUT")
    kernel = get_watermark_kernel(_device_config_from_args(args), cfg, inverse_cdf)
    embedding = kernel.embed(
        binding=binding,
        bit_sequence=codeword_bits,
        sample_set=sample_set,
    )
    alpha_fixed, scale_q30, alpha_effective, scale_factor = alpha_fixed_and_scale_q30(
        cfg.image.alpha
    )
    latent_scaled = (embedding.latent_noise.astype(np.float32) * scale_factor).astype(np.float32)
    gaussian_fixed_all, canonical_ch0 = canonicalize_scaled_channel(
        embedding.noise_artifacts.gaussian,
        embedding.noise_artifacts.spread,
        scale_factor_q30=scale_q30,
        alpha_effective_fixed=alpha_fixed,
    )
    canonical_ch0 = canonical_ch0.reshape(cfg.image.height, cfg.image.width)
    latent_scaled[0] = canonical_ch0

    latent_path: Path | None = None
    z0_path: Path | None = None
    if retain_private:
        latent_path = private_dir / "canonical_latent_noise.npy"
        np.save(latent_path, latent_scaled)

    png_path = output_dir / "watermarked.png"
    guidance_scale = 7.5
    num_inference_steps = 50
    negative_prompt = ""
    try:
        import torch  # type: ignore
        from diffusers import (  # type: ignore
            DDIMScheduler,
            StableDiffusionPipeline,
            StableDiffusionXLPipeline,
        )

        use_sdxl = _is_sdxl_model(model_id)
        pipe_cls = StableDiffusionXLPipeline if use_sdxl else StableDiffusionPipeline
        model_kwargs: dict[str, Any] = {"torch_dtype": torch.float16}
        if use_sdxl:
            model_kwargs["variant"] = "fp16"
        try:
            pipe = pipe_cls.from_pretrained(model_id, **model_kwargs)
        except (TypeError, ValueError):
            model_kwargs.pop("variant", None)
            pipe = pipe_cls.from_pretrained(model_id, **model_kwargs)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
        z0prime = (
            latent_scaled
            if latent_scaled.ndim == 3
            else np.repeat(latent_scaled[None, :, :], 4, axis=0)
        )
        torch_latents = torch.from_numpy(z0prime[None, ...]).to(
            device="cuda", dtype=torch.float16
        )
        generator = torch.Generator(device="cuda")
        generator.manual_seed(int.from_bytes(seed, "big") % (2**63 - 1))
        result = pipe(
            prompt=args.prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            latents=torch_latents,
            output_type="np",
            generator=generator,
        )
        image_u8 = np.clip(result.images[0] * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(image_u8).save(png_path)
        if retain_private:
            z0_path = private_dir / "z0_latents.npy"
            np.save(z0_path, z0prime)
    except Exception as exc:  # pragma: no cover - GPU/model integration
        raise RuntimeError(f"image generation failed: {exc}") from exc

    hash_size = (int(HASH_SPEC["resize"][0]), int(HASH_SPEC["resize"][1]))
    with Image.open(png_path) as image:
        image_hash = _hash_normalized_image(image, hash_size)

    sample_trace = SampleTrace()
    gaussian_fixed: list[int] = []
    combined_fixed: list[int] = []
    bits_for_samples: list[int] = []
    sp1_samples: list[SP1SampleWitness] = []
    canonical_flat = canonical_ch0.reshape(-1)
    for index in sample_set.indices:
        bit, expected_gaussian, expected_combined = expected_sample_relation(
            binding=binding_bytes,
            codeword_bits=codeword_bits,
            flattened_index=index,
            width=cfg.image.width,
            inverse_cdf=inverse_cdf,
            scale_factor_q30=scale_q30,
            alpha_effective_fixed=alpha_fixed,
        )
        actual_combined = int(round(float(canonical_flat[index]) * FIXED_SCALE))
        if expected_gaussian != int(gaussian_fixed_all[index]):
            raise RuntimeError(f"canonical Gaussian mismatch at latent index {index}")
        if expected_combined != actual_combined:
            raise RuntimeError(f"canonical combined value mismatch at latent index {index}")
        gaussian_fixed.append(expected_gaussian)
        combined_fixed.append(expected_combined)
        bits_for_samples.append(bit)
        sample_trace.record(
            index,
            0.0,
            expected_gaussian / FIXED_SCALE,
            expected_combined / FIXED_SCALE,
        )
        sp1_samples.append(
            SP1SampleWitness(
                index=index,
                gaussian_fixed=expected_gaussian,
                combined_fixed=expected_combined,
                bit=bit,
            )
        )

    sample_root, _ = build_sample_merkle_sha(
        sample_trace.entries,
        gaussian_fixed=gaussian_fixed,
        combined_fixed=combined_fixed,
        bits=bits_for_samples,
    )
    trace_commitment = hash_trace_commitment(
        [sample.to_json() for sample in sp1_samples]
    )
    opening_k = min(OPENING_K_DEFAULT, len(sp1_samples))
    challenge_seed = derive_streaming_challenge_seed(
        image_hash, ctx_hash, trace_commitment
    )
    positions = derive_opening_positions(challenge_seed, opening_k, len(sp1_samples))
    opened_entries = [
        {
            "pos": position,
            "index": sp1_samples[position].index,
            "gaussian_fixed": sp1_samples[position].gaussian_fixed,
            "combined_fixed": sp1_samples[position].combined_fixed,
            "bit": sp1_samples[position].bit,
        }
        for position in positions
    ]
    statement = StreamingAttestationStatement(
        protocol_version=STREAMING_PROTOCOL_VERSION,
        ctx_hash=ctx_hash,
        binding=binding_bytes,
        producer_key_commitment=producer_key_commitment(secret_key),
        image_hash=image_hash,
        trace_commitment=trace_commitment,
        sample_indices_hash=hash_sample_indices(sample_set.indices),
        challenge_seed=challenge_seed,
        opening_digest=opening_digest(opened_entries),
        lut_hash=lut_hash,
        opening_k=opening_k,
        alpha_effective_fixed=alpha_fixed,
        scale_factor_q30=scale_q30,
        sample_count=len(sp1_samples),
        width=cfg.image.width,
        height=cfg.image.height,
        rs_n=codec.n,
        rs_k=codec.k,
    )
    statement.validate_derived_fields()

    package_paths = prepare_sp1_package(
        statement.to_json(),
        SP1Witness(secret_key=secret_key, codeword=codeword, samples=sp1_samples),
        output_dir / "sp1",
    )
    skip_proof = bool(getattr(args, "skip_proof", False))
    if skip_proof:
        prover_time = 0.0
        print("[prover] --skip-proof: private witness retained for later proving")
    else:
        os.environ.setdefault("SP1_PROVER", "cuda")
        os.environ.setdefault("SP1_PROOF_MODE", "core")
        prover_time = run_sp1_prover(
            package_paths,
            cfg.zk.prover_cmd,
            retain_private_witness=retain_private,
        )

    if retain_private:
        trace_path = private_dir / "sample_trace.bin"
        sample_trace.save_binary(trace_path)
    else:
        trace_path = None

    channels = cfg.model.latent_dim[0] if cfg.model.latent_dim else 4
    metadata: dict[str, Any] = {
        "version": ATTEST_METADATA_VERSION,
        "paper_modes": {
            "detect": "transform-tolerant claim-conditioned watermark detection",
            "attest": "bit-exact image-bound proof plus Detect",
        },
        "prompt": args.prompt,
        "prompt_hash": "0x" + prompt_hash_bytes.hex(),
        "ctx_hash": "0x" + ctx_hash.hex(),
        "binding": "0x" + binding_bytes.hex(),
        "producer_key_commitment": "0x" + statement.producer_key_commitment.hex(),
        "seed_hex": seed.hex(),
        "message_hex": binding_bytes.hex(),
        "codeword_hex": codeword.hex(),
        "payload_hash": _sha256_hex(binding_bytes),
        "sample_count": len(sample_set.indices),
        "commitment_scheme": "streaming-full-trace-v2",
        "trace_commitment": "0x" + trace_commitment.hex(),
        # Retained only as a reproducibility diagnostic for historical v1
        # comparisons; the v2 receipt does not rely on this Merkle root.
        "sample_root": sample_root.hex(),
        "sample_indices_hash": "0x" + statement.sample_indices_hash.hex(),
        "lut_hash": "0x" + lut_hash.hex(),
        "width": cfg.image.width,
        "height": cfg.image.height,
        "pixel_width": cfg.image.pixel_width,
        "pixel_height": cfg.image.pixel_height,
        "channels": channels,
        "model_id": model_id,
        "guidance_scale": guidance_scale,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "scheduler": "ddim",
        "alpha": cfg.image.alpha,
        "alpha_effective": alpha_effective,
        "scale_factor": scale_factor,
        "hash_spec": HASH_SPEC,
        "image_png": _relative_artifact(png_path, output_dir),
        "attestation_statement": statement.to_json(),
        "sp1": {
            "public": _relative_artifact(package_paths.public, output_dir),
            "receipt": _relative_artifact(package_paths.receipt, output_dir),
            "proof_status": "pending" if skip_proof else "generated",
        },
        "timings": {"sp1_attest_prover_sec": prover_time},
    }
    if getattr(args, "usecase_tag", None):
        metadata["usecase_tag"] = args.usecase_tag
    if getattr(args, "user_tag", None):
        metadata["user_tag"] = args.user_tag
    if args.registry_address or args.vk_cid or args.proof_cid:
        metadata["producer_registry"] = {
            "registry_address": args.registry_address,
            "vk_cid": args.vk_cid,
            "proof_cid": args.proof_cid,
        }
    if retain_private:
        private_debug: dict[str, str] = {}
        if package_paths.witness is not None:
            private_debug["witness"] = _relative_artifact(package_paths.witness, output_dir)
        if trace_path is not None:
            private_debug["sample_trace"] = _relative_artifact(trace_path, output_dir)
        if latent_path is not None:
            private_debug["latent_noise"] = _relative_artifact(latent_path, output_dir)
        if z0_path is not None:
            private_debug["z0_latents"] = _relative_artifact(z0_path, output_dir)
        metadata["private_debug_do_not_publish"] = private_debug

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[prover] wrote public Detect/Attest metadata to {meta_path}")


def run_prover(args: argparse.Namespace) -> None:
    cfg = GlobalConfig.load(Path(args.config))
    _ensure_sp1_backend(cfg, bool(getattr(args, "allow_legacy_backend", False)))
    if cfg.zk.backend.lower() == "sp1":
        _run_sp1_attest_prover(args, cfg)
        return
    output_dir = Path(args.output).resolve()
    _ensure_dir(output_dir)

    codec = ReedSolomonCodec(n=cfg.watermark.rs_n, k=cfg.watermark.rs_k)
    keypair = _load_secret(args.secret_hex, Path(args.secret_file) if args.secret_file else None)
    seed = secrets.token_bytes(32) if args.seed_hex is None else bytes.fromhex(args.seed_hex)

    zk_backend = cfg.zk.backend.lower()
    if zk_backend == "risc0":
        payload = build_payload_risc0(
            prompt=args.prompt,
            seed=seed,
            secret=keypair.secret_bytes(),
            codec=codec,
        )
    elif getattr(args, "legacy_poseidon", False):
        payload = build_payload_bn254(prompt=args.prompt, seed=seed, secret=keypair.secret_bytes(), codec=codec)
    else:
        payload = build_payload_pallas(prompt=args.prompt, seed=seed, secret=keypair.secret_bytes(), codec=codec)
    sample_set = deterministic_sample(
        total_pixels=cfg.image.total_pixels,
        width=cfg.image.width,
        height=cfg.image.height,
        key_material=payload.codeword,
        count=cfg.image.sample_count(),
    )

    inverse_cdf_path = cfg.tables.inverse_cdf_path
    inverse_cdf = InverseCDFTable.from_file(inverse_cdf_path)
    lut_hash = _file_sha256(inverse_cdf_path)
    kernel = get_watermark_kernel(_device_config_from_args(args), cfg, inverse_cdf)
    embedding = kernel.embed(binding=payload.binding, bit_sequence=payload.bits, sample_set=sample_set)
    scale_factor = 1.0 / float(np.sqrt(1.0 + cfg.image.alpha * cfg.image.alpha))
    alpha_effective = float(cfg.image.alpha) * scale_factor
    latent = embedding.latent_noise.astype(np.float32)
    latent_scaled = latent * scale_factor
    guidance_scale = 7.5
    num_inference_steps = 50
    negative_prompt = ""

    latent_path = output_dir / "latent_noise.npy"
    np.save(latent_path, latent_scaled)

    # Build z0 = watermark noise (normalized) and run SDXL denoise to produce the attacked RGB image.
    png_path = output_dir / "watermarked.png"
    z0_path = output_dir / "z0_latents.npy"
    vae_model = args.model_id or cfg.model.base
    try:
        import torch  # type: ignore
        from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline  # type: ignore

        dtype = torch.float16
        use_sdxl = _is_sdxl_model(vae_model)
        pipe_cls = StableDiffusionXLPipeline if use_sdxl else StableDiffusionPipeline
        kwargs = {"torch_dtype": dtype}
        if use_sdxl:
            kwargs["variant"] = "fp16"
        try:
            pipe = pipe_cls.from_pretrained(vae_model, **kwargs)
        except (TypeError, ValueError):
            kwargs.pop("variant", None)
            pipe = pipe_cls.from_pretrained(vae_model, **kwargs)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        scheduler_name = "ddim"
        device = "cuda"
        pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        if getattr(args, "use_random_latents", False):
            print(">> [TEST] Generating Standard Numpy Noise...")
            z0prime = np.random.randn(4, cfg.image.height, cfg.image.width).astype(np.float32)
        else:
            z0prime = latent_scaled if latent_scaled.ndim == 3 else np.repeat(latent_scaled[None, :, :], 4, axis=0)
        torch_latents = torch.from_numpy(z0prime[None, ...]).to(device=device, dtype=dtype)

        gen = torch.Generator(device=device)
        gen.manual_seed(int.from_bytes(seed, "big") % (2**63 - 1))

        result = pipe(
            prompt=args.prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            latents=torch_latents,
            output_type="np",
            generator=gen,
        )
        imgs = result.images[0]
        imgs_u8 = np.clip(imgs * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(imgs_u8).save(png_path)
        np.save(z0_path, z0prime)
    except Exception as exc:  # pragma: no cover - optional dependency path
        print(f"[prover] ERROR: SDXL decode failed ({exc}); aborting.")
        raise
    trace_path = output_dir / "sample_trace.bin"
    sample_trace = SampleTrace()
    z0_ch0 = z0prime[0]
    alpha_val = alpha_effective
    bit_idx_map = build_bit_index_map(
        binding=payload.binding,
        width=cfg.image.width,
        height=cfg.image.height,
        bit_len=len(payload.bits),
        backend=cfg.zk.sample_backend,
    )
    for idx in sample_set.indices:
        y = idx // cfg.image.width
        x = idx % cfg.image.width
        combined_val = float(z0_ch0[y, x])
        bit = payload.bits[int(bit_idx_map[y, x])]
        gaussian_val = combined_val - alpha_val * (2 * bit - 1)
        sample_trace.record(idx, 0.0, gaussian_val, combined_val)
    sample_trace.save_binary(trace_path)

    key_path = output_dir / "key_info.json"
    key_info = {
        "public_x": hex(keypair.public[0]),
        "public_y": hex(keypair.public[1]),
    }
    key_path.write_text(json.dumps(key_info, indent=2), encoding="utf-8")

    alpha_fixed = int(round(alpha_effective * FIXED_SCALE))
    bits_for_samples = [payload.bits[int(bit_idx_map[idx // cfg.image.width, idx % cfg.image.width])] for idx in sample_set.indices]
    gaussian_fixed = [int(round(obs.z_expected * FIXED_SCALE)) for obs in sample_trace.entries]
    combined_fixed = [int(round(obs.z_observed * FIXED_SCALE)) for obs in sample_trace.entries]
    if zk_backend == "risc0":
        # The legacy RISC0 guest enforces the exact fixed-point embedding relation
        # combined = gaussian + alpha * sign on the opened subset. We synthesize
        # combined_fixed from gaussian_fixed so the circuit check holds without tolerance.
        # The ~LSB divergence from the raw rounded z_observed is absorbed by opening_eps
        # on the verifier side (opening_check in verify flow).
        combined_fixed = [
            g + alpha_fixed * (2 * bit - 1) for g, bit in zip(gaussian_fixed, bits_for_samples)
        ]
    opening_info = None

    if zk_backend == "risc0":
        sample_root, merkle_paths = build_sample_merkle_sha(
            sample_trace.entries,
            gaussian_fixed=gaussian_fixed,
            combined_fixed=combined_fixed,
            bits=bits_for_samples,
        )
        merkle_depth = len(merkle_paths[0]) if merkle_paths else 0
        public_inputs_payload = {
            "prompt_hash": "0x" + payload.prompt_hash.to_bytes(32, "big").hex(),
            "binding": "0x" + payload.binding.to_bytes(32, "big").hex(),
            "sample_merkle_root": "0x" + sample_root.hex(),
            "alpha": str(alpha_fixed),
            "sample_merkle_depth": merkle_depth,
        }
        witness = Witness(
            secret_key=keypair.secret_bytes(),
            seed=seed,
            codeword=payload.codeword,
            sample_observations=list(sample_trace.entries),
            sample_merkle_paths=merkle_paths,
            gaussian_fixed=gaussian_fixed,
            combined_fixed=combined_fixed,
            bits=bits_for_samples,
            alpha_fixed=alpha_fixed,
        )
        risc0_dir = output_dir / "risc0"
        package_paths = prepare_risc0_package(public_inputs_payload, witness, risc0_dir)
        prover_time = run_risc0_prover(package_paths, cfg.zk.prover_cmd)
    else:
        sample_root, merkle_paths = build_sample_merkle(
            sample_trace.entries,
            gaussian_fixed=gaussian_fixed,
            combined_fixed=combined_fixed,
            bits=bits_for_samples,
        )
        merkle_depth = len(merkle_paths[0]) if merkle_paths else 0
        public_inputs = PublicInputs(
            prompt_hash=payload.prompt_hash,
            binding=payload.binding,
            sample_merkle_root=sample_root,
            alpha=alpha_fixed,
            sample_merkle_depth=merkle_depth,
        )
        witness = Witness(
            secret_key=keypair.secret_bytes(),
            seed=seed,
            codeword=payload.codeword,
            sample_observations=list(sample_trace.entries),
            sample_merkle_paths=merkle_paths,
            gaussian_fixed=gaussian_fixed,
            combined_fixed=combined_fixed,
            bits=bits_for_samples,
            alpha_fixed=alpha_fixed,
        )
        halo2_dir = output_dir / "halo2"
        package = Halo2Package(public_inputs=public_inputs, witness=witness, proof_path=halo2_dir / "proof.bin")
        halo2_paths = prepare_prover_package(package, sample_set, halo2_dir)
        prover_time = run_halo2_prover(halo2_paths, cfg.zk)

    channels = cfg.model.latent_dim[0] if cfg.model.latent_dim else 4
    metadata = {
        "prompt": args.prompt,
        "prompt_hash": hex(payload.prompt_hash),
        "binding": hex(payload.binding),
        "seed_hex": seed.hex(),
        "message_hex": payload.message_bytes.hex(),
        "payload_hash": _sha256_hex(payload.message_bytes),
        "codeword_hex": payload.codeword.hex(),
        "latent_noise": str(latent_path),
        "sample_trace": str(trace_path),
        "sample_count": len(sample_set.indices),
        "sample_root": sample_root.hex(),
        "key_info": str(key_path),
        "lut_path": str(cfg.tables.inverse_cdf_path),
        "lut_hash": lut_hash,
        "grid_w": cfg.image.width,
        "grid_h": cfg.image.height,
        "pixel_width": cfg.image.pixel_width,
        "pixel_height": cfg.image.pixel_height,
        "channels": channels,
        "guidance_scale": guidance_scale,
        "negative_prompt": negative_prompt,
        "num_inference_steps": num_inference_steps,
        "scheduler": scheduler_name,
        "alpha": cfg.image.alpha,
        "alpha_effective": alpha_effective,
        "scale_factor": scale_factor,
        "version": PPMARK_VERSION,
        "image_png": str(png_path) if png_path else None,
        "z0_path": str(z0_path),
    }
    if opening_info is not None:
        metadata["hash_spec"] = HASH_SPEC
        metadata["opening"] = opening_info
    if args.model_id:
        metadata["model_id"] = args.model_id
    if getattr(args, "usecase_tag", None):
        metadata["usecase_tag"] = args.usecase_tag
    if getattr(args, "user_tag", None):
        metadata["user_tag"] = args.user_tag
    metadata["width"] = cfg.image.width
    metadata["height"] = cfg.image.height
    if zk_backend == "risc0":
        metadata["timings"] = {"risc0_prover_sec": prover_time}
        metadata["risc0"] = {
            "public": str(package_paths.public),
            "witness": str(package_paths.witness),
            "receipt": str(package_paths.receipt),
        }
    else:
        metadata["timings"] = {"halo2_prover_sec": prover_time}
        metadata["halo2"] = {
            "public": str(halo2_paths.public),
            "witness": str(halo2_paths.witness),
            "proof": str(halo2_paths.proof),
        }
    if args.model_id or args.registry_address or args.vk_cid or args.proof_cid:
        metadata["blockchain"] = {
            "model_id": args.model_id,
            "registry_address": args.registry_address,
            "vk_cid": args.vk_cid,
            "proof_cid": args.proof_cid,
        }

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[prover] wrote payload metadata to {meta_path}")


def _recover_attest_score_latent(
    *,
    args: argparse.Namespace,
    metadata: Dict[str, Any],
    cfg: GlobalConfig,
    binding: int,
    sample_indices: np.ndarray,
    codeword_bits: np.ndarray,
    inverse_cdf: InverseCDFTable,
) -> np.ndarray:
    score_latent = getattr(args, "score_latent", None)
    source_path = Path(score_latent) if score_latent else Path(args.image)
    if source_path.suffix.lower() == ".npy":
        latent = np.asarray(np.load(source_path), dtype=np.float32)
        while latent.ndim > 2:
            latent = latent[0]
        if latent.shape != (cfg.image.height, cfg.image.width):
            raise RuntimeError(
                f"score latent shape {latent.shape} != {(cfg.image.height, cfg.image.width)}"
            )
    else:
        with Image.open(source_path) as image:
            image_rgb = image.convert("RGB")
            image_array = np.array(image_rgb, dtype=np.float32)
        model_id = getattr(args, "ddim_model_id", None) or metadata.get("model_id")
        if not model_id:
            raise RuntimeError("Detect requires --ddim-model-id when no model_id is in metadata")
        prompt = metadata.get("prompt")
        if prompt is None:
            raise RuntimeError("metadata prompt is required for DDIM inversion")
        guidance_scale = (
            float(args.ddim_guidance_scale)
            if getattr(args, "ddim_guidance_scale", None) is not None
            else float(metadata.get("guidance_scale", 7.5))
        )
        negative_prompt = (
            str(args.ddim_negative_prompt)
            if getattr(args, "ddim_negative_prompt", None) is not None
            else str(metadata.get("negative_prompt", ""))
        )
        inverter = DiffusersDDIMInverter(
            DiffusersDDIMConfig(
                model_id=str(model_id),
                device=args.ddim_device,
                torch_dtype=args.ddim_dtype,
                num_steps=args.ddim_steps,
                scheduler=str(metadata.get("scheduler", "ddim")),
                prompt=str(prompt),
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
            )
        ).invert
        crop_keep_ratio = float(getattr(args, "crop_keep_ratio", 0.0) or 0.0)
        if crop_keep_ratio > 0:
            latent, crop_stats = _crop_search_latent(
                base_img=Image.fromarray(image_array.astype(np.uint8)),
                invert_fn=lambda arr: ddim_invert(
                    arr,
                    target_resolution=(cfg.image.width, cfg.image.height),
                    normalize=True,
                    external_inverter=inverter,
                ),
                binding=binding,
                config=cfg,
                inverse_cdf=inverse_cdf,
                sample_indices=sample_indices,
                bits=codeword_bits,
                keep_ratio=crop_keep_ratio,
                grid=int(getattr(args, "crop_grid", 3)),
                refine=bool(getattr(args, "crop_refine", False)),
                refine_grid=int(getattr(args, "crop_refine_grid", 5)),
                refine_step=float(getattr(args, "crop_refine_step", 0.05)),
                sigma_meta=float(metadata.get("init_noise_sigma", 1.0)),
            )
            print(
                f"[Detect] crop alignment selected {crop_stats['best_crop_box']} "
                f"from {crop_stats['crop_candidates_count']} candidates"
            )
        else:
            latent = ddim_invert(
                image_array,
                target_resolution=(cfg.image.width, cfg.image.height),
                normalize=True,
                external_inverter=inverter,
            )
            sigma = float(metadata.get("init_noise_sigma", 1.0))
            if sigma != 0:
                latent = latent / sigma
        latent = np.asarray(latent, dtype=np.float32)
        while latent.ndim > 2:
            latent = latent[0]
        if latent.shape != (cfg.image.height, cfg.image.width):
            raise RuntimeError(
                f"DDIM inversion shape {latent.shape} != {(cfg.image.height, cfg.image.width)}"
            )

    max_shift = int(getattr(args, "max_shift", 0) or 0)
    max_rotation = float(getattr(args, "max_rotation", 0.0) or 0.0)
    if (max_shift > 0 or max_rotation > 0) and getattr(args, "sync_mode", "auto") != "off":
        result, latent = sync_search(
            latent=latent,
            binding=binding,
            config=cfg,
            inverse_cdf=inverse_cdf,
            sample_indices=sample_indices,
            bits=codeword_bits,
            max_shift=max_shift,
            max_rotation=max_rotation,
            rotation_step=float(getattr(args, "rotation_step", 1.0) or 1.0),
            rotation_strategy=str(getattr(args, "rotation_strategy", "auto")),
        )
        print(
            f"[Detect] alignment dx={result.dx}, dy={result.dy}, "
            f"theta={result.theta:.3f}"
        )
    return np.asarray(latent, dtype=np.float32)


def _run_sp1_attest_verifier(
    args: argparse.Namespace,
    cfg: GlobalConfig,
    meta_path: Path,
    metadata: Dict[str, Any],
) -> None:
    mode = str(getattr(args, "mode", "attest")).lower()
    if mode not in {"detect", "attest"}:
        raise ValueError("verification mode must be 'detect' or 'attest'")
    if cfg.zk.sample_backend != "sha256":
        raise RuntimeError("SP1 Detect/Attest requires zk.sample_backend='sha256'")
    if not getattr(args, "image", None):
        raise RuntimeError("Detect/Attest verification requires --image")
    score_threshold = _resolve_score_threshold(args, metadata)
    if score_threshold is None or not math.isfinite(float(score_threshold)):
        raise RuntimeError("Detect requires a finite --tau or --tau-file")

    raw_statement = metadata.get("attestation_statement")
    if not isinstance(raw_statement, dict):
        raise RuntimeError("metadata is missing the canonical attestation_statement")
    statement = StreamingAttestationStatement.from_mapping(raw_statement)
    statement.validate_derived_fields()
    if (statement.width, statement.height) != (cfg.image.width, cfg.image.height):
        raise RuntimeError("statement latent dimensions do not match verifier config")
    if (statement.rs_n, statement.rs_k) != (
        cfg.watermark.rs_n,
        cfg.watermark.rs_k,
    ):
        raise RuntimeError("statement RS parameters do not match verifier config")

    local_lut_hash = hash_lut_file(cfg.tables.inverse_cdf_path)
    if statement.lut_hash != local_lut_hash:
        raise RuntimeError("statement inverse-CDF LUT hash does not match verifier LUT")
    alpha_fixed, scale_q30, alpha_effective, _ = alpha_fixed_and_scale_q30(
        float(metadata["alpha"])
    )
    if (
        statement.alpha_effective_fixed != alpha_fixed
        or statement.scale_factor_q30 != scale_q30
    ):
        raise RuntimeError("statement fixed-point alpha/normalization parameters are inconsistent")

    prompt = str(metadata["prompt"])
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).digest()
    if metadata.get("prompt_hash") != "0x" + prompt_hash.hex():
        raise RuntimeError("metadata prompt_hash does not match prompt")
    seed = bytes.fromhex(str(metadata["seed_hex"]))
    if len(seed) != 32:
        raise RuntimeError("metadata seed must be exactly 32 bytes")
    model_id = str(metadata.get("model_id", ""))
    if args.model_id and args.model_id != model_id:
        raise RuntimeError(f"model ID mismatch: metadata={model_id}, requested={args.model_id}")
    reconstructed_ctx = make_ctx_hash(
        prompt_hash=int.from_bytes(prompt_hash, "big"),
        model_id=model_id,
        seed=seed,
        width=cfg.image.width,
        height=cfg.image.height,
        usecase_tag=metadata.get("usecase_tag"),
        user_tag=metadata.get("user_tag"),
    ).to_bytes(32, "big")
    if reconstructed_ctx != statement.ctx_hash:
        raise RuntimeError("P1 public context does not match ctx_hash")
    if metadata.get("ctx_hash") != "0x" + statement.ctx_hash.hex():
        raise RuntimeError("metadata ctx_hash diverges from the public statement")
    if metadata.get("binding") != "0x" + statement.binding.hex():
        raise RuntimeError("metadata binding diverges from the public statement")

    expected_commitment_text = getattr(args, "expected_key_commitment", None) or os.environ.get(
        "PPMARK_EXPECTED_KEY_COMMITMENT"
    )
    if expected_commitment_text:
        expected_commitment = parse_hex32(
            str(expected_commitment_text), "expected_key_commitment"
        )
        if statement.producer_key_commitment != expected_commitment:
            raise RuntimeError("producer key commitment is not the verifier-trusted key")
    elif mode == "attest" and not bool(
        getattr(args, "allow_untrusted_producer_key", False)
    ):
        raise RuntimeError(
            "Attest requires --expected-key-commitment (or an authenticated registry); "
            "use --allow-untrusted-producer-key only for self-consistency testing"
        )
    else:
        print("[verifier] producer key is self-asserted; producer identity is not authenticated")

    codec = ReedSolomonCodec(n=statement.rs_n, k=statement.rs_k)
    codeword = codec.encode(statement.binding)
    if metadata.get("codeword_hex") != codeword.hex():
        raise RuntimeError("P4 metadata codeword is not RS(binding)")
    codeword_bits = bytes_to_bits(codeword)
    sample_set = deterministic_sample(
        total_pixels=cfg.image.total_pixels,
        width=cfg.image.width,
        height=cfg.image.height,
        key_material=codeword,
        count=statement.sample_count,
    )
    if hash_sample_indices(sample_set.indices) != statement.sample_indices_hash:
        raise RuntimeError("P2 deterministic sample-set hash mismatch")
    if metadata.get("commitment_scheme") != "streaming-full-trace-v2":
        raise RuntimeError("metadata does not declare the production streaming commitment")
    if metadata.get("trace_commitment") != "0x" + statement.trace_commitment.hex():
        raise RuntimeError("metadata trace commitment diverges from the public statement")

    sp1_meta = metadata.get("sp1")
    if not isinstance(sp1_meta, dict):
        raise RuntimeError("SP1 artifact paths are missing")
    public_path = _resolve(meta_path.parent, str(sp1_meta["public"]))
    public_statement = StreamingAttestationStatement.from_mapping(
        json.loads(public_path.read_text(encoding="utf-8"))
    )
    if public_statement != statement:
        raise RuntimeError("standalone SP1 statement diverges from metadata")

    image_path = Path(args.image)
    if mode == "attest":
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            raise RuntimeError("Attest requires the exact image artifact, not a latent array")
        if metadata.get("hash_spec") != HASH_SPEC:
            raise RuntimeError("image hash normalization specification mismatch")
        with Image.open(image_path) as image:
            image_hash = _hash_normalized_image(
                image,
                (int(HASH_SPEC["resize"][0]), int(HASH_SPEC["resize"][1])),
            )
        if image_hash != statement.image_hash:
            raise RuntimeError("P3 image hash mismatch: this is not the bit-exact attested artifact")
        receipt_path = _resolve(meta_path.parent, str(sp1_meta["receipt"]))
        if sp1_meta.get("proof_status") != "generated" or not receipt_path.exists():
            raise RuntimeError("Attest receipt has not been generated")
        os.environ.setdefault("SP1_PROVER", "cpu")
        _resolve_vk_path(cfg, model_id)
        verify_sp1_receipt(
            SP1Paths(public=public_path, receipt=receipt_path),
            _load_vk_from_env(cfg),
        )

    inverse_cdf = InverseCDFTable.from_file(cfg.tables.inverse_cdf_path)
    sample_indices = np.asarray(sample_set.indices, dtype=np.int64)
    bits_array = np.asarray(codeword_bits, dtype=np.int8)
    latent = _recover_attest_score_latent(
        args=args,
        metadata=metadata,
        cfg=cfg,
        binding=int.from_bytes(statement.binding, "big"),
        sample_indices=sample_indices,
        codeword_bits=bits_array,
        inverse_cdf=inverse_cdf,
    )
    bit_map = build_bit_index_map(
        binding=int.from_bytes(statement.binding, "big"),
        width=cfg.image.width,
        height=cfg.image.height,
        bit_len=len(codeword_bits),
        backend="sha256",
    )
    ys = sample_indices // cfg.image.width
    xs = sample_indices % cfg.image.width
    sample_bits = bits_array[bit_map[ys, xs]]
    signs = 2 * sample_bits.astype(np.float32) - 1.0
    watermark = float(alpha_effective) * signs
    raw_score = float(np.dot(latent[ys, xs], watermark)) / float(
        np.linalg.norm(watermark) + 1e-9
    )
    score = abs(raw_score)
    passed = score >= float(score_threshold)
    print(
        f"[Detect] score={score:.4f} (raw={raw_score:.4f}, "
        f"tau={float(score_threshold):.4f}, pass={passed})"
    )
    if not passed:
        raise RuntimeError("Detect score is below the frozen threshold")
    if mode == "attest":
        print("[Attest] PASS: trusted key, exact image, P1-P4 receipt, and Detect all verified")
    else:
        print("[Detect] PASS: watermark signal detected; generator compliance is not attested")


def run_verifier(args: argparse.Namespace) -> None:
    cfg = GlobalConfig.load(Path(args.config))
    _ensure_sp1_backend(cfg, bool(getattr(args, "allow_legacy_backend", False)))
    zk_backend = cfg.zk.backend.lower()
    meta_path = Path(args.metadata)
    metadata: Dict[str, Any]
    with meta_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if zk_backend == "sp1":
        if metadata.get("version") != ATTEST_METADATA_VERSION:
            raise RuntimeError(
                "legacy SP1 metadata is refused; regenerate a canonical ppmark_attest_v2 artifact"
            )
        _run_sp1_attest_verifier(args, cfg, meta_path, metadata)
        return
    alpha_effective = float(metadata.get("alpha_effective", cfg.image.alpha))
    require_decode = args.require_decode if args.require_decode is not None else False
    legacy_metadata_only = bool(getattr(args, "legacy_metadata_only", False))
    require_root = bool(getattr(args, "require_root", False))
    score_threshold = _resolve_score_threshold(args, metadata)
    if not legacy_metadata_only:
        if not getattr(args, "image", None):
            raise RuntimeError("Score-based verification requires --image (or a .npy latent). Use --legacy-metadata-only to skip.")
        if score_threshold is None or not math.isfinite(float(score_threshold)):
            raise RuntimeError("Score threshold is required; set --tau/--tau-file (or --score-threshold).")

    # Basic metadata sanity: resolution, model_id, sample_count.
    meta_width = metadata.get("width")
    meta_height = metadata.get("height")
    if meta_width is not None and meta_width != cfg.image.width:
        raise RuntimeError(f"Metadata width {meta_width} != config width {cfg.image.width}")
    if meta_height is not None and meta_height != cfg.image.height:
        raise RuntimeError(f"Metadata height {meta_height} != config height {cfg.image.height}")
    if args.model_id and metadata.get("model_id") and args.model_id != metadata["model_id"]:
        raise RuntimeError(f"Model ID mismatch: metadata={metadata['model_id']} vs args={args.model_id}")

    trace_path = _resolve(meta_path.parent, metadata["sample_trace"])
    sample_trace = SampleTrace.load_binary(trace_path)
    codec = ReedSolomonCodec(n=cfg.watermark.rs_n, k=cfg.watermark.rs_k)
    codeword = bytes.fromhex(metadata["codeword_hex"])
    sample_set = deterministic_sample(
        total_pixels=cfg.image.total_pixels,
        width=cfg.image.width,
        height=cfg.image.height,
        key_material=codeword,
        count=cfg.image.sample_count(),
    )
    if len(sample_trace.entries) != len(sample_set.indices):
        raise RuntimeError(f"Sample trace length mismatch: trace={len(sample_trace.entries)}, expected={len(sample_set.indices)}")
    alpha_fixed = int(round(alpha_effective * FIXED_SCALE))
    bits = bytes_to_bits(codeword)
    binding_int = int(metadata["binding"], 16)
    bit_idx_map = build_bit_index_map(
        binding=binding_int,
        width=cfg.image.width,
        height=cfg.image.height,
        bit_len=len(bits),
        backend=cfg.zk.sample_backend,
    )
    bits_for_samples = [bits[int(bit_idx_map[idx // cfg.image.width, idx % cfg.image.width])] for idx in sample_set.indices]
    gaussian_fixed = [int(round(obs.z_expected * FIXED_SCALE)) for obs in sample_trace.entries]
    combined_fixed = [int(round(obs.z_observed * FIXED_SCALE)) for obs in sample_trace.entries]
    score = None
    score_pass = True
    opening_ok = None
    opening_checked = False
    require_opening = bool(metadata.get("opening"))
    opening_entries: list[dict[str, int]] | None = None
    # Optional image path: run inversion for image-side checks (decode/score) and diagnostics.
    if getattr(args, "image", None):
        image_path = Path(args.image)
        if image_path.suffix == ".npy":
            img_arr = np.load(image_path)
        else:
            with Image.open(image_path) as img:
                img_arr = np.array(img.convert("RGB"), dtype=np.float32)
        inverse_cdf = InverseCDFTable.from_file(cfg.tables.inverse_cdf_path)
        lut_hash = _file_sha256(cfg.tables.inverse_cdf_path)
        if "lut_hash" in metadata and metadata["lut_hash"] != lut_hash:
            raise RuntimeError("LUT hash mismatch between config and metadata")
        opening_mode = getattr(args, "opening_mode", "strict")
        opening_eps = float(getattr(args, "opening_eps", 0.0) or 0.0)
        if metadata.get("opening"):
            hash_spec = metadata.get("hash_spec", {})
            if hash_spec != HASH_SPEC:
                raise RuntimeError("hash_spec mismatch between metadata and verifier")
            if image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff"}:
                with Image.open(image_path) as img:
                    image_hash = _hash_normalized_image(
                        img,
                        (int(HASH_SPEC["resize"][0]), int(HASH_SPEC["resize"][1])),
                    )
            elif image_path.suffix.lower() == ".npy":
                image_hash_hex = str(metadata.get("opening", {}).get("image_hash", ""))
                if image_hash_hex.startswith("0x"):
                    image_hash_hex = image_hash_hex[2:]
                if not image_hash_hex:
                    raise RuntimeError("opening verification for .npy requires metadata opening.image_hash")
                image_hash = bytes.fromhex(image_hash_hex)
            else:
                raise RuntimeError("opening verification requires an image file path")
            ctx_hash_hex = metadata.get("ctx_hash")
            if not ctx_hash_hex:
                raise RuntimeError("ctx_hash missing for opening verification")
            ctx_hash = int(ctx_hash_hex, 16).to_bytes(32, "big")
            sample_root_hex = metadata.get("sample_root")
            if not sample_root_hex:
                raise RuntimeError("sample_root missing for opening verification")
            challenge_seed = hashlib.sha256(
                OPENING_DOMAIN + image_hash + ctx_hash + bytes.fromhex(sample_root_hex)
            ).digest()
            opening = metadata["opening"]
            if opening.get("version") != OPENING_VERSION:
                raise RuntimeError("opening version mismatch")
            if opening.get("challenge_seed") != "0x" + challenge_seed.hex():
                raise RuntimeError("opening challenge_seed mismatch")
            openings_path = _resolve(meta_path.parent, opening["path"])
            with openings_path.open("r", encoding="utf-8") as handle:
                openings_doc = json.load(handle)
            if openings_doc.get("version") != OPENING_VERSION:
                raise RuntimeError("openings version mismatch")
            if openings_doc.get("k") != opening.get("k"):
                raise RuntimeError("openings k mismatch")
            if openings_doc.get("digest") != opening.get("digest"):
                raise RuntimeError("openings digest mismatch (metadata)")
            entries = openings_doc.get("entries", [])
            if len(entries) != int(opening.get("k", 0)):
                raise RuntimeError("openings entries length mismatch")
            positions = openings_doc.get("positions", [])
            if positions and [int(entry["pos"]) for entry in entries] != list(positions):
                raise RuntimeError("opening positions mismatch")
            opening_digest = _opening_digest(entries)
            if "0x" + opening_digest.hex() != opening.get("digest"):
                raise RuntimeError("openings digest mismatch (entries)")
            opening_entries = entries
        external_inverter = None
        normalize_latent = not image_path.suffix.lower().endswith(".npy")
        allow_latent_png = bool(getattr(args, "allow_latent_png", False))
        sigma_meta = float(metadata.get("init_noise_sigma", 1.0))
        sample_indices_np = np.array(sample_set.indices, dtype=np.int64)
        bits_np = np.array(bits, dtype=np.int8)
        # If the image is a scaled latent (from prover), allow direct unscale to preserve exact values (debug only).
        latent: np.ndarray
        if allow_latent_png and image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tiff"} and "image_scale_min" in metadata and "image_scale_max" in metadata:
            mn = float(metadata["image_scale_min"])
            mx = float(metadata["image_scale_max"])
            scale = mx - mn if mx != mn else 1.0
            latent = img_arr.astype(np.float32) / 65535.0 * scale + mn
            normalize_latent = False
            external_inverter = None
            sigma_applied = False
        elif image_path.suffix.lower() == ".npy" and getattr(args, "exact_latent", False):
            if img_arr.ndim == 3:
                latent = img_arr[0].astype(np.float32)
            else:
                latent = img_arr.astype(np.float32)
            normalize_latent = False
            external_inverter = None
            sigma_applied = True
        else:
            if getattr(args, "ddim_model_id", None):
                if not metadata.get("prompt"):
                    raise RuntimeError("Metadata missing prompt for DDIM inversion")
                prompt_text = str(metadata.get("prompt", ""))
                guidance_scale = (
                    float(args.ddim_guidance_scale)
                    if getattr(args, "ddim_guidance_scale", None) is not None
                    else float(metadata.get("guidance_scale", 7.5))
                )
                negative_prompt = (
                    str(args.ddim_negative_prompt)
                    if getattr(args, "ddim_negative_prompt", None) is not None
                    else str(metadata.get("negative_prompt", ""))
                )
                cfg_ddim = DiffusersDDIMConfig(
                    model_id=args.ddim_model_id,
                    device=args.ddim_device,
                    torch_dtype=args.ddim_dtype,
                    num_steps=args.ddim_steps,
                    scheduler=str(metadata.get("scheduler", "ddim")),
                    prompt=prompt_text,
                    guidance_scale=guidance_scale,
                    negative_prompt=negative_prompt,
                )
                external_inverter = DiffusersDDIMInverter(cfg_ddim).invert
                print(f"[verifier] using diffusers DDIM inversion: model={args.ddim_model_id}, steps={args.ddim_steps}, device={args.ddim_device}")
            else:
                raise RuntimeError("DDIM model id is required for image-based verification")
            crop_keep_ratio = float(getattr(args, "crop_keep_ratio", 0.0) or 0.0)
            if crop_keep_ratio > 0 and not image_path.suffix.lower().endswith(".npy"):
                base_img = Image.fromarray(img_arr.astype(np.uint8))
                latent, crop_stats = _crop_search_latent(
                    base_img=base_img,
                    invert_fn=lambda arr: ddim_invert(
                        arr,
                        target_resolution=(cfg.image.width, cfg.image.height),
                        normalize=normalize_latent,
                        external_inverter=external_inverter,
                    ),
                    binding=int(metadata["binding"], 16),
                    config=cfg,
                    inverse_cdf=inverse_cdf,
                    sample_indices=sample_indices_np,
                    bits=bits_np,
                    keep_ratio=crop_keep_ratio,
                    grid=int(getattr(args, "crop_grid", 3)),
                    refine=bool(getattr(args, "crop_refine", False)),
                    refine_grid=int(getattr(args, "crop_refine_grid", 5)),
                    refine_step=float(getattr(args, "crop_refine_step", 0.05)),
                    sigma_meta=sigma_meta,
                )
                print(
                    f"[verifier] crop search best_box={crop_stats['best_crop_box']} "
                    f"score_top={crop_stats['crop_top3_scores'][0]['score'] if crop_stats['crop_top3_scores'] else 'n/a'}"
                )
                sigma_applied = True
            else:
                latent = ddim_invert(
                    img_arr,
                    target_resolution=(cfg.image.width, cfg.image.height),
                    normalize=normalize_latent,
                    external_inverter=external_inverter,
                )
                sigma_applied = False
        if not sigma_applied and sigma_meta != 0:
            latent = latent / sigma_meta
        aligned_latent = latent
        max_shift = int(getattr(args, "max_shift", 0) or 0)
        max_rotation = float(getattr(args, "max_rotation", 0.0) or 0.0)
        rotation_step = float(getattr(args, "rotation_step", 0.0) or 1.0)
        rotation_strategy = str(getattr(args, "rotation_strategy", "auto"))
        backend = "sha" if zk_backend == "risc0" else "poseidon"
        sync_mode = getattr(args, "sync_mode", "auto")
        search_allowed = (max_shift > 0 or max_rotation > 0) and sync_mode != "off"
        sync_dx = 0
        sync_dy = 0

        root_match = None
        if "sample_root" in metadata:
            zero_root, _ = recompute_sample_root(
                latent=latent,
                binding=int(metadata["binding"], 16),
                config=cfg,
                sample_indices=sample_indices_np,
                bits=bits_np,
                alpha_fixed=alpha_fixed,
                backend=backend,
                dx=0,
                dy=0,
                alpha_effective=alpha_effective,
            )
            root_match = bool(zero_root.hex() == metadata["sample_root"])
            if root_match:
                print("[verifier] sample root matches at zero offset; sync search skipped")
            else:
                if search_allowed:
                    sync_result, aligned_latent = sync_search(
                        latent=latent,
                        binding=int(metadata["binding"], 16),
                        config=cfg,
                        inverse_cdf=inverse_cdf,
                        sample_indices=sample_indices_np,
                        bits=bits_np,
                        max_shift=max_shift,
                        max_rotation=max_rotation,
                        rotation_step=rotation_step,
                        rotation_strategy=rotation_strategy,
                    )
                    sync_dx = sync_result.dx
                    sync_dy = sync_result.dy
                    recomputed_root, _ = recompute_sample_root(
                        latent=aligned_latent,
                        binding=int(metadata["binding"], 16),
                        config=cfg,
                        sample_indices=sample_indices_np,
                        bits=bits_np,
                        alpha_fixed=alpha_fixed,
                        backend=backend,
                        dx=sync_dx,
                        dy=sync_dy,
                        alpha_effective=alpha_effective,
                    )
                    root_match = bool(recomputed_root.hex() == metadata["sample_root"])
                    if root_match:
                        print(
                            f"[verifier] sample root matches after sync (dx={sync_result.dx}, dy={sync_result.dy}, "
                            f"theta={sync_result.theta:.3f}, score={sync_result.score:.4f})"
                        )
                    else:
                        print(
                            "[verifier] sample root mismatch after sync: "
                            f"dx={sync_result.dx}, dy={sync_result.dy}, "
                            f"theta={sync_result.theta:.3f}, score={sync_result.score:.4f}"
                        )
                else:
                    print("[verifier] sample root mismatch (sync search disabled)")
        if require_root:
            if root_match is None:
                raise RuntimeError("Sample root missing; strict root check requested")
            if not root_match:
                raise RuntimeError("Sample root mismatch under strict root check")

        if opening_entries is not None:
            opening_checked = True
            alpha = float(alpha_effective)
            opening_ok = True
            opening_reason = None
            for entry in opening_entries:
                pos = int(entry["pos"])
                if pos < 0 or pos >= len(sample_set.indices):
                    opening_ok = False
                    opening_reason = "opening_pos_oob"
                    break
                idx = sample_set.indices[pos]
                if int(entry["index"]) != int(idx):
                    opening_ok = False
                    opening_reason = "opening_index_mismatch"
                    break
                y = idx // cfg.image.width
                x = idx % cfg.image.width
                y_shift = y + sync_dy
                x_shift = x + sync_dx
                if y_shift < 0 or y_shift >= cfg.image.height or x_shift < 0 or x_shift >= cfg.image.width:
                    opening_ok = False
                    opening_reason = "opening_shift_oob"
                    break
                combined_val = float(aligned_latent[y_shift, x_shift])
                bit_expected = int(bits_for_samples[pos])
                combined_fixed_expected = int(round(combined_val * FIXED_SCALE))
                if int(entry["bit"]) != bit_expected:
                    opening_ok = False
                    opening_reason = "opening_bit_mismatch"
                    break
                gaussian_fixed_expected = int(
                    round((combined_val - alpha * (2 * bit_expected - 1)) * FIXED_SCALE)
                )
                if opening_mode == "strict":
                    if int(entry["combined_fixed"]) != combined_fixed_expected:
                        opening_ok = False
                        opening_reason = "opening_combined_mismatch"
                        break
                    if int(entry["gaussian_fixed"]) != gaussian_fixed_expected:
                        opening_ok = False
                        opening_reason = "opening_gaussian_mismatch"
                        break
                else:
                    eps_fixed = int(round(opening_eps * FIXED_SCALE))
                    if abs(int(entry["combined_fixed"]) - combined_fixed_expected) > eps_fixed:
                        opening_ok = False
                        opening_reason = "opening_combined_eps"
                        break
                    if abs(int(entry["gaussian_fixed"]) - gaussian_fixed_expected) > eps_fixed:
                        opening_ok = False
                        opening_reason = "opening_gaussian_eps"
                        break
            if not opening_ok:
                reason = opening_reason or "opening_mismatch"
                print(f"[verifier] opening mismatch (image-derived trace): {reason}")
            if require_opening and not opening_ok:
                reason = opening_reason or "opening_mismatch"
                raise RuntimeError(f"Opening verification failed: {reason}")

        decoded_ok = None
        if require_decode:
            try:
                message, bit_flip, threshold = _decode_payload_from_latent(
                    latent=aligned_latent,
                    binding=binding_int,
                    config=cfg,
                    inverse_cdf=inverse_cdf,
                    codec=codec,
                    bit_idx_map=bit_idx_map,
                    alpha=alpha_effective,
                )
                expected_hash = _get_expected_payload_hash(metadata)
                rs_ok = message is not None
                decoded_hash = _sha256_hex(message) if message is not None else None
                payload_match = bool(decoded_hash == expected_hash) if decoded_hash is not None else False
                decoded_ok = bool(rs_ok and payload_match)
                print(
                    f"[verifier] decoded payload match={decoded_ok} (rs_ok={rs_ok}, "
                    f"payload_match={payload_match}, bit_flip={bit_flip}, threshold={threshold:.4f})"
                )
                if not decoded_ok:
                    raise RuntimeError("Decoded payload mismatch")
            except Exception as exc:
                raise RuntimeError("Decoded payload check failed") from exc
        elif require_decode is False:
            try:
                message, bit_flip, threshold = _decode_payload_from_latent(
                    latent=aligned_latent,
                    binding=binding_int,
                    config=cfg,
                    inverse_cdf=inverse_cdf,
                    codec=codec,
                    bit_idx_map=bit_idx_map,
                    alpha=alpha_effective,
                )
                expected_hash = _get_expected_payload_hash(metadata)
                rs_ok = message is not None
                decoded_hash = _sha256_hex(message) if message is not None else None
                payload_match = bool(decoded_hash == expected_hash) if decoded_hash is not None else False
                decoded_ok = bool(rs_ok and payload_match)
                print(
                    f"[verifier] decoded payload match={decoded_ok} (rs_ok={rs_ok}, "
                    f"payload_match={payload_match}, bit_flip={bit_flip}, threshold={threshold:.4f})"
                )
            except Exception as exc:
                print(f"[verifier] decoded payload check skipped ({exc})")

        ys = np.array(sample_set.indices, dtype=np.int64) // cfg.image.width
        xs = np.array(sample_set.indices, dtype=np.int64) % cfg.image.width
        samples = aligned_latent[ys, xs]
        signs = 2 * np.array(bits_for_samples, dtype=np.float32) - 1.0
        watermark = alpha_effective * signs
        numerator = float(np.dot(samples, watermark))
        denom = float(np.linalg.norm(watermark) + 1e-9)
        raw_score = numerator / denom
        score = abs(raw_score)
        if score_threshold is not None and math.isfinite(float(score_threshold)):
            score_pass = score >= float(score_threshold)
            print(
                f"[verifier] correlation score = {score:.4f} (raw={raw_score:.4f}, "
                f"tau={float(score_threshold):.4f}, pass={score_pass})"
            )
        else:
            print(f"[verifier] correlation score = {score:.4f} (raw={raw_score:.4f})")
    else:
        if require_decode:
            raise RuntimeError("Decoded payload check requires --image")
        if zk_backend == "risc0":
            sample_root, _ = build_sample_merkle_sha(
                sample_trace.entries,
                gaussian_fixed=gaussian_fixed,
                combined_fixed=combined_fixed,
                bits=bits_for_samples,
            )
        else:
            sample_root, _ = build_sample_merkle(
                sample_trace.entries,
                gaussian_fixed=gaussian_fixed,
                combined_fixed=combined_fixed,
                bits=bits_for_samples,
            )
        if sample_root.hex() != metadata["sample_root"]:
            if require_root:
                raise RuntimeError("Sample root mismatch between sample trace and metadata")
            print("[verifier] sample root mismatch between sample trace and metadata")

    if zk_backend == "risc0":
        risc0_meta = metadata.get("risc0")
        if not risc0_meta:
            raise RuntimeError("RISC0 metadata missing")
        risc0_paths = Risc0Paths(
            public=_resolve(meta_path.parent, risc0_meta["public"]),
            witness=_resolve(meta_path.parent, risc0_meta["witness"]),
            receipt=_resolve(meta_path.parent, risc0_meta["receipt"]),
        )
        risc0_time = verify_risc0_receipt(risc0_paths, cfg.zk.verifier_cmd)
        print(f"[verifier] sample trace validated; RISC0 receipt verified in {risc0_time:.3f}s")
    else:
        halo2_meta = metadata.get("halo2")
        if not halo2_meta:
            raise RuntimeError("Halo2 metadata missing")
        halo2_paths = Halo2Paths(
            public=_resolve(meta_path.parent, halo2_meta["public"]),
            witness=_resolve(meta_path.parent, halo2_meta["witness"]),
            proof=_resolve(meta_path.parent, halo2_meta["proof"]),
        )
        halo2_time = verify_halo2_proof(halo2_paths, cfg.zk)
        print(f"[verifier] sample trace validated; halo2 verification completed in {halo2_time:.3f}s")

    if not legacy_metadata_only and not score_pass:
        raise RuntimeError(
            f"Score below threshold ({float(score):.4f} < {float(score_threshold):.4f}); verification failed"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="PP-Mark v0.3 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    prover = sub.add_parser("prover", help="Run prover pipeline")
    prover.add_argument("--config", required=True)
    prover.add_argument("--prompt", required=True)
    prover.add_argument("--output", required=True)
    prover.add_argument(
        "--allow-legacy-backend",
        action="store_true",
        help="Allow legacy backends (non-SP1).",
    )
    prover.add_argument("--seed-hex")
    prover.add_argument("--secret-hex")
    prover.add_argument("--secret-file")
    prover.add_argument(
        "--secret-output",
        help="Optional producer-secret output path; must be outside the public artifact directory.",
    )
    prover.add_argument(
        "--retain-private-witness",
        action="store_true",
        help="Retain private witness/latent diagnostics under output/private (never publish them).",
    )
    prover.add_argument("--model-id")
    prover.add_argument("--usecase-tag")
    prover.add_argument("--user-tag")
    prover.add_argument("--registry-address")
    prover.add_argument("--vk-cid")
    prover.add_argument("--proof-cid")
    prover.add_argument("--device-backend", default="cpu", choices=["cpu", "cuda"])
    prover.add_argument(
        "--uniform-backend",
        default="auto",
        choices=["auto", "cpu", "gpu_xorshift"],
        help="Uniform RNG backend: auto selects gpu_xorshift for CUDA and cpu Poseidon otherwise.",
    )
    prover.add_argument(
        "--use-random-latents",
        action="store_true",
        help="TEST ONLY: bypass watermark noise and feed pure numpy randn latents to SDXL.",
    )
    prover.add_argument(
        "--legacy-poseidon",
        action="store_true",
        help="Legacy: use BN254 Poseidon binding (Pallas binding is default).",
    )
    prover.add_argument(
        "--skip-proof",
        action="store_true",
        help="Skip ZKP proof generation; save all metadata/traces for later proof.",
    )
    prover.set_defaults(func=run_prover)

    verifier = sub.add_parser("verifier", help="Run verifier pipeline")
    verifier.add_argument("--config", required=True)
    verifier.add_argument("--metadata", required=True)
    verifier.add_argument(
        "--image",
        help="Image/latent path required for score-gated verification (use --legacy-metadata-only to skip).",
    )
    verifier.add_argument(
        "--mode",
        choices=["detect", "attest"],
        default="attest",
        help="Detect checks the robust signal; Attest additionally checks exact image binding and P1-P4 receipt.",
    )
    verifier.add_argument(
        "--score-latent",
        help="Debug-only recovered .npy latent for Detect scoring while --image remains the attested image.",
    )
    verifier.add_argument(
        "--expected-key-commitment",
        help="Trusted 32-byte producer-key commitment (hex) required for authenticated Attest.",
    )
    verifier.add_argument(
        "--allow-untrusted-producer-key",
        action="store_true",
        help="Test-only: verify receipt self-consistency without authenticating the producer key.",
    )
    verifier.add_argument("--model-id", help="Optional model identifier for registry/VK selection")
    verifier.add_argument(
        "--allow-legacy-backend",
        action="store_true",
        help="Allow legacy backends (non-SP1).",
    )
    verifier.add_argument(
        "--legacy-metadata-only",
        action="store_true",
        help="Legacy mode: allow metadata-only verification without score gating.",
    )
    verifier.add_argument(
        "--uniform-backend",
        default="poseidon",
        choices=["poseidon", "gpu_xorshift"],
        help="Uniform RNG backend (kept for compatibility; unused when recomputing from recovered latent).",
    )
    verifier.add_argument(
        "--allow-latent-png",
        action="store_true",
        help="Debug-only: allow latent PNG min/max rescale path (do not use in main experiments).",
    )
    verifier.add_argument(
        "--max-shift",
        type=int,
        default=4,
        help="Max pixel shift for sync search (translation); set 0 to disable",
    )
    verifier.add_argument(
        "--max-rotation",
        type=float,
        default=2.0,
        help="Max rotation (degrees) for sync search; set 0 to disable",
    )
    verifier.add_argument(
        "--rotation-step",
        type=float,
        default=1.0,
        help="Rotation step (degrees) for sync search",
    )
    verifier.add_argument(
        "--rotation-strategy",
        choices=["auto", "grid", "coarse_to_fine"],
        default="auto",
        help="Rotation search strategy for sync: auto/grid/coarse_to_fine.",
    )
    verifier.add_argument(
        "--crop-keep-ratio",
        type=float,
        default=0.0,
        help="Enable crop&scale search with keep ratio (e.g., 0.75).",
    )
    verifier.add_argument("--crop-grid", type=int, default=3, help="Crop grid size for coarse search.")
    verifier.add_argument("--crop-refine", action="store_true", help="Enable crop refinement stage.")
    verifier.add_argument("--crop-refine-grid", type=int, default=5, help="Crop refine grid size.")
    verifier.add_argument("--crop-refine-step", type=float, default=0.05, help="Crop refine step as fraction of size.")
    verifier.add_argument(
        "--sync-mode",
        choices=["auto", "on", "off"],
        default="auto",
        help="Sync search mode: auto runs search only on mismatch; on always attempts search; off disables search",
    )
    verifier.add_argument("--ddim-model-id", help="Optional diffusers model id for UNet-based DDIM inversion")
    verifier.add_argument("--ddim-device", default="cpu", help="Device for DDIM inversion (default cpu)")
    verifier.add_argument("--ddim-steps", type=int, default=50, help="DDIM steps for inversion model (default 50)")
    verifier.add_argument("--ddim-dtype", default="float32", help="Torch dtype for inversion model (default float32)")
    verifier.add_argument(
        "--ddim-guidance-scale",
        type=float,
        help="Guidance scale for DDIM inversion (defaults to prover/metadata value when omitted)",
    )
    verifier.add_argument(
        "--ddim-negative-prompt",
        help="Negative prompt for DDIM inversion (defaults to prover/metadata value when omitted)",
    )
    verifier.add_argument(
        "--require-root",
        action="store_true",
        help="Legacy: require sample_root recomputation match.",
    )
    verifier.add_argument(
        "--require-decode",
        dest="require_decode",
        action="store_true",
        help="Require decoded payload match from inversion (diagnostic).",
    )
    verifier.add_argument(
        "--no-decode",
        dest="require_decode",
        action="store_false",
        help="Disable decoded payload check even when --image is provided.",
    )
    verifier.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Legacy alias for --tau.",
    )
    verifier.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Soft-detect threshold (score >= tau).",
    )
    verifier.add_argument(
        "--tau-file",
        default="",
        help="JSON file containing tau_score.",
    )
    verifier.add_argument(
        "--exact-latent",
        action="store_true",
        help="Use exact latent from .npy (skip inversion) for strict opening checks.",
    )
    verifier.add_argument(
        "--opening-mode",
        default="strict",
        choices=["strict", "robust"],
        help="Opening verification mode (strict exact match or robust tolerance).",
    )
    verifier.add_argument(
        "--opening-eps",
        type=float,
        default=0.0,
        help="Tolerance for robust opening checks (latent units).",
    )
    verifier.set_defaults(func=run_verifier, require_decode=None)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
