#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from PIL import Image

from ppmark_v03.config import GlobalConfig
from ppmark_v03.embedding import inject_watermark
from ppmark_v03.payload import bytes_to_bits, int_to_fixed_bytes, make_ctx_hash
from ppmark_v03.rs import ReedSolomonCodec
from ppmark_v03.sampling import deterministic_sample
from ppmark_v03.tables import InverseCDFTable


def _load_manifest(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _seed_to_bytes(seed: int) -> bytes:
    if seed < 0:
        raise ValueError("Seed must be non-negative")
    return seed.to_bytes(32, "big")


def _load_secret(args: argparse.Namespace) -> bytes:
    if args.secret_hex:
        return bytes.fromhex(args.secret_hex.strip().lower().removeprefix("0x"))
    if args.secret_file:
        value = Path(args.secret_file).read_text(encoding="utf-8").strip()
        return bytes.fromhex(value.lower().removeprefix("0x"))
    if args.secret_env:
        value = os.environ.get(args.secret_env, "").strip()
        if value:
            return bytes.fromhex(value.lower().removeprefix("0x"))
    raise RuntimeError("Missing secret; set --secret-hex, --secret-file, or --secret-env")


def _build_payload_bits(
    prompt: str,
    model_id: str,
    seed_bytes: bytes,
    secret_bytes: bytes,
    width: int,
    height: int,
    codec: ReedSolomonCodec,
) -> tuple[int, bytes, List[int]]:
    prompt_hash = int.from_bytes(hashlib.sha256(prompt.encode("utf-8")).digest(), "big")
    ctx_hash = make_ctx_hash(
        prompt_hash=prompt_hash,
        model_id=model_id,
        seed=seed_bytes,
        width=width,
        height=height,
        usecase_tag=None,
        user_tag=None,
    )
    binding_bytes = hashlib.sha256(b"ctx_hash" + ctx_hash.to_bytes(32, "big") + secret_bytes).digest()
    binding = int.from_bytes(binding_bytes, "big")
    message_bytes = int_to_fixed_bytes(binding, codec.k)
    codeword = codec.encode(message_bytes)
    return binding, codeword, bytes_to_bits(codeword)


def _is_sdxl_model(model_id: str) -> bool:
    tag = model_id.lower()
    return "sdxl" in tag or "stable-diffusion-xl" in tag


def _load_pipe(model_id: str, device: str, dtype: str, scheduler: str):
    from diffusers import DDIMScheduler, StableDiffusionPipeline, StableDiffusionXLPipeline  # type: ignore

    torch_dtype = getattr(torch, dtype, None)
    if torch_dtype is None:
        raise ValueError(f"Unsupported torch dtype: {dtype}")
    variant = "fp16" if dtype in {"float16", "bfloat16"} else None
    use_sdxl = _is_sdxl_model(model_id)
    pipe_cls = StableDiffusionXLPipeline if use_sdxl else StableDiffusionPipeline
    kwargs = {"torch_dtype": torch_dtype}
    if variant and use_sdxl:
        kwargs["variant"] = variant
    try:
        pipe = pipe_cls.from_pretrained(model_id, **kwargs)
    except (TypeError, ValueError):
        kwargs.pop("variant", None)
        pipe = pipe_cls.from_pretrained(model_id, **kwargs)
    scheduler_name = scheduler.lower()
    if scheduler_name != "ddim":
        raise ValueError(f"Unsupported scheduler: {scheduler}")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _link_existing(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    raise ValueError(f"Unknown link mode: {mode}")


def _resolve_value(row: dict, key: str, fallback):
    meta = row.get("meta", {})
    if key in row and row[key] is not None:
        return row[key]
    if key in meta and meta[key] is not None:
        return meta[key]
    return fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate quality images from a manifest without proofs.")
    parser.add_argument("--manifest", required=True, help="Manifest JSONL with prompt/seed pairs.")
    parser.add_argument("--output-root", required=True, help="Root directory for output images.")
    parser.add_argument("--config", default="config_alpha4_k1000.json")
    parser.add_argument("--model-id", default="")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--scheduler", default="ddim")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--secret-hex", default="")
    parser.add_argument("--secret-file", default="")
    parser.add_argument("--secret-env", default="PPMARK_SECRET_HEX")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--link-existing", action="store_true", help="Link/copy source_image_path when present.")
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    args = parser.parse_args()

    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows = _load_manifest(Path(args.manifest))
    if args.start_index:
        rows = rows[args.start_index :]
    if args.max_images:
        rows = rows[: args.max_images]

    cfg = GlobalConfig.load(Path(args.config))
    inverse_cdf = InverseCDFTable.from_file(cfg.tables.inverse_cdf_path)
    codec = ReedSolomonCodec(n=cfg.watermark.rs_n, k=cfg.watermark.rs_k)
    secret_bytes = _load_secret(args)

    model_id = args.model_id or cfg.model.base
    pipe = _load_pipe(model_id, args.device, args.dtype, args.scheduler)

    for idx, row in enumerate(rows, 1):
        prompt = str(row["prompt_text"])
        seed = int(row["seed"])
        alpha = float(row["alpha"])
        rel_path = Path(row["image_path"])
        out_path = out_root / rel_path
        if out_path.exists() and not args.overwrite:
            continue

        source_path = row.get("source_image_path")
        if source_path and args.link_existing:
            src = Path(source_path)
            if src.exists():
                _link_existing(src, out_path, args.link_mode)
                continue

        cfg.image.alpha = alpha
        steps = int(_resolve_value(row, "num_inference_steps", args.steps or 50))
        guidance = float(_resolve_value(row, "guidance_scale", args.guidance_scale or 7.5))
        negative_prompt = _resolve_value(row, "negative_prompt", args.negative_prompt)
        if negative_prompt is None:
            negative_prompt = ""

        seed_bytes = _seed_to_bytes(seed)
        binding, codeword, bits = _build_payload_bits(
            prompt=prompt,
            model_id=model_id,
            seed_bytes=seed_bytes,
            secret_bytes=secret_bytes,
            width=cfg.image.width,
            height=cfg.image.height,
            codec=codec,
        )
        sample_set = deterministic_sample(
            total_pixels=cfg.image.total_pixels,
            width=cfg.image.width,
            height=cfg.image.height,
            key_material=codeword,
            count=cfg.image.sample_count(),
        )
        embedding = inject_watermark(
            binding=binding,
            bit_sequence=bits,
            config=cfg,
            inverse_cdf=inverse_cdf,
            sample_set=sample_set,
        )
        scale_factor = 1.0 / float(np.sqrt(1.0 + alpha * alpha))
        latent_scaled = embedding.latent_noise.astype(np.float32) * scale_factor
        torch_latents = torch.from_numpy(latent_scaled[None, ...]).to(device=args.device, dtype=pipe.dtype)
        gen = torch.Generator(device=args.device)
        gen.manual_seed(seed)
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance,
            latents=torch_latents,
            generator=gen,
            output_type="np",
        )
        img = result.images[0]
        img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img_u8).save(out_path)

        if idx % 25 == 0:
            print(f"[quality-generate] {idx}/{len(rows)} done")


if __name__ == "__main__":
    main()
