#!/usr/bin/env python3
"""Generate transformed copies of watermarked images for robustness evaluation.

The transforms and their parameters correspond to the post-transformation
detection table in the paper. Each transform writes its own JSONL manifest in
the schema accepted by ``eval_attack_manifest.py``, so the generated collection
can be scored without further preparation. Public metadata is carried over
unchanged from the original image, because detection is evaluated against the
claim attached to that original.

Random draws use fixed per-transform generators (crop=0, noise=1,
brightness=2, chained=3), so repeated runs over the same inputs are identical.
The realised crop boxes and brightness factors are written next to the
manifests; the sequential crop-resize-JPEG evaluation consumes ``crop_boxes.json``.
"""

from __future__ import annotations

import argparse
import io
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

JPEG_SUBSAMPLING = 2  # 4:2:0, matching the reported JPEG condition
DEFAULT_CROP_KEEP_RATIO = 0.75
DEFAULT_ROTATION_DEG = 75.0
DEFAULT_JPEG_QUALITY = 25
DEFAULT_BLUR_KERNEL = 8
DEFAULT_NOISE_SIGMA = 0.1
DEFAULT_BRIGHTNESS_JITTER = 0.6
DEFAULT_RESIZE_SCALE = 0.75
LEGACY_JPEG_QUALITY = 75


def _iter_manifest(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _resolve(base: Path, raw: str) -> Path:
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def _select_image_field(row: dict) -> str:
    for key in ("image_path", "image_relpath", "image"):
        value = row.get(key)
        if value:
            return str(value)
    raise RuntimeError(f"Manifest row has no image path: {row}")


def _select_metadata_field(row: dict) -> str:
    for key in ("metadata_path", "metadata_relpath", "ref_metadata_path"):
        value = row.get(key)
        if value:
            return str(value)
    raise RuntimeError(f"Manifest row has no metadata path: {row}")


def _attack_jpeg(image: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, subsampling=JPEG_SUBSAMPLING)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _attack_blur(image: Image.Image, kernel: int) -> Image.Image:
    radius = max(1, int(round(kernel / 2)))
    return image.filter(ImageFilter.BoxBlur(radius))


def _attack_noise(image: Image.Image, sigma: float, rng: np.random.Generator) -> Image.Image:
    arr = np.array(image, dtype=np.float32) / 255.0
    noise = rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0.0, 1.0)
    return Image.fromarray((arr * 255.0).astype(np.uint8))


def _attack_brightness_jitter(
    image: Image.Image, jitter: float, rng: np.random.Generator
) -> Tuple[Image.Image, float]:
    factor = 1.0 + float(rng.uniform(-jitter, jitter))
    return ImageEnhance.Brightness(image).enhance(factor), factor


def _attack_rotate(image: Image.Image, degrees: float) -> Image.Image:
    return image.rotate(degrees, resample=Image.BICUBIC, expand=False)


def _attack_resize(image: Image.Image, scale: float) -> Image.Image:
    width, height = image.size
    down = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(down, resample=Image.BICUBIC).resize(
        (width, height), resample=Image.BICUBIC
    )


def _random_crop_box(
    width: int, height: int, keep_ratio: float, rng: np.random.Generator
) -> Tuple[int, int, int, int]:
    scale = math.sqrt(keep_ratio)
    crop_w = max(1, int(round(width * scale)))
    crop_h = max(1, int(round(height * scale)))
    max_x = max(0, width - crop_w)
    max_y = max(0, height - crop_h)
    x0 = int(rng.integers(0, max_x + 1)) if max_x > 0 else 0
    y0 = int(rng.integers(0, max_y + 1)) if max_y > 0 else 0
    return (x0, y0, x0 + crop_w, y0 + crop_h)


def _attack_crop_scale(
    image: Image.Image, keep_ratio: float, rng: np.random.Generator
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    width, height = image.size
    box = _random_crop_box(width, height, keep_ratio, rng)
    cropped = image.crop(box)
    return cropped.resize((width, height), resample=Image.BICUBIC), box


ALL_TRANSFORMS = (
    "clean",
    "jpeg_q25",
    "blur_8x8",
    "noise_sigma_0.1",
    "brightness_jitter_0_6",
    "crop_scale_0.75",
    "rot_75",
    "jpeg_q75",
    "resize_0.75",
    "chained",
)
DEFAULT_TRANSFORMS = (
    "clean",
    "jpeg_q25",
    "blur_8x8",
    "noise_sigma_0.1",
    "brightness_jitter_0_6",
    "crop_scale_0.75",
    "rot_75",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply evaluation transforms to a watermarked image collection."
    )
    parser.add_argument("--manifest", required=True, help="Input JSONL manifest of original images.")
    parser.add_argument(
        "--output-root", required=True, help="Directory for transformed images and manifests."
    )
    parser.add_argument("--image-root", default="", help="Base path for relative image paths.")
    parser.add_argument("--metadata-root", default="", help="Base path for relative metadata paths.")
    parser.add_argument(
        "--transforms",
        default=",".join(DEFAULT_TRANSFORMS),
        help=f"Comma-separated subset of: {','.join(ALL_TRANSFORMS)}. Use 'all' for every transform.",
    )
    parser.add_argument("--max-images", type=int, default=0, help="Limit the number of rows processed.")
    parser.add_argument("--crop-keep-ratio", type=float, default=DEFAULT_CROP_KEEP_RATIO)
    parser.add_argument("--rotation-deg", type=float, default=DEFAULT_ROTATION_DEG)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--blur-kernel", type=int, default=DEFAULT_BLUR_KERNEL)
    parser.add_argument("--noise-sigma", type=float, default=DEFAULT_NOISE_SIGMA)
    parser.add_argument("--brightness-jitter", type=float, default=DEFAULT_BRIGHTNESS_JITTER)
    parser.add_argument("--resize-scale", type=float, default=DEFAULT_RESIZE_SCALE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest_dir = manifest_path.parent
    image_root = Path(args.image_root).resolve() if args.image_root else manifest_dir
    metadata_root = Path(args.metadata_root).resolve() if args.metadata_root else manifest_dir
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    requested = (
        list(ALL_TRANSFORMS)
        if args.transforms.strip().lower() == "all"
        else [name.strip() for name in args.transforms.split(",") if name.strip()]
    )
    unknown = [name for name in requested if name not in ALL_TRANSFORMS]
    if unknown:
        raise SystemExit(f"Unknown transforms: {', '.join(unknown)}")

    rows = list(_iter_manifest(manifest_path))
    if args.max_images:
        rows = rows[: args.max_images]
    if not rows:
        raise SystemExit("Manifest has no rows to process.")

    rng_crop = np.random.default_rng(0)
    rng_noise = np.random.default_rng(1)
    rng_bright = np.random.default_rng(2)
    rng_chain = np.random.default_rng(3)

    manifests: Dict[str, List[dict]] = {name: [] for name in requested}
    crop_boxes: Dict[str, List[int]] = {}
    brightness_factors: Dict[str, float] = {}

    for position, row in enumerate(rows):
        image_path = _resolve(image_root, _select_image_field(row))
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")
        metadata_path = _resolve(metadata_root, _select_metadata_field(row))
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata: {metadata_path}")

        item = f"img_{position:03d}"
        (output_root / item).mkdir(parents=True, exist_ok=True)
        base = Image.open(image_path).convert("RGB")

        produced: Dict[str, Image.Image] = {}
        if "clean" in requested:
            produced["clean"] = base
        if "jpeg_q25" in requested:
            produced["jpeg_q25"] = _attack_jpeg(base, args.jpeg_quality)
        if "blur_8x8" in requested:
            produced["blur_8x8"] = _attack_blur(base, args.blur_kernel)
        if "noise_sigma_0.1" in requested:
            produced["noise_sigma_0.1"] = _attack_noise(base, args.noise_sigma, rng_noise)
        if "brightness_jitter_0_6" in requested:
            image, factor = _attack_brightness_jitter(base, args.brightness_jitter, rng_bright)
            produced["brightness_jitter_0_6"] = image
            brightness_factors[item] = factor
        if "crop_scale_0.75" in requested:
            image, box = _attack_crop_scale(base, args.crop_keep_ratio, rng_crop)
            produced["crop_scale_0.75"] = image
            crop_boxes[item] = [int(value) for value in box]
        if "rot_75" in requested:
            produced["rot_75"] = _attack_rotate(base, args.rotation_deg)
        if "jpeg_q75" in requested:
            produced["jpeg_q75"] = _attack_jpeg(base, LEGACY_JPEG_QUALITY)
        if "resize_0.75" in requested:
            produced["resize_0.75"] = _attack_resize(base, args.resize_scale)
        if "chained" in requested:
            chained = _attack_jpeg(base, args.jpeg_quality)
            chained, _ = _attack_crop_scale(chained, args.crop_keep_ratio, rng_chain)
            chained = _attack_rotate(chained, args.rotation_deg)
            chained = _attack_blur(chained, args.blur_kernel)
            chained = _attack_noise(chained, args.noise_sigma, rng_chain)
            chained, _ = _attack_brightness_jitter(chained, args.brightness_jitter, rng_chain)
            produced["chained"] = chained

        for name in requested:
            relative = f"{item}/{name}.png"
            produced[name].save(output_root / relative)
            manifests[name].append(
                {
                    "idx": int(row.get("idx", position)),
                    "label": str(row.get("label", "pos")),
                    "attack": name,
                    "prompt_id": row.get("prompt_id"),
                    "seed": row.get("seed"),
                    "image_path": relative,
                    "metadata_path": str(metadata_path),
                }
            )

        print(f"[transform] {position + 1}/{len(rows)} {item}: {', '.join(requested)}", flush=True)

    for name, entries in manifests.items():
        target = output_root / f"manifest_{name}.jsonl"
        target.write_text(
            "\n".join(json.dumps(entry, ensure_ascii=True) for entry in entries) + "\n",
            encoding="utf-8",
        )

    if crop_boxes:
        (output_root / "crop_boxes.json").write_text(
            json.dumps(crop_boxes, indent=2) + "\n", encoding="utf-8"
        )
    if brightness_factors:
        (output_root / "brightness_factors.json").write_text(
            json.dumps(brightness_factors, indent=2) + "\n", encoding="utf-8"
        )

    settings = {
        "transforms": requested,
        "count": len(rows),
        "crop_keep_ratio": args.crop_keep_ratio,
        "rotation_deg": args.rotation_deg,
        "jpeg_quality": args.jpeg_quality,
        "jpeg_subsampling": "4:2:0",
        "blur_kernel": args.blur_kernel,
        "noise_sigma": args.noise_sigma,
        "brightness_jitter": args.brightness_jitter,
        "resize_scale": args.resize_scale,
        "rng_seeds": {"crop": 0, "noise": 1, "brightness": 2, "chained": 3},
        "note": (
            "brightness_jitter is the half-width of a uniform multiplicative factor drawn "
            "from [1-j, 1+j]; the realised per-image factors are in brightness_factors.json."
        ),
    }
    (output_root / "transform_settings.json").write_text(
        json.dumps(settings, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[transform] wrote {len(requested)} manifests to {output_root}")


if __name__ == "__main__":
    main()
