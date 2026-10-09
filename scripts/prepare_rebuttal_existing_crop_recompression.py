#!/usr/bin/env python3
"""Add JPEG recompression to the submitted random-crop-and-resize images."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


TAU = 2.4665623073926706


def find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "config_alpha4_k1000.json").exists():
            return candidate
    raise RuntimeError("Could not locate PP-Mark repository root")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jpeg_roundtrip(source: Path, quality: int) -> Image.Image:
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, subsampling=2)
    buffer.seek(0)
    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def parse_indices(raw: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("indices must be a non-empty unique comma-separated list")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", default="0,1,2")
    parser.add_argument("--qualities", default="75,25")
    parser.add_argument("--crop-root", required=True, help="Directory containing crop images, manifest_crop.jsonl, crop_boxes.json and crop scores")
    parser.add_argument("--metadata-root", help="Base for metadata paths in the crop manifest; defaults to --crop-root")
    parser.add_argument(
        "--output-dir", default="outputs/rebuttal_existing_crop_recompression_pilot"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    indices = parse_indices(args.indices)
    qualities = parse_indices(args.qualities)
    if any(quality < 1 or quality > 100 for quality in qualities):
        raise ValueError("JPEG qualities must be in [1, 100]")

    root = find_repo_root(Path(__file__).resolve())
    output = (root / args.output_dir).resolve()
    outputs_root = (root / "outputs").resolve()
    if outputs_root not in output.parents:
        raise ValueError("Choose an output subdirectory under this repository's outputs/")
    if output.exists():
        if not args.force:
            raise FileExistsError(f"Output exists: {output}; use --force")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    crop_root = (root / args.crop_root).resolve()
    pp_eval_root = (root / args.metadata_root).resolve() if args.metadata_root else crop_root
    crop_rows = {
        int(row["idx"]): row for row in read_jsonl(crop_root / "manifest_crop.jsonl")
    }
    with (crop_root / "eval_crop_full100_maxlevers/scores.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        score_rows = {int(row["idx"]): row for row in csv.DictReader(handle)}
    crop_boxes = json.loads((crop_root / "crop_boxes.json").read_text(encoding="utf-8"))

    manifests: dict[int, list[dict[str, Any]]] = {
        quality: [] for quality in qualities
    }
    provenance: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    for index in indices:
        row = crop_rows.get(index)
        score_row = score_rows.get(index)
        if row is None or score_row is None:
            raise KeyError(f"Missing submitted crop artifact for idx={index}")

        source = crop_root / str(row["image_path"])
        if not source.exists():
            raise FileNotFoundError(source)
        metadata_source = pp_eval_root / str(row["metadata_path"])
        if not metadata_source.exists():
            raise FileNotFoundError(metadata_source)

        metadata_relative = Path("metadata") / f"img_{index:04d}.json"
        metadata_copy = output / metadata_relative
        metadata_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(metadata_source, metadata_copy)

        baseline_rows.append(
            {
                "idx": index,
                "score": float(score_row["score"]),
                "pass": float(score_row["score"]) >= TAU,
                "runtime_sec": float(score_row["runtime_sec"]),
            }
        )

        for quality in qualities:
            condition = f"submitted_crop75_resize_then_jpeg_q{quality}"
            image_relative = Path(f"jpeg_q{quality}") / f"img_{index:04d}.png"
            destination = output / image_relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            jpeg_roundtrip(source, quality).save(destination, format="PNG")

            manifest_row = {
                "idx": index,
                "prompt_id": row.get("prompt_id"),
                "seed": row.get("seed"),
                "label": "pos",
                "attack": condition,
                "strength": condition,
                "image_path": image_relative.as_posix(),
                "image_relpath": image_relative.as_posix(),
                "metadata_path": metadata_copy.relative_to(root).as_posix(),
                "metadata_relpath": metadata_copy.relative_to(root).as_posix(),
            }
            manifests[quality].append(manifest_row)
            provenance.append(
                {
                    **manifest_row,
                    "submitted_crop_box": crop_boxes[f"img_{index:03d}"],
                    "submitted_crop_source": str(source),
                    "submitted_crop_source_sha256": sha256(source),
                    "recompressed_sha256": sha256(destination),
                    "metadata_sha256": sha256(metadata_copy),
                    "submitted_crop_score": float(score_row["score"]),
                    "submitted_crop_pass": float(score_row["score"]) >= TAU,
                }
            )

    for quality, rows in manifests.items():
        write_jsonl(output / f"manifest_q{quality}.jsonl", rows)
        write_jsonl(output / f"manifest_q{quality}_gpu0.jsonl", rows[0::2])
        write_jsonl(output / f"manifest_q{quality}_gpu1.jsonl", rows[1::2])
    write_jsonl(output / "provenance.jsonl", provenance)
    (output / "baseline_crop_scores.json").write_text(
        json.dumps(baseline_rows, indent=2), encoding="utf-8"
    )
    (output / "config.json").write_text(
        json.dumps(
            {
                "indices": list(indices),
                "qualities": list(qualities),
                "source": "submitted random crop 75% plus bicubic resize",
                "source_manifest": str(
                    crop_root / "manifest_crop.jsonl"
                ).replace("\\", "/"),
                "alignment": {
                    "ddim_dtype": "float32",
                    "ddim_steps": 100,
                    "max_shift": 16,
                    "crop_keep_ratio": 0.75,
                    "crop_grid": 9,
                    "crop_refine_grid": 9,
                    "crop_refine_step": 0.03,
                },
                "tau": TAU,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "PREPARED").write_text("validated\n", encoding="ascii")
    print(
        f"Prepared {len(indices)} submitted crop images at "
        f"JPEG qualities {qualities} in {output}"
    )


if __name__ == "__main__":
    main()
