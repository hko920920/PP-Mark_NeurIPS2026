#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image


Pair = Tuple[Path, str]


def _batch_iter(items: Sequence[Pair], batch_size: int) -> Iterable[Sequence[Pair]]:
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def _bootstrap_mean_ci(values: np.ndarray, draws: int, seed: int) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _load_method_pairs(manifest_path: Path) -> Tuple[List[str], Dict[str, List[Pair]]]:
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise RuntimeError(f"Empty manifest: {manifest_path}")

    methods = list(rows[0]["images"])
    pairs: Dict[str, List[Pair]] = {method: [] for method in methods}
    for row_index, row in enumerate(rows):
        if list(row["images"]) != methods:
            raise RuntimeError(f"Inconsistent methods at row {row_index}")
        prompt = str(row["prompt_text"])
        for method, raw_path in row["images"].items():
            image_path = Path(raw_path)
            if not image_path.is_absolute():
                image_path = manifest_path.parent / image_path
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            pairs[method].append((image_path, prompt))
    return methods, pairs


def _compute_scores(
    pairs: Sequence[Pair],
    model,
    processor,
    device: str,
    batch_size: int,
) -> np.ndarray:
    scores: List[float] = []
    with torch.no_grad():
        for batch in _batch_iter(pairs, batch_size):
            images = [Image.open(path).convert("RGB") for path, _ in batch]
            prompts = [prompt for _, prompt in batch]
            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=model.config.text_config.max_position_embeddings,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            image_features = outputs.image_embeds
            text_features = outputs.text_embeds
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            scores.extend((image_features * text_features).sum(dim=-1).cpu().tolist())
    return np.asarray(scores, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute matched CLIPScore for all rebuttal quality baselines."
    )
    parser.add_argument(
        "--manifest",
        required=True,
    )
    parser.add_argument(
        "--output",
        default="outputs/quality/clipscore",
    )
    parser.add_argument("--model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.bootstrap_draws <= 0:
        parser.error("--batch-size and --bootstrap-draws must be positive")

    from transformers import CLIPModel, CLIPProcessor
    import transformers

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods, pairs_by_method = _load_method_pairs(manifest_path)

    model = CLIPModel.from_pretrained(args.model).to(device)
    processor = CLIPProcessor.from_pretrained(args.model)
    model.eval()

    labels = {
        "clean": "Clean",
        "pp_mark": "PP-Mark",
        "gaussian_shading": "Gaussian Shading",
        "hidden": "HiDDeN",
        "invismark": "InvisMark",
        "ringid": "RingID",
        "stable_signature": "Stable Signature",
        "tree_ring": "Tree-Ring",
        "wind": "WIND",
        "trustmark": "TrustMark",
    }

    summary: Dict[str, dict] = {}
    scores_by_method: Dict[str, np.ndarray] = {}
    per_image_rows: List[dict] = []
    for method_index, method in enumerate(methods):
        scores = _compute_scores(
            pairs_by_method[method],
            model,
            processor,
            device,
            args.batch_size,
        )
        scores_by_method[method] = scores
        ci = _bootstrap_mean_ci(
            scores,
            draws=args.bootstrap_draws,
            seed=args.seed + method_index,
        )
        summary[method] = {
            "label": labels[method],
            "n": int(len(scores)),
            "mean": float(scores.mean()),
            "std": float(scores.std(ddof=1)),
            "mean_ci95": list(ci),
            "mean_x100": float(scores.mean() * 100.0),
            "std_x100": float(scores.std(ddof=1) * 100.0),
            "mean_ci95_x100": [float(ci[0] * 100.0), float(ci[1] * 100.0)],
        }
        for pair_index, (pair, score) in enumerate(zip(pairs_by_method[method], scores)):
            per_image_rows.append(
                {
                    "method": method,
                    "label": labels[method],
                    "index": pair_index,
                    "image_path": str(pair[0].resolve()),
                    "prompt_text": pair[1],
                    "clipscore": f"{score:.12f}",
                }
            )
        print(f"[CLIPScore] {method_index + 1}/{len(methods)} {labels[method]}", flush=True)

    clean_scores = scores_by_method["clean"]
    for method_index, method in enumerate(methods):
        delta = scores_by_method[method] - clean_scores
        delta_ci = _bootstrap_mean_ci(
            delta,
            draws=args.bootstrap_draws,
            seed=args.seed + 100 + method_index,
        )
        summary[method]["paired_mean_delta_vs_clean"] = float(delta.mean())
        summary[method]["paired_mean_delta_ci95"] = list(delta_ci)
        summary[method]["paired_mean_delta_vs_clean_x100"] = float(delta.mean() * 100.0)
        summary[method]["paired_mean_delta_ci95_x100"] = [
            float(delta_ci[0] * 100.0),
            float(delta_ci[1] * 100.0),
        ]

    protocol = {
        "manifest": str(manifest_path),
        "common_pairs": len(next(iter(pairs_by_method.values()))),
        "model": args.model,
        "definition": "cosine similarity between normalized CLIP image and text embeddings",
        "text_truncation": model.config.text_config.max_position_embeddings,
        "device": device,
        "bootstrap_draws": args.bootstrap_draws,
        "seed": args.seed,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps({"protocol": protocol, "methods": summary}, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "label",
                "n",
                "clipscore_mean_x100",
                "clipscore_std_x100",
                "clipscore_mean_ci95_low_x100",
                "clipscore_mean_ci95_high_x100",
                "paired_mean_delta_vs_clean_x100",
                "paired_delta_ci95_low_x100",
                "paired_delta_ci95_high_x100",
            ]
        )
        for method in methods:
            row = summary[method]
            writer.writerow(
                [
                    method,
                    row["label"],
                    row["n"],
                    f"{row['mean_x100']:.6f}",
                    f"{row['std_x100']:.6f}",
                    f"{row['mean_ci95_x100'][0]:.6f}",
                    f"{row['mean_ci95_x100'][1]:.6f}",
                    f"{row['paired_mean_delta_vs_clean_x100']:.6f}",
                    f"{row['paired_mean_delta_ci95_x100'][0]:.6f}",
                    f"{row['paired_mean_delta_ci95_x100'][1]:.6f}",
                ]
            )
    with (output_dir / "per_image.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(per_image_rows[0]))
        writer.writeheader()
        writer.writerows(per_image_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
