#!/usr/bin/env python3
"""Summarize persistent detection from already computed checkpoint scores."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_scores(path: Path) -> dict[int, float]:
    scores: dict[int, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            index, score = int(row["idx"]), float(row["score"])
            if index in scores or not math.isfinite(score):
                raise ValueError(f"Duplicate index or non-finite score in {path}: {index}")
            scores[index] = score
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-scores", type=Path, required=True)
    parser.add_argument("--jpeg-scores", type=Path, required=True)
    parser.add_argument("--tau", type=float, default=2.4665623073926706)
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    crop, jpeg = read_scores(args.crop_scores), read_scores(args.jpeg_scores)
    if not math.isfinite(args.tau) or args.expected_count <= 0:
        raise ValueError("Require a finite threshold and a positive expected count")
    if crop.keys() != jpeg.keys() or len(crop) != args.expected_count:
        raise ValueError("Both checkpoints must contain the same complete set of indices")
    result = {
        "n": len(crop),
        "tau": args.tau,
        "criterion": "score >= tau at both crop-resize and crop-resize-JPEG checkpoints",
        "crop_pass": sum(score >= args.tau for score in crop.values()),
        "final_jpeg_pass": sum(score >= args.tau for score in jpeg.values()),
        "persistent_pass": sum(crop[i] >= args.tau and jpeg[i] >= args.tau for i in crop),
        "crop_scores": str(args.crop_scores),
        "jpeg_scores": str(args.jpeg_scores),
    }
    text = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
