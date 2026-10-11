#!/usr/bin/env python3
"""Derive a Score detection threshold from clean-image scores at a target FPR.

Input is one or more ``scores.csv`` files produced by ``eval_attack_manifest.py``
over a negative (clean, non-watermarked) collection, each row pairing a clean
image with watermark metadata so the recorded score is a false-positive score.
The output JSON uses the same schema as the threshold files at the repository
root and can be passed straight to ``--tau-file``.

Two estimators are provided:

``percentile``  the (1 - fpr) quantile with linear interpolation. This is the
                default and is the estimator behind the n=1000 SD2.1 reference
                calibration.
``kth-largest`` the k-th largest score with k = max(1, round(fpr * n)), a
                distribution-free order statistic that is more conservative on
                small collections.

The frozen thresholds shipped at the repository root are preserved constants
from the reported evaluation; their source negative collections are not
redistributed, so this tool reproduces the calibration *procedure* rather than
those exact numbers. Calibrate a fresh threshold whenever the model, alpha,
sample count, or inversion settings change, and keep the calibration and
evaluation conditions identical.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List

import numpy as np


def _read_scores(paths: List[Path], column: str, label: str) -> np.ndarray:
    values: List[float] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise SystemExit(f"{path}: no '{column}' column (found {reader.fieldnames})")
            has_label = "label" in reader.fieldnames
            for row in reader:
                if label and has_label and str(row.get("label", "")).strip() != label:
                    continue
                raw = str(row.get(column, "")).strip()
                if raw in ("", "nan", "None"):
                    continue
                values.append(float(raw))
    if not values:
        raise SystemExit("No usable score rows were found.")
    return np.sort(np.asarray(values, dtype=np.float64))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a Score threshold at a target false-positive rate."
    )
    parser.add_argument(
        "--scores",
        nargs="+",
        required=True,
        help="One or more scores.csv files over a clean (negative) collection.",
    )
    parser.add_argument("--output", required=True, help="Destination threshold JSON path.")
    parser.add_argument("--fpr", type=float, default=0.01, help="Target false-positive rate.")
    parser.add_argument(
        "--estimator",
        default="percentile",
        choices=["percentile", "kth-largest"],
        help="Threshold estimator (default: percentile).",
    )
    parser.add_argument("--score-column", default="score", help="CSV column holding the score.")
    parser.add_argument(
        "--label",
        default="",
        help="Keep only rows with this 'label' value, when the column is present (e.g. 'neg').",
    )
    parser.add_argument("--model-family", default="", help="Recorded for provenance, e.g. SD2.1.")
    parser.add_argument("--alpha", type=float, default=None, help="Embedding strength used.")
    parser.add_argument("--sample-count", type=int, default=None, help="N used.")
    parser.add_argument("--inversion-dtype", default="float32")
    parser.add_argument("--inversion-steps", type=int, default=100)
    parser.add_argument("--source", default="", help="Free-text provenance note.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.fpr < 1.0:
        raise SystemExit("--fpr must lie in (0, 1)")

    paths = [Path(p).resolve() for p in args.scores]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("Missing score files: " + ", ".join(missing))

    scores = _read_scores(paths, args.score_column, args.label)
    count = int(scores.size)

    if args.estimator == "percentile":
        tau = float(np.percentile(scores, 100.0 * (1.0 - args.fpr)))
        estimator_detail = f"(1-fpr) quantile, linear interpolation, n={count}"
    else:
        k = max(1, int(round(args.fpr * count)))
        if k > count:
            raise SystemExit(f"k={k} exceeds n={count}")
        tau = float(scores[-k])
        estimator_detail = f"{k}-th largest of n={count}"

    payload = {
        "tau_score": tau,
        "fpr_target": args.fpr,
        "estimator": args.estimator,
        "estimator_detail": estimator_detail,
        "count": count,
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=0)),
        "min": float(scores[0]),
        "max": float(scores[-1]),
        "p95": float(np.percentile(scores, 95)),
        "p99": float(np.percentile(scores, 99)),
        "p999": float(np.percentile(scores, 99.9)),
        "inversion_dtype": args.inversion_dtype,
        "inversion_steps": args.inversion_steps,
        "scores_inputs": [p.name for p in paths],
    }
    if args.model_family:
        payload["model_family"] = args.model_family
    if args.alpha is not None:
        payload["alpha"] = args.alpha
    if args.sample_count is not None:
        payload["sample_count"] = args.sample_count
    payload["source"] = args.source or (
        "Calibrated with scripts/calibrate_threshold.py on the listed clean-score inputs."
    )
    payload["note"] = (
        "Valid only under the calibration conditions recorded here. Recalibrate when the "
        "model, alpha, sample count, or inversion settings change."
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"[calibrate] n={count} estimator={args.estimator} ({estimator_detail})")
    print(f"[calibrate] tau_score={tau!r} at FPR={args.fpr}")
    print(f"[calibrate] wrote {output}")


if __name__ == "__main__":
    main()
