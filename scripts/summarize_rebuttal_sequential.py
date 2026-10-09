#!/usr/bin/env python3
"""Summarize sequential-transform score CSVs with exact binomial intervals."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta


def clopper_pearson(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    alpha = 1.0 - confidence
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return lower, upper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", nargs="+", required=True)
    parser.add_argument("--tau", type=float, default=2.4665623073926706)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for raw_path in args.scores:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                grouped[(row["strength"], row["label"])].append(float(row["score"]))

    results: dict[str, dict[str, object]] = {}
    for (condition, label), scores in sorted(grouped.items()):
        values = np.asarray(scores, dtype=np.float64)
        successes = int(np.sum(values >= args.tau))
        total = int(values.size)
        interval = clopper_pearson(successes, total)
        results.setdefault(condition, {})[label] = {
            "successes": successes,
            "total": total,
            "rate": successes / total,
            "ci95": list(interval),
            "score_mean": float(np.mean(values)),
            "score_median": float(np.median(values)),
            "score_p10": float(np.quantile(values, 0.10)),
            "score_min": float(np.min(values)),
            "margin_median": float(np.median(values - args.tau)),
        }

    payload = {
        "tau": args.tau,
        "pilot": bool(args.pilot),
        "reporting_warning": (
            "Pilot negative rates are diagnostics only because pilot negatives come from the "
            "threshold calibration split."
            if args.pilot
            else None
        ),
        "conditions": results,
    }
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Sequential-transform results",
        "",
        f"Frozen threshold: `{args.tau:.12f}`",
        "",
    ]
    if args.pilot:
        lines.extend(
            [
                "Pilot negative rates are diagnostics only; final FPR requires a disjoint held-out set.",
                "",
            ]
        )
    lines.extend(
        [
            "| Condition | Positive TPR (95% CI) | Negative FPR (95% CI) | Positive median score | Positive p10 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition, labels in sorted(results.items()):
        pos = labels.get("pos")
        neg = labels.get("neg")
        if not pos or not neg:
            raise RuntimeError(f"Condition {condition} requires both pos and neg rows")
        pos_ci = pos["ci95"]
        neg_ci = neg["ci95"]
        lines.append(
            f"| {condition} | {pos['successes']}/{pos['total']}={pos['rate']:.3f} "
            f"[{pos_ci[0]:.3f},{pos_ci[1]:.3f}] | "
            f"{neg['successes']}/{neg['total']}={neg['rate']:.3f} "
            f"[{neg_ci[0]:.3f},{neg_ci[1]:.3f}] | "
            f"{pos['score_median']:.3f} | {pos['score_p10']:.3f} |"
        )
    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
