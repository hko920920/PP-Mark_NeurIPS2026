#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

from eval_rebuttal_quality_baselines import _polynomial_mmd


Key = Tuple[int, int]

SUBMITTED_PP_MARK = {
    "brisque_n": 100,
    "brisque_design": "25 prompts x 4 seeds",
    "brisque_mean": 9.92451937,
    "brisque_p95": 34.144476,
    "kid_n": 1000,
    "kid_subset_size": 500,
    "kid_splits": 20,
    "kid_mean": 0.0011013573985044322,
    "kid_std": 0.00020202137688984587,
}

METHOD_ORDER = (
    "pp_mark",
    "gaussian_shading",
    "hidden",
    "invismark",
    "ringid",
    "stable_signature",
    "tree_ring",
    "wind",
    "trustmark",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_manifest_keys(path: Path) -> List[Key]:
    keys: List[Key] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        keys.append((int(row["prompt_id"]), int(row["seed"])))
    if len(keys) != len(set(keys)):
        raise RuntimeError(f"Duplicate prompt/seed keys in {path}")
    return keys


def _reference_resampled_kid(
    method_features: np.ndarray,
    clean_reference_features: np.ndarray,
    splits: int,
    seed: int,
) -> np.ndarray:
    if len(method_features) < 2:
        raise RuntimeError("At least two method samples are required")
    if len(clean_reference_features) < len(method_features):
        raise RuntimeError("The disjoint clean reference must be at least as large as the method set")

    rng = np.random.default_rng(seed)
    x = torch.from_numpy(method_features).to(dtype=torch.float64)
    y = torch.from_numpy(clean_reference_features).to(dtype=torch.float64)
    scores: List[float] = []
    for _ in range(splits):
        indices = rng.choice(len(y), size=len(x), replace=False)
        scores.append(float(_polynomial_mmd(x, y[indices])))
    return np.asarray(scores, dtype=np.float64)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _format_kid(value: float) -> str:
    return f"{value * 1000:.3f}"


def _latex_escape(label: str) -> str:
    return label.replace("&", r"\&")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build auditable NeurIPS rebuttal tables from submitted and matched quality results."
    )
    parser.add_argument(
        "--matched-dir",
        default="outputs/rebuttal_quality_baselines_n100_corrected",
    )
    parser.add_argument(
        "--full-pp-dir",
        default="outputs/rebuttal_quality_ppmark_full1000_corrected",
    )
    parser.add_argument("--kid-reference-splits", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    matched_dir = Path(args.matched_dir)
    full_pp_dir = Path(args.full_pp_dir)
    matched_summary = _read_json(matched_dir / "summary.json")

    matched_keys = set(_read_manifest_keys(matched_dir / "matched_manifest.jsonl"))
    full_keys = _read_manifest_keys(full_pp_dir / "matched_manifest.jsonl")
    clean_full = np.load(full_pp_dir / "inception_features" / "clean.npy")
    if len(clean_full) != len(full_keys):
        raise RuntimeError("Full clean feature rows do not match the full manifest")

    disjoint_indices = [index for index, key in enumerate(full_keys) if key not in matched_keys]
    clean_disjoint = clean_full[disjoint_indices]
    if len(clean_disjoint) != 900:
        raise RuntimeError(f"Expected 900 disjoint clean references, found {len(clean_disjoint)}")

    independent_kid: Dict[str, dict] = {}
    split_rows: List[dict] = []
    for method_index, slug in enumerate(METHOD_ORDER):
        method_features = np.load(matched_dir / "inception_features" / f"{slug}.npy")
        if len(method_features) != 100:
            raise RuntimeError(f"Expected 100 features for {slug}, found {len(method_features)}")
        scores = _reference_resampled_kid(
            method_features,
            clean_disjoint,
            splits=args.kid_reference_splits,
            seed=args.seed + method_index,
        )
        label = matched_summary["brisque"][slug]["label"]
        independent_kid[slug] = {
            "label": label,
            "method_n": int(len(method_features)),
            "clean_reference_n": int(len(clean_disjoint)),
            "subset_size": int(len(method_features)),
            "splits": int(args.kid_reference_splits),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=1)),
            "median": float(np.median(scores)),
            "p025": float(np.quantile(scores, 0.025)),
            "p975": float(np.quantile(scores, 0.975)),
        }
        split_rows.extend(
            {
                "method": slug,
                "label": label,
                "split": split_index,
                "kid": f"{score:.12f}",
            }
            for split_index, score in enumerate(scores)
        )

    protocol = {
        "method_count": 100,
        "method_design": "100 common prompt/seed pairs shared by all methods",
        "clean_reference_count": int(len(clean_disjoint)),
        "clean_reference": (
            "The 900 clean images in the validated 1,000-image clean manifest "
            "whose prompt/seed keys are disjoint from the common-100 method set."
        ),
        "kid_estimator": "Unbiased polynomial MMD, (x^T y / d + 1)^3",
        "kid_features": "Cached torchvision Inception_V3_Weights.DEFAULT features, fc=Identity",
        "kid_subset_size": 100,
        "kid_reference_resamples": args.kid_reference_splits,
        "seed": args.seed,
        "finite_sample_note": (
            "Unbiased KID can be slightly negative at finite sample sizes. "
            "Negative values near zero are not interpreted as better-than-identical quality."
        ),
    }
    (matched_dir / "independent_reference_kid.json").write_text(
        json.dumps({"protocol": protocol, "methods": independent_kid}, indent=2),
        encoding="utf-8",
    )
    _write_csv(
        matched_dir / "independent_reference_kid_split_scores.csv",
        ("method", "label", "split", "kid"),
        split_rows,
    )

    controlled_rows: List[dict] = []
    for slug in METHOD_ORDER:
        brisque = matched_summary["brisque"][slug]
        kid = independent_kid[slug]
        controlled_rows.append(
            {
                "method": slug,
                "label": brisque["label"],
                "n": brisque["n"],
                "brisque_mean": f"{brisque['mean']:.6f}",
                "brisque_mean_ci95_low": f"{brisque['mean_ci95'][0]:.6f}",
                "brisque_mean_ci95_high": f"{brisque['mean_ci95'][1]:.6f}",
                "brisque_p95": f"{brisque['p95']:.6f}",
                "brisque_p95_ci95_low": f"{brisque['p95_ci95'][0]:.6f}",
                "brisque_p95_ci95_high": f"{brisque['p95_ci95'][1]:.6f}",
                "kid_mean": f"{kid['mean']:.12f}",
                "kid_std": f"{kid['std']:.12f}",
                "kid_x1000_mean": _format_kid(kid["mean"]),
                "kid_x1000_std": _format_kid(kid["std"]),
                "source": "matched_rebuttal_n100",
            }
        )
    controlled_fields = tuple(controlled_rows[0])
    _write_csv(matched_dir / "controlled_quality_table.csv", controlled_fields, controlled_rows)

    submitted_anchor_rows = [
        {
            "method": "pp_mark",
            "label": "PP-Mark (submitted)",
            "brisque_n": SUBMITTED_PP_MARK["brisque_n"],
            "brisque_design": SUBMITTED_PP_MARK["brisque_design"],
            "brisque_mean": f"{SUBMITTED_PP_MARK['brisque_mean']:.6f}",
            "brisque_p95": f"{SUBMITTED_PP_MARK['brisque_p95']:.6f}",
            "kid_n": SUBMITTED_PP_MARK["kid_n"],
            "kid_subset_size": SUBMITTED_PP_MARK["kid_subset_size"],
            "kid_splits": SUBMITTED_PP_MARK["kid_splits"],
            "kid_mean": f"{SUBMITTED_PP_MARK['kid_mean']:.12f}",
            "kid_std": f"{SUBMITTED_PP_MARK['kid_std']:.12f}",
            "kid_x1000_mean": _format_kid(SUBMITTED_PP_MARK["kid_mean"]),
            "kid_x1000_std": _format_kid(SUBMITTED_PP_MARK["kid_std"]),
            "source": "submitted_manuscript_and_saved_evaluation",
        }
    ]
    _write_csv(
        matched_dir / "submitted_ppmark_quality_anchor.csv",
        tuple(submitted_anchor_rows[0]),
        submitted_anchor_rows,
    )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Matched quality comparison on 100 common prompt--seed pairs. "
        r"BRISQUE is lower-is-better; KID is reported as $10^3\times$ mean$\pm$"
        r"reference-resampling standard deviation against 900 disjoint clean images.}",
        r"\label{tab:rebuttal-quality-matched}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & BRISQUE mean / p95 $\downarrow$ & KID $\times 10^3$ $\downarrow$ \\",
        r"\midrule",
    ]
    for row in controlled_rows:
        lines.append(
            f"{_latex_escape(row['label'])} & "
            f"{float(row['brisque_mean']):.2f} / {float(row['brisque_p95']):.2f} & "
            f"{row['kid_x1000_mean']} $\\pm$ {row['kid_x1000_std']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\begin{minipage}{0.98\linewidth}\footnotesize",
            r"Unbiased KID may be slightly negative at finite $n$; near-zero negative "
            r"estimates are not interpreted as better-than-zero distances. "
            r"The submitted PP-Mark results remain BRISQUE mean/p95 "
            r"$9.92/34.14$ ($n=100$) and KID "
            r"$1.101\pm0.202\times10^{-3}$ ($n=1000$, subset 500, 20 splits).",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    (matched_dir / "controlled_quality_table.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    anchored_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Image-quality comparison. PP-Mark retains the submitted values; "
        r"baseline metrics are newly measured on the common 100-image evaluation pool.}",
        r"\label{tab:rebuttal-quality-submitted-anchor}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & BRISQUE mean / p95 $\downarrow$ & KID $\times 10^3$ $\downarrow$ \\",
        r"\midrule",
        (
            r"PP-Mark (submitted)$^\dagger$ & "
            f"{SUBMITTED_PP_MARK['brisque_mean']:.2f} / "
            f"{SUBMITTED_PP_MARK['brisque_p95']:.2f} & "
            f"{_format_kid(SUBMITTED_PP_MARK['kid_mean'])} "
            f"$\\pm$ {_format_kid(SUBMITTED_PP_MARK['kid_std'])} \\\\"
        ),
    ]
    for row in controlled_rows:
        if row["method"] == "pp_mark":
            continue
        anchored_lines.append(
            f"{_latex_escape(row['label'])}$^\\ddagger$ & "
            f"{float(row['brisque_mean']):.2f} / {float(row['brisque_p95']):.2f} & "
            f"{row['kid_x1000_mean']} $\\pm$ {row['kid_x1000_std']} \\\\"
        )
    anchored_lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{2pt}",
            r"\begin{minipage}{0.98\linewidth}\footnotesize",
            r"$^\dagger$Submitted PP-Mark: BRISQUE $n=100$ (25 prompts, four seeds); "
            r"KID $n=1000$, subset 500, 20 splits. "
            r"$^\ddagger$New baselines: the same 100 prompt--seed pairs across methods; "
            r"KID against a disjoint 900-image clean pool with 1,000 reference resamples. "
            r"Unbiased KID can be slightly negative at finite $n$.",
            r"\end{minipage}",
            r"\end{table}",
        ]
    )
    (matched_dir / "submitted_anchored_quality_table.tex").write_text(
        "\n".join(anchored_lines) + "\n",
        encoding="utf-8",
    )

    markdown = [
        "# Rebuttal quality tables",
        "",
        "## Submitted PP-Mark anchor (not replaced)",
        "",
        "| Metric | Submitted value | Protocol |",
        "|---|---:|---|",
        (
            "| BRISQUE mean / p95 | "
            f"{SUBMITTED_PP_MARK['brisque_mean']:.2f} / {SUBMITTED_PP_MARK['brisque_p95']:.2f} | "
            f"n={SUBMITTED_PP_MARK['brisque_n']}, {SUBMITTED_PP_MARK['brisque_design']} |"
        ),
        (
            "| KID x 1,000 | "
            f"{_format_kid(SUBMITTED_PP_MARK['kid_mean'])} +/- "
            f"{_format_kid(SUBMITTED_PP_MARK['kid_std'])} | "
            f"n={SUBMITTED_PP_MARK['kid_n']}, subset="
            f"{SUBMITTED_PP_MARK['kid_subset_size']}, splits={SUBMITTED_PP_MARK['kid_splits']} |"
        ),
        "",
        "## Submitted-anchored comparison",
        "",
        "| Method | BRISQUE mean | BRISQUE p95 | KID x 1,000 (mean +/- SD) | Source |",
        "|---|---:|---:|---:|---|",
        (
            f"| PP-Mark | {SUBMITTED_PP_MARK['brisque_mean']:.2f} | "
            f"{SUBMITTED_PP_MARK['brisque_p95']:.2f} | "
            f"{_format_kid(SUBMITTED_PP_MARK['kid_mean'])} +/- "
            f"{_format_kid(SUBMITTED_PP_MARK['kid_std'])} | submitted |"
        ),
    ]
    for row in controlled_rows:
        if row["method"] == "pp_mark":
            continue
        markdown.append(
            f"| {row['label']} | {float(row['brisque_mean']):.2f} | "
            f"{float(row['brisque_p95']):.2f} | "
            f"{row['kid_x1000_mean']} +/- {row['kid_x1000_std']} | new baseline, n=100 |"
        )
    markdown.extend(
        [
            "",
            "The submitted PP-Mark row and new baseline rows preserve their actual "
            "evaluation protocols; they are not falsely described as one paired experiment.",
            "",
        ]
    )
    markdown.extend(
        [
        "## Controlled common-100 comparison",
        "",
        "| Method | BRISQUE mean | BRISQUE p95 | KID x 1,000 (mean +/- ref-resampling SD) |",
        "|---|---:|---:|---:|",
        ]
    )
    for row in controlled_rows:
        markdown.append(
            f"| {row['label']} | {float(row['brisque_mean']):.2f} | "
            f"{float(row['brisque_p95']):.2f} | "
            f"{row['kid_x1000_mean']} +/- {row['kid_x1000_std']} |"
        )
    markdown.extend(
        [
            "",
            "The controlled table uses 100 shared prompt/seed pairs for every method. "
            "KID uses the 900 clean images disjoint from those 100 pairs as its reference pool.",
            "",
            "Unbiased KID may be slightly negative at finite sample sizes. "
            "Values close to zero are treated as statistically indistinguishable from zero, "
            "not as better-than-zero distributional distance.",
        ]
    )
    (matched_dir / "rebuttal_quality_tables.md").write_text(
        "\n".join(markdown) + "\n",
        encoding="utf-8",
    )

    sample_size_assessment = f"""# Sample-size assessment

## Decision

No additional image generation is required for the rebuttal if the claim is
limited to reporting the missing baseline quality axis and interpreting the
quality--robustness trade-off descriptively.

## Why n=100 is adequate for this response

- It matches the submitted BRISQUE experiment size and the main attack
  evaluation size.
- Every method is represented by the same 100 prompt--seed pairs in the
  controlled comparison.
- BRISQUE confidence intervals expose the remaining content variance instead
  of hiding it.
- Independent-reference KID is reported with
  {args.kid_reference_splits:,} deterministic reference resamples. HiDDeN is
  clearly separated from zero; the remaining methods correctly appear as a
  near-zero group rather than a forced ranking.
- The reviewer requested matched baseline metrics, not a statistically
  significant proof that PP-Mark dominates every baseline in visual quality.

## What n=100 does not support

- Fine ranking among the near-zero KID methods.
- A precise comparison of BRISQUE p95 values: n=100 places only about five
  observations in the upper five-percent tail.
- A claim that PP-Mark is the best-quality method.

## When expansion would be warranted

Expand to at least n=500 per method, preferably n=1,000 for KID, only if the
rebuttal intends to claim statistically resolved quality superiority or a
precise tail-quality ranking. Reaching n=500 would require at least 400 new
outputs for each method currently limited to n=100, with no direct need for
the narrower claim made here.
"""
    (matched_dir / "sample_size_assessment.md").write_text(
        sample_size_assessment,
        encoding="utf-8",
    )

    print(json.dumps({"protocol": protocol, "methods": independent_kid}, indent=2))


if __name__ == "__main__":
    main()
