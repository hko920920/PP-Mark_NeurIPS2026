#!/usr/bin/env python3
"""Run secure Bare-Sig and Context-Sig controls for the PP-Mark rebuttal."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppmark_v03.signature_controls import (  # noqa: E402
    load_clean_images,
    load_positive_cases,
    run_control_experiment,
    write_report,
)


def _paths(values: list[str]) -> list[Path]:
    return [Path(value).resolve() for value in values]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate secure signatures over PP-Mark's canonical image hash, "
            "with and without public context fields."
        )
    )
    parser.add_argument("--positive-manifest", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--positive-image-root", action="append", default=[])
    parser.add_argument("--positive-metadata-root", action="append", default=[])
    parser.add_argument("--clean-image-root", action="append", default=[])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")

    positive_manifest = args.positive_manifest.resolve()
    clean_manifest = args.clean_manifest.resolve()
    positive_cases = load_positive_cases(
        positive_manifest,
        image_roots=_paths(args.positive_image_root),
        metadata_roots=_paths(args.positive_metadata_root),
        limit=args.limit,
    )
    clean_images = load_clean_images(
        clean_manifest,
        image_roots=_paths(args.clean_image_root),
        limit=args.limit,
    )
    report = run_control_experiment(positive_cases, clean_images)
    write_report(report, args.output.resolve())

    summary = report["summary"]
    print(f"[signature-controls] n={summary['n']}")
    print(
        "[signature-controls] "
        f"bare signed-clean={summary['bare_sig']['signed_clean']}/{summary['n']}; "
        f"context signed-clean={summary['context_sig']['signed_clean']}/{summary['n']}"
    )
    print(
        "[signature-controls] "
        f"all_expected_checks_pass={report['all_expected_checks_pass']}"
    )
    print(f"[signature-controls] report={args.output.resolve()}")
    return 0 if report["all_expected_checks_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
