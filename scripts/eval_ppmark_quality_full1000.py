#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval_rebuttal_quality_baselines import (
    _compute_brisque,
    _compute_kid,
    _environment_payload,
    _prepare_matched_rows,
    _write_matched_manifest,
    load_method_specs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate PP-Mark quality on all metadata-aligned SD2.1 image pairs."
    )
    parser.add_argument(
        "--output",
        default="outputs/rebuttal_quality_ppmark_full1000_corrected",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--methods-file", type=Path, required=True, help="JSON method list with portable manifest paths")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260725)
    args = parser.parse_args()

    specs = tuple(spec for spec in load_method_specs(args.methods_file) if spec.slug in {"clean", "pp_mark"})
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    common, manifests = _prepare_matched_rows(specs, expected_count=1000)
    _write_matched_manifest(output_dir / "matched_manifest.jsonl", specs, common, manifests)

    brisque = _compute_brisque(
        specs,
        common,
        manifests,
        output_dir,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )
    (output_dir / "brisque_summary.json").write_text(
        json.dumps(brisque, indent=2),
        encoding="utf-8",
    )

    kid = _compute_kid(
        specs,
        common,
        manifests,
        output_dir,
        device=args.device,
        batch_size=args.batch_size,
        subset_size=500,
        splits=20,
        seed=2025,
    )
    (output_dir / "kid_summary.json").write_text(
        json.dumps(kid, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "protocol": {
                    "count": len(common),
                    "pair_key_source": "PP-Mark per-image metadata matched to clean manifest",
                    "kid_subset_size": 500,
                    "kid_splits": 20,
                },
                "environment": _environment_payload(args.device),
                "brisque": brisque,
                "kid": kid,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"brisque": brisque, "kid": kid}, indent=2))


if __name__ == "__main__":
    main()
