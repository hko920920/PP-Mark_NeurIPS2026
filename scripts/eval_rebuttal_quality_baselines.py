#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models import Inception_V3_Weights, inception_v3


Key = Tuple[int, int]


@dataclass(frozen=True)
class MethodSpec:
    slug: str
    label: str
    manifest: Path
    metadata_derived_keys: bool = False


METHODS = (
    MethodSpec(
        "clean",
        "Clean",
        Path("outputs/attacks/muller_forgery_report_prep/clean/eval/manifest.jsonl"),
    ),
    MethodSpec(
        "pp_mark",
        "PP-Mark",
        Path("outputs/attacks/muller_forgery_report_prep/methods/pp_mark/eval/manifest_fixed.jsonl"),
        metadata_derived_keys=True,
    ),
    MethodSpec(
        "gaussian_shading",
        "Gaussian Shading",
        Path("outputs/attacks/muller_forgery_report_prep/methods/gaussian_shading/eval/manifest.jsonl"),
    ),
    MethodSpec(
        "hidden",
        "HiDDeN",
        Path("outputs/attacks/muller_forgery_report_prep/methods/hidden_revisited/eval/manifest.jsonl"),
    ),
    MethodSpec(
        "invismark",
        "InvisMark",
        Path("outputs/attacks/muller_forgery_report_prep/methods/invismark/eval/manifest.jsonl"),
    ),
    MethodSpec(
        "ringid",
        "RingID",
        Path("outputs/attacks/muller_forgery_report_prep/methods/ringid/eval/manifest.jsonl"),
    ),
    MethodSpec(
        "stable_signature",
        "Stable Signature",
        Path("outputs/attacks/muller_forgery_report_prep/methods/stable_signature/eval/manifest.jsonl"),
    ),
    MethodSpec(
        "tree_ring",
        "Tree-Ring",
        Path("outputs/attacks/muller_forgery_report_prep/methods/tree_ring/eval/manifest.jsonl"),
    ),
    MethodSpec(
        "wind",
        "WIND",
        Path("outputs/rebuttal_quality_baselines_n100/wind_common100/manifest.jsonl"),
    ),
    MethodSpec(
        "trustmark",
        "TrustMark",
        Path("outputs/attacks/muller_forgery_report_prep/methods/trustmark_hidden/eval/manifest.jsonl"),
    ),
)


def load_method_specs(path: Path | None = None) -> Tuple[MethodSpec, ...]:
    """Read portable manifest locations; preserve historical defaults when omitted."""
    if path is None:
        return METHODS
    rows = json.loads(path.read_text(encoding="utf-8"))
    specs = []
    for row in rows:
        manifest = Path(row["manifest"])
        if not manifest.is_absolute():
            manifest = (path.parent / manifest).resolve()
        specs.append(MethodSpec(
            slug=str(row["slug"]), label=str(row.get("label", row["slug"])),
            manifest=manifest,
            metadata_derived_keys=bool(row.get("metadata_derived_keys", False)),
        ))
    slugs = [spec.slug for spec in specs]
    if "clean" not in slugs or len(slugs) != len(set(slugs)):
        raise ValueError("Method specifications need a clean reference and unique slugs")
    return tuple(specs)


def _read_manifest(
    spec: MethodSpec,
    clean_lookup: Dict[Tuple[str, int], Key] | None = None,
) -> Dict[Key, dict]:
    path = spec.manifest
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    rows: Dict[Key, dict] = {}

    if spec.metadata_derived_keys:
        if clean_lookup is None:
            raise RuntimeError(f"Clean lookup is required for {spec.slug}")
        metadata_files = sorted((path.parent / "images").glob("img_*/metadata.json"))
        if not metadata_files:
            raise RuntimeError(f"No per-image metadata found next to {path}")
        for metadata_path in metadata_files:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            prompt_text = str(metadata["prompt"])
            seed = int(str(metadata["seed_hex"]).removeprefix("0x"), 16)
            try:
                key = clean_lookup[(prompt_text, seed)]
            except KeyError as exc:
                raise RuntimeError(
                    f"Metadata key is absent from the clean reference at {metadata_path}"
                ) from exc
            if key in rows:
                raise RuntimeError(f"Duplicate metadata-derived key {key} at {metadata_path}")
            image_path = metadata_path.parent / "watermarked.png"
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image for {key}: {image_path}")
            rows[key] = {
                "prompt_id": key[0],
                "seed": key[1],
                "prompt_text": prompt_text,
                "image_path": image_path,
                "meta": metadata,
            }
        return rows

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            metadata = row.get("meta", {})
            key = (int(row["prompt_id"]), int(row["seed"]))
            prompt_text = str(row["prompt_text"])
            if key in rows:
                raise RuntimeError(f"Duplicate key {key} in {path}:{line_number}")
            image_path = path.parent / str(row["image_path"])
            if not image_path.exists():
                raise FileNotFoundError(f"Missing image for {key}: {image_path}")
            rows[key] = {
                "prompt_id": key[0],
                "seed": key[1],
                "prompt_text": prompt_text,
                "image_path": image_path,
                "meta": metadata,
            }
    if not rows:
        raise RuntimeError(f"Empty manifest: {path}")
    return rows


def _prepare_matched_rows(
    specs: Sequence[MethodSpec],
    expected_count: int,
) -> Tuple[List[Key], Dict[str, Dict[Key, dict]]]:
    clean_spec = next(spec for spec in specs if spec.slug == "clean")
    clean_rows = _read_manifest(clean_spec)
    clean_lookup = {
        (str(row["prompt_text"]), key[1]): key
        for key, row in clean_rows.items()
    }
    manifests = {"clean": clean_rows}
    for spec in specs:
        if spec.slug == "clean":
            continue
        manifests[spec.slug] = _read_manifest(spec, clean_lookup=clean_lookup)

    key_sets = [set(rows) for rows in manifests.values()]
    common = sorted(set.intersection(*key_sets))
    if len(common) != expected_count:
        counts = {slug: len(rows) for slug, rows in manifests.items()}
        raise RuntimeError(
            f"Expected {expected_count} common prompt/seed pairs, found {len(common)}; "
            f"manifest counts={counts}"
        )

    for key in common:
        expected_prompt = clean_rows[key]["prompt_text"]
        for spec in specs:
            row = manifests[spec.slug][key]
            if row["prompt_text"] != expected_prompt:
                raise RuntimeError(f"Prompt mismatch for {spec.slug} at key {key}")

    return common, manifests


def _write_matched_manifest(
    output: Path,
    specs: Sequence[MethodSpec],
    common: Sequence[Key],
    manifests: Dict[str, Dict[Key, dict]],
) -> None:
    with output.open("w", encoding="utf-8") as handle:
        for key in common:
            row = {
                "prompt_id": key[0],
                "seed": key[1],
                "prompt_text": manifests["clean"][key]["prompt_text"],
                "images": {
                    spec.slug: str(manifests[spec.slug][key]["image_path"].resolve())
                    for spec in specs
                },
            }
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _batch_iter(items: Sequence[Path], batch_size: int) -> Iterable[Sequence[Path]]:
    for idx in range(0, len(items), batch_size):
        yield items[idx : idx + batch_size]


def _load_unit_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _bootstrap_interval(
    values: np.ndarray,
    statistic: str,
    draws: int,
    seed: int,
) -> Tuple[float, float]:
    if statistic == "mean":
        reducer = np.mean
    elif statistic == "p95":
        reducer = lambda x, axis: np.quantile(x, 0.95, axis=axis)
    else:
        raise ValueError(f"Unsupported bootstrap statistic: {statistic}")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    samples = values[indices]
    estimates = reducer(samples, axis=1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _compute_brisque(
    specs: Sequence[MethodSpec],
    common: Sequence[Key],
    manifests: Dict[str, Dict[Key, dict]],
    output_dir: Path,
    bootstrap_draws: int,
    seed: int,
) -> Dict[str, dict]:
    import pyiqa

    metric = pyiqa.create_metric("brisque", device="cpu")
    per_image_path = output_dir / "brisque_per_image.csv"
    values_by_method: Dict[str, List[float]] = {spec.slug: [] for spec in specs}

    with per_image_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "label", "prompt_id", "seed", "brisque", "image_path"])
        total = len(specs) * len(common)
        completed = 0
        for spec in specs:
            for key in common:
                image_path = manifests[spec.slug][key]["image_path"]
                tensor = _load_unit_tensor(image_path)
                with torch.no_grad():
                    value = float(metric(tensor).item())
                values_by_method[spec.slug].append(value)
                writer.writerow(
                    [spec.slug, spec.label, key[0], key[1], f"{value:.9f}", str(image_path.resolve())]
                )
                completed += 1
                if completed % 100 == 0 or completed == total:
                    print(f"[BRISQUE] {completed}/{total}", flush=True)

    clean = np.asarray(values_by_method["clean"], dtype=np.float64)
    summary: Dict[str, dict] = {}
    for method_index, spec in enumerate(specs):
        values = np.asarray(values_by_method[spec.slug], dtype=np.float64)
        mean_ci = _bootstrap_interval(values, "mean", bootstrap_draws, seed + method_index)
        p95_ci = _bootstrap_interval(values, "p95", bootstrap_draws, seed + 100 + method_index)
        delta = values - clean
        delta_ci = _bootstrap_interval(delta, "mean", bootstrap_draws, seed + 200 + method_index)
        summary[spec.slug] = {
            "label": spec.label,
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.90)),
            "p95": float(np.quantile(values, 0.95)),
            "mean_ci95": list(mean_ci),
            "p95_ci95": list(p95_ci),
            "paired_mean_delta_vs_clean": float(np.mean(delta)),
            "paired_mean_delta_ci95": list(delta_ci),
        }
    return summary


def _build_inception(device: str):
    weights = Inception_V3_Weights.DEFAULT
    transform = weights.transforms()
    model = inception_v3(weights=weights, aux_logits=True, transform_input=False)
    model.fc = torch.nn.Identity()
    model.to(device)
    model.eval()
    return model, transform


def _extract_inception_features(
    model,
    transform,
    image_paths: Sequence[Path],
    device: str,
    batch_size: int,
) -> np.ndarray:
    features: List[np.ndarray] = []
    with torch.no_grad():
        for batch in _batch_iter(image_paths, batch_size):
            tensors = [transform(Image.open(path).convert("RGB")) for path in batch]
            output = model(torch.stack(tensors, dim=0).to(device))
            features.append(output.detach().cpu().numpy())
    return np.concatenate(features, axis=0)


def _image_set_fingerprint(image_paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in image_paths:
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _polynomial_mmd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dim = x.shape[1]
    k_xx = ((x @ x.T) / dim + 1.0) ** 3
    k_yy = ((y @ y.T) / dim + 1.0) ** 3
    k_xy = ((x @ y.T) / dim + 1.0) ** 3
    m = x.shape[0]
    n = y.shape[0]
    sum_xx = (k_xx.sum() - k_xx.diag().sum()) / (m * (m - 1))
    sum_yy = (k_yy.sum() - k_yy.diag().sum()) / (n * (n - 1))
    return sum_xx + sum_yy - 2.0 * k_xy.mean()


def _compute_kid_scores(
    features: np.ndarray,
    clean_features: np.ndarray,
    subset_size: int,
    splits: int,
    seed: int,
    device: str,
) -> np.ndarray:
    subset = min(subset_size, len(features), len(clean_features))
    if subset < 2:
        raise RuntimeError("At least two samples are required for KID")

    rng = np.random.default_rng(seed)
    x_all = torch.from_numpy(features).to(device=device, dtype=torch.float64)
    y_all = torch.from_numpy(clean_features).to(device=device, dtype=torch.float64)
    scores: List[float] = []
    for _ in range(splits):
        x_indices = rng.choice(len(features), size=subset, replace=False)
        y_indices = rng.choice(len(clean_features), size=subset, replace=False)
        scores.append(float(_polynomial_mmd(x_all[x_indices], y_all[y_indices]).cpu()))
    return np.asarray(scores, dtype=np.float64)


def _compute_kid(
    specs: Sequence[MethodSpec],
    common: Sequence[Key],
    manifests: Dict[str, Dict[Key, dict]],
    output_dir: Path,
    device: str,
    batch_size: int,
    subset_size: int,
    splits: int,
    seed: int,
) -> Dict[str, dict]:
    feature_dir = output_dir / "inception_features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    model, transform = _build_inception(device)
    features_by_method: Dict[str, np.ndarray] = {}

    for index, spec in enumerate(specs, 1):
        cache_path = feature_dir / f"{spec.slug}.npy"
        cache_meta_path = feature_dir / f"{spec.slug}.json"
        image_paths = [manifests[spec.slug][key]["image_path"] for key in common]
        fingerprint = _image_set_fingerprint(image_paths)
        cache_valid = False
        if cache_path.exists() and cache_meta_path.exists():
            cache_meta = json.loads(cache_meta_path.read_text(encoding="utf-8"))
            cache_valid = (
                cache_meta.get("fingerprint") == fingerprint
                and int(cache_meta.get("count", -1)) == len(common)
            )
        if cache_valid:
            features = np.load(cache_path)
            if features.shape[0] != len(common):
                raise RuntimeError(f"Invalid feature cache count for {spec.slug}: {features.shape}")
        else:
            features = _extract_inception_features(model, transform, image_paths, device, batch_size)
            np.save(cache_path, features)
            cache_meta_path.write_text(
                json.dumps(
                    {
                        "count": len(common),
                        "fingerprint": fingerprint,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        features_by_method[spec.slug] = features
        print(f"[KID features] {index}/{len(specs)} {spec.label}: {features.shape}", flush=True)

    clean = features_by_method["clean"]
    summary: Dict[str, dict] = {
        "clean": {
            "label": "Clean",
            "n": int(len(clean)),
            "mean": 0.0,
            "std": 0.0,
            "subset_size": subset_size,
            "splits": splits,
            "reference": True,
        }
    }
    split_rows: List[List[str]] = []
    for spec in specs:
        if spec.slug == "clean":
            continue
        scores = _compute_kid_scores(
            features_by_method[spec.slug],
            clean,
            subset_size=subset_size,
            splits=splits,
            seed=seed,
            device=device,
        )
        summary[spec.slug] = {
            "label": spec.label,
            "n": int(len(features_by_method[spec.slug])),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=1)),
            "median": float(np.median(scores)),
            "p025": float(np.quantile(scores, 0.025)),
            "p975": float(np.quantile(scores, 0.975)),
            "subset_size": subset_size,
            "splits": splits,
            "reference": False,
        }
        split_rows.extend(
            [[spec.slug, spec.label, str(index), f"{score:.12f}"] for index, score in enumerate(scores)]
        )

    with (output_dir / "kid_split_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "label", "split", "kid"])
        writer.writerows(split_rows)
    return summary


def _write_summary_csv(
    output: Path,
    specs: Sequence[MethodSpec],
    brisque: Dict[str, dict],
    kid: Dict[str, dict],
) -> None:
    fields = [
        "method",
        "label",
        "n",
        "brisque_mean",
        "brisque_std",
        "brisque_p95",
        "brisque_mean_ci95_low",
        "brisque_mean_ci95_high",
        "brisque_p95_ci95_low",
        "brisque_p95_ci95_high",
        "brisque_paired_mean_delta_vs_clean",
        "brisque_paired_delta_ci95_low",
        "brisque_paired_delta_ci95_high",
        "kid_mean",
        "kid_std",
        "kid_subset_size",
        "kid_splits",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for spec in specs:
            b = brisque[spec.slug]
            k = kid[spec.slug]
            writer.writerow(
                {
                    "method": spec.slug,
                    "label": spec.label,
                    "n": b["n"],
                    "brisque_mean": f"{b['mean']:.9f}",
                    "brisque_std": f"{b['std']:.9f}",
                    "brisque_p95": f"{b['p95']:.9f}",
                    "brisque_mean_ci95_low": f"{b['mean_ci95'][0]:.9f}",
                    "brisque_mean_ci95_high": f"{b['mean_ci95'][1]:.9f}",
                    "brisque_p95_ci95_low": f"{b['p95_ci95'][0]:.9f}",
                    "brisque_p95_ci95_high": f"{b['p95_ci95'][1]:.9f}",
                    "brisque_paired_mean_delta_vs_clean": f"{b['paired_mean_delta_vs_clean']:.9f}",
                    "brisque_paired_delta_ci95_low": f"{b['paired_mean_delta_ci95'][0]:.9f}",
                    "brisque_paired_delta_ci95_high": f"{b['paired_mean_delta_ci95'][1]:.9f}",
                    "kid_mean": f"{k['mean']:.12f}",
                    "kid_std": f"{k['std']:.12f}",
                    "kid_subset_size": k["subset_size"],
                    "kid_splits": k["splits"],
                }
            )


def _environment_payload(device: str) -> dict:
    import pyiqa

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "pyiqa": pyiqa.__version__,
        "requested_device": device,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched BRISQUE and KID evaluation for the NeurIPS 2026 PP-Mark rebuttal."
    )
    parser.add_argument("--output", default="outputs/rebuttal_quality_baselines_n100")
    parser.add_argument("--methods-file", type=Path, help="JSON method list with paths relative to that file")
    parser.add_argument("--expected-count", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--kid-subset-size", type=int, default=50)
    parser.add_argument("--kid-splits", type=int, default=100)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--reuse-brisque",
        action="store_true",
        help="Reuse a complete brisque_summary.json in the output directory.",
    )
    args = parser.parse_args()
    methods = load_method_specs(args.methods_file)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    common, manifests = _prepare_matched_rows(methods, args.expected_count)
    _write_matched_manifest(output_dir / "matched_manifest.jsonl", methods, common, manifests)

    environment = _environment_payload(device)
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )
    protocol = {
        "common_pairs": len(common),
        "pair_key": ["prompt_id", "seed"],
        "methods": [
            {
                "slug": spec.slug,
                "label": spec.label,
                "manifest": str(spec.manifest),
                "metadata_derived_keys": spec.metadata_derived_keys,
            }
            for spec in methods
        ],
        "brisque": {
            "implementation": "pyiqa",
            "device": "cpu",
            "input_range": "[0,1]",
            "bootstrap_draws": args.bootstrap_draws,
        },
        "kid": {
            "features": "torchvision Inception_V3_Weights.DEFAULT, fc=Identity",
            "kernel": "(x^T y / d + 1)^3",
            "reference": f"same clean common-{len(common)} set",
            "subset_size": args.kid_subset_size,
            "splits": args.kid_splits,
            "device": device,
        },
        "seed": args.seed,
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    brisque_summary_path = output_dir / "brisque_summary.json"
    if args.reuse_brisque:
        if not brisque_summary_path.exists():
            raise FileNotFoundError(f"Cannot reuse missing summary: {brisque_summary_path}")
        brisque = json.loads(brisque_summary_path.read_text(encoding="utf-8"))
        for spec in methods:
            if spec.slug not in brisque or int(brisque[spec.slug]["n"]) != len(common):
                raise RuntimeError(f"Incomplete reusable BRISQUE summary for {spec.slug}")
        print(f"[BRISQUE] reused {brisque_summary_path}", flush=True)
    else:
        brisque = _compute_brisque(
            methods,
            common,
            manifests,
            output_dir,
            bootstrap_draws=args.bootstrap_draws,
            seed=args.seed,
        )
        brisque_summary_path.write_text(
            json.dumps(brisque, indent=2),
            encoding="utf-8",
        )

    kid = _compute_kid(
        methods,
        common,
        manifests,
        output_dir,
        device=device,
        batch_size=args.batch_size,
        subset_size=args.kid_subset_size,
        splits=args.kid_splits,
        seed=args.seed,
    )
    (output_dir / "kid_summary.json").write_text(json.dumps(kid, indent=2), encoding="utf-8")

    _write_summary_csv(output_dir / "summary.csv", methods, brisque, kid)
    payload = {
        "protocol": protocol,
        "environment": environment,
        "brisque": brisque,
        "kid": kid,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[done] {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
