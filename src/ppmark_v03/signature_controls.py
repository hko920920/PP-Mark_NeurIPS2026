"""Secure image-signature controls for the PP-Mark rebuttal experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

BARE_DOMAIN = b"PPMARK_BARE_SIG_V1\x00"
CONTEXT_DOMAIN = b"PPMARK_CONTEXT_SIG_V1\x00"

# This must remain byte-for-byte compatible with ppmark_v03.cli.HASH_SPEC.
HASH_SPEC = {
    "version": "norm_v1",
    "color_space": "sRGB",
    "resize": [512, 512],
    "interpolation": "bicubic",
    "rounding": "round",
    "bytes": "uint8_rgb",
    "exif_transpose": True,
}


@dataclass(frozen=True, slots=True)
class PositiveCase:
    case_id: str
    image_path: Path
    metadata_path: Path
    image_hash: bytes
    ctx_hash: bytes
    binding: bytes


def _require_32(value: bytes, name: str) -> bytes:
    if len(value) != 32:
        raise ValueError(f"{name} must be exactly 32 bytes; got {len(value)}")
    return value


def _parse_hex32(
    value: Any,
    name: str,
    *,
    allow_left_padding: bool = False,
) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a hexadecimal string")
    text = value[2:] if value.startswith("0x") else value
    if len(text) % 2:
        text = "0" + text
    try:
        parsed = bytes.fromhex(text)
    except ValueError as exc:
        raise ValueError(f"{name} is not valid hexadecimal") from exc
    if allow_left_padding:
        if len(parsed) > 32:
            raise ValueError(f"{name} exceeds 32 bytes")
        parsed = parsed.rjust(32, b"\x00")
    return _require_32(parsed, name)


def bare_message(image_hash: bytes) -> bytes:
    """Domain-separated message for the reviewer-requested image-hash signature."""
    return BARE_DOMAIN + _require_32(image_hash, "image_hash")


def context_message(image_hash: bytes, ctx_hash: bytes, binding: bytes) -> bytes:
    """Image-hash signature strengthened with PP-Mark public context fields."""
    return (
        CONTEXT_DOMAIN
        + _require_32(image_hash, "image_hash")
        + _require_32(ctx_hash, "ctx_hash")
        + _require_32(binding, "binding")
    )


def sign(private_key: Ed25519PrivateKey, message: bytes) -> bytes:
    return private_key.sign(message)


def verifies(public_key: Ed25519PublicKey, signature: bytes, message: bytes) -> bool:
    try:
        public_key.verify(signature, message)
    except InvalidSignature:
        return False
    return True


def public_key_raw(public_key: Ed25519PublicKey) -> bytes:
    return public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def _bicubic_resample() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.BICUBIC
    return Image.BICUBIC


def normalize_image_for_hash(image: Image.Image) -> np.ndarray:
    """Apply the submitted PP-Mark norm_v1 canonicalization."""
    normalized = ImageOps.exif_transpose(image)
    normalized = normalized.convert("RGB")
    normalized = normalized.resize(
        (int(HASH_SPEC["resize"][0]), int(HASH_SPEC["resize"][1])),
        resample=_bicubic_resample(),
    )
    array = np.asarray(normalized, dtype=np.float32)
    return np.clip(np.round(array), 0, 255).astype(np.uint8)


def canonical_image_hash(image: Image.Image) -> bytes:
    return hashlib.sha256(normalize_image_for_hash(image).tobytes()).digest()


def canonical_image_hash_path(path: Path) -> bytes:
    with Image.open(path) as image:
        return canonical_image_hash(image)


def jpeg_variant(image: Image.Image, quality: int = 25) -> Image.Image:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    variant = Image.open(buffer).convert("RGB")
    variant.load()
    return variant


def crop_resize_variant(image: Image.Image, fraction: float = 0.75) -> Image.Image:
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie strictly between zero and one")
    rgb = image.convert("RGB")
    width, height = rgb.size
    crop_width = max(1, int(round(width * fraction)))
    crop_height = max(1, int(round(height * fraction)))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = rgb.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), resample=_bicubic_resample())


def _load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _first_string(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    raise ValueError(f"none of the required path fields are present: {tuple(keys)}")


def _resolve_existing(raw_path: str, manifest_dir: Path, roots: Sequence[Path]) -> Path:
    supplied = Path(raw_path)
    candidates = [supplied] if supplied.is_absolute() else [manifest_dir / supplied]
    if not supplied.is_absolute():
        candidates.extend(root / supplied for root in roots)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"could not resolve {raw_path!r}; tried: {rendered}")


def load_positive_cases(
    manifest_path: Path,
    *,
    image_roots: Sequence[Path] = (),
    metadata_roots: Sequence[Path] = (),
    limit: int | None = None,
) -> list[PositiveCase]:
    rows = _load_jsonl(manifest_path, limit)
    cases: list[PositiveCase] = []
    for index, row in enumerate(rows):
        image_raw = _first_string(row, ("image_path", "image_relpath"))
        metadata_raw = _first_string(row, ("metadata_path", "metadata_relpath"))
        image_path = _resolve_existing(image_raw, manifest_path.parent, image_roots)
        metadata_path = _resolve_existing(metadata_raw, manifest_path.parent, metadata_roots)

        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        if not isinstance(metadata.get("opening"), dict):
            opening_variant = metadata_path.with_name("metadata_opening.json")
            if opening_variant.is_file():
                metadata_path = opening_variant.resolve()
                with metadata_path.open("r", encoding="utf-8") as handle:
                    metadata = json.load(handle)
        if metadata.get("hash_spec") != HASH_SPEC:
            raise ValueError(f"{metadata_path}: hash_spec does not match submitted norm_v1")

        opening = metadata.get("opening")
        if isinstance(opening, dict):
            hash_fields = opening
            hash_field_name = "opening.image_hash"
        else:
            hash_fields = metadata.get("attestation_statement")
            hash_field_name = "attestation_statement.image_hash"
            if not isinstance(hash_fields, dict):
                raise ValueError(f"{metadata_path}: image-bound public metadata is missing")
        recorded_hash = _parse_hex32(hash_fields.get("image_hash"), hash_field_name)
        observed_hash = canonical_image_hash_path(image_path)
        if observed_hash != recorded_hash:
            raise ValueError(f"{image_path}: canonical image hash does not match metadata")

        case_id = str(row.get("idx", row.get("prompt_id", index)))
        cases.append(
            PositiveCase(
                case_id=case_id,
                image_path=image_path,
                metadata_path=metadata_path,
                image_hash=observed_hash,
                ctx_hash=_parse_hex32(
                    metadata.get("ctx_hash"),
                    "ctx_hash",
                    allow_left_padding=True,
                ),
                binding=_parse_hex32(
                    metadata.get("binding"),
                    "binding",
                    allow_left_padding=True,
                ),
            )
        )
    if not cases:
        raise ValueError(f"{manifest_path}: no positive cases were loaded")
    return cases


def load_clean_images(
    manifest_path: Path,
    *,
    image_roots: Sequence[Path] = (),
    limit: int | None = None,
) -> list[Path]:
    rows = _load_jsonl(manifest_path, limit)
    images: list[Path] = []
    for row in rows:
        raw_path = _first_string(row, ("image_path", "image_relpath"))
        images.append(_resolve_existing(raw_path, manifest_path.parent, image_roots))
    if not images:
        raise ValueError(f"{manifest_path}: no clean images were loaded")
    return images


def _flip_first_bit(value: bytes) -> bytes:
    return bytes([value[0] ^ 0x01]) + value[1:]


def _count_true(rows: Sequence[dict[str, Any]], path: tuple[str, ...]) -> int:
    count = 0
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        count += int(bool(value))
    return count


def run_control_experiment(
    positive_cases: Sequence[PositiveCase],
    clean_images: Sequence[Path],
) -> dict[str, Any]:
    """Run the minimal reviewer-requested functional comparison."""
    if not positive_cases:
        raise ValueError("positive_cases must not be empty")
    if not clean_images:
        raise ValueError("clean_images must not be empty")

    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    public_raw = public_key_raw(public_key)

    clean_hashes = [canonical_image_hash_path(path) for path in clean_images]
    rows: list[dict[str, Any]] = []

    for index, case in enumerate(positive_cases):
        if len(positive_cases) > 1:
            other = positive_cases[(index + 1) % len(positive_cases)]
            other_hash = other.image_hash
            swapped_ctx_hash = other.ctx_hash
            swapped_binding = other.binding
        else:
            other_hash = clean_hashes[index % len(clean_hashes)]
            swapped_ctx_hash = _flip_first_bit(case.ctx_hash)
            swapped_binding = _flip_first_bit(case.binding)

        clean_hash = clean_hashes[index % len(clean_hashes)]
        bare_signature = sign(private_key, bare_message(case.image_hash))
        context_signature = sign(
            private_key,
            context_message(case.image_hash, case.ctx_hash, case.binding),
        )

        clean_bare_signature = sign(private_key, bare_message(clean_hash))
        clean_context_signature = sign(
            private_key,
            context_message(clean_hash, case.ctx_hash, case.binding),
        )

        with Image.open(case.image_path) as source:
            source.load()
            jpeg_hash = canonical_image_hash(jpeg_variant(source, quality=25))
            crop_hash = canonical_image_hash(crop_resize_variant(source, fraction=0.75))

        row = {
            "case_id": case.case_id,
            "canonical_hash_checks": {
                "cross_image_changed": other_hash != case.image_hash,
                "jpeg_q25_changed": jpeg_hash != case.image_hash,
                "crop_resize_75_changed": crop_hash != case.image_hash,
                "clean_image_changed": clean_hash != case.image_hash,
            },
            "bare_sig": {
                "original": verifies(
                    public_key, bare_signature, bare_message(case.image_hash)
                ),
                "cross_image_replay": verifies(
                    public_key, bare_signature, bare_message(other_hash)
                ),
                "context_swap": verifies(
                    public_key, bare_signature, bare_message(case.image_hash)
                ),
                "signed_clean": verifies(
                    public_key, clean_bare_signature, bare_message(clean_hash)
                ),
                "jpeg_q25": verifies(
                    public_key, bare_signature, bare_message(jpeg_hash)
                ),
                "crop_resize_75": verifies(
                    public_key, bare_signature, bare_message(crop_hash)
                ),
            },
            "context_sig": {
                "original": verifies(
                    public_key,
                    context_signature,
                    context_message(case.image_hash, case.ctx_hash, case.binding),
                ),
                "cross_image_replay": verifies(
                    public_key,
                    context_signature,
                    context_message(other_hash, case.ctx_hash, case.binding),
                ),
                "context_swap": verifies(
                    public_key,
                    context_signature,
                    context_message(
                        case.image_hash, swapped_ctx_hash, swapped_binding
                    ),
                ),
                "signed_clean": verifies(
                    public_key,
                    clean_context_signature,
                    context_message(clean_hash, case.ctx_hash, case.binding),
                ),
                "jpeg_q25": verifies(
                    public_key,
                    context_signature,
                    context_message(jpeg_hash, case.ctx_hash, case.binding),
                ),
                "crop_resize_75": verifies(
                    public_key,
                    context_signature,
                    context_message(crop_hash, case.ctx_hash, case.binding),
                ),
            },
        }
        rows.append(row)

    total = len(rows)
    summary = {
        "n": total,
        "bare_sig": {
            key: _count_true(rows, ("bare_sig", key))
            for key in rows[0]["bare_sig"]
        },
        "context_sig": {
            key: _count_true(rows, ("context_sig", key))
            for key in rows[0]["context_sig"]
        },
        "canonical_hash_checks": {
            key: _count_true(rows, ("canonical_hash_checks", key))
            for key in rows[0]["canonical_hash_checks"]
        },
    }
    expected = {
        "all_originals_verify": (
            summary["bare_sig"]["original"] == total
            and summary["context_sig"]["original"] == total
        ),
        "all_cross_image_replays_rejected": (
            summary["bare_sig"]["cross_image_replay"] == 0
            and summary["context_sig"]["cross_image_replay"] == 0
        ),
        "bare_does_not_bind_context": summary["bare_sig"]["context_swap"] == total,
        "context_sig_rejects_swaps": summary["context_sig"]["context_swap"] == 0,
        "both_accept_intentionally_signed_clean_images": (
            summary["bare_sig"]["signed_clean"] == total
            and summary["context_sig"]["signed_clean"] == total
        ),
        "both_reject_canonical_transform_changes": (
            summary["bare_sig"]["jpeg_q25"] == 0
            and summary["context_sig"]["jpeg_q25"] == 0
            and summary["bare_sig"]["crop_resize_75"] == 0
            and summary["context_sig"]["crop_resize_75"] == 0
        ),
        "all_tested_images_changed_as_expected": all(
            value == total for value in summary["canonical_hash_checks"].values()
        ),
    }

    return {
        "protocol": {
            "signature": "Ed25519",
            "bare_domain": BARE_DOMAIN.decode("ascii", errors="replace"),
            "context_domain": CONTEXT_DOMAIN.decode("ascii", errors="replace"),
            "hash_spec": HASH_SPEC,
            "public_key_hex": public_raw.hex(),
            "public_key_fingerprint_sha256": hashlib.sha256(public_raw).hexdigest(),
            "private_key_exported": False,
        },
        "summary": summary,
        "expected_checks": expected,
        "all_expected_checks_pass": all(expected.values()),
        "cases": rows,
    }


def write_report(report: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
