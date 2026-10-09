import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("cryptography")

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from ppmark_v03.signature_controls import (
    HASH_SPEC,
    PositiveCase,
    bare_message,
    canonical_image_hash_path,
    context_message,
    load_clean_images,
    load_positive_cases,
    run_control_experiment,
    sign,
    verifies,
)


def _write_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


def _hex(value: bytes) -> str:
    return "0x" + value.hex()


def test_signature_messages_separate_image_and_context_properties():
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    image_hash = b"\x11" * 32
    other_hash = b"\x12" * 32
    ctx_hash = b"\x22" * 32
    other_ctx = b"\x23" * 32
    binding = b"\x33" * 32

    bare_signature = sign(private_key, bare_message(image_hash))
    assert verifies(public_key, bare_signature, bare_message(image_hash))
    assert not verifies(public_key, bare_signature, bare_message(other_hash))
    assert verifies(public_key, bare_signature, bare_message(image_hash))

    context_signature = sign(
        private_key, context_message(image_hash, ctx_hash, binding)
    )
    assert verifies(
        public_key,
        context_signature,
        context_message(image_hash, ctx_hash, binding),
    )
    assert not verifies(
        public_key,
        context_signature,
        context_message(image_hash, other_ctx, binding),
    )


def test_full_control_matrix(tmp_path: Path):
    positive_rows = []
    clean_rows = []
    for index in range(2):
        positive_path = tmp_path / f"positive_{index}.png"
        clean_path = tmp_path / f"clean_{index}.png"
        metadata_path = tmp_path / f"metadata_{index}.json"
        _write_image(positive_path, seed=index)
        _write_image(clean_path, seed=100 + index)

        image_hash = canonical_image_hash_path(positive_path)
        metadata = {
            "hash_spec": HASH_SPEC,
            "opening": {"image_hash": _hex(image_hash)},
            "ctx_hash": _hex(bytes([0x20 + index]) * 32),
            "binding": _hex(bytes([0x40 + index]) * 32),
        }
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        positive_rows.append(
            {
                "idx": index,
                "image_path": positive_path.name,
                "metadata_path": metadata_path.name,
            }
        )
        clean_rows.append({"image_path": clean_path.name})

    positive_manifest = tmp_path / "positive.jsonl"
    positive_manifest.write_text(
        "\n".join(json.dumps(row) for row in positive_rows) + "\n",
        encoding="utf-8",
    )
    clean_manifest = tmp_path / "clean.jsonl"
    clean_manifest.write_text(
        "\n".join(json.dumps(row) for row in clean_rows) + "\n",
        encoding="utf-8",
    )

    positive_cases = load_positive_cases(positive_manifest, limit=2)
    clean_images = load_clean_images(clean_manifest, limit=2)
    report = run_control_experiment(positive_cases, clean_images)

    assert report["all_expected_checks_pass"]
    assert report["summary"]["n"] == 2
    assert report["summary"]["bare_sig"]["original"] == 2
    assert report["summary"]["bare_sig"]["cross_image_replay"] == 0
    assert report["summary"]["bare_sig"]["context_swap"] == 2
    assert report["summary"]["bare_sig"]["signed_clean"] == 2
    assert report["summary"]["context_sig"]["context_swap"] == 0
    assert report["summary"]["context_sig"]["signed_clean"] == 2


def test_metadata_hash_mismatch_fails_closed(tmp_path: Path):
    image_path = tmp_path / "positive.png"
    metadata_path = tmp_path / "metadata.json"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_image(image_path, seed=9)
    metadata_path.write_text(
        json.dumps(
            {
                "hash_spec": HASH_SPEC,
                "opening": {"image_hash": _hex(b"\x00" * 32)},
                "ctx_hash": _hex(b"\x11" * 32),
                "binding": _hex(b"\x22" * 32),
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "image_path": image_path.name,
                "metadata_path": metadata_path.name,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical image hash does not match"):
        load_positive_cases(manifest_path)


def test_loader_uses_opening_metadata_variant(tmp_path: Path):
    image_path = tmp_path / "positive.png"
    metadata_path = tmp_path / "metadata.json"
    opening_path = tmp_path / "metadata_opening.json"
    manifest_path = tmp_path / "manifest.jsonl"
    _write_image(image_path, seed=11)
    metadata_path.write_text(json.dumps({"version": "without-opening"}), encoding="utf-8")
    opening_path.write_text(
        json.dumps(
            {
                "hash_spec": HASH_SPEC,
                "opening": {
                    "image_hash": _hex(canonical_image_hash_path(image_path))
                },
                "ctx_hash": "0x1",
                "binding": "0x2",
            }
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "image_path": image_path.name,
                "metadata_path": metadata_path.name,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_positive_cases(manifest_path)
    assert len(cases) == 1
    assert cases[0].metadata_path == opening_path.resolve()
    assert cases[0].ctx_hash == b"\x00" * 31 + b"\x01"
    assert cases[0].binding == b"\x00" * 31 + b"\x02"
