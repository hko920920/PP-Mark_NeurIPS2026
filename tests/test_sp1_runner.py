from __future__ import annotations

import json
from pathlib import Path

from ppmark_v03 import sp1_runner
from ppmark_v03.sp1_runner import (
    SP1Paths,
    SP1SampleWitness,
    SP1Witness,
    prepare_sp1_package,
    run_sp1_prover,
    verify_sp1_receipt,
)


def test_sp1_package_separates_public_and_private_data(tmp_path: Path) -> None:
    public = {"protocol_version": 1, "image_hash": "0x" + "11" * 32}
    witness = SP1Witness(
        secret_key=bytes([9]) * 32,
        codeword=bytes(range(64)),
        samples=[SP1SampleWitness(3, -4, 5, 1)],
    )
    paths = prepare_sp1_package(public, witness, tmp_path)
    assert json.loads(paths.public.read_text(encoding="utf-8")) == public
    assert (bytes([9]) * 32).hex() not in paths.public.read_text(encoding="utf-8")
    private = json.loads(paths.witness.read_text(encoding="utf-8"))  # type: ignore[union-attr]
    assert private["secret_key"] == "0x" + (bytes([9]) * 32).hex()
    assert "sample_merkle_paths" not in private


def test_prover_removes_private_witness_and_verifier_passes_public_statement(
    tmp_path: Path, monkeypatch
) -> None:
    public = tmp_path / "statement.json"
    witness = tmp_path / "witness.json"
    receipt = tmp_path / "receipt.bin"
    public.write_text("{}", encoding="utf-8")
    witness.write_text("{}", encoding="utf-8")
    calls: list[list[str]] = []
    monkeypatch.setattr(sp1_runner, "_run_command", lambda command: calls.append(command))
    paths = SP1Paths(public=public, receipt=receipt, witness=witness)
    run_sp1_prover(paths, "host-bin")
    assert paths.witness is None
    assert not witness.exists()
    assert calls[0] == [
        "host-bin",
        "prove",
        "--public",
        str(public),
        "--witness",
        str(witness),
        "--receipt",
        str(receipt),
    ]
    verify_sp1_receipt(paths, "host-bin")
    assert calls[1] == [
        "host-bin",
        "verify",
        "--public",
        str(public),
        "--receipt",
        str(receipt),
    ]

