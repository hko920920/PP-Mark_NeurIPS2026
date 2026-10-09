"""SP1 Attest package, prover, and verifier integration."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class SP1SampleWitness:
    index: int
    gaussian_fixed: int
    combined_fixed: int
    bit: int

    def to_json(self) -> dict[str, int | str]:
        if self.index < 0 or self.index > 0xFFFFFFFF:
            raise ValueError(f"sample index out of u32 range: {self.index}")
        if self.bit not in (0, 1):
            raise ValueError(f"sample bit must be 0 or 1: {self.bit}")
        return {
            "index": self.index,
            "gaussian_fixed": str(self.gaussian_fixed),
            "combined_fixed": str(self.combined_fixed),
            "bit": self.bit,
        }


@dataclass(frozen=True, slots=True)
class SP1Witness:
    secret_key: bytes
    codeword: bytes
    samples: Sequence[SP1SampleWitness]

    def to_json(self) -> dict[str, Any]:
        if len(self.secret_key) != 32:
            raise ValueError("SP1 secret key must be exactly 32 bytes")
        if not self.codeword:
            raise ValueError("SP1 codeword must not be empty")
        if not self.samples:
            raise ValueError("SP1 witness must contain samples")
        return {
            "secret_key": "0x" + self.secret_key.hex(),
            "codeword": "0x" + self.codeword.hex(),
            "samples": [sample.to_json() for sample in self.samples],
        }


@dataclass(slots=True)
class SP1Paths:
    public: Path
    receipt: Path
    witness: Path | None = None


def _command_prefix(command: str | None) -> list[str]:
    return shlex.split(command or "./target/sp1/release/sp1-genguard-host")


def _run_command(args: list[str]) -> None:
    subprocess.run(args, check=True, env=os.environ.copy())


def prepare_sp1_package(
    public_statement: Mapping[str, Any],
    witness: SP1Witness,
    output_dir: Path,
) -> SP1Paths:
    """Write the public statement and short-lived private proving witness."""
    output_dir.mkdir(parents=True, exist_ok=True)
    public_path = output_dir / "attestation_statement.json"
    witness_path = output_dir / "private_witness.json"
    receipt_path = output_dir / "attest_receipt.bin"

    public_path.write_text(
        json.dumps(dict(public_statement), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    witness_path.write_text(
        json.dumps(witness.to_json(), indent=2),
        encoding="utf-8",
    )
    return SP1Paths(public=public_path, receipt=receipt_path, witness=witness_path)


def run_sp1_prover(
    paths: SP1Paths,
    prover_cmd: str | None,
    *,
    retain_private_witness: bool = False,
) -> float:
    if paths.witness is None:
        raise ValueError("SP1 proving requires a private witness path")
    start = time.perf_counter()
    command = [
        *_command_prefix(prover_cmd),
        "prove",
        "--public",
        str(paths.public),
        "--witness",
        str(paths.witness),
        "--receipt",
        str(paths.receipt),
    ]
    _run_command(command)
    elapsed = time.perf_counter() - start
    if not retain_private_witness:
        paths.witness.unlink(missing_ok=True)
        paths.witness = None
    print(f"[sp1] Attest prover elapsed: {elapsed:.3f}s")
    return elapsed


def verify_sp1_receipt(paths: SP1Paths, verifier_cmd: str | None) -> float:
    """Verify both the proof and its exact expected public statement."""
    start = time.perf_counter()
    command = [
        *_command_prefix(verifier_cmd),
        "verify",
        "--public",
        str(paths.public),
        "--receipt",
        str(paths.receipt),
    ]
    _run_command(command)
    elapsed = time.perf_counter() - start
    print(f"[sp1] Attest receipt and statement verified in {elapsed:.3f}s")
    return elapsed
