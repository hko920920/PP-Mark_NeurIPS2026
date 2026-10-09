"""RISC Zero prover/verifier scaffolding."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .halo2_interface import Witness
from .sampling import SampleObservation


@dataclass(slots=True)
class Risc0Paths:
    public: Path
    witness: Path
    receipt: Path


def _run_command(cmd: str, env: Dict[str, str]) -> None:
    args = shlex.split(cmd)
    subprocess.run(args, check=True, env=env)


def _serialize_observation(obs: SampleObservation) -> Dict[str, float | int]:
    return {
        "index": obs.index,
        "uniform": obs.uniform,
        "z_expected": obs.z_expected,
        "z_observed": obs.z_observed,
    }


def _serialize_witness(witness: Witness) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "secret_key": witness.secret_key.hex(),
        "seed": witness.seed.hex(),
        "codeword": witness.codeword.hex(),
        "sample_observations": [_serialize_observation(obs) for obs in witness.sample_observations],
    }
    if witness.gaussian_fixed is not None:
        payload["gaussian_fixed"] = [str(x) for x in witness.gaussian_fixed]
    if witness.combined_fixed is not None:
        payload["combined_fixed"] = [str(x) for x in witness.combined_fixed]
    if witness.bits is not None:
        payload["bits"] = witness.bits
    if witness.alpha_fixed is not None:
        payload["alpha_fixed"] = str(witness.alpha_fixed)
    if witness.sample_merkle_paths is not None:
        payload["sample_merkle_paths"] = [
            ["0x" + chunk.hex() for chunk in path] for path in witness.sample_merkle_paths
        ]
    return payload


def prepare_risc0_package(public_inputs: Dict[str, str | int], witness: Witness, output_dir: Path) -> Risc0Paths:
    output_dir.mkdir(parents=True, exist_ok=True)
    witness.require_fixed()
    public_path = output_dir / "public_inputs.json"
    witness_path = output_dir / "witness.json"
    receipt_path = output_dir / "receipt.bin"

    with public_path.open("w", encoding="utf-8") as handle:
        json.dump(public_inputs, handle, indent=2)

    witness_payload = _serialize_witness(witness)
    with witness_path.open("w", encoding="utf-8") as handle:
        json.dump(witness_payload, handle, indent=2)

    return Risc0Paths(public=public_path, witness=witness_path, receipt=receipt_path)


def run_risc0_prover(paths: Risc0Paths, prover_cmd: str | None) -> float:
    start = time.time()
    cmd = prover_cmd or "./risc0/host/target/release/zk-genguard-host"
    full = f"{cmd} prove --public {paths.public} --witness {paths.witness} --receipt {paths.receipt}"
    _run_command(full, os.environ.copy())
    elapsed = time.time() - start
    print(f"[risc0] prover elapsed: {elapsed:.3f}s")
    return elapsed


def verify_risc0_receipt(paths: Risc0Paths, verifier_cmd: str | None) -> float:
    start = time.time()
    cmd = verifier_cmd or "./risc0/host/target/release/zk-genguard-host"
    full = f"{cmd} verify --receipt {paths.receipt}"
    _run_command(full, os.environ.copy())
    elapsed = time.time() - start
    print(f"[risc0] receipt verified in {elapsed:.3f}s")
    return elapsed


# === SHA-256 Merkle helpers for the RISC Zero path ===
def _hash_pair(left: bytes, right: bytes) -> bytes:
    return sha256(left + right).digest()


def _hash_leaf(index: int, gaussian_fixed: int, combined_fixed: int, bit: int) -> bytes:
    data = (
        index.to_bytes(4, "little")
        + gaussian_fixed.to_bytes(8, "little", signed=True)
        + combined_fixed.to_bytes(8, "little", signed=True)
        + bytes([bit & 0xFF])
    )
    return sha256(data).digest()


def build_sample_merkle_sha(
    sample_trace: Iterable[SampleObservation],
    gaussian_fixed: Sequence[int],
    combined_fixed: Sequence[int],
    bits: Sequence[int],
) -> Tuple[bytes, List[List[bytes]]]:
    observations = list(sample_trace)
    if not observations:
        raise ValueError("sample_trace must be non-empty for Merkle commitment")
    if not (len(observations) == len(gaussian_fixed) == len(combined_fixed) == len(bits)):
        raise ValueError("sample trace, gaussian_fixed, combined_fixed, and bits lengths must match")

    leaves: List[bytes] = []
    for obs, g_fixed, c_fixed, bit in zip(observations, gaussian_fixed, combined_fixed, bits):
        leaves.append(_hash_leaf(obs.index, g_fixed, c_fixed, bit))

    target = 1
    while target < len(leaves):
        target <<= 1
    if len(leaves) < target:
        leaves.extend(b"\x00" * 32 for _ in range(target - len(leaves)))

    paths: List[List[bytes]] = [[] for _ in leaves]
    level: List[Tuple[bytes, List[int]]] = [(leaf, [idx]) for idx, leaf in enumerate(leaves)]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        next_level: List[Tuple[bytes, List[int]]] = []
        for i in range(0, len(level), 2):
            left_val, left_indices = level[i]
            right_val, right_indices = level[i + 1]
            parent = _hash_pair(left_val, right_val)
            for idx in left_indices:
                paths[idx].append(right_val)
            for idx in right_indices:
                paths[idx].append(left_val)
            next_level.append((parent, left_indices + right_indices))
        level = next_level

    root = level[0][0]
    paths = paths[: len(observations)]
    return root, paths
