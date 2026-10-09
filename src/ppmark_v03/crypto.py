"""Poseidon helpers and deterministic samplers."""

from __future__ import annotations

import hashlib
import subprocess
import random
from pathlib import Path
from typing import List, Sequence

from .pallas_poseidon import PALLAS_MODULUS

FIELD_MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617


class PoseidonUnavailableError(RuntimeError):
    """Raised when poseidon bindings are not installed."""


def _run_poseidon_debug_bn254(inputs: Sequence[int]) -> int:
    """Compute BN254 Poseidon hash via the Rust helper binary to match the circuit."""
    if len(inputs) != 3:
        raise ValueError("poseidon_debug_bn254 expects exactly three inputs")
    root = Path(__file__).resolve().parents[2]
    binary = root / "halo2_prover" / "target" / "release" / "poseidon_debug"
    if not binary.exists():
        raise PoseidonUnavailableError(f"poseidon_debug binary not found: {binary}")
    args = [str(binary)] + [hex(v) for v in inputs]
    proc = subprocess.run(args, capture_output=True, text=True, check=True)
    for line in proc.stdout.splitlines():
        if line.startswith("result=0x"):
            return int(line.split("=", 1)[1], 16)
    raise RuntimeError("poseidon_debug output missing result line")


def hash_prompt_to_field(prompt: str) -> int:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % FIELD_MODULUS


def hash_prompt_to_pallas_field(prompt: str) -> int:
    """Hash prompt into the Pallas base field."""
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % PALLAS_MODULUS


def poseidon_hash_elements(elements: Sequence[int]) -> int:
    try:
        from poseidon_py.poseidon_hash import poseidon_hash  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise PoseidonUnavailableError(
            "poseidon_py is required for Poseidon hashing. Install poseidon-py."
        ) from exc

    iterator = iter(elements)
    try:
        acc = next(iterator) % FIELD_MODULUS
    except StopIteration as exc:  # pragma: no cover
        raise ValueError("poseidon_hash_elements requires a non-empty sequence") from exc
    for elem in iterator:
        acc = poseidon_hash(acc, elem % FIELD_MODULUS) % FIELD_MODULUS
    return acc % FIELD_MODULUS


def poseidon_hash_elements_pallas(elements: Sequence[int]) -> int:
    """Pallas-compatible Poseidon hash using poseidon_py parameters."""
    from .pallas_poseidon import poseidon_hash_pallas

    if not elements:
        raise ValueError("poseidon_hash_elements_pallas requires a non-empty sequence")
    return poseidon_hash_pallas(elements)


def poseidon_hash_elements_bn254_halo2(elements: Sequence[int]) -> int:
    """BN254 Poseidon hash aligned with the Halo2 circuit (ConstantLength<3>, rate=2)."""
    if len(elements) != 3:
        raise ValueError("poseidon_hash_elements_bn254_halo2 requires exactly 3 inputs")
    return _run_poseidon_debug_bn254([e % FIELD_MODULUS for e in elements])


def poseidon_hash_bytes(payload: bytes) -> bytes:
    ints = [int.from_bytes(payload[i : i + 31], "big") % FIELD_MODULUS for i in range(0, len(payload), 31)]
    digest = poseidon_hash_elements(ints)
    return digest.to_bytes(32, "big")


def sha256_hash_elements(elements: Sequence[int]) -> int:
    """Deterministic SHA256-based hash over integer elements."""
    hasher = hashlib.sha256()
    for elem in elements:
        val = elem % (1 << 256)
        hasher.update(val.to_bytes(32, "big"))
    return int.from_bytes(hasher.digest(), "big")


def bind_payload(prompt: str, seed: bytes, secret: bytes) -> int:
    """BN254 binding that matches the Halo2 Poseidon gadget."""
    if not seed or not secret:
        raise ValueError("Seed and secret must be non-empty")
    h_p = hash_prompt_to_field(prompt)
    seed_field = int.from_bytes(seed, "big") % FIELD_MODULUS
    secret_field = int.from_bytes(secret, "big") % FIELD_MODULUS
    return poseidon_hash_elements_bn254_halo2([h_p, seed_field, secret_field])


def shuffle_seeded_indices(length: int, key_material: bytes, sample_count: int) -> List[int]:
    if length <= 0:
        raise ValueError("length must be positive")
    if sample_count <= 0 or sample_count > length:
        raise ValueError("sample_count must be in [1, length]")
    digest = hashlib.blake2s(key_material).digest()
    rng = random.Random(int.from_bytes(digest, "big"))
    indices = list(range(length))
    rng.shuffle(indices)
    return indices[:sample_count]
