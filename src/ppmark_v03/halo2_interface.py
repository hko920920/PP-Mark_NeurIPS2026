"""Halo2 circuit interface definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from .sampling import SampleObservation
from .pallas_poseidon import normalize_pallas_field, poseidon_hash_pair

FIXED_SCALE = 1 << 18  # Q18 fixed-point scale for gaussian/combined/alpha


@dataclass(slots=True)
class PublicInputs:
    prompt_hash: int
    binding: int
    sample_merkle_root: bytes
    alpha: int | None = None
    codeword_hash: bytes | None = None
    sample_merkle_depth: int | None = None


@dataclass(slots=True)
class Witness:
    secret_key: bytes
    seed: bytes
    codeword: bytes
    sample_observations: List[SampleObservation]
    sample_merkle_paths: List[List[bytes]] | None = None
    gaussian_fixed: List[int] | None = None
    combined_fixed: List[int] | None = None
    bits: List[int] | None = None
    alpha_fixed: int | None = None

    def require_fixed(self) -> None:
        if self.gaussian_fixed is None or self.combined_fixed is None or self.bits is None:
            raise ValueError("Fixed-point fields (gaussian_fixed, combined_fixed, bits) are required for ZK witness")
        if self.sample_merkle_paths is None:
            raise ValueError("Sample merkle paths are required for ZK witness")
        if self.alpha_fixed is None:
            raise ValueError("alpha_fixed is required for ZK witness")
        if len(self.sample_observations) != len(self.gaussian_fixed) or len(self.gaussian_fixed) != len(self.combined_fixed):
            raise ValueError("Sample observations must align with gaussian_fixed and combined_fixed lengths")
        if len(self.bits) != len(self.sample_observations):
            raise ValueError("Bit list must align with sample observations")
        if len(self.sample_merkle_paths) != len(self.sample_observations):
            raise ValueError("Merkle paths must align with sample observations")

    def to_json(self) -> dict:
        """Serialize witness to the JSON shape expected by RISC0/SP1 hosts."""
        def _obs(o: SampleObservation) -> dict:
            return {
                "index": o.index,
                "uniform": o.uniform,
                "z_expected": o.z_expected,
                "z_observed": o.z_observed,
            }

        self.require_fixed()
        payload: dict = {
            "secret_key": self.secret_key.hex(),
            "seed": self.seed.hex(),
            "codeword": self.codeword.hex(),
            "sample_observations": [_obs(o) for o in self.sample_observations],
            "gaussian_fixed": [str(x) for x in self.gaussian_fixed],  # type: ignore[arg-type]
            "combined_fixed": [str(x) for x in self.combined_fixed],  # type: ignore[arg-type]
            "bits": self.bits,  # type: ignore[arg-type]
            "alpha_fixed": str(self.alpha_fixed),
            "sample_merkle_paths": [
                ["0x" + chunk.hex() for chunk in path] for path in self.sample_merkle_paths  # type: ignore[arg-type]
            ],
        }
        return payload


def merkle_tree_poseidon(leaves: List[int]) -> Tuple[int, List[List[int]]]:
    """Compute Poseidon Merkle root and per-leaf authentication paths over integer leaves."""
    if not leaves:
        raise ValueError("merkle_tree_poseidon requires non-empty leaves")
    paths: List[List[int]] = [[] for _ in leaves]
    level: List[Tuple[int, List[int]]] = [(leaf, [idx]) for idx, leaf in enumerate(leaves)]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        next_level: List[Tuple[int, List[int]]] = []
        for i in range(0, len(level), 2):
            left_val, left_indices = level[i]
            right_val, right_indices = level[i + 1]
            parent = poseidon_hash_pair(left_val, right_val)
            for idx in left_indices:
                paths[idx].append(right_val)
            for idx in right_indices:
                paths[idx].append(left_val)
            next_level.append((parent, left_indices + right_indices))
        level = next_level
    return level[0][0] if isinstance(level[0], tuple) else level[0], paths


@dataclass(slots=True)
class Halo2Package:
    public_inputs: PublicInputs
    witness: Witness
    proof_path: Path | None = None


def build_sample_merkle(
    sample_trace: Iterable[SampleObservation],
    gaussian_fixed: Sequence[int],
    combined_fixed: Sequence[int],
    bits: Sequence[int],
) -> Tuple[bytes, List[List[bytes]]]:
    """Build Poseidon Merkle tree over per-sample observations.

    Leaves = Poseidon(index, gaussian_fixed, combined_fixed, bit).
    Tree is left-padded to a full binary tree so every path has equal depth.
    """
    observations = list(sample_trace)
    if not observations:
        raise ValueError("sample_trace must be non-empty for Merkle commitment")
    if not (len(observations) == len(gaussian_fixed) == len(combined_fixed) == len(bits)):
        raise ValueError("sample trace, gaussian_fixed, combined_fixed, and bits lengths must match")

    leaves: List[int] = []
    for obs, g_fixed, c_fixed, bit in zip(observations, gaussian_fixed, combined_fixed, bits):
        leaf = poseidon_hash_pair(
            normalize_pallas_field(obs.index),
            normalize_pallas_field(g_fixed),
        )
        leaf = poseidon_hash_pair(leaf, normalize_pallas_field(c_fixed))
        leaf = poseidon_hash_pair(leaf, normalize_pallas_field(bit))
        leaves.append(leaf)
    # Pad to next power of two for consistent path depths.
    target = 1
    while target < len(leaves):
        target <<= 1
    if len(leaves) < target:
        padding = [0] * (target - len(leaves))
        leaves = leaves + padding

    root_int, paths_int = merkle_tree_poseidon(leaves)
    paths_int = paths_int[: len(observations)]
    root_bytes = root_int.to_bytes(32, "big")
    paths_bytes = [[p.to_bytes(32, "big") for p in path] for path in paths_int]
    return root_bytes, paths_bytes
