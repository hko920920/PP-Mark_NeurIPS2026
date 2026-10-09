"""Spread spectrum noise generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .tables import InverseCDFTable
from .crypto import poseidon_hash_elements, sha256_hash_elements


@dataclass(slots=True)
class NoiseArtifacts:
    uniforms: np.ndarray
    gaussian: np.ndarray
    spread: np.ndarray
    combined: np.ndarray


def _hash_elements(elements: Sequence[int], backend: str) -> int:
    if backend == "poseidon":
        return poseidon_hash_elements(elements)
    if backend == "sha256":
        return sha256_hash_elements(elements)
    raise ValueError(f"Unsupported hash backend: {backend}")


def hash_uniforms(binding: int, count: int, backend: str = "sha256") -> np.ndarray:
    # float64 preserves the exact u32 / 2^32 value used by the SP1 relation.
    # Casting to float32 before LUT indexing can move values across a bin edge.
    uniforms = np.empty(count, dtype=np.float64)
    seed = binding
    for i in range(count):
        digest = _hash_elements([seed, i + 1], backend)
        uniforms[i] = (digest % (1 << 32)) / float(1 << 32)
    return uniforms


def poseidon_uniforms(binding: int, count: int) -> np.ndarray:
    return hash_uniforms(binding, count, backend="poseidon")


def spread_bits(bits: Sequence[int], total_pixels: int) -> np.ndarray:
    if len(bits) == 0:
        raise ValueError("Bit sequence must be non-empty")
    repeats = int(np.ceil(total_pixels / len(bits)))
    tiled = np.tile(bits, repeats)[:total_pixels]
    return (np.array(tiled, dtype=np.float32) * 2.0 - 1.0).astype(np.float32)


def generate_noise_artifacts(
    binding: int,
    bit_sequence: Sequence[int],
    total_pixels: int,
    inverse_cdf: InverseCDFTable,
    alpha: float,
    backend: str = "sha256",
) -> NoiseArtifacts:
    uniforms = hash_uniforms(binding, total_pixels, backend=backend)
    gaussian = inverse_cdf.batch_lookup(uniforms)
    spread = spread_bits(bit_sequence, total_pixels)
    combined = gaussian + alpha * spread
    return NoiseArtifacts(uniforms=uniforms, gaussian=gaussian, spread=spread, combined=combined)


def generate_noise_artifacts_2d(
    binding: int,
    bit_sequence: Sequence[int],
    height: int,
    width: int,
    inverse_cdf: InverseCDFTable,
    alpha: float,
    backend: str = "sha256",
) -> tuple[NoiseArtifacts, np.ndarray]:
    if len(bit_sequence) == 0:
        raise ValueError("Bit sequence must be non-empty")
    bit_len = len(bit_sequence)
    uniforms_2d = np.empty((height, width), dtype=np.float32)
    gaussian_2d = np.empty_like(uniforms_2d)
    spread_2d = np.empty_like(uniforms_2d)

    idx = 0
    for y in range(height):
        for x in range(width):
            digest = _hash_elements([binding, idx + 1], backend)
            u = (digest % (1 << 32)) / float(1 << 32)
            uniforms_2d[y, x] = u
            # Lookup before storing u in the float32 artifact array.  The
            # Python generator and the integer-only guest therefore select
            # exactly the same inverse-CDF entry.
            gaussian_2d[y, x] = inverse_cdf.lookup(u)
            # Spatial scrambling: pick bit via hash of coordinates to avoid column resonance.
            rand_val = _hash_elements([binding, x, y, 777], backend)
            bit_idx = rand_val % bit_len
            bit = bit_sequence[bit_idx]
            spread_2d[y, x] = (float(bit) * 2.0) - 1.0
            idx += 1

    combined_2d = gaussian_2d + alpha * spread_2d
    artifacts = NoiseArtifacts(
        uniforms=uniforms_2d.reshape(-1),
        gaussian=gaussian_2d.reshape(-1),
        spread=spread_2d.reshape(-1),
        combined=combined_2d.reshape(-1),
    )
    return artifacts, combined_2d


def build_bit_index_map(
    binding: int,
    width: int,
    height: int,
    bit_len: int,
    backend: str = "sha256",
) -> np.ndarray:
    idx_map = np.empty((height, width), dtype=np.int64)
    for y in range(height):
        for x in range(width):
            rand_val = _hash_elements([binding, x, y, 777], backend)
            idx_map[y, x] = rand_val % bit_len
    return idx_map
