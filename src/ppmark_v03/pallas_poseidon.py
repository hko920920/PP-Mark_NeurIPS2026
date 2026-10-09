"""Pallas Poseidon hash utilities aligned with halo2_gadgets defaults."""

from __future__ import annotations

from typing import Iterable, List

from .pallas_poseidon_constants import MDS, ROUND_CONSTANTS

PALLAS_MODULUS = int("40000000000000000000000000000000224698fc094cf91b992d30ed00000001", 16)
"""Pallas base field modulus (Fp), matches halo2_proofs::pasta::Fp."""

_FULL_ROUNDS = 8
_PARTIAL_ROUNDS = 56
_STATE_WIDTH = 3
_RATE = 2


def normalize_pallas_field(value: int) -> int:
    """Return value modulo the Pallas base field."""
    return value % PALLAS_MODULUS


def _pow5(x: int) -> int:
    return pow(x, 5, PALLAS_MODULUS)


def _mds_multiply(state: List[int]) -> List[int]:
    """Apply MDS matrix multiplication over the field."""
    out = [0] * _STATE_WIDTH
    for i in range(_STATE_WIDTH):
        acc = 0
        for j in range(_STATE_WIDTH):
            acc += state[j] * MDS[i][j]
        out[i] = acc % PALLAS_MODULUS
    return out


def poseidon_permute_pallas(state: List[int]) -> List[int]:
    """Poseidon permutation with Pallas constants (P128Pow5T3, width=3, rate=2)."""
    if len(state) != _STATE_WIDTH:
        raise ValueError("poseidon_permute_pallas expects a state of length 3")
    s = [normalize_pallas_field(x) for x in state]
    total_rounds = _FULL_ROUNDS + _PARTIAL_ROUNDS
    for round_idx in range(total_rounds):
        # Add round constants.
        s = [(s[i] + ROUND_CONSTANTS[round_idx][i]) % PALLAS_MODULUS for i in range(_STATE_WIDTH)]

        # Apply S-boxes.
        if round_idx < _FULL_ROUNDS // 2 or round_idx >= total_rounds - _FULL_ROUNDS // 2:
            s = [_pow5(x) for x in s]
        else:
            s[0] = _pow5(s[0])

        # Mix layer.
        s = _mds_multiply(s)
    return s


def _poseidon_hash_constant_length(elements: List[int]) -> int:
    """Replicate halo2 Poseidon Hash(ConstantLength<len(elements)>, width=3, rate=2)."""
    length = len(elements)
    if length == 0:
        raise ValueError("poseidon_hash_pallas requires non-empty input")
    state = [0, 0, normalize_pallas_field(length << 64)]  # capacity encodes length
    buffer: List[int] = []

    for idx, value in enumerate(elements):
        buffer.append(value)
        # If the buffer is full and more input remains, absorb+permute now.
        if len(buffer) == _RATE and idx + 1 < length:
            for i, v in enumerate(buffer):
                state[i] = (state[i] + v) % PALLAS_MODULUS
            state = poseidon_permute_pallas(state)
            buffer = []

    # Pad final block to the rate, then absorb and permute once (finish_absorbing).
    pad_len = (_RATE - len(buffer) % _RATE) % _RATE
    buffer.extend([0] * pad_len)
    for i, v in enumerate(buffer):
        state[i] = (state[i] + v) % PALLAS_MODULUS
    state = poseidon_permute_pallas(state)

    return state[0]


def poseidon_hash_pallas(elements: Iterable[int]) -> int:
    """Poseidon hash using the Pallas field parameters and ConstantLength domain tag."""
    data: List[int] = [normalize_pallas_field(int(x)) for x in elements]
    return _poseidon_hash_constant_length(data)


def poseidon_hash_pair(left: int, right: int) -> int:
    """Convenience wrapper for 2-ary Poseidon hash (ConstantLength<2>)."""
    return poseidon_hash_pallas([left, right])
