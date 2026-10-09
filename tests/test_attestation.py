from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ppmark_v03.attestation import (
    AttestationStatement,
    StreamingAttestationStatement,
    alpha_fixed_and_scale_q30,
    canonicalize_scaled_channel,
    derive_challenge_seed,
    derive_streaming_challenge_seed,
    derive_opening_positions,
    expected_sample_relation,
    hash_lut_file,
    hash_sample_indices,
    hash_trace_commitment,
    hash_u32,
    lut_index_from_u32,
    producer_key_commitment,
)
from ppmark_v03.tables import InverseCDFTable


ROOT = Path(__file__).resolve().parents[1]
LUT_PATH = ROOT / "tables" / "invcdf_gaussian.bin"


def test_cross_language_fixed_vectors() -> None:
    binding = bytes(range(32))
    table = InverseCDFTable.from_file(LUT_PATH)
    assert table.size == 65_536
    assert hash_lut_file(LUT_PATH).hex() == (
        "1cfbba1e07e34386a57626f82aeb43e6ed54ba70a3ea6b7d964dc66a841074a8"
    )
    assert producer_key_commitment(binding).hex() == (
        "a436559a1eecc1836a9c08bd7ad20cff14fd30153b6f25eadbbdd4ace063c01c"
    )
    assert hash_u32(binding, 1) == 1_311_525_112
    assert lut_index_from_u32(hash_u32(binding, 1), table.size) == 20_011
    alpha_fixed, scale_q30, _, _ = alpha_fixed_and_scale_q30(4.0)
    assert (alpha_fixed, scale_q30) == (254_317, 260_420_644)
    assert expected_sample_relation(
        binding=binding,
        codeword_bits=[0, 1] * 256,
        flattened_index=17,
        width=128,
        inverse_cdf=table,
        scale_factor_q30=scale_q30,
        alpha_effective_fixed=alpha_fixed,
    ) == (1, -24_470, 229_847)


def test_canonical_channel_is_exactly_q18() -> None:
    table = InverseCDFTable.from_file(LUT_PATH)
    binding = bytes(range(32))
    bits = [0, 1] * 256
    alpha_fixed, scale_q30, _, _ = alpha_fixed_and_scale_q30(4.0)
    indices = np.arange(64, dtype=np.int64)
    gaussian = np.empty(64, dtype=np.float32)
    spread = np.empty(64, dtype=np.float32)
    expected: list[tuple[int, int, int]] = []
    for index in indices:
        relation = expected_sample_relation(
            binding=binding,
            codeword_bits=bits,
            flattened_index=int(index),
            width=8,
            inverse_cdf=table,
            scale_factor_q30=scale_q30,
            alpha_effective_fixed=alpha_fixed,
        )
        expected.append(relation)
        bit, gaussian_fixed, _ = relation
        # Undo only for the canonicalizer input; it requantizes deterministically.
        gaussian[index] = table.values[
            lut_index_from_u32(hash_u32(binding, int(index) + 1), table.size)
        ]
        spread[index] = 2 * bit - 1
    gaussian_fixed, combined = canonicalize_scaled_channel(
        gaussian,
        spread,
        scale_factor_q30=scale_q30,
        alpha_effective_fixed=alpha_fixed,
    )
    for index, (_, expected_gaussian, expected_combined) in enumerate(expected):
        assert int(gaussian_fixed[index]) == expected_gaussian
        assert int(round(float(combined[index]) * (1 << 18))) == expected_combined


def _statement() -> AttestationStatement:
    image_hash = bytes([4]) * 32
    ctx_hash = bytes([1]) * 32
    root = bytes([5]) * 32
    return AttestationStatement(
        protocol_version=1,
        ctx_hash=ctx_hash,
        binding=bytes([2]) * 32,
        producer_key_commitment=bytes([3]) * 32,
        image_hash=image_hash,
        sample_merkle_root=root,
        sample_indices_hash=hash_sample_indices([1, 2]),
        challenge_seed=derive_challenge_seed(image_hash, ctx_hash, root),
        opening_digest=bytes([8]) * 32,
        lut_hash=hash_lut_file(LUT_PATH),
        opening_k=1,
        alpha_effective_fixed=100,
        scale_factor_q30=1 << 29,
        sample_count=2,
        width=2,
        height=2,
        rs_n=64,
        rs_k=32,
    )


def test_statement_round_trip_and_challenge_tamper() -> None:
    statement = _statement()
    assert AttestationStatement.from_mapping(statement.to_json()) == statement
    statement.validate_derived_fields()
    tampered = statement.to_json()
    tampered["image_hash"] = "0x" + "ff" * 32
    with pytest.raises(ValueError, match="challenge_seed"):
        AttestationStatement.from_mapping(tampered).validate_derived_fields()


def test_opening_positions_are_unique_and_deterministic() -> None:
    seed = bytes([7]) * 32
    positions = derive_opening_positions(seed, 32, 1_000)
    assert positions == derive_opening_positions(seed, 32, 1_000)
    assert len(positions) == len(set(positions)) == 32


def test_statement_rejects_unknown_fields() -> None:
    statement = _statement().to_json()
    statement["self_asserted_trust"] = True
    with pytest.raises(ValueError, match="unknown"):
        AttestationStatement.from_mapping(statement)


def _streaming_statement() -> StreamingAttestationStatement:
    image_hash = bytes([4]) * 32
    ctx_hash = bytes([1]) * 32
    trace = hash_trace_commitment(
        [
            {"index": 1, "gaussian_fixed": -2, "combined_fixed": 3, "bit": 1},
            {"index": 2, "gaussian_fixed": 4, "combined_fixed": -5, "bit": 0},
        ]
    )
    return StreamingAttestationStatement(
        protocol_version=2,
        ctx_hash=ctx_hash,
        binding=bytes([2]) * 32,
        producer_key_commitment=bytes([3]) * 32,
        image_hash=image_hash,
        trace_commitment=trace,
        sample_indices_hash=hash_sample_indices([1, 2]),
        challenge_seed=derive_streaming_challenge_seed(image_hash, ctx_hash, trace),
        opening_digest=bytes([8]) * 32,
        lut_hash=hash_lut_file(LUT_PATH),
        opening_k=1,
        alpha_effective_fixed=100,
        scale_factor_q30=1 << 29,
        sample_count=2,
        width=2,
        height=2,
        rs_n=64,
        rs_k=32,
    )


def test_streaming_statement_round_trip_and_commitment_binding() -> None:
    statement = _streaming_statement()
    assert StreamingAttestationStatement.from_mapping(statement.to_json()) == statement
    statement.validate_derived_fields()
    reversed_trace = hash_trace_commitment(
        [
            {"index": 2, "gaussian_fixed": 4, "combined_fixed": -5, "bit": 0},
            {"index": 1, "gaussian_fixed": -2, "combined_fixed": 3, "bit": 1},
        ]
    )
    assert reversed_trace != statement.trace_commitment
    tampered = statement.to_json()
    tampered["trace_commitment"] = "0x" + "ff" * 32
    with pytest.raises(ValueError, match="challenge_seed"):
        StreamingAttestationStatement.from_mapping(tampered).validate_derived_fields()


def test_streaming_commitment_cross_language_vector() -> None:
    digest = hash_trace_commitment(
        [
            {"index": 7, "gaussian_fixed": -11, "combined_fixed": 13, "bit": 1},
            {"index": 9, "gaussian_fixed": 17, "combined_fixed": -19, "bit": 0},
        ]
    )
    assert digest.hex() == (
        "3f2618cf432fef251e643d44b0585bf02edfb7bd25d542e78a40cb9ad28013ae"
    )
