"""Canonical PP-Mark Attest statement and fixed-point relation helpers.

This module is deliberately free of model dependencies.  The prover, public
verifier, SP1 host, and tests all use the same byte-level statement contract.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .halo2_interface import FIXED_SCALE
from .tables import InverseCDFTable


PROTOCOL_VERSION = 1
STREAMING_PROTOCOL_VERSION = 2
Q30_SCALE = 1 << 30
KEY_COMMIT_DOMAIN = b"ppmark_key_v1"
OPENING_DOMAIN = b"ppmark_opening_v1"
STREAMING_OPENING_DOMAIN = b"ppmark_opening_v2"
SAMPLE_INDICES_DOMAIN = b"ppmark_sample_indices_v1"
TRACE_COMMITMENT_DOMAIN = b"ppmark_trace_v2"
STATEMENT_FIELDS = frozenset(
    {
        "protocol_version",
        "ctx_hash",
        "binding",
        "producer_key_commitment",
        "image_hash",
        "sample_merkle_root",
        "sample_indices_hash",
        "challenge_seed",
        "opening_digest",
        "lut_hash",
        "opening_k",
        "alpha_effective_fixed",
        "scale_factor_q30",
        "sample_count",
        "width",
        "height",
        "rs_n",
        "rs_k",
    }
)
STREAMING_STATEMENT_FIELDS = frozenset(
    (STATEMENT_FIELDS - {"sample_merkle_root"}) | {"trace_commitment"}
)


def bytes32(value: bytes | bytearray | memoryview, name: str) -> bytes:
    raw = bytes(value)
    if len(raw) != 32:
        raise ValueError(f"{name} must be exactly 32 bytes; received {len(raw)}")
    return raw


def parse_hex32(value: str, name: str) -> bytes:
    text = str(value).strip()
    if text.lower().startswith("0x"):
        text = text[2:]
    if len(text) > 64:
        raise ValueError(f"{name} exceeds 32 bytes")
    text = text.zfill(64)
    try:
        return bytes32(bytes.fromhex(text), name)
    except ValueError as exc:
        raise ValueError(f"{name} is not valid 32-byte hex") from exc


def hex32(value: bytes) -> str:
    return "0x" + bytes32(value, "value").hex()


def int_bytes32(value: int, name: str = "value") -> bytes:
    if value < 0 or value >= 1 << 256:
        raise ValueError(f"{name} must be in [0, 2^256)")
    return value.to_bytes(32, "big")


def producer_key_commitment(secret_key: bytes) -> bytes:
    return hashlib.sha256(KEY_COMMIT_DOMAIN + bytes32(secret_key, "secret_key")).digest()


def derive_challenge_seed(
    image_hash: bytes,
    ctx_hash: bytes,
    sample_merkle_root: bytes,
) -> bytes:
    return hashlib.sha256(
        OPENING_DOMAIN
        + bytes32(image_hash, "image_hash")
        + bytes32(ctx_hash, "ctx_hash")
        + bytes32(sample_merkle_root, "sample_merkle_root")
    ).digest()


def derive_streaming_challenge_seed(
    image_hash: bytes,
    ctx_hash: bytes,
    trace_commitment: bytes,
) -> bytes:
    return hashlib.sha256(
        STREAMING_OPENING_DOMAIN
        + bytes32(image_hash, "image_hash")
        + bytes32(ctx_hash, "ctx_hash")
        + bytes32(trace_commitment, "trace_commitment")
    ).digest()


def derive_opening_positions(seed: bytes, k: int, sample_count: int) -> list[int]:
    seed = bytes32(seed, "challenge_seed")
    if k <= 0 or k > sample_count:
        raise ValueError("opening k must be in [1, sample_count]")
    positions: list[int] = []
    seen: set[int] = set()
    counter = 0
    while len(positions) < k:
        if counter > 0xFFFFFFFF:
            raise OverflowError("opening challenge counter overflow")
        digest = hashlib.sha256(seed + counter.to_bytes(4, "little")).digest()
        position = int.from_bytes(digest, "big") % sample_count
        if position not in seen:
            seen.add(position)
            positions.append(position)
        counter += 1
    return positions


def opening_digest(entries: Sequence[Mapping[str, Any]]) -> bytes:
    hasher = hashlib.sha256()
    for entry in entries:
        hasher.update(int(entry["pos"]).to_bytes(4, "little", signed=False))
        hasher.update(int(entry["index"]).to_bytes(4, "little", signed=False))
        hasher.update(int(entry["gaussian_fixed"]).to_bytes(8, "little", signed=True))
        hasher.update(int(entry["combined_fixed"]).to_bytes(8, "little", signed=True))
        hasher.update(int(entry["bit"]).to_bytes(1, "little", signed=False))
    return hasher.digest()


def hash_sample_indices(indices: Iterable[int]) -> bytes:
    hasher = hashlib.sha256()
    hasher.update(SAMPLE_INDICES_DOMAIN)
    for index in indices:
        value = int(index)
        if value < 0 or value > 0xFFFFFFFF:
            raise ValueError(f"sample index out of u32 range: {value}")
        hasher.update(value.to_bytes(4, "little"))
    return hasher.digest()


def hash_trace_commitment(entries: Sequence[Mapping[str, Any]]) -> bytes:
    """Hash a complete ordered trace using the protocol-v2 canonical encoding."""
    if len(entries) > 0xFFFFFFFF:
        raise ValueError("trace length exceeds u32")
    hasher = hashlib.sha256()
    hasher.update(TRACE_COMMITMENT_DOMAIN)
    hasher.update(len(entries).to_bytes(4, "little"))
    for entry in entries:
        index = int(entry["index"])
        gaussian = int(entry["gaussian_fixed"])
        combined = int(entry["combined_fixed"])
        bit = int(entry["bit"])
        if index < 0 or index > 0xFFFFFFFF:
            raise ValueError(f"sample index out of u32 range: {index}")
        if bit not in (0, 1):
            raise ValueError(f"sample bit must be binary: {bit}")
        hasher.update(index.to_bytes(4, "little"))
        hasher.update(gaussian.to_bytes(8, "little", signed=True))
        hasher.update(combined.to_bytes(8, "little", signed=True))
        hasher.update(bit.to_bytes(1, "little"))
    return hasher.digest()


def hash_lut_file(path: Path) -> bytes:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.digest()


def alpha_fixed_and_scale_q30(alpha: float) -> tuple[int, int, float, float]:
    scale = 1.0 / float(np.sqrt(1.0 + float(alpha) * float(alpha)))
    alpha_effective = float(alpha) * scale
    return (
        int(round(alpha_effective * FIXED_SCALE)),
        int(round(scale * Q30_SCALE)),
        alpha_effective,
        scale,
    )


def _round_div_signed(value: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    half = denominator // 2
    if value >= 0:
        return (value + half) // denominator
    return -((-value + half) // denominator)


def canonicalize_scaled_channel(
    gaussian: np.ndarray,
    spread: np.ndarray,
    *,
    scale_factor_q30: int,
    alpha_effective_fixed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the canonical Q18 Gaussian and combined channel.

    The returned floating channel consists of exact binary Q18 values and is
    therefore identical to the values checked by the SP1 guest after a
    float32 round trip.
    """
    gaussian_arr = np.asarray(gaussian, dtype=np.float64)
    spread_arr = np.asarray(spread, dtype=np.float64)
    if gaussian_arr.shape != spread_arr.shape:
        raise ValueError("gaussian and spread arrays must have the same shape")
    if not np.all(np.logical_or(spread_arr == -1.0, spread_arr == 1.0)):
        raise ValueError("spread must contain only -1 and +1")
    raw_fixed = np.rint(gaussian_arr * FIXED_SCALE).astype(np.int64)
    products = raw_fixed * np.int64(scale_factor_q30)
    half = np.int64(Q30_SCALE // 2)
    scaled_fixed = np.where(
        products >= 0,
        (products + half) // np.int64(Q30_SCALE),
        -((-products + half) // np.int64(Q30_SCALE)),
    ).astype(np.int64)
    combined_fixed = scaled_fixed + np.int64(alpha_effective_fixed) * spread_arr.astype(np.int64)
    return scaled_fixed, (combined_fixed.astype(np.float64) / FIXED_SCALE).astype(np.float32)


def hash_u32(binding: bytes, index: int) -> int:
    if index < 0 or index > 0xFFFFFFFF:
        raise ValueError(f"index out of u32 range: {index}")
    digest = hashlib.sha256(
        bytes32(binding, "binding") + int(index).to_bytes(32, "big")
    ).digest()
    return int.from_bytes(digest[-4:], "big")


def lut_index_from_u32(uniform_u32: int, lut_size: int) -> int:
    if uniform_u32 < 0 or uniform_u32 > 0xFFFFFFFF:
        raise ValueError("uniform_u32 out of range")
    if lut_size < 2:
        raise ValueError("LUT must have at least two entries")
    return (int(uniform_u32) * (int(lut_size) - 1)) >> 32


def scaled_gaussian_fixed(
    binding: bytes,
    flattened_index: int,
    inverse_cdf: InverseCDFTable,
    scale_factor_q30: int,
) -> int:
    uniform = hash_u32(binding, flattened_index + 1)
    lut_index = lut_index_from_u32(uniform, inverse_cdf.size)
    raw_value = float(inverse_cdf.values[lut_index])
    raw_fixed = int(round(raw_value * FIXED_SCALE))
    return _round_div_signed(raw_fixed * int(scale_factor_q30), Q30_SCALE)


def bit_index(binding: bytes, x: int, y: int, bit_length: int) -> int:
    if bit_length <= 0:
        raise ValueError("bit_length must be positive")
    if min(x, y) < 0 or max(x, y) > 0xFFFFFFFF:
        raise ValueError("coordinates must fit u32")
    hasher = hashlib.sha256()
    hasher.update(bytes32(binding, "binding"))
    for value in (x, y, 777):
        hasher.update(int(value).to_bytes(32, "big"))
    return int.from_bytes(hasher.digest(), "big") % bit_length


def expected_sample_relation(
    *,
    binding: bytes,
    codeword_bits: Sequence[int],
    flattened_index: int,
    width: int,
    inverse_cdf: InverseCDFTable,
    scale_factor_q30: int,
    alpha_effective_fixed: int,
) -> tuple[int, int, int]:
    if width <= 0:
        raise ValueError("width must be positive")
    x = int(flattened_index) % int(width)
    y = int(flattened_index) // int(width)
    mapped = bit_index(binding, x, y, len(codeword_bits))
    bit = 1 if int(codeword_bits[mapped]) else 0
    gaussian_fixed = scaled_gaussian_fixed(
        binding,
        int(flattened_index),
        inverse_cdf,
        scale_factor_q30,
    )
    combined_fixed = gaussian_fixed + int(alpha_effective_fixed) * (2 * bit - 1)
    return bit, gaussian_fixed, combined_fixed


@dataclass(frozen=True, slots=True)
class AttestationStatement:
    protocol_version: int
    ctx_hash: bytes
    binding: bytes
    producer_key_commitment: bytes
    image_hash: bytes
    sample_merkle_root: bytes
    sample_indices_hash: bytes
    challenge_seed: bytes
    opening_digest: bytes
    lut_hash: bytes
    opening_k: int
    alpha_effective_fixed: int
    scale_factor_q30: int
    sample_count: int
    width: int
    height: int
    rs_n: int
    rs_k: int

    def __post_init__(self) -> None:
        for name in (
            "ctx_hash",
            "binding",
            "producer_key_commitment",
            "image_hash",
            "sample_merkle_root",
            "sample_indices_hash",
            "challenge_seed",
            "opening_digest",
            "lut_hash",
        ):
            bytes32(getattr(self, name), name)
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"unsupported protocol version: {self.protocol_version}")
        if self.opening_k <= 0 or self.opening_k > self.sample_count:
            raise ValueError("opening_k must be in [1, sample_count]")
        if min(self.sample_count, self.width, self.height, self.rs_n, self.rs_k) <= 0:
            raise ValueError("statement dimensions must be positive")
        if self.rs_k != 32 or self.rs_n <= self.rs_k or self.rs_n > 255:
            raise ValueError("Attest currently requires a 32-byte systematic RS message")
        if self.alpha_effective_fixed <= 0 or self.alpha_effective_fixed > 16 * FIXED_SCALE:
            raise ValueError("alpha_effective_fixed is outside the supported range")
        if self.scale_factor_q30 <= 0 or self.scale_factor_q30 > Q30_SCALE:
            raise ValueError("scale_factor_q30 must be in (0, 2^30]")
        if self.sample_count > self.width * self.height:
            raise ValueError("sample_count exceeds the latent grid")

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "ctx_hash": hex32(self.ctx_hash),
            "binding": hex32(self.binding),
            "producer_key_commitment": hex32(self.producer_key_commitment),
            "image_hash": hex32(self.image_hash),
            "sample_merkle_root": hex32(self.sample_merkle_root),
            "sample_indices_hash": hex32(self.sample_indices_hash),
            "challenge_seed": hex32(self.challenge_seed),
            "opening_digest": hex32(self.opening_digest),
            "lut_hash": hex32(self.lut_hash),
            "opening_k": self.opening_k,
            "alpha_effective_fixed": str(self.alpha_effective_fixed),
            "scale_factor_q30": str(self.scale_factor_q30),
            "sample_count": self.sample_count,
            "width": self.width,
            "height": self.height,
            "rs_n": self.rs_n,
            "rs_k": self.rs_k,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "AttestationStatement":
        fields = set(raw)
        if fields != STATEMENT_FIELDS:
            missing = sorted(STATEMENT_FIELDS - fields)
            extra = sorted(fields - STATEMENT_FIELDS)
            details: list[str] = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"unknown={extra}")
            raise ValueError("non-canonical Attest statement fields: " + ", ".join(details))
        return cls(
            protocol_version=int(raw["protocol_version"]),
            ctx_hash=parse_hex32(str(raw["ctx_hash"]), "ctx_hash"),
            binding=parse_hex32(str(raw["binding"]), "binding"),
            producer_key_commitment=parse_hex32(
                str(raw["producer_key_commitment"]), "producer_key_commitment"
            ),
            image_hash=parse_hex32(str(raw["image_hash"]), "image_hash"),
            sample_merkle_root=parse_hex32(
                str(raw["sample_merkle_root"]), "sample_merkle_root"
            ),
            sample_indices_hash=parse_hex32(
                str(raw["sample_indices_hash"]), "sample_indices_hash"
            ),
            challenge_seed=parse_hex32(str(raw["challenge_seed"]), "challenge_seed"),
            opening_digest=parse_hex32(str(raw["opening_digest"]), "opening_digest"),
            lut_hash=parse_hex32(str(raw["lut_hash"]), "lut_hash"),
            opening_k=int(raw["opening_k"]),
            alpha_effective_fixed=int(raw["alpha_effective_fixed"]),
            scale_factor_q30=int(raw["scale_factor_q30"]),
            sample_count=int(raw["sample_count"]),
            width=int(raw["width"]),
            height=int(raw["height"]),
            rs_n=int(raw["rs_n"]),
            rs_k=int(raw["rs_k"]),
        )

    def validate_derived_fields(self) -> None:
        expected = derive_challenge_seed(
            self.image_hash,
            self.ctx_hash,
            self.sample_merkle_root,
        )
        if self.challenge_seed != expected:
            raise ValueError("challenge_seed does not match image/context/root")


@dataclass(frozen=True, slots=True)
class StreamingAttestationStatement:
    """Protocol-v2 statement for a complete ordered streaming commitment."""

    protocol_version: int
    ctx_hash: bytes
    binding: bytes
    producer_key_commitment: bytes
    image_hash: bytes
    trace_commitment: bytes
    sample_indices_hash: bytes
    challenge_seed: bytes
    opening_digest: bytes
    lut_hash: bytes
    opening_k: int
    alpha_effective_fixed: int
    scale_factor_q30: int
    sample_count: int
    width: int
    height: int
    rs_n: int
    rs_k: int

    def __post_init__(self) -> None:
        for name in (
            "ctx_hash",
            "binding",
            "producer_key_commitment",
            "image_hash",
            "trace_commitment",
            "sample_indices_hash",
            "challenge_seed",
            "opening_digest",
            "lut_hash",
        ):
            bytes32(getattr(self, name), name)
        if self.protocol_version != STREAMING_PROTOCOL_VERSION:
            raise ValueError(f"unsupported streaming protocol version: {self.protocol_version}")
        if self.opening_k <= 0 or self.opening_k > self.sample_count:
            raise ValueError("opening_k must be in [1, sample_count]")
        if min(self.sample_count, self.width, self.height, self.rs_n, self.rs_k) <= 0:
            raise ValueError("statement dimensions must be positive")
        if self.rs_k != 32 or self.rs_n <= self.rs_k or self.rs_n > 255:
            raise ValueError("Attest currently requires a 32-byte systematic RS message")
        if self.alpha_effective_fixed <= 0 or self.alpha_effective_fixed > 16 * FIXED_SCALE:
            raise ValueError("alpha_effective_fixed is outside the supported range")
        if self.scale_factor_q30 <= 0 or self.scale_factor_q30 > Q30_SCALE:
            raise ValueError("scale_factor_q30 must be in (0, 2^30]")
        if self.sample_count > self.width * self.height:
            raise ValueError("sample_count exceeds the latent grid")

    def to_json(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "ctx_hash": hex32(self.ctx_hash),
            "binding": hex32(self.binding),
            "producer_key_commitment": hex32(self.producer_key_commitment),
            "image_hash": hex32(self.image_hash),
            "trace_commitment": hex32(self.trace_commitment),
            "sample_indices_hash": hex32(self.sample_indices_hash),
            "challenge_seed": hex32(self.challenge_seed),
            "opening_digest": hex32(self.opening_digest),
            "lut_hash": hex32(self.lut_hash),
            "opening_k": self.opening_k,
            "alpha_effective_fixed": str(self.alpha_effective_fixed),
            "scale_factor_q30": str(self.scale_factor_q30),
            "sample_count": self.sample_count,
            "width": self.width,
            "height": self.height,
            "rs_n": self.rs_n,
            "rs_k": self.rs_k,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "StreamingAttestationStatement":
        fields = set(raw)
        if fields != STREAMING_STATEMENT_FIELDS:
            missing = sorted(STREAMING_STATEMENT_FIELDS - fields)
            extra = sorted(fields - STREAMING_STATEMENT_FIELDS)
            details: list[str] = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"unknown={extra}")
            raise ValueError(
                "non-canonical streaming Attest statement fields: " + ", ".join(details)
            )
        return cls(
            protocol_version=int(raw["protocol_version"]),
            ctx_hash=parse_hex32(str(raw["ctx_hash"]), "ctx_hash"),
            binding=parse_hex32(str(raw["binding"]), "binding"),
            producer_key_commitment=parse_hex32(
                str(raw["producer_key_commitment"]), "producer_key_commitment"
            ),
            image_hash=parse_hex32(str(raw["image_hash"]), "image_hash"),
            trace_commitment=parse_hex32(
                str(raw["trace_commitment"]), "trace_commitment"
            ),
            sample_indices_hash=parse_hex32(
                str(raw["sample_indices_hash"]), "sample_indices_hash"
            ),
            challenge_seed=parse_hex32(str(raw["challenge_seed"]), "challenge_seed"),
            opening_digest=parse_hex32(str(raw["opening_digest"]), "opening_digest"),
            lut_hash=parse_hex32(str(raw["lut_hash"]), "lut_hash"),
            opening_k=int(raw["opening_k"]),
            alpha_effective_fixed=int(raw["alpha_effective_fixed"]),
            scale_factor_q30=int(raw["scale_factor_q30"]),
            sample_count=int(raw["sample_count"]),
            width=int(raw["width"]),
            height=int(raw["height"]),
            rs_n=int(raw["rs_n"]),
            rs_k=int(raw["rs_k"]),
        )

    def validate_derived_fields(self) -> None:
        expected = derive_streaming_challenge_seed(
            self.image_hash,
            self.ctx_hash,
            self.trace_commitment,
        )
        if self.challenge_seed != expected:
            raise ValueError("challenge_seed does not match image/context/trace commitment")
