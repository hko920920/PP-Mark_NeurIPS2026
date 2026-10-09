"""Payload construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from hashlib import sha256

from .crypto import bind_payload, hash_prompt_to_field, hash_prompt_to_pallas_field, poseidon_hash_elements_pallas
from .rs import ReedSolomonCodec


@dataclass(slots=True)
class PayloadArtifacts:
    prompt_hash: int
    binding: int
    message_bytes: bytes
    codeword: bytes
    bits: List[int]


def int_to_fixed_bytes(value: int, length: int) -> bytes:
    data = value.to_bytes(length, "big")
    if len(data) > length:
        raise ValueError("Value exceeds fixed length")
    return data[-length:]


def bytes_to_bits(payload: bytes) -> List[int]:
    bits: List[int] = []
    for byte in payload:
        for shift in range(7, -1, -1):
            bits.append((byte >> shift) & 1)
    return bits


def bits_to_bytes(bits: List[int]) -> bytes:
    """Pack a list of bits (MSB-first) into bytes."""
    if not bits:
        return b""
    normalized = [1 if b else 0 for b in bits]
    pad = (-len(normalized)) % 8
    if pad:
        normalized.extend([0] * pad)
    out = bytearray()
    for i in range(0, len(normalized), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | normalized[i + j]
        out.append(byte)
    return bytes(out)


def _encode_len_prefixed(data: bytes) -> bytes:
    if len(data) > (1 << 32) - 1:
        raise ValueError("Field too long for length-prefix encoding")
    return len(data).to_bytes(4, "big") + data


def _to_bytes(value: str | bytes | None) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8")


def make_ctx_hash(
    prompt_hash: int,
    model_id: str | bytes | None,
    seed: bytes,
    width: int,
    height: int,
    usecase_tag: str | bytes | None = None,
    user_tag: str | bytes | None = None,
) -> int:
    """Canonical context hash for SP1 binding."""
    segments: List[bytes] = []
    segments.append(prompt_hash.to_bytes(32, "big"))
    segments.append(_encode_len_prefixed(_to_bytes(model_id)))
    segments.append(_encode_len_prefixed(seed))
    segments.append(width.to_bytes(4, "big"))
    segments.append(height.to_bytes(4, "big"))
    segments.append(_encode_len_prefixed(_to_bytes(usecase_tag)))
    segments.append(_encode_len_prefixed(_to_bytes(user_tag)))
    digest = sha256(b"ctx_hash" + b"".join(segments)).digest()
    return int.from_bytes(digest, "big")


def build_payload(prompt: str, seed: bytes, secret: bytes, codec: ReedSolomonCodec) -> PayloadArtifacts:
    if not seed or not secret:
        raise ValueError("Seed and secret must be non-empty")
    prompt_hash = hash_prompt_to_pallas_field(prompt)
    seed_field = int.from_bytes(seed, "big")
    secret_field = int.from_bytes(secret, "big")
    binding = poseidon_hash_elements_pallas([prompt_hash, seed_field, secret_field])
    message_bytes = int_to_fixed_bytes(binding, codec.k)
    codeword = codec.encode(message_bytes)
    bits = bytes_to_bits(codeword)
    return PayloadArtifacts(
        prompt_hash=prompt_hash,
        binding=binding,
        message_bytes=message_bytes,
        codeword=codeword,
        bits=bits,
    )


def build_payload_pallas(prompt: str, seed: bytes, secret: bytes, codec: ReedSolomonCodec) -> PayloadArtifacts:
    """Build payload with Pallas Poseidon binding to match the Halo2 circuit."""
    if not seed or not secret:
        raise ValueError("Seed and secret must be non-empty")
    prompt_hash = hash_prompt_to_pallas_field(prompt)
    seed_field = int.from_bytes(seed, "big")
    secret_field = int.from_bytes(secret, "big")
    binding = poseidon_hash_elements_pallas([prompt_hash, seed_field, secret_field])
    message_bytes = int_to_fixed_bytes(binding, codec.k)
    codeword = codec.encode(message_bytes)
    bits = bytes_to_bits(codeword)
    return PayloadArtifacts(
        prompt_hash=prompt_hash,
        binding=binding,
        message_bytes=message_bytes,
        codeword=codeword,
        bits=bits,
    )


def build_payload_bn254(prompt: str, seed: bytes, secret: bytes, codec: ReedSolomonCodec) -> PayloadArtifacts:
    """Legacy BN254 Poseidon binding (for backward compatibility)."""
    if not seed or not secret:
        raise ValueError("Seed and secret must be non-empty")
    prompt_hash = hash_prompt_to_field(prompt)
    binding = bind_payload(prompt, seed, secret)
    message_bytes = int_to_fixed_bytes(binding, codec.k)
    codeword = codec.encode(message_bytes)
    bits = bytes_to_bits(codeword)
    return PayloadArtifacts(
        prompt_hash=prompt_hash,
        binding=binding,
        message_bytes=message_bytes,
        codeword=codeword,
        bits=bits,
    )


def build_payload_risc0(prompt: str, seed: bytes, secret: bytes, codec: ReedSolomonCodec) -> PayloadArtifacts:
    """SHA-256 binding for the RISC Zero backend (fast zkVM-friendly hash)."""
    if not seed or not secret:
        raise ValueError("Seed and secret must be non-empty")
    prompt_hash_bytes = sha256(prompt.encode("utf-8")).digest()
    binding_bytes = sha256(prompt_hash_bytes + seed + secret).digest()
    binding = int.from_bytes(binding_bytes, "big")
    prompt_hash = int.from_bytes(prompt_hash_bytes, "big")
    message_bytes = binding_bytes[: codec.k]
    codeword = codec.encode(message_bytes)
    bits = bytes_to_bits(codeword)
    return PayloadArtifacts(
        prompt_hash=prompt_hash,
        binding=binding,
        message_bytes=message_bytes,
        codeword=codeword,
        bits=bits,
    )
