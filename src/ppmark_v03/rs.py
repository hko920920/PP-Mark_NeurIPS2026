"""Reed-Solomon codec wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

try:  # pragma: no cover
    from reedsolo import RSCodec
except ModuleNotFoundError:  # pragma: no cover
    RSCodec = None  # type: ignore


class ReedSolomonDependencyError(RuntimeError):
    """Raised when reedsolo is unavailable."""


@dataclass(slots=True)
class ReedSolomonCodec:
    n: int = 64
    k: int = 32
    primitive_polynomial: int = 0x11D
    _codec: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if RSCodec is None:
            raise ReedSolomonDependencyError("Install the 'reedsolo' package for RS encoding")
        nsym = self.n - self.k
        if nsym <= 0:
            raise ValueError("n must be greater than k")
        self._codec = RSCodec(nsym)

    @property
    def parity_symbols(self) -> int:
        return self.n - self.k

    def encode(self, payload: bytes) -> bytes:
        if len(payload) != self.k:
            raise ValueError(f"Payload must be {self.k} bytes; received {len(payload)}")
        codeword = self._codec.encode(payload)
        if len(codeword) != self.n:
            raise RuntimeError("Encoded length mismatch")
        return bytes(codeword)

    def decode(self, codeword: bytes) -> bytes:
        if len(codeword) != self.n:
            raise ValueError(f"Codeword must be {self.n} bytes; received {len(codeword)}")
        decoded = self._codec.decode(bytearray(codeword))
        message = decoded[0] if isinstance(decoded, tuple) else decoded
        if len(message) < self.k:
            raise RuntimeError("Decoded message truncated")
        return bytes(message[: self.k])

    def interleave_bits(self, payload_bits: Iterable[int]) -> bytes:
        bits = [1 if bit else 0 for bit in payload_bits]
        if len(bits) % 8 != 0:
            bits.extend([0] * (8 - (len(bits) % 8)))
        out = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            out.append(byte)
        return bytes(out)
