"""BN254 EdDSA key helpers."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from typing import Tuple

from .crypto import FIELD_MODULUS

try:  # pragma: no cover - optional dependency
    from py_ecc.bn128.bn128_curve import G1, curve_order, multiply
    from py_ecc.fields.field_elements import FQ
except ModuleNotFoundError as exc:  # pragma: no cover
    G1 = None  # type: ignore
    curve_order = None  # type: ignore
    multiply = None  # type: ignore
    FQ = None  # type: ignore
    DEP_ERROR = exc
else:
    DEP_ERROR = None


class KeygenDependencyError(RuntimeError):
    """Raised when py-ecc is missing."""


@dataclass(slots=True)
class KeyPair:
    secret: int
    public: Tuple[int, int]

    def secret_bytes(self) -> bytes:
        return self.secret.to_bytes(32, "big")

    def public_bytes(self) -> bytes:
        x, y = self.public
        return x.to_bytes(32, "big") + y.to_bytes(32, "big")


def _require_curve() -> None:
    if DEP_ERROR is not None or G1 is None or multiply is None or curve_order is None or FQ is None:
        raise KeygenDependencyError("py-ecc is required for BN254 key generation")


def generate_secret_scalar() -> int:
    _require_curve()
    rng = secrets.SystemRandom()
    return rng.randrange(1, curve_order)  # type: ignore[arg-type]


def derive_public_point(secret: int) -> Tuple[int, int]:
    _require_curve()
    if not (1 <= secret < curve_order):  # type: ignore[operator]
        raise ValueError("Secret scalar is out of range")
    point = multiply(G1, secret)  # type: ignore[arg-type]
    x, y = point[0], point[1]
    if isinstance(x, FQ):
        x = x.n
    if isinstance(y, FQ):
        y = y.n
    return x % FIELD_MODULUS, y % FIELD_MODULUS


def generate_keypair() -> KeyPair:
    secret = generate_secret_scalar()
    public = derive_public_point(secret)
    return KeyPair(secret=secret, public=public)
