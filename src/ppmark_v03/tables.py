"""Inverse CDF lookup table helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


class InverseCDFTable:
    def __init__(self, values: np.ndarray) -> None:
        if values.ndim != 1:
            raise ValueError("InverseCDFTable expects 1D array")
        self.values = values.astype(np.float32)

    @classmethod
    def from_file(cls, path: Path) -> "InverseCDFTable":
        if not path.exists():
            raise FileNotFoundError(f"Inverse CDF table not found: {path}")
        data = np.fromfile(path, dtype=np.float32)
        if data.size == 0:
            raise ValueError("Inverse CDF table is empty")
        return cls(data)

    @property
    def size(self) -> int:
        return int(self.values.size)

    def lookup(self, u: float) -> float:
        clipped = min(max(u, 0.0), 0.999999)
        idx = int(clipped * (self.size - 1))
        return float(self.values[idx])

    def batch_lookup(self, uniforms: Iterable[float]) -> np.ndarray:
        # Keep u32-derived uniforms in float64 until after index selection.
        # This is byte-for-byte aligned with floor(u32*(size-1)/2^32) in SP1.
        arr = np.array(list(uniforms), dtype=np.float64)
        idx = np.clip((arr * (self.size - 1)).astype(np.int64), 0, self.size - 1)
        return self.values[idx]
