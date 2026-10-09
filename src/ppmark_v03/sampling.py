"""Deterministic sampling utilities."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

from .crypto import shuffle_seeded_indices


@dataclass(slots=True)
class SampleSet:
    indices: List[int]
    width: int
    height: int

    def as_coords(self) -> List[Tuple[int, int]]:
        return [self.index_to_coord(idx) for idx in self.indices]

    def index_to_coord(self, idx: int) -> Tuple[int, int]:
        if idx < 0 or idx >= self.width * self.height:
            raise ValueError("Sample index out of bounds")
        x = idx % self.width
        y = idx // self.width
        return x, y

    @classmethod
    def create(cls, total_pixels: int, width: int, height: int, key_material: bytes, count: int) -> "SampleSet":
        indices = shuffle_seeded_indices(total_pixels, key_material, count)
        return cls(indices=indices, width=width, height=height)


@dataclass(slots=True)
class SampleObservation:
    index: int
    uniform: float
    z_expected: float
    z_observed: float


@dataclass(slots=True)
class SampleTrace:
    entries: List[SampleObservation] = field(default_factory=list)

    def record(self, index: int, uniform: float, z_expected: float, z_observed: float) -> None:
        self.entries.append(SampleObservation(index, uniform, z_expected, z_observed))

    def extend(self, observations: Sequence[SampleObservation]) -> None:
        self.entries.extend(observations)

    def __len__(self) -> int:
        return len(self.entries)

    def save_binary(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fmt = struct.Struct("<Ifff")
        with path.open("wb") as handle:
            for entry in self.entries:
                handle.write(fmt.pack(entry.index, entry.uniform, entry.z_expected, entry.z_observed))

    @classmethod
    def load_binary(cls, path: Path) -> "SampleTrace":
        fmt = struct.Struct("<Ifff")
        entries: List[SampleObservation] = []
        with path.open("rb") as handle:
            while chunk := handle.read(fmt.size):
                if len(chunk) != fmt.size:
                    raise IOError("Corrupted sample trace file")
                idx, uniform, z_expected, z_observed = fmt.unpack(chunk)
                entries.append(SampleObservation(idx, uniform, z_expected, z_observed))
        return cls(entries=entries)


def deterministic_sample(total_pixels: int, width: int, height: int, key_material: bytes, count: int) -> SampleSet:
    return SampleSet.create(total_pixels=total_pixels, width=width, height=height, key_material=key_material, count=count)
