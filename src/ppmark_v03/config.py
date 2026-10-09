"""Configuration dataclasses for PP-Mark v0.3."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple


@dataclass(slots=True)
class ImageConfig:
    resolution: Tuple[int, int] = (1080, 1080)  # latent grid (W, H)
    pixel_resolution: Tuple[int, int] | None = None  # optional pixel-space resolution (W, H)
    sample_rate: float = 0.01
    sample_count_value: int | None = None  # override sample count; when None, fall back to sample_rate
    alpha: float = 2.0

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]

    @property
    def total_pixels(self) -> int:
        return self.width * self.height

    def sample_count(self) -> int:
        if self.sample_count_value is not None:
            return max(1, int(self.sample_count_value))
        return max(1, round(self.total_pixels * self.sample_rate))

    @property
    def pixel_width(self) -> int:
        return (self.pixel_resolution or self.resolution)[0]

    @property
    def pixel_height(self) -> int:
        return (self.pixel_resolution or self.resolution)[1]

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ImageConfig":
        res = tuple(raw.get("resolution", cls().resolution))
        pixel_res = raw.get("pixel_resolution")
        pixel_res = tuple(pixel_res) if pixel_res else None
        sample_rate = raw.get("sample_rate", cls().sample_rate)
        sample_count_value = raw.get("sample_count", raw.get("sample_count_value"))
        alpha = raw.get("alpha", cls().alpha)
        return cls(
            resolution=res,
            pixel_resolution=pixel_res,
            sample_rate=sample_rate,
            sample_count_value=sample_count_value,
            alpha=alpha,
        )


@dataclass(slots=True)
class ModelConfig:
    base: str = "SDXL_1.0"
    latent_dim: Tuple[int, int, int] = (4, 128, 128)


@dataclass(slots=True)
class ZKConfig:
    backend: str = "sp1"
    batch_time_min: int = 10
    sample_backend: str = "sha256"
    k: int = 10
    prover_cmd: str | None = None
    verifier_cmd: str | None = None
    vk_path: str | None = None
    vk_registry: Dict[str, str] | None = None


@dataclass(slots=True)
class WatermarkConfig:
    msg_bits: int = 256
    rs_n: int = 64
    rs_k: int = 32


@dataclass(slots=True)
class TableConfig:
    inverse_cdf_path: Path = Path("tables/invcdf_gaussian.bin")


@dataclass(slots=True)
class GlobalConfig:
    image: ImageConfig = field(default_factory=ImageConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    zk: ZKConfig = field(default_factory=ZKConfig)
    watermark: WatermarkConfig = field(default_factory=WatermarkConfig)
    tables: TableConfig = field(default_factory=TableConfig)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["tables"]["inverse_cdf_path"] = str(self.tables.inverse_cdf_path)
        # Rename the sample count override field for clarity in serialized configs.
        if "image" in payload:
            img = payload["image"]
            if "sample_count_value" in img:
                img["sample_count"] = img.pop("sample_count_value")
        return payload

    def dump(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "GlobalConfig":
        table_cfg = TableConfig(Path(raw.get("tables", {}).get("inverse_cdf_path", TableConfig().inverse_cdf_path)))
        return cls(
            image=ImageConfig.from_dict(raw.get("image", {})),
            model=ModelConfig(**raw.get("model", {})),
            zk=ZKConfig(**raw.get("zk", {})),
            watermark=WatermarkConfig(**raw.get("watermark", {})),
            tables=table_cfg,
        )

    @classmethod
    def load(cls, path: Path) -> "GlobalConfig":
        with path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        base_dir = path.parent
        table_entry = raw.get("tables", {})
        inverse_path = Path(table_entry.get("inverse_cdf_path", TableConfig().inverse_cdf_path))
        if not inverse_path.is_absolute():
            inverse_path = (base_dir / inverse_path).resolve()
        table_entry["inverse_cdf_path"] = inverse_path
        raw["tables"] = table_entry
        return cls.from_dict(raw)
