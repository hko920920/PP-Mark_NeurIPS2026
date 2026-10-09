"""Reference watermark embedding pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import GlobalConfig
from .noise import NoiseArtifacts, generate_noise_artifacts_2d
from .sampling import SampleSet, SampleTrace
from .tables import InverseCDFTable


@dataclass(slots=True)
class EmbeddingResult:
    latent_noise: np.ndarray
    noise_artifacts: NoiseArtifacts
    sample_trace: SampleTrace


def inject_watermark(
    binding: int,
    bit_sequence: Sequence[int],
    config: GlobalConfig,
    inverse_cdf: InverseCDFTable,
    sample_set: SampleSet,
) -> EmbeddingResult:
    total_pixels = config.image.total_pixels
    channels = config.model.latent_dim[0] if config.model.latent_dim else 4
    combined_channels = []
    artifacts_ch0 = None
    backend = config.zk.sample_backend
    for ch in range(channels):
        artifacts, combined_2d = generate_noise_artifacts_2d(
            binding=binding + ch,
            bit_sequence=bit_sequence,
            height=config.image.height,
            width=config.image.width,
            inverse_cdf=inverse_cdf,
            alpha=config.image.alpha,
            backend=backend,
        )
        combined_channels.append(combined_2d)
        if ch == 0:
            artifacts_ch0 = artifacts
    artifacts = artifacts_ch0
    if artifacts is None:
        raise RuntimeError("Failed to generate watermark artifacts for channel 0")
    sample_trace = SampleTrace()
    for idx in sample_set.indices:
        sample_trace.record(
            idx,
            float(artifacts.uniforms[idx]),
            float(artifacts.gaussian[idx]),
            float(artifacts.combined[idx]),
        )
    combined_hwc = np.stack(combined_channels, axis=2) if channels > 1 else combined_channels[0][:, :, None]
    latent_noise = np.transpose(combined_hwc, (2, 0, 1))
    return EmbeddingResult(latent_noise=latent_noise, noise_artifacts=artifacts, sample_trace=sample_trace)
