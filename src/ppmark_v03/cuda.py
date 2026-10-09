"""CUDA integration scaffolding."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .config import GlobalConfig
from .embedding import EmbeddingResult, inject_watermark
from .noise import NoiseArtifacts, generate_noise_artifacts_2d, hash_uniforms
from .sampling import SampleSet, SampleTrace
from .tables import InverseCDFTable

try:  # pragma: no cover - optional dependency
    import cupy as cp
except Exception:  # pragma: no cover
    cp = None


@dataclass(slots=True)
class DeviceConfig:
    backend: str = "cpu"
    device_id: int = 0
    stream: int | None = None
    log_buffer: str | None = None
    # "auto" picks gpu_xorshift for CUDA, CPU Poseidon otherwise. Override to "cpu" for deterministic parity.
    uniform_backend: str = "auto"


class WatermarkKernel(Protocol):
    def embed(
        self,
        binding: int,
        bit_sequence: Sequence[int],
        sample_set: SampleSet,
    ) -> EmbeddingResult: ...


class CPUWatermarkKernel:
    def __init__(self, config: GlobalConfig, inverse_cdf: InverseCDFTable, device_cfg: DeviceConfig) -> None:
        self.config = config
        self.inverse_cdf = inverse_cdf
        self.device_cfg = device_cfg

    def embed(
        self,
        binding: int,
        bit_sequence: Sequence[int],
        sample_set: SampleSet,
    ) -> EmbeddingResult:
        return inject_watermark(
            binding=binding,
            bit_sequence=bit_sequence,
            config=self.config,
            inverse_cdf=self.inverse_cdf,
            sample_set=sample_set,
        )


class CUDAWatermarkKernel:
    def __init__(self, config: GlobalConfig, inverse_cdf: InverseCDFTable, device_cfg: DeviceConfig) -> None:
        if cp is None:
            raise RuntimeError("CuPy is required for CUDA backend")
        self._ensure_cuda_available()
        self.config = config
        self.inverse_cdf = inverse_cdf
        self.device_cfg = device_cfg
        self.uniform_backend = self._resolve_uniform_backend(device_cfg.uniform_backend)
        self.inverse_table_gpu = cp.asarray(inverse_cdf.values)
        # Precompute scale factor for LUT lookup to avoid repeated size checks in the kernel.
        self._lut_max_index = self.inverse_table_gpu.size - 1
        self._spread_kernel = self._build_spread_kernel()
        self._sample_kernel = self._build_sample_kernel()
        self._uniform_kernel = self._build_uniform_kernel()

    def embed(
        self,
        binding: int,
        bit_sequence: Sequence[int],
        sample_set: SampleSet,
    ) -> EmbeddingResult:
        total_pixels = self.config.image.total_pixels
        if total_pixels <= 0:
            raise ValueError("total_pixels must be positive")
        bits = np.array(bit_sequence, dtype=np.int8)
        if bits.size == 0:
            raise ValueError("Bit sequence must be non-empty")

        channels = self.config.model.latent_dim[0] if self.config.model.latent_dim else 4
        combined_channels = []
        sample_trace = SampleTrace()
        artifacts = None

        for ch in range(channels):
            artifacts_ch, combined_2d = generate_noise_artifacts_2d(
                binding=binding + ch,
                bit_sequence=bits,
                height=self.config.image.height,
                width=self.config.image.width,
                inverse_cdf=self.inverse_cdf,
                alpha=self.config.image.alpha,
                backend=self.config.zk.sample_backend,
            )
            combined_channels.append(combined_2d)
            if ch == 0:
                artifacts = artifacts_ch
                for idx in sample_set.indices:
                    sample_trace.record(
                        idx,
                        float(artifacts.uniforms[idx]),
                        float(artifacts.gaussian[idx]),
                        float(artifacts.combined[idx]),
                    )

        if artifacts is None:
            raise RuntimeError("Failed to generate watermark artifacts for channel 0")

        combined_hwc = np.stack(combined_channels, axis=2)
        latent_noise = np.transpose(combined_hwc, (2, 0, 1))
        return EmbeddingResult(latent_noise=latent_noise, noise_artifacts=artifacts, sample_trace=sample_trace)

    def _ensure_cuda_available(self) -> None:
        try:
            device_count = cp.cuda.runtime.getDeviceCount()  # type: ignore[attr-defined]
        except Exception as exc:  # pragma: no cover - runtime availability check
            raise RuntimeError("CuPy is installed but CUDA runtime is unavailable") from exc
        if device_count <= 0:  # pragma: no cover - hardware guard
            raise RuntimeError("CUDA backend requested but no CUDA devices were detected")
        # Help CuPy locate CUDA assets when running from a conda-only install (no /usr/local/cuda).
        if not os.environ.get("CUDA_PATH"):
            candidate_roots = [
                Path(sys.prefix) / "targets" / "x86_64-linux",
                Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "nvidia" / "cuda_runtime",
            ]
            for root in candidate_roots:
                include_dir = root / "include"
                if include_dir.exists():
                    os.environ["CUDA_PATH"] = str(root)
                    try:
                        import cupy as _cp  # delayed import to avoid circulars

                        _cp._environment._cuda_path = str(root)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    break
        try:
            import cupy_backends.cuda.libs.nvrtc as nvrtc  # type: ignore

            # nvrtc.getVersion() throws if the library is missing.
            _ = nvrtc.getVersion()  # pragma: no cover - sanity check
        except Exception as exc:  # pragma: no cover - environment guard
            raise RuntimeError("CuPy CUDA NVRTC support is unavailable") from exc

    def _resolve_uniform_backend(self, mode: str) -> str:
        normalized = mode.lower()
        if normalized == "auto":
            return "gpu_xorshift"
        if normalized not in {"cpu", "gpu_xorshift"}:
            raise ValueError(f"Unsupported uniform_backend: {mode}")
        return normalized

    def _hash_uniforms(self, binding: int, count: int) -> "cp.ndarray":
        uniforms = hash_uniforms(binding, count, backend=self.config.zk.sample_backend)
        return cp.asarray(uniforms, dtype=cp.float32)

    def _lookup_inverse_cdf(self, uniforms: "cp.ndarray") -> "cp.ndarray":
        idx = cp.clip((uniforms * self._lut_max_index).astype(cp.int64), 0, self._lut_max_index)
        return self.inverse_table_gpu[idx]

    def _generate_uniforms(self, binding: int, total_pixels: int) -> "cp.ndarray":
        if self.uniform_backend == "gpu_xorshift":
            out = cp.empty(total_pixels, dtype=cp.float32)
            block = 256
            grid = (total_pixels + block - 1) // block
            seed = np.uint64(binding & 0xFFFFFFFFFFFFFFFF)
            self._uniform_kernel((grid,), (block,), (np.int32(total_pixels), seed, out))
            return out
        # Default: CPU Poseidon path (parity with reference).
        return self._hash_uniforms(binding, total_pixels)

    def _build_spread_kernel(self) -> "cp.RawKernel":
        code = r"""
        extern "C" __global__ void spread_lookup_kernel(
            const int total_pixels,
            const int bit_length,
            const signed char* bits,
            const float* uniforms,
            const float* inverse_cdf,
            const int lut_size,
            const float alpha,
            float* gaussian_out,
            float* spread_out,
            float* combined_out
        ) {
            int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
            if (idx >= total_pixels) return;

            float u = uniforms[idx];
            int lut_idx = (int)(u * (lut_size - 1));
            if (lut_idx < 0) lut_idx = 0;
            if (lut_idx >= lut_size) lut_idx = lut_size - 1;
            float g = inverse_cdf[lut_idx];
            gaussian_out[idx] = g;

            int bit_idx = idx % bit_length;
            float s = (float)bits[bit_idx] * 2.0f - 1.0f;
            spread_out[idx] = s;

            float combined = g + alpha * s;
            combined_out[idx] = combined;
        }
        """
        return cp.RawKernel(code, "spread_lookup_kernel")

    def _build_sample_kernel(self) -> "cp.RawKernel":
        code = r"""
        extern "C" __global__ void extract_sample_logs(
            const int sample_count,
            const int* sample_indices,
            const float* uniforms,
            const float* gaussian,
            const float* combined,
            int* out_index,
            float* out_uniform,
            float* out_gaussian,
            float* out_combined
        ) {
            int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
            if (i >= sample_count) return;
            int idx = sample_indices[i];
            out_index[i] = idx;
            out_uniform[i] = uniforms[idx];
            out_gaussian[i] = gaussian[idx];
            out_combined[i] = combined[idx];
        }
        """
        return cp.RawKernel(code, "extract_sample_logs")

    def _build_uniform_kernel(self) -> "cp.RawKernel":
        code = r"""
        extern "C" __global__ void generate_uniforms(
            const int total_pixels,
            const unsigned long long seed,
            float* uniforms_out
        ) {
            int idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
            if (idx >= total_pixels) return;
            unsigned long long state = seed ^ (0x9E3779B97F4A7C15ULL * (unsigned long long)idx);
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state *= 0x2545F4914F6CDD1DULL;
            const double scale = 1.0 / 9007199254740992.0; // 2^53
            double uni = (double)(state >> 11) * scale;
            uniforms_out[idx] = (float)uni;
        }
        """
        return cp.RawKernel(code, "generate_uniforms")


def get_watermark_kernel(
    device_cfg: DeviceConfig,
    config: GlobalConfig,
    inverse_cdf: InverseCDFTable,
) -> WatermarkKernel:
    if device_cfg.backend.lower() == "cuda":
        return CUDAWatermarkKernel(config, inverse_cdf, device_cfg)
    return CPUWatermarkKernel(config, inverse_cdf, device_cfg)
