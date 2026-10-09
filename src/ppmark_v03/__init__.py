"""PP-Mark v0.3 core package."""

from .config import GlobalConfig, ImageConfig, ModelConfig, TableConfig, WatermarkConfig, ZKConfig
from .crypto import (
    FIELD_MODULUS,
    PALLAS_MODULUS,
    bind_payload,
    hash_prompt_to_field,
    hash_prompt_to_pallas_field,
    poseidon_hash_bytes,
    shuffle_seeded_indices,
)
from .keys import KeyPair, generate_keypair, generate_secret_scalar, derive_public_point
from .payload import PayloadArtifacts, bits_to_bytes, bytes_to_bits, build_payload, build_payload_bn254, build_payload_pallas
from .rs import ReedSolomonCodec
from .sampling import SampleObservation, SampleSet, SampleTrace, deterministic_sample
from .tables import InverseCDFTable
from .embedding import EmbeddingResult, inject_watermark
from .cuda import DeviceConfig, get_watermark_kernel
from .halo2_interface import PublicInputs, Witness, Halo2Package, build_sample_merkle
from .halo2_runner import Halo2Paths, prepare_prover_package, run_halo2_prover, verify_halo2_proof
from .sync import ExtractionResult, SyncResult, coarse_align, ddim_invert, extract_watermark, recover_bits, sync_search
from .ddim_unet import DiffusersDDIMConfig, DiffusersDDIMInverter

__all__ = [
    "GlobalConfig",
    "ImageConfig",
    "ModelConfig",
    "TableConfig",
    "WatermarkConfig",
    "ZKConfig",
    "FIELD_MODULUS",
    "PALLAS_MODULUS",
    "bind_payload",
    "hash_prompt_to_field",
    "hash_prompt_to_pallas_field",
    "poseidon_hash_bytes",
    "shuffle_seeded_indices",
    "bits_to_bytes",
    "bytes_to_bits",
    "KeyPair",
    "generate_keypair",
    "generate_secret_scalar",
    "derive_public_point",
    "PayloadArtifacts",
    "build_payload",
    "build_payload_bn254",
    "build_payload_pallas",
    "ReedSolomonCodec",
    "SampleObservation",
    "SampleSet",
    "SampleTrace",
    "deterministic_sample",
    "InverseCDFTable",
    "EmbeddingResult",
    "inject_watermark",
    "DeviceConfig",
    "get_watermark_kernel",
    "PublicInputs",
    "Witness",
    "Halo2Package",
    "Halo2Paths",
    "build_sample_merkle",
    "prepare_prover_package",
    "run_halo2_prover",
    "verify_halo2_proof",
    "ExtractionResult",
    "SyncResult",
    "coarse_align",
    "ddim_invert",
    "sync_search",
    "recover_bits",
    "extract_watermark",
    "DiffusersDDIMConfig",
    "DiffusersDDIMInverter",
]
