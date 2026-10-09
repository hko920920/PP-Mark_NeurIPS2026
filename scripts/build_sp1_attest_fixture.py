"""Build a model-free SP1 Attest fixture for honest and tamper execution tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from ppmark_v03.attestation import (
    AttestationStatement,
    STREAMING_PROTOCOL_VERSION,
    StreamingAttestationStatement,
    alpha_fixed_and_scale_q30,
    derive_challenge_seed,
    derive_streaming_challenge_seed,
    derive_opening_positions,
    expected_sample_relation,
    hash_lut_file,
    hash_sample_indices,
    hash_trace_commitment,
    opening_digest,
    producer_key_commitment,
)
from ppmark_v03.halo2_interface import FIXED_SCALE
from ppmark_v03.payload import bytes_to_bits, make_ctx_hash
from ppmark_v03.risc0_runner import build_sample_merkle_sha
from ppmark_v03.rs import ReedSolomonCodec
from ppmark_v03.sampling import SampleTrace, deterministic_sample
from ppmark_v03.sp1_runner import (
    SP1SampleWitness,
    SP1Witness,
    prepare_sp1_package,
)
from ppmark_v03.tables import InverseCDFTable


TAMPERS = (
    "none",
    "p1-binding",
    "p1-key",
    "p2-root",
    "p2-index",
    "p3-image",
    "p3-opening",
    "p4-codeword",
    "p4-bit",
    "p4-gaussian",
    "p4-lut",
)

HASH_SPEC = {
    "version": "norm_v1",
    "color_space": "sRGB",
    "resize": [512, 512],
    "interpolation": "bicubic",
    "rounding": "round",
    "bytes": "uint8_rgb",
    "exif_transpose": True,
}


def _image_hash(image: Image.Image) -> bytes:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((512, 512), resample=Image.Resampling.BICUBIC)
    array = np.asarray(image, dtype=np.float32)
    normalized = np.clip(np.round(array), 0, 255).astype(np.uint8)
    return hashlib.sha256(normalized.tobytes()).digest()


def build_fixture(
    output_dir: Path,
    tamper: str = "none",
    proof_status: str = "pending",
    width: int = 8,
    height: int = 8,
    sample_count: int = 32,
    opening_k: int = 8,
    commitment_scheme: str = "streaming",
) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not 1 <= sample_count <= width * height:
        raise ValueError("sample_count must be in [1, width * height]")
    if not 1 <= opening_k <= sample_count:
        raise ValueError("opening_k must be in [1, sample_count]")
    if commitment_scheme not in {"merkle", "streaming"}:
        raise ValueError("commitment_scheme must be 'merkle' or 'streaming'")
    root = Path(__file__).resolve().parents[1]
    lut_path = root / "tables" / "invcdf_gaussian.bin"
    table = InverseCDFTable.from_file(lut_path)
    secret = bytes(range(32))
    seed = bytes(reversed(range(32)))
    prompt = "PP-Mark Attest model-free fixture"
    model_id = "fixture-model"
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).digest()
    ctx_hash = make_ctx_hash(
        prompt_hash=int.from_bytes(prompt_hash, "big"),
        model_id=model_id,
        seed=seed,
        width=width,
        height=height,
    ).to_bytes(32, "big")
    binding = hashlib.sha256(b"ctx_hash" + ctx_hash + secret).digest()
    codec = ReedSolomonCodec(64, 32)
    codeword = codec.encode(binding)
    bits = bytes_to_bits(codeword)
    sample_set = deterministic_sample(
        total_pixels=width * height,
        width=width,
        height=height,
        key_material=codeword,
        count=sample_count,
    )
    alpha_fixed, scale_q30, _, _ = alpha_fixed_and_scale_q30(4.0)

    samples: list[SP1SampleWitness] = []
    for index in sample_set.indices:
        bit, gaussian, combined = expected_sample_relation(
            binding=binding,
            codeword_bits=bits,
            flattened_index=index,
            width=width,
            inverse_cdf=table,
            scale_factor_q30=scale_q30,
            alpha_effective_fixed=alpha_fixed,
        )
        samples.append(SP1SampleWitness(index, gaussian, combined, bit))

    # Coherent malicious traces pass P2/P3 and must be rejected specifically by P4.
    if tamper == "p4-bit":
        samples = [replace(sample, bit=1 - sample.bit) for sample in samples]
    elif tamper == "p4-gaussian":
        samples = [
            replace(
                sample,
                gaussian_fixed=sample.gaussian_fixed + 1,
                combined_fixed=sample.combined_fixed + 1,
            )
            for sample in samples
        ]

    trace = SampleTrace()
    for sample in samples:
        trace.record(
            sample.index,
            0.0,
            sample.gaussian_fixed / FIXED_SCALE,
            sample.combined_fixed / FIXED_SCALE,
        )
    sample_root, _ = build_sample_merkle_sha(
        trace.entries,
        [sample.gaussian_fixed for sample in samples],
        [sample.combined_fixed for sample in samples],
        [sample.bit for sample in samples],
    )
    image_array = np.zeros((32, 32, 3), dtype=np.uint8)
    image_array[:, :, 0] = np.arange(32, dtype=np.uint8)[None, :]
    image_array[:, :, 1] = np.arange(32, dtype=np.uint8)[:, None]
    image_array[:, :, 2] = 127
    image = Image.fromarray(image_array, mode="RGB")
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "fixture.png"
    image.save(image_path)
    image_hash = _image_hash(image)
    trace_entries = [sample.to_json() for sample in samples]
    trace_commitment = hash_trace_commitment(trace_entries)
    if commitment_scheme == "streaming":
        challenge = derive_streaming_challenge_seed(
            image_hash,
            ctx_hash,
            trace_commitment,
        )
    else:
        challenge = derive_challenge_seed(image_hash, ctx_hash, sample_root)
    positions = derive_opening_positions(challenge, opening_k, len(samples))
    entries = [
        {
            "pos": position,
            "index": samples[position].index,
            "gaussian_fixed": samples[position].gaussian_fixed,
            "combined_fixed": samples[position].combined_fixed,
            "bit": samples[position].bit,
        }
        for position in positions
    ]
    common_statement = {
        "ctx_hash": ctx_hash,
        "binding": binding,
        "producer_key_commitment": producer_key_commitment(secret),
        "image_hash": image_hash,
        "sample_indices_hash": hash_sample_indices(sample_set.indices),
        "challenge_seed": challenge,
        "opening_digest": opening_digest(entries),
        "lut_hash": hash_lut_file(lut_path),
        "opening_k": opening_k,
        "alpha_effective_fixed": alpha_fixed,
        "scale_factor_q30": scale_q30,
        "sample_count": len(samples),
        "width": width,
        "height": height,
        "rs_n": 64,
        "rs_k": 32,
    }
    if commitment_scheme == "streaming":
        statement: AttestationStatement | StreamingAttestationStatement = (
            StreamingAttestationStatement(
                protocol_version=STREAMING_PROTOCOL_VERSION,
                trace_commitment=trace_commitment,
                **common_statement,
            )
        )
    else:
        statement = AttestationStatement(
            protocol_version=1,
            sample_merkle_root=sample_root,
            **common_statement,
        )

    witness_secret = secret
    witness_codeword = codeword
    if tamper == "p1-binding":
        statement = replace(statement, binding=bytes([0xAA]) * 32)
    elif tamper == "p1-key":
        statement = replace(statement, producer_key_commitment=bytes([0xAA]) * 32)
    elif tamper == "p2-root":
        if commitment_scheme == "streaming":
            statement = replace(statement, trace_commitment=bytes([0xAA]) * 32)
        else:
            statement = replace(statement, sample_merkle_root=bytes([0xAA]) * 32)
    elif tamper == "p2-index":
        samples[0] = replace(samples[0], index=(samples[0].index + 1) % (width * height))
    elif tamper == "p3-image":
        statement = replace(statement, image_hash=bytes([0xAA]) * 32)
    elif tamper == "p3-opening":
        statement = replace(statement, opening_digest=bytes([0xAA]) * 32)
    elif tamper == "p4-codeword":
        witness_codeword = bytes([codeword[0] ^ 1]) + codeword[1:]
    elif tamper == "p4-lut":
        statement = replace(statement, lut_hash=bytes([0xAA]) * 32)
    elif tamper not in {"none", "p4-bit", "p4-gaussian"}:
        raise ValueError(f"unknown tamper: {tamper}")

    paths = prepare_sp1_package(
        statement.to_json(),
        SP1Witness(
            secret_key=witness_secret,
            codeword=witness_codeword,
            samples=samples,
        ),
        output_dir,
    )
    # A full canonical latent lets the CLI exercise Detect without a diffusion
    # model. It is a test diagnostic, never a public deployment artifact.
    latent = np.empty((height, width), dtype=np.float32)
    for index in range(width * height):
        _, _, combined = expected_sample_relation(
            binding=binding,
            codeword_bits=bits,
            flattened_index=index,
            width=width,
            inverse_cdf=table,
            scale_factor_q30=scale_q30,
            alpha_effective_fixed=alpha_fixed,
        )
        latent[index // width, index % width] = combined / FIXED_SCALE
    latent_path = output_dir / "score_latent.npy"
    np.save(latent_path, latent)

    config = {
        "image": {
            "resolution": [width, height],
            "pixel_resolution": [32, 32],
            "sample_count": len(samples),
            "alpha": 4.0,
        },
        "model": {"base": model_id, "latent_dim": [1, height, width]},
        "zk": {
            "backend": "sp1",
            "sample_backend": "sha256",
            "prover_cmd": str(root / "target/sp1/release/sp1-genguard-host"),
            "verifier_cmd": str(root / "target/sp1/release/sp1-genguard-host"),
        },
        "watermark": {"msg_bits": 256, "rs_n": 64, "rs_k": 32},
        "tables": {"inverse_cdf_path": str(lut_path)},
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metadata = {
        "version": f"ppmark_attest_v{statement.protocol_version}",
        "prompt": prompt,
        "prompt_hash": "0x" + prompt_hash.hex(),
        "ctx_hash": "0x" + statement.ctx_hash.hex(),
        "binding": "0x" + statement.binding.hex(),
        "producer_key_commitment": "0x" + statement.producer_key_commitment.hex(),
        "seed_hex": seed.hex(),
        "message_hex": binding.hex(),
        "codeword_hex": witness_codeword.hex(),
        "payload_hash": hashlib.sha256(binding).hexdigest(),
        "sample_count": len(samples),
        "sample_root": sample_root.hex(),
        "trace_commitment": "0x" + trace_commitment.hex(),
        "commitment_scheme": (
            "streaming-full-trace-v2"
            if commitment_scheme == "streaming"
            else "merkle-v1-diagnostic"
        ),
        "sample_indices_hash": "0x" + statement.sample_indices_hash.hex(),
        "lut_hash": "0x" + statement.lut_hash.hex(),
        "width": width,
        "height": height,
        "model_id": model_id,
        "alpha": 4.0,
        "alpha_effective": alpha_fixed / FIXED_SCALE,
        "scale_factor": scale_q30 / (1 << 30),
        "hash_spec": HASH_SPEC,
        "image_png": image_path.name,
        "attestation_statement": statement.to_json(),
        "sp1": {
            "public": paths.public.name,
            "receipt": paths.receipt.name,
            "proof_status": proof_status,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(paths.public)
    print(paths.witness)
    print("0x" + producer_key_commitment(secret).hex())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tamper", choices=TAMPERS, default="none")
    parser.add_argument("--proof-status", choices=["pending", "generated"], default="pending")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--sample-count", type=int, default=32)
    parser.add_argument("--opening-k", type=int, default=8)
    parser.add_argument(
        "--commitment-scheme",
        choices=["merkle", "streaming"],
        default="streaming",
        help="Production defaults to protocol-v2 streaming; Merkle is diagnostic only.",
    )
    args = parser.parse_args()
    build_fixture(
        args.output.resolve(),
        args.tamper,
        args.proof_status,
        args.width,
        args.height,
        args.sample_count,
        args.opening_k,
        args.commitment_scheme,
    )


if __name__ == "__main__":
    main()
