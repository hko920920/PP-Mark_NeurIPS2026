# Implementation map

The default path is `src/ppmark_v03/cli.py` with the SP1 workspace in `sp1/`.
`config_alpha4_k1000.json` selects SD2.1, alpha 4, and N=1000;
`config_alpha4_k1000_sdxl.json` provides the SDXL shape/model configuration.

| Component | Source |
|---|---|
| Context binding, payload, Reed–Solomon coding | `payload.py`, `rs.py`, `crypto.py` |
| Sampling, Gaussian lookup, latent embedding | `sampling.py`, `tables.py`, `noise.py`, `embedding.py`, `cuda.py` |
| Public statement, commitments and fixed-point relation | `attestation.py` |
| Diffusion generation and image verification | `cli.py`, `ddim_unet.py`, `sync.py` |
| SP1 serialization and process invocation | `sp1_runner.py` |
| Receipt constraints and host verification | `sp1/guest/src/lib.rs`, `sp1/host/src/main.rs` |
| Signature controls | `signature_controls.py` |

## Prover

The producer derives a binding from the context and producer key, encodes the
payload, constructs the watermarked latent and sampled trace, and generates an
image. A canonical public statement binds the context, producer-key commitment,
canonical image hash, ordered trace commitment, index-set hash, opening digest,
LUT hash, and fixed-point parameters. The private witness is passed separately
to the SP1 prover. Public metadata refers to the public statement and receipt.

Protocol v2 uses a domain-separated SHA-256 commitment over the complete ordered
trace. Its P4 relation is checked at the image-derived opening positions K.
The full-trace commitment and opened-subset embedding checks have distinct roles.
The v1 Merkle path remains for historical comparison, and a diagnostic sample root
may still appear in metadata; the default production receipt is v2.

## Verifier

Score (`detect`) uses the declared context/binding and model configuration for
DDIM recovery and statistical detection. Accept (`attest`) additionally checks
the trusted producer commitment, canonical image binding, and receipt against
the expected public statement. A modified derivative can retain a Score signal
while failing the original receipt's Accept rule.

The standalone `eval_attack_manifest.py` script is the historical statistical
evaluation harness. Use the main `cli.py verifier --mode attest` command for the
current authenticated Accept path; a historical evaluation column is not a
substitute for that verification command.

RISC0/Halo2 Python interfaces remain because shared types/utilities are imported
by the current implementation. This release's supported proof backend is SP1.
