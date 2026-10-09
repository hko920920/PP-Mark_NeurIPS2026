# Evaluation guide

Follow the [installation and image-generation example](../README.md), then
install `requirements-eval.txt` for the optional evaluation tools. Run commands
from the repository root. Supply generated images and public metadata through
JSONL manifests; use the same prompt/seed pairs across comparison methods.

The SD2.1 reference configuration uses alpha = 4, N = 1000, FP32 DDIM-100
recovery, and the threshold in `tau_fpr1_alpha4_sd21.json`. Calibrate separately
when the model or evaluation conditions change. Record model revisions,
software versions and hardware with each run.

## Statistical detection and transformations

After generating `outputs/demo/` with the README commands, create a manifest:

```bash
mkdir -p outputs/evaluation
cat > outputs/evaluation/manifest.jsonl <<'JSONL'
{"idx":0,"label":"pos","attack":"clean","image_path":"../demo/watermarked.png","metadata_path":"../demo/metadata.json"}
JSONL

python scripts/eval_attack_manifest.py \
  --manifest outputs/evaluation/manifest.jsonl \
  --config config_alpha4_k1000.json --tau-file tau_fpr1_alpha4_sd21.json \
  --ddim-device cuda --ddim-dtype float32 --ddim-steps 100 \
  --skip-decode --skip-consistency \
  --scores-out outputs/evaluation/scores.csv \
  --summary-out outputs/evaluation/summary.json
```

Relative image/metadata paths resolve from the manifest directory unless
`--image-root` or `--metadata-root` is supplied. Add rows for additional images.
For transformed images, point `image_path` to the transformed file and retain
the public metadata for its corresponding original. Give each condition an
`attack` label. This is the statistical evaluation harness; use the README's
`verifier --mode attest` command for authenticated Accept verification.

## BRISQUE, KID and CLIP

Prepare a JSONL manifest for each method. For example, a clean-image row is:

```json
{"prompt_id":0,"seed":1,"prompt_text":"A red bicycle beside a stone wall","image_path":"images/p000.png","alpha":0.0}
```

An example PP-Mark row with public metadata is:

```json
{"idx":0,"prompt_id":0,"seed":1,"prompt_text":"A red bicycle beside a stone wall","image_path":"images/item_0000/watermarked.png","metadata_path":"images/item_0000/metadata.json","alpha":4.0}
```

Create complete matched collections; the commands below expect 100 pairs.
Image/metadata paths are relative to each manifest. Generate PP-Mark artifacts
with the main CLI, encoding integer seeds as 32-byte, zero-padded `--seed-hex`
values. The image-only helper `generate_quality_images.py` can generate clean
(alpha 0) or watermarked (alpha 4) images from the same row format:

```bash
python scripts/generate_quality_images.py \
  --manifest outputs/quality_inputs/clean/manifest.jsonl \
  --output-root outputs/quality_inputs/clean \
  --config config_alpha4_k1000.json --secret-file private_keys/producer.hex
```

This helper writes images only. Use the main CLI when metadata and receipts
are needed. Obtain comparison implementations from the [model guide](MODELS.md).

Create `outputs/quality_inputs/methods.json` with manifest paths relative to it:

```json
[
  {"slug":"clean","label":"Clean","manifest":"clean/manifest.jsonl"},
  {"slug":"pp_mark","label":"PP-Mark","manifest":"pp_mark/manifest.jsonl"}
]
```

Additional methods use the same format, with unique slugs and a `clean` reference.

```bash
python scripts/eval_rebuttal_quality_baselines.py \
  --methods-file outputs/quality_inputs/methods.json \
  --output outputs/quality --expected-count 100

python scripts/eval_rebuttal_clipscore_baselines.py \
  --manifest outputs/quality/matched_manifest.jsonl \
  --output outputs/quality/clipscore
```

BRISQUE and KID are computed on the matched collection. CLIP reports normalized
image/text cosine similarity. The default KID calculation uses subsets of 50
and 100 splits; adjust `--expected-count`, `--kid-subset-size` and `--kid-splits`
together for another dataset size. KID needs at least two samples per subset.
Slightly negative unbiased KID estimates can occur with finite samples.

`eval_ppmark_quality_full1000.py` accepts the same `--methods-file` interface
and expects 1,000 matched clean/PP-Mark pairs for its fixed-size evaluation.

## Signature controls

Use populated clean/PP-Mark manifests in the format above. PP-Mark metadata
must belong to the corresponding original image:

```bash
python scripts/experiment_secure_signature_controls.py \
  --positive-manifest outputs/quality_inputs/pp_mark/manifest.jsonl \
  --clean-manifest outputs/quality_inputs/clean/manifest.jsonl \
  --limit 100 --output outputs/signature_controls/report.json
```

This evaluates Ed25519 signatures over the canonical image hash (Bare-Sig),
and over the image hash, context hash and binding (Context-Sig). It checks
original verification, cross-image replay, context substitution, signer-authorized
clean images and post-signing transforms. PP-Mark Score/Accept decisions are
obtained with the PP-Mark verifier separately.

## Sequential crop-resize-JPEG evaluation

Prepare cropped images retaining 75% of the original area, resized back to
512 x 512 with bicubic interpolation. Keep each image paired with its original
public metadata. Under a directory such as `outputs/sequential_inputs/`, supply:

- `manifest_crop.jsonl`, using the statistical manifest schema above;
- the crop images and metadata referenced by that manifest;
- `crop_boxes.json`, mapping `img_000`, `img_001`, etc. to original-image
  `[left, top, right, bottom]` crop coordinates;
- `eval_crop_full100_maxlevers/scores.csv`, computed on those crop images.

Compute crop checkpoint scores with `eval_attack_manifest.py`, adding:

```text
--max-shift 16 --max-rotation 0 --crop-keep-ratio 0.75 --crop-grid 9
--crop-refine --crop-refine-grid 9 --crop-refine-step 0.03
```

Use the reference threshold and FP32 DDIM-100 settings above. Keep all indices,
including failed detections. To recompress a 100-image collection:

```bash
python scripts/prepare_rebuttal_existing_crop_recompression.py \
  --indices "$(seq -s, 0 99)" --qualities 25 \
  --crop-root outputs/sequential_inputs --output-dir outputs/sequential
```

Evaluate `outputs/sequential/manifest_q25.jsonl` with the same statistical
settings, `--image-root outputs/sequential` and `--metadata-root .`. Save the
result to `outputs/sequential/eval_q25/scores.csv`, then aggregate:

```bash
python scripts/summarize_persistent_pipeline.py \
  --crop-scores outputs/sequential_inputs/eval_crop_full100_maxlevers/scores.csv \
  --jpeg-scores outputs/sequential/eval_q25/scores.csv \
  --output outputs/sequential/persistent_summary.json
```

Persistent detection requires passing both checkpoints with the same threshold.
The helper joins by `idx` and requires the same complete set at both checkpoints.
For another sample count, specify that count with `--expected-count` and list
its indices in the preparation command. JPEG uses quality 25 and 4:2:0 subsampling.

## SP1 timing

Use the build and fixture preparation in [sp1/README.md](../sp1/README.md).

```bash
CC="${CC:-gcc}" CXX="${CXX:-g++}" CARGO_TARGET_DIR="$PWD/target/sp1" cargo build --release \
  --manifest-path sp1/Cargo.toml -p sp1-genguard-host \
  --example timing_campaign --features cuda

SP1_PROVER=cuda target/sp1/release/examples/timing_campaign streaming \
  --elf PATH_TO_BUILT_STREAMING_ELF \
  --public outputs/fixture/attestation_statement.json \
  --witness outputs/fixture/private_witness.json \
  --warmup 1 --runs 5 --output outputs/timing/streaming.json
```

Replace `PATH_TO_BUILT_STREAMING_ELF` with your built streaming guest ELF.
The campaign uses core receipts and records setup, execution, proving,
serialization and receipt verification separately. The fixture excludes diffusion
and inversion. Report the proof mode, hardware, sample/opening counts and timing
boundaries with results.
