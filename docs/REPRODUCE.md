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

## Generating transformed images

`apply_transforms.py` produces the transformed collections for the
post-transformation detection results. It takes a manifest of original
watermarked images, writes one subdirectory per image, and emits a separate
manifest per transform in the schema above, carrying each original's public
metadata forward unchanged.

```bash
python scripts/apply_transforms.py \
  --manifest outputs/quality_inputs/pp_mark/manifest.jsonl \
  --output-root outputs/transforms \
  --metadata-root . \
  --transforms clean,jpeg_q25,blur_8x8,noise_sigma_0.1,brightness_jitter_0_6,crop_scale_0.75,rot_75
```

The reported parameters are the defaults: JPEG `q=25` with 4:2:0 subsampling,
box blur with an 8-pixel kernel, additive Gaussian noise `sigma=0.1`,
brightness jitter of half-width `0.6` (a uniform factor on `[0.4, 1.6]`),
random crop retaining 75% of the area resized back with bicubic
interpolation, and rotation by 75 degrees. `--transforms all` adds `jpeg_q75`,
`resize_0.75` and a `chained` condition that applies JPEG, crop-resize,
rotation, blur, noise and brightness in that order. Random draws use fixed
per-transform generators, so repeated runs over the same inputs are identical;
`transform_settings.json`, `crop_boxes.json` and `brightness_factors.json`
record the settings and the realised draws.

Score each manifest with `eval_attack_manifest.py` under the reference
inversion settings. Geometric conditions need their alignment search enabled:

```bash
python scripts/eval_attack_manifest.py \
  --manifest outputs/transforms/manifest_rot_75.jsonl \
  --config config_alpha4_k1000.json --tau-file tau_fpr1_alpha4_sd21.json \
  --image-root outputs/transforms --metadata-root . \
  --ddim-device cuda --ddim-dtype float32 --ddim-steps 100 \
  --max-rotation 75 --rotation-step 1 --rotation-strategy coarse_to_fine \
  --skip-decode --skip-consistency \
  --scores-out outputs/transforms/eval_rot_75/scores.csv \
  --summary-out outputs/transforms/eval_rot_75/summary.json
```

Crop conditions use `--max-shift 16 --crop-keep-ratio 0.75 --crop-grid 9
--crop-refine --crop-refine-grid 9 --crop-refine-step 0.03` instead. Pixel-only
conditions (JPEG, blur, noise, brightness) need no alignment flags. The
detection rate for a condition is the `score_pass` fraction in its summary.

## Threshold calibration

The thresholds at the repository root are frozen constants from the reported
evaluation, and their original negative collections are not redistributed.
`calibrate_threshold.py` reproduces the calibration procedure so a threshold
can be derived for a new model or new evaluation conditions.

Generate a clean (non-watermarked) collection, score it against watermark
metadata so each row yields a false-positive score, then take the quantile:

```bash
python scripts/eval_attack_manifest.py \
  --manifest outputs/calibration/manifest.jsonl \
  --config config_alpha4_k1000.json --tau 0 \
  --ddim-device cuda --ddim-dtype float32 --ddim-steps 100 \
  --skip-decode --skip-consistency \
  --scores-out outputs/calibration/scores.csv \
  --summary-out outputs/calibration/summary.json

python scripts/calibrate_threshold.py \
  --scores outputs/calibration/scores.csv \
  --output outputs/calibration/tau_fpr1.json \
  --fpr 0.01 --model-family SD2.1 --alpha 4.0 --sample-count 1000 \
  --inversion-dtype float32 --inversion-steps 100
```

The default `percentile` estimator takes the `(1 - fpr)` quantile with linear
interpolation; `--estimator kth-largest` takes the `round(fpr * n)`-th largest
score, which is more conservative on small collections. Record the model, alpha,
sample count and inversion settings with any threshold, and never reuse a
threshold across different inversion settings. `datasets/calibration_recipe.json`
records how the reported prompt collections were selected from the public
prompt dataset, so equivalent collections can be rebuilt.

Two thresholds are provided: `tau_fpr1_alpha4_sd21.json` for SD2.1 under FP32
DDIM-100 inversion, and `tau_fpr1_alpha4_sdxl.json` for SDXL. SDXL inversion
must run in `bfloat16`; `float16` silently collapses the recovered latent to
zero and produces meaningless scores.

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
are needed. Watermarked images use the current prover's fixed-point embedding
channel. Obtain comparison implementations from the [model guide](MODELS.md).

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

For a 1,000-pair clean/PP-Mark collection, use the same evaluator with
`--expected-count 1000 --kid-subset-size 500 --kid-splits 20 --seed 2025`.
Choose a separate output directory for each collection.

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

## White-box forgery attack

`attack_whitebox_forge.py` runs the white-box PGD forgery attack against the
Score detector. The adversary is given the detector, its gradients, a cover
image and the public metadata of the claim being forged, and maximises the
Score statistic under an l-infinity budget. Acceptance is evaluated under the
any-step criterion: an image counts as accepted if its score reaches the
threshold at any optimization step, and the reported PSNR for an accepted image
is measured at its first-breach step.

```bash
bash scripts/run_wb_pgd_ppmark.sh \
  outputs/quality_inputs/pp_mark/manifest.jsonl \
  outputs/attacks/wb_pgd_sd21 sd21 150 100
```

The wrapper selects the configuration, threshold, model id and inversion dtype
for `sd21` or `sdxl` and applies the reported settings (`eps = 2/255`,
150 steps). Call the script directly for other budgets:

```bash
python scripts/attack_whitebox_forge.py \
  --manifest outputs/quality_inputs/pp_mark/manifest.jsonl \
  --output-root outputs/attacks/wb_pgd_sd21 \
  --config config_alpha4_k1000.json \
  --threshold-file tau_fpr1_alpha4_sd21.json \
  --metadata-root . \
  --ddim-device cuda --ddim-dtype float32 --ddim-steps 50 \
  --steps 150 --eps 0.00784313725490196 \
  --max-images 100 --trace-all
```

`summary.json` reports `far_any_step`, `avg_steps_to_fa` and `psnr_mean`;
`metrics.csv` carries the per-image `breach_step`, `score_max` and PSNR, and
`trace_scores.csv` the per-step score and PSNR trajectories used for the
cumulative FAR curves. Completed images are skipped unless `--overwrite` is
given, so a long run is restart-safe.

Gradients flow through DDIM inversion, so the cost is dominated by inversion
and a 100-image run takes hours to days on one GPU. On SDXL pass
`--ddim-dtype bfloat16` with `--ddim-grad-checkpointing` and
`--ddim-vae-slicing`; `float16` is unusable for inversion.

This attack targets Score only. It cannot produce an SP1 receipt, so a forged
image fails Accept verification whatever its Score. Confirm this by running the
main verifier in `attest` mode on a forged output, which is the separation the
two assurance levels are meant to provide.

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
CC="${CC:-gcc}" CXX="${CXX:-g++}" CARGO_TARGET_DIR="$PWD/target/sp1" cargo build --locked --release \
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
