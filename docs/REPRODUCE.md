# Reproduction inputs and experiment entry points

Run commands from the repository root after the installation in the
[README](../README.md). Evaluation additionally needs `requirements-eval.txt`.
The default configuration is SD2.1, alpha = 4, N = 1000. The historical detection
threshold is 2.4665623073926706, with FP32 DDIM-100 recovery. These commands are
provided for users of the release; no GPU experiment or timing campaign was
rerun while assembling it.

## What the source release contains

| Experiment | Entry point | Included input description / record |
|---|---|---|
| Current image generation, Score and Accept | `python -m ppmark_v03.cli` | Root README, two model configurations, inverse-CDF LUT, SP1 source |
| Statistical transformation evaluation | `scripts/eval_attack_manifest.py` | Image/metadata manifest interface below |
| Crop-resize followed by JPEG | `scripts/prepare_rebuttal_existing_crop_recompression.py` | Crop boxes, historical checkpoint scores, normalized crop manifest |
| Bare-Sig and Context-Sig | `scripts/experiment_secure_signature_controls.py` | Preserved report and current/legacy public-metadata reader |
| BRISQUE / KID / CLIP | `scripts/eval_rebuttal_quality_baselines.py`, companion scripts below | Matched prompt/seed identities, portable method specifications, reference tables |
| Receipt runtime | `sp1/host/examples/timing_campaign.rs` | Current SP1 host/guest, fixture builder, separate timing boundaries |

Generated images and model weights are not bundled. Obtain baseline models
from their [own projects](MODELS.md). Image paths in the included manifests are
locations to populate, not included downloads. Frozen historical results and
new runs must remain distinguishable: new keys, model revisions, libraries and
the corrected post-submission implementation can change the generated pixels
and measured values. This release does not promise byte-identical regeneration
of every historical image.

## Image collections and manifest paths

`data/rebuttal_prompt_seed_pairs_100.jsonl` records the 100 common quality pairs.
Each method has a manifest at `data/quality_inputs/<method>/manifest.jsonl`.
Image and metadata paths resolve relative to that manifest's directory. The
PP-Mark layout uses `images/img_0000/watermarked.png` and the adjacent
`metadata.json`; the other image names encode the prompt and seed. Keep the
same prompt/seed key across methods. Use the current image CLI to produce
PP-Mark image/public-metadata pairs, with the listed prompt and `--seed-hex`
set to the seed encoded as exactly 32 bytes (64 hexadecimal digits, left-padded
with zeroes, without a `0x` prefix).

`data/quality_methods.json` lists these manifests. Its manifest paths resolve
relative to the JSON file itself, so it can also point to external datasets.
`data/quality_methods_1000.json` identifies the separate full rebuttal pool.
Its 1,000 keys contain the common 100 keys and 900 disjoint clean reference keys.
The older `datasets/quality_prompts_sd21_100/` prompt/seed files are preserved
separately; they are not this rebuttal pool.

For image-only quality generation, the existing helper accepts the supplied
clean or PP-Mark manifests. For example:

```bash
python scripts/generate_quality_images.py \
  --manifest data/quality_1000/clean/manifest.jsonl \
  --output-root data/quality_1000/clean \
  --config config_alpha4_k1000.json --secret-file private_keys/producer.hex
```

The manifest sets clean alpha to 0 and PP-Mark alpha to 4. This historical
quality helper writes images only; it does not create receipts or the public
metadata needed by signature and Score/Accept evaluation. The current CLI in
the README supplies that complete artifact flow.

## Quality comparisons

Populate the image collections, then run:

```bash
python scripts/eval_rebuttal_quality_baselines.py \
  --methods-file data/quality_methods.json \
  --output outputs/quality_matched100 --expected-count 100

python scripts/eval_ppmark_quality_full1000.py \
  --methods-file data/quality_methods_1000.json \
  --output outputs/quality_full1000

python scripts/build_rebuttal_quality_tables.py \
  --matched-dir outputs/quality_matched100 \
  --full-pp-dir outputs/quality_full1000 --kid-reference-splits 1000

python scripts/eval_rebuttal_clipscore_baselines.py \
  --manifest outputs/quality_matched100/matched_manifest.jsonl \
  --output outputs/quality_matched100/clipscore
```

The first two scripts extract image features and BRISQUE values. The table
builder uses all 100 method features and resamples the disjoint 900-image clean
pool 1,000 times for the rebuttal baseline KID results. The initial matched-pool
KID summary from the first command uses a different reference protocol; use
`independent_reference_kid.json` for the disjoint-reference comparison.
Slightly negative unbiased KID estimates can occur with finite samples.

The submitted PP-Mark anchor is retained separately: BRISQUE p95 = 34.144476
(100 images) and KID = 1.101 +/- 0.202 x 10^-3 (1,000 images, subset 500,
20 splits). The table builder copies those explicitly marked historical values;
it does not recompute that anchor. The independently measured matched-100
PP-Mark row has different values and remains in `quality_controlled.csv`.
CLIP uses normalized image/text embedding cosine similarity, reported x100.

## Signature controls

This command needs populated clean and PP-Mark images plus the public metadata
corresponding to each PP-Mark original:

```bash
python scripts/experiment_secure_signature_controls.py \
  --positive-manifest data/quality_inputs/pp_mark/manifest.jsonl \
  --clean-manifest data/quality_inputs/clean/manifest.jsonl \
  --limit 100 --output outputs/signature_controls/report.json
```

It tests Ed25519 signatures over the canonical image hash (Bare-Sig), and over
the image hash, context hash and binding (Context-Sig). It checks original
verification, cross-image replay, context substitution, signer-authorized clean
images, and post-signing transforms. These are signature outcomes. The PP-Mark
Score/Accept columns in the rebuttal came from separate PP-Mark evaluations;
the signature script does not fabricate or substitute those results.

## Sequential transformations

The sequential collection is an independent historical image/metadata pairing.
Do not infer its pairing from the quality manifest or treat its legacy
prompt/seed labels as authoritative generation inputs. To use original records,
supply the matching crop images and public metadata. To evaluate newly generated
images, prepare a new manifest and compute both checkpoints on those same images.
Do not combine historical crop scores with new JPEG scores.

For a new run, put `manifest_crop.jsonl`, `crop_boxes.json`, each crop image at
`img_<idx:03d>/crop_075.png`, and its original public metadata at
`metadata/img_<idx:04d>.json` under `outputs/sequential_inputs/`. The included
`data/sequential/` files show the schema and historical boxes. Each box retains
approximately 75% of the original 512 x 512 area; resize the crop back to
512 x 512 with bicubic interpolation. Keep all 100 indices, including failures.

```bash
python scripts/eval_attack_manifest.py \
  --manifest outputs/sequential_inputs/manifest_crop.jsonl \
  --config config_alpha4_k1000.json \
  --image-root outputs/sequential_inputs --metadata-root outputs/sequential_inputs \
  --tau 2.4665623073926706 \
  --ddim-device cuda --ddim-dtype float32 --ddim-steps 100 \
  --max-shift 16 --max-rotation 0 --crop-keep-ratio 0.75 --crop-grid 9 \
  --crop-refine --crop-refine-grid 9 --crop-refine-step 0.03 \
  --skip-decode --skip-consistency \
  --scores-out outputs/sequential_inputs/eval_crop_full100_maxlevers/scores.csv \
  --summary-out outputs/sequential_inputs/eval_crop_full100_maxlevers/summary.json

python scripts/prepare_rebuttal_existing_crop_recompression.py \
  --indices "$(seq -s, 0 99)" --qualities 25 \
  --crop-root outputs/sequential_inputs --metadata-root outputs/sequential_inputs \
  --output-dir outputs/sequential

python scripts/eval_attack_manifest.py \
  --manifest outputs/sequential/manifest_q25.jsonl \
  --config config_alpha4_k1000.json \
  --image-root outputs/sequential --metadata-root . --tau 2.4665623073926706 \
  --ddim-device cuda --ddim-dtype float32 --ddim-steps 100 \
  --max-shift 16 --max-rotation 0 --crop-keep-ratio 0.75 --crop-grid 9 \
  --crop-refine --crop-refine-grid 9 --crop-refine-step 0.03 \
  --skip-decode --skip-consistency \
  --scores-out outputs/sequential/eval_q25/scores.csv \
  --summary-out outputs/sequential/eval_q25/summary.json

python scripts/summarize_persistent_pipeline.py \
  --crop-scores outputs/sequential_inputs/eval_crop_full100_maxlevers/scores.csv \
  --jpeg-scores outputs/sequential/eval_q25/scores.csv \
  --output outputs/sequential/persistent_summary.json
```

JPEG recompression uses quality 25 and 4:2:0 subsampling. Persistent success
requires score >= threshold at **both** checkpoints. The historical count was
80/100, while final-stage-only detection was 91/100. The generic
`summarize_rebuttal_sequential.py` reports individual conditions; it does not
implement this two-checkpoint criterion. `summarize_persistent_pipeline.py`
joins on `idx`, rejects missing/duplicate rows, and reports the separate counts.
It can also summarize the two included historical score CSVs without models.

## SP1 timing

Use the build and input preparation in [sp1/README.md](../sp1/README.md).
With the generated public/witness files and the built streaming guest ELF:

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

Replace `PATH_TO_BUILT_STREAMING_ELF` with the streaming guest ELF from your
build. This campaign explicitly uses core receipts. It records client setup,
execution, proving, serialization and receipt verification separately; the
fixture does not include diffusion generation or inversion. Full image-command
timing and receipt-only timing have different boundaries. Report hardware,
SDK, proof mode, sample/opening counts and those boundaries with new results.

## Preserved results

[`results/reference`](../results/reference) contains compact existing records:
submitted and matched quality CSVs, independent-reference KID, BRISQUE and CLIP
summaries, the quality environment, the signature-control report, and final
JPEG scores. Crop checkpoint scores are under `data/sequential/`.
Machine-specific input paths were removed from the signature/score exports;
their recorded decisions and numerical values were retained. These are
historical records, not outputs of a fresh release-validation run.
