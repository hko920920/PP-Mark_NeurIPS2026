#!/usr/bin/env bash
# White-box PGD forgery attack on the PP-Mark Score detector.
#
# Usage, from the repository root:
#   bash scripts/run_wb_pgd_ppmark.sh MANIFEST OUTPUT_DIR sd21|sdxl [STEPS] [MAX_IMAGES]
#
# Reported configuration: eps = 2/255, 150 steps, 100 images. Gradients flow
# through DDIM inversion, so expect hours to days on a single GPU; the run is
# restart-safe because completed images are skipped unless --overwrite is given.
set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 5 ]]; then
  printf 'Usage: %s MANIFEST OUTPUT_DIR sd21|sdxl [STEPS] [MAX_IMAGES]\n' "$0" >&2
  exit 2
fi

MANIFEST="$1"
OUT="$2"
MODEL="$3"
STEPS="${4:-150}"
MAX_IMAGES="${5:-100}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
PYTHON="${PPMARK_PYTHON:-python}"

case "$MODEL" in
  sd21)
    CONFIG="config_alpha4_k1000.json"
    THRESHOLD="tau_fpr1_alpha4_sd21.json"
    MODEL_ID="Manojb/stable-diffusion-2-1-base"
    DDIM_DTYPE="float32"
    DDIM_STEPS=50
    EXTRA_FLAGS=()
    ;;
  sdxl)
    CONFIG="config_alpha4_k1000_sdxl.json"
    THRESHOLD="tau_fpr1_alpha4_sdxl.json"
    MODEL_ID="stabilityai/stable-diffusion-xl-base-1.0"
    # float16 silently collapses the recovered SDXL latent to zero.
    DDIM_DTYPE="bfloat16"
    DDIM_STEPS=50
    EXTRA_FLAGS=(--ddim-vae-slicing --ddim-grad-checkpointing)
    ;;
  *)
    printf 'Unknown model: %s (expected sd21 or sdxl)\n' "$MODEL" >&2
    exit 2
    ;;
esac

test -s "$MANIFEST"
test -s "$CONFIG"
test -s "$THRESHOLD"
mkdir -p "$OUT"

EPS="$("$PYTHON" -c 'print(2/255)')"
MANIFEST_DIR="$(cd "$(dirname "$MANIFEST")" && pwd)"

printf 'PP-Mark white-box PGD: model=%s steps=%s eps=%s images=%s\n' \
  "$MODEL" "$STEPS" "$EPS" "$MAX_IMAGES"

"$PYTHON" -u scripts/attack_whitebox_forge.py \
  --manifest "$MANIFEST" \
  --output-root "$OUT" \
  --config "$CONFIG" \
  --threshold-file "$THRESHOLD" \
  --image-root "$MANIFEST_DIR" \
  --metadata-root "$ROOT" \
  --ddim-model-id "$MODEL_ID" \
  --ddim-device cuda --ddim-dtype "$DDIM_DTYPE" --ddim-steps "$DDIM_STEPS" \
  "${EXTRA_FLAGS[@]}" \
  --steps "$STEPS" --eps "$EPS" \
  --max-images "$MAX_IMAGES" \
  --trace-all \
  --trace-path "$OUT/trace_scores.csv" \
  --metrics-out "$OUT/metrics.csv" \
  --summary-out "$OUT/summary.json" \
  --manifest-out "$OUT/manifest_forged.jsonl"

printf '\nSummary:\n'
"$PYTHON" - "$OUT/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
for key in ("count", "tau", "far_any_step", "avg_steps_to_fa", "score_max_mean", "psnr_mean"):
    print(f"  {key:16} {summary.get(key)}")
PY

printf '\nForged images carry no valid receipt. Confirm Accept still rejects them:\n'
printf '  python -m ppmark_v03.cli verifier --config %s --mode attest ...\n' "$CONFIG"
