#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
  printf 'Usage: %s OUTPUT_DIR JPEG_QUALITY [GPU_COUNT]\n' "$0" >&2
  exit 2
fi

OUT="$1"
QUALITY="$2"
GPU_COUNT="${3:-2}"
PYTHON="${PPMARK_PYTHON:-python}"
TAU="2.4665623073926706"

run_shard() {
  local shard="$1"
  local gpu="$2"
  local result="$OUT/eval_q${QUALITY}/$shard"
  mkdir -p "$result"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/eval_attack_manifest.py \
    --manifest "$OUT/manifest_q${QUALITY}_${shard}.jsonl" \
    --config config_alpha4_k1000.json \
    --image-root "$OUT" \
    --metadata-root "$ROOT" \
    --tau "$TAU" \
    --ddim-device cuda --ddim-dtype float32 --ddim-steps 100 \
    --max-shift 16 --max-rotation 0 \
    --crop-keep-ratio 0.75 --crop-grid 9 \
    --crop-refine --crop-refine-grid 9 --crop-refine-step 0.03 \
    --skip-decode --skip-consistency \
    --scores-out "$result/scores.csv" \
    --summary-out "$result/summary.json" \
    >"$result/run.log" 2>&1
}

test -f "$OUT/PREPARED"

pids=()
inputs=()
for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
  shard="gpu${gpu}"
  test -s "$OUT/manifest_q${QUALITY}_${shard}.jsonl"
  run_shard "$shard" "$gpu" &
  pids+=("$!")
  inputs+=("$OUT/eval_q${QUALITY}/$shard/scores.csv")
done

status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=1
done
if [[ "$status" -ne 0 ]]; then
  tail -n 30 "$OUT"/eval_q${QUALITY}/gpu*/run.log >&2 || true
  exit "$status"
fi

"$PYTHON" scripts/merge_score_csv.py \
  --inputs "${inputs[@]}" \
  --output "$OUT/eval_q${QUALITY}/scores.csv"

printf 'Recompression extension completed: q%s on %s GPUs.\n' "$QUALITY" "$GPU_COUNT"
