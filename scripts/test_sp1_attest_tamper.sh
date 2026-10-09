#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${PPMARK_SP1_HOST:-$ROOT/target/sp1/release/sp1-genguard-host}"

if [[ ! -x "$HOST" ]]; then
  echo "SP1 release host not found: $HOST" >&2
  exit 2
fi

tamper_cases=(
  p1-binding
  p1-key
  p2-root
  p2-index
  p3-image
  p3-opening
  p4-codeword
  p4-bit
  p4-gaussian
  p4-lut
)

cd "$ROOT"
for tamper in "${tamper_cases[@]}"; do
  fixture="$(mktemp -d /tmp/ppmark_attest_tamper.XXXXXX)"
  PYTHONPATH=src python3 scripts/build_sp1_attest_fixture.py \
    --output "$fixture" --tamper "$tamper" --commitment-scheme streaming >/dev/null
  if SP1_PROVER=cpu "$HOST" execute \
    --public "$fixture/attestation_statement.json" \
    --witness "$fixture/private_witness.json" \
    >"$fixture/execute.log" 2>&1; then
    echo "UNEXPECTED_ACCEPT $tamper"
    exit 1
  fi
  reason="$(grep -Eo 'P[1-4]: [^\"]+' "$fixture/execute.log" | head -1 || true)"
  echo "REJECTED $tamper :: ${reason:-constraint failure}"
done
