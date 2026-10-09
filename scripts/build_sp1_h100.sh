#!/usr/bin/env bash
set -euo pipefail

# Build helper for the pinned SP1 v6.3.1 host/guest on H100-class GPUs.
# Usage: ./scripts/build_sp1_h100.sh [cpu|cuda]  (default: cuda)

MODE=${1:-cuda}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/sp1"

if [[ "$MODE" != "cpu" && "$MODE" != "cuda" ]]; then
  echo "Usage: $0 [cpu|cuda]" >&2
  exit 1
fi

if ! rustc +succinct --print target-list 2>/dev/null \
  | grep -qx 'riscv64im-succinct-zkvm-elf'; then
  echo "SP1 v6.3.1 guest toolchain is missing." >&2
  echo "Install it with: ~/.sp1/bin/sp1up --version 6.3.1" >&2
  exit 2
fi
if [[ -z "${PROTOC:-}" ]] && ! command -v protoc >/dev/null 2>&1; then
  echo "protoc is required by SP1 v6.3.1 (Ubuntu: apt install protobuf-compiler)." >&2
  exit 2
fi

export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-"$ROOT/target/sp1"}
if [[ -z "${CC:-}" ]]; then
  if command -v gcc-12 >/dev/null 2>&1; then
    export CC="$(command -v gcc-12)"
  else
    export CC="$(command -v gcc)"
  fi
fi
if [[ -z "${CXX:-}" ]]; then
  if command -v g++-12 >/dev/null 2>&1; then
    export CXX="$(command -v g++-12)"
  else
    export CXX="$(command -v g++)"
  fi
fi
export RUSTFLAGS=${RUSTFLAGS:-"-C target-cpu=native"}
FEATURES=()
if [[ "$MODE" == "cuda" ]]; then
  FEATURES+=(--features cuda)
  echo "[sp1] CUDA mode enabled (CC=$CC)"
else
  echo "[sp1] CPU mode build"
fi

cargo build --locked --release -p sp1-genguard-host "${FEATURES[@]}"
echo "[sp1] build complete -> ${CARGO_TARGET_DIR}/release/sp1-genguard-host"
