#!/bin/bash
#
# Run strong and weak scaling benchmarks for every sample_sort binary by
# calling run_benchmarks.sh for each (binary, np) pair.
#
# Strong scaling: fixed total_N (default 1<<24 = 16M) for every np.
# Weak scaling:   total_N(np) = WEAK_PER_RANK * np   (default WEAK_PER_RANK=1<<20).
#
# CSV output layout (mirrors run_3d_stencil_scaling.sh):
#   results/strong/<np>/<basename>_results.csv
#   results/weak/<np>/<basename>_results.csv
#
# Requires: run_benchmarks.sh in the same directory; binaries under
#           bin/sample_sort/{agnostic,optimizations}/; platforms/<np>/*.xml.
# Missing binaries are skipped — so this script is usable now (with only the
# agnostic baseline) and "just picks up" optimization variants as you add them.
#
# Usage: ./run_sample_sort_scaling.sh [strong|weak|all]
#        Default: all. Override sizes with env vars STRONG_N / WEAK_PER_RANK.
#
# Loop order: by np first, then binary — at each process count, run all
# variants before increasing np.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUN_BENCH="$SCRIPT_DIR/run_benchmarks.sh"
BIN_ROOT="bin/sample_sort"

# Relative paths under $BIN_ROOT. Add to this list as you implement variants.
SAMPLE_BINS=(
    "agnostic/sample_sort"
    "optimizations/sample_sort_ring"
    "optimizations/sample_sort_torus"
    "optimizations/sample_sort_hypercube"
    "optimizations/sample_sort_fat_tree"
    "optimizations/sample_sort_dragonfly"
)

NP_LIST=(16 32 64 128 256)

STRONG_N="${STRONG_N:-$((1 << 24))}"          # ~16M keys
WEAK_PER_RANK="${WEAK_PER_RANK:-$((1 << 20))}" # ~1M keys per rank

run_strong() {
    export RESULTS_SUBDIR=strong
    echo "======== Strong scaling (TOTAL_N=$STRONG_N for all np) ========"
    echo "Batches: ${#NP_LIST[@]} np × ${#SAMPLE_BINS[@]} binaries (skipped if not built)."
    for np in "${NP_LIST[@]}"; do
        for relbin in "${SAMPLE_BINS[@]}"; do
            local path="$BIN_ROOT/$relbin"
            if [ ! -f "$path" ]; then
                echo "Skip missing binary: $path" >&2
                continue
            fi
            echo ""
            echo "--- STRONG  np=$np  $relbin  total_N=$STRONG_N ---"
            bash "$RUN_BENCH" "$path" "$np" "$STRONG_N"
        done
    done
    unset RESULTS_SUBDIR
}

run_weak() {
    export RESULTS_SUBDIR=weak
    echo "======== Weak scaling (PER_RANK=$WEAK_PER_RANK keys; total = per_rank*np) ========"
    echo "Batches: ${#NP_LIST[@]} np × ${#SAMPLE_BINS[@]} binaries (skipped if not built)."
    for np in "${NP_LIST[@]}"; do
        local total_N=$((WEAK_PER_RANK * np))
        for relbin in "${SAMPLE_BINS[@]}"; do
            local path="$BIN_ROOT/$relbin"
            if [ ! -f "$path" ]; then
                echo "Skip missing binary: $path" >&2
                continue
            fi
            echo ""
            echo "--- WEAK  np=$np  $relbin  total_N=$total_N ---"
            bash "$RUN_BENCH" "$path" "$np" "$total_N"
        done
    done
    unset RESULTS_SUBDIR
}

if [ ! -f "$RUN_BENCH" ]; then
    echo "Error: $RUN_BENCH not found" >&2
    exit 1
fi

MODE="${1:-all}"
case "$MODE" in
    strong) run_strong ;;
    weak)   run_weak ;;
    all)
        echo "Running BOTH studies: strong first, then weak. For only weak: $0 weak"
        echo ""
        run_strong
        echo ""
        echo "======== Strong finished; starting weak scaling ========"
        echo ""
        run_weak
        ;;
    *)
        echo "Usage: $0 [strong|weak|all]" >&2
        exit 1
        ;;
esac

echo ""
echo "Done. Results under results/strong/<np>/ and results/weak/<np>/ (when both ran)."
