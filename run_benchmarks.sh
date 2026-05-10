#!/bin/bash

# Usage: ./run_benchmarks.sh <binary> <np> [program arguments...]
# Examples:
#   ./run_benchmarks.sh bin/stencil 64
#   ./run_benchmarks.sh bin/3d_stencil/3d_stencil_torus 128 96 96 512

if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <binary> <np> [program arguments...]"
    exit 1
fi

BINARY=$1
NP=$2
shift 2

FLOPS=5.262Gf
PLATFORM_DIR=platforms/$NP   # look for platform files for this node count
# Optional: RESULTS_SUBDIR=strong → results/strong/<np>/   (avoids clobbering other studies)
if [ -n "${RESULTS_SUBDIR:-}" ]; then
    RESULTS_DIR="results/${RESULTS_SUBDIR}/$NP"
else
    RESULTS_DIR="results/$NP"
fi

if [ ! -f "$BINARY" ]; then
    echo "Error: binary '$BINARY' not found"
    exit 1
fi

if [ ! -d "$PLATFORM_DIR" ]; then
    echo "Error: platform directory '$PLATFORM_DIR' not found"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

BENCHMARK=$(basename "$BINARY")
OUTFILE="$RESULTS_DIR/${BENCHMARK}_results.csv"

# Resume support: if OUTFILE already exists, treat any successfully-recorded
# (non-ERROR) row as "already done" and skip those topologies. ERROR rows
# are dropped on resume so they get retried.
DONE_TOPOLOGIES=""
if [ -f "$OUTFILE" ]; then
    # Treat a row as "done" only if the time is a valid floating-point number
    # (e.g. 0.001234, 1.23e-05). This catches partial regex captures like "e"
    # from runs that crashed mid-output.
    DONE_TOPOLOGIES=$(awk -F, 'NR>1 && $2 ~ /^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$/ {print $1}' "$OUTFILE" | tr '\n' ' ')
    TMPFILE=$(mktemp)
    echo "topology,time_s" > "$TMPFILE"
    awk -F, 'NR>1 && $2 ~ /^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$/ {print}' "$OUTFILE" >> "$TMPFILE"
    mv "$TMPFILE" "$OUTFILE"
else
    echo "topology,time_s" > "$OUTFILE"
fi

echo "Running $BENCHMARK across all topologies with $NP ranks..."
if [ "$#" -gt 0 ]; then
    echo "(extra program args: $*)"
fi
if [ -n "$DONE_TOPOLOGIES" ]; then
    echo "(resuming; already done: $DONE_TOPOLOGIES)"
fi
echo "-----------------------------------------------------------"

for PLATFORM in "$PLATFORM_DIR"/*.xml; do
    TOPOLOGY=$(basename "$PLATFORM" .xml)

    # Skip if already recorded for this CSV
    if echo " $DONE_TOPOLOGIES " | grep -q " $TOPOLOGY "; then
        printf "  %-20s ... (skip; already in CSV)\n" "$TOPOLOGY"
        continue
    fi

    printf "  %-20s ... " "$TOPOLOGY"
    OUTPUT=$(smpirun -np "$NP" \
                     -platform "$PLATFORM" \
                     --cfg=smpi/host-speed:auto \
                     --cfg=smpi/coll-selector:ompi \
                     "$BINARY" "$@" 2>&1)

    TIME=$(echo "$OUTPUT" | grep -oP '[\d.e+-]+(?= s)' | tail -1)

    if [ -z "$TIME" ]; then
        echo "FAILED"
        echo "$TOPOLOGY,ERROR" >> "$OUTFILE"
        echo "    Output was: $OUTPUT"
    else
        echo "$TIME s"
        echo "$TOPOLOGY,$TIME" >> "$OUTFILE"
    fi
done

echo "-----------------------------------------------------------"
echo "Results saved to $OUTFILE"