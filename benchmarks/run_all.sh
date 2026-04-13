#!/bin/bash
#
# Benchmark script for value iteration implementations.
# Runs discrete_vi and unified_vi across grid sizes and
# saves timing results to a CSV file.

set -e

# Grid size range
START=${1:-50}
END=${2:-2000}
STEP=${3:-50}

# Output directory
RESULTS_DIR="benchmarks/results"
mkdir -p "$RESULTS_DIR"

# Timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Detect GPU name from either implementation
GPU_NAME="unknown"
if [ -x "./discrete_vi" ]; then
    GPU_NAME=$(./discrete_vi 10 10 2>/dev/null | grep "^GPU:" | sed 's/GPU: //' | tr ' ' '_')
elif [ -x "./unified_vi" ]; then
    GPU_NAME=$(./unified_vi 10 10 2>/dev/null | grep "^GPU:" | sed 's/GPU: //' | tr ' ' '_')
fi

CSV_FILE="${RESULTS_DIR}/benchmark_${GPU_NAME}_${TIMESTAMP}.csv"

echo "=== Value Iteration Benchmark ==="
echo "Grid sizes: ${START} to ${END} (step ${STEP})"
echo "Output: ${CSV_FILE}"
echo ""

# CSV header - Updated for explicit breakdown
echo "implementation,grid_size,num_states,iterations,build_ms,copy_to_gpu_ms,iteration_loop_ms,copy_from_gpu_ms,avg_iter_ms,total_ms" > "$CSV_FILE"

# Function to parse timing from discrete_vi output
parse_discrete() {
    local output="$1"
    local size="$2"
    local num_states=$((size * size))

    local iterations=$(echo "$output" | grep -E "(Converged at iteration|Warning: did not converge)" | head -1 | grep -oP '\d+' | head -1)
    local build=$(echo "$output" | grep "Build time:" | grep -oP '[\d.]+' | head -1)
    local copy_to=$(echo "$output" | grep "Copy to GPU:" | grep -oP '[\d.]+' | head -1)
    local loop=$(echo "$output" | grep "Iteration loop:" | grep -oP '[\d.]+' | head -1)
    local avg_iter=$(echo "$output" | grep "Avg per iteration:" | grep -oP '[\d.]+' | head -1)
    local copy_from=$(echo "$output" | grep "Copy from GPU:" | grep -oP '[\d.]+' | head -1)
    local total=$(echo "$output" | grep "^Total:" | grep -oP '[\d.]+' | head -1)

    echo "discrete,${size},${num_states},${iterations},${build},${copy_to},${loop},${copy_from},${avg_iter},${total}"
}

# Function to parse timing from unified_vi output
parse_unified() {
    local output="$1"
    local size="$2"
    local num_states=$((size * size))
    local impl_name="$3"

    local iterations=$(echo "$output" | grep -E "(Converged at iteration|Warning: did not converge)" | head -1 | grep -oP '\d+' | head -1)
    local build=$(echo "$output" | grep "Build time:" | tail -1 | grep -oP '[\d.]+' | head -1)
    local loop=$(echo "$output" | grep "Iteration loop:" | grep -oP '[\d.]+' | head -1)
    local avg_iter=$(echo "$output" | grep "Avg per iteration:" | grep -oP '[\d.]+' | head -1)
    local total=$(echo "$output" | grep "^Total:" | grep -oP '[\d.]+' | head -1)

    # Unified uses 0.0 for explicit transfers to keep columns perfectly aligned
    echo "${impl_name},${size},${num_states},${iterations},${build},0.0,${loop},0.0,${avg_iter},${total}"
}

# Run benchmarks
for size in $(seq $START $STEP $END); do
    echo "--- Grid size: ${size}x${size} ($(( size * size )) states) ---"

    # Run discrete_vi
    if [ -x "./discrete_vi" ]; then
        echo -n "  discrete_vi... "
        output=$(./discrete_vi $size $size 2>&1)
        result=$(parse_discrete "$output" "$size")
        echo "$result" >> "$CSV_FILE"
        total=$(echo "$result" | cut -d',' -f10)
        echo "done (${total} ms)"
    else
        echo "  discrete_vi not found, skipping"
    fi

    # Run unified_vi
    if [ -x "./unified_vi" ]; then
        echo -n "  unified_vi... "
        output=$(./unified_vi $size $size 2>&1)
        result=$(parse_unified "$output" "$size" "unified")
        echo "$result" >> "$CSV_FILE"
        total=$(echo "$result" | cut -d',' -f10)
        echo "done (${total} ms)"
    else
        echo "  unified_vi not found, skipping"
    fi

    # Run unified_vi_prefetch if available
    if [ -x "./unified_vi_prefetch" ]; then
        echo -n "  unified_vi_prefetch... "
        output=$(./unified_vi_prefetch $size $size 2>&1)
        result=$(parse_unified "$output" "$size" "unified_prefetch")
        echo "$result" >> "$CSV_FILE"
        total=$(echo "$result" | cut -d',' -f10)
        echo "done (${total} ms)"
    fi

    echo ""
done

echo "=== Benchmark complete ==="
echo "Results saved to: ${CSV_FILE}"
echo ""
echo "Preview:"
head -5 "$CSV_FILE"
echo "..."
tail -3 "$CSV_FILE"
