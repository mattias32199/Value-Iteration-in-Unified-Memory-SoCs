// unified_vi.cu

/*
 * Unified memory CUDA value iteration.
 *
 * Uses cudaMallocManaged for all allocations:
 *   - Single pointer accessible by both CPU and GPU
 *   - No explicit cudaMemcpy transfers
 *   - CPU builds grid world directly in managed memory
 *   - Same Bellman update and reduction kernels as discrete version
 *
 * On discrete GPU (e.g., Colab T4/L4): the CUDA runtime migrates
 * pages on demand via page faults. Performance depends on access
 * patterns and driver-level migration.
 *
 * On unified memory SoC (e.g., Jetson Orin Nano): CPU and GPU
 * share the same physical memory. No migration needed, just
 * cache coherency management.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifndef USE_PREFETCH
#define USE_PREFETCH 0
#endif

extern "C" {
#include "gridworld.h"
}

/* ----------------------------------------------------------------
 * Error checking macro
 * ---------------------------------------------------------------- */
#define CUDA_CHECK(call)                                                  \
    do {                                                                  \
        cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

/* ----------------------------------------------------------------
 * Bellman update kernel (identical to discrete version)
 * ---------------------------------------------------------------- */
__global__ void bellman_update_kernel(
    const int *transitions,
    const float *rewards,
    const int8_t *is_terminal,
    const int8_t *is_obstacle,
    const float *V_curr,
    float *V_next,
    float *deltas,
    int *policy,
    int num_states,
    float gamma)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_states) return;

    if (is_terminal[s] || is_obstacle[s]) {
        V_next[s] = 0.0f;
        deltas[s] = 0.0f;
        policy[s] = 0;
        return;
    }

    float best_value = -FLT_MAX;
    int best_action = 0;

    #pragma unroll
    for (int a = 0; a < NUM_ACTIONS; a++) {
        int idx = s * NUM_ACTIONS + a;
        int next_state = transitions[idx];
        float reward = rewards[idx];

        float q_value = reward + gamma * V_curr[next_state];

        if (q_value > best_value) {
            best_value = q_value;
            best_action = a;
        }
    }

    V_next[s] = best_value;
    policy[s] = best_action;
    deltas[s] = fabsf(best_value - V_curr[s]);
}

/* ----------------------------------------------------------------
 * Parallel max reduction kernel (identical to discrete version)
 * ---------------------------------------------------------------- */
__global__ void max_reduce_kernel(
    const float *input,
    float *output,
    int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float val = 0.0f;
    if (i < n) val = input[i];
    if (i + blockDim.x < n) val = fmaxf(val, input[i + blockDim.x]);
    sdata[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/* ----------------------------------------------------------------
 * Host-side max reduction
 * ---------------------------------------------------------------- */
float gpu_max_reduce(float *d_input, float *d_workspace, int n) {
    int threads = 256;
    float h_result;

    float *src = d_input;
    float *dst = d_workspace;

    while (n > 1) {
        int blocks = (n + (threads * 2) - 1) / (threads * 2);
        max_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(
            src, dst, n);
        CUDA_CHECK(cudaGetLastError());

        n = blocks;

        float *tmp = src;
        src = dst;
        dst = tmp;
    }

    /*
     * In unified memory, we could read *src directly on the CPU
     * after a cudaDeviceSynchronize(). Using cudaMemcpy here for
     * consistency with the discrete version — it also acts as
     * an implicit sync. On the Jetson, you could instead do:
     *   cudaDeviceSynchronize();
     *   return *src;
     */
    CUDA_CHECK(cudaMemcpy(&h_result, src, sizeof(float),
                          cudaMemcpyDeviceToHost));
    return h_result;
}

/* ----------------------------------------------------------------
 * Build grid world directly into managed memory.
 *
 * Instead of building in host memory and copying to device,
 * we allocate managed memory first and have the CPU write
 * the grid world data directly into it.
 * ---------------------------------------------------------------- */
typedef struct {
    int *transitions;
    float *rewards;
    int8_t *is_terminal;
    int8_t *is_obstacle;
    int num_states;
    int rows;
    int cols;
} ManagedGridWorld;

ManagedGridWorld build_managed_gridworld(int rows, int cols,
                                         float step_reward,
                                         float goal_reward) {
    ManagedGridWorld mgw;
    mgw.rows = rows;
    mgw.cols = cols;
    mgw.num_states = rows * cols;
    int N = mgw.num_states;

    size_t trans_size  = (size_t)N * NUM_ACTIONS * sizeof(int);
    size_t reward_size = (size_t)N * NUM_ACTIONS * sizeof(float);
    size_t flag_size   = (size_t)N * sizeof(int8_t);

    /* Allocate all grid world data in managed memory */
    CUDA_CHECK(cudaMallocManaged(&mgw.transitions, trans_size));
    CUDA_CHECK(cudaMallocManaged(&mgw.rewards, reward_size));
    CUDA_CHECK(cudaMallocManaged(&mgw.is_terminal, flag_size));
    CUDA_CHECK(cudaMallocManaged(&mgw.is_obstacle, flag_size));

    /* Zero out the flag arrays using cudaMemset for managed memory */
    CUDA_CHECK(cudaMemset(mgw.is_terminal, 0, flag_size));
    CUDA_CHECK(cudaMemset(mgw.is_obstacle, 0, flag_size));
    CUDA_CHECK(cudaDeviceSynchronize());

    /*
     * Reuse the existing gridworld functions by creating a
     * temporary GridWorld struct that points to managed memory.
     * The CPU writes directly into the managed allocations.
     */
    GridWorld tmp;
    tmp.rows = rows;
    tmp.cols = cols;
    tmp.num_states = N;
    tmp.step_reward = step_reward;
    tmp.goal_reward = goal_reward;
    tmp.transitions = mgw.transitions;
    tmp.rewards = mgw.rewards;
    tmp.is_terminal = mgw.is_terminal;
    tmp.is_obstacle = mgw.is_obstacle;

    gridworld_default_layout(&tmp);
    gridworld_build_transitions(&tmp);

    return mgw;
}

/* ----------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------- */
int main(int argc, char **argv) {
    int rows = 1000;
    int cols = 1000;
    float gamma_val = 0.99f;
    float threshold = 1e-4f;
    int max_iters = 10000;

    if (argc >= 3) {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
    }
    if (argc >= 4) gamma_val = atof(argv[3]);
    if (argc >= 5) threshold = atof(argv[4]);

    int N = rows * cols;

    printf("=== Unified Memory CUDA Value Iteration ===\n");
    printf("Grid: %dx%d (%d states)\n", rows, cols, N);
    printf("Gamma: %.4f\n", gamma_val);
    printf("Threshold: %.1e\n", threshold);
    printf("Max iterations: %d\n\n", max_iters);

    /* Print GPU info */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Global memory: %.0f MB\n", prop.totalGlobalMem / (1024.0 * 1024.0));
    printf("Managed memory: %s\n", prop.managedMemory ? "yes" : "no");
    printf("Concurrent managed access: %s\n\n",
           prop.concurrentManagedAccess ? "yes" : "no");

    /* --------------------------------------------------------
     * Build grid world directly in managed memory
     * -------------------------------------------------------- */
    printf("Building grid world in managed memory...\n");

    cudaEvent_t build_start, build_end;
    CUDA_CHECK(cudaEventCreate(&build_start));
    CUDA_CHECK(cudaEventCreate(&build_end));

    CUDA_CHECK(cudaEventRecord(build_start));

    ManagedGridWorld mgw = build_managed_gridworld(rows, cols, -1.0f, 0.0f);

    CUDA_CHECK(cudaEventRecord(build_end));
    CUDA_CHECK(cudaEventSynchronize(build_end));

    float build_ms;
    CUDA_CHECK(cudaEventElapsedTime(&build_ms, build_start, build_end));

    int n_obstacles = 0, n_terminals = 0;
    for (int i = 0; i < N; i++) {
        n_obstacles += mgw.is_obstacle[i];
        n_terminals += mgw.is_terminal[i];
    }
    printf("Grid world ready (%d obstacles, %d terminals)\n",
           n_obstacles, n_terminals);
    printf("Build time: %.3f ms\n\n", build_ms);

    /* --------------------------------------------------------
     * Allocate value iteration arrays in managed memory
     * -------------------------------------------------------- */
    float *V_curr, *V_next;
    float *deltas, *reduce_workspace;
    int *policy;

    size_t value_size  = (size_t)N * sizeof(float);
    size_t policy_size = (size_t)N * sizeof(int);

    int reduce_blocks = (N + 511) / 512;
    size_t reduce_size = reduce_blocks * sizeof(float);

    CUDA_CHECK(cudaMallocManaged(&V_curr, value_size));
    CUDA_CHECK(cudaMallocManaged(&V_next, value_size));
    CUDA_CHECK(cudaMallocManaged(&deltas, value_size));
    CUDA_CHECK(cudaMallocManaged(&reduce_workspace, reduce_size));
    CUDA_CHECK(cudaMallocManaged(&policy, policy_size));

    CUDA_CHECK(cudaMemset(V_curr, 0, value_size));
    CUDA_CHECK(cudaMemset(V_next, 0, value_size));

    size_t total_managed = (size_t)N * NUM_ACTIONS * sizeof(int)
                         + (size_t)N * NUM_ACTIONS * sizeof(float)
                         + (size_t)N * 2 * sizeof(int8_t)
                         + 3 * value_size
                         + reduce_size + policy_size;
    printf("Total managed memory: %.2f MB\n\n",
           total_managed / (1024.0 * 1024.0));

    /* --------------------------------------------------------
     * Optional: prefetch managed memory to GPU.
     *
     * On discrete GPUs, this avoids page-fault storms on the
     * first iteration by migrating data before the loop starts.
     * On Jetson (unified SoC), this is essentially a no-op
     * since CPU and GPU share the same physical memory.
     *
     * Set USE_PREFETCH=1 at compile time to enable:
     *   nvcc -DUSE_PREFETCH=1 ...
     * -------------------------------------------------------- */
#if USE_PREFETCH
    {
        int device;
        CUDA_CHECK(cudaGetDevice(&device));

        #if CUDART_VERSION >= 13000
                cudaMemLocation loc = {};
                loc.type = cudaMemLocationTypeDevice;
                loc.id = device;
                // On new CUDA, PREFETCH_DEST turns into: loc, 0
                #define PREFETCH_DEST loc, 0
            #else
                // On old CUDA, PREFETCH_DEST turns into: device
                #define PREFETCH_DEST device
        #endif

        size_t trans_size  = (size_t)N * NUM_ACTIONS * sizeof(int);
        size_t reward_size = (size_t)N * NUM_ACTIONS * sizeof(float);
        size_t flag_size   = (size_t)N * sizeof(int8_t);

        printf("Prefetching managed memory to GPU...\n");
        // CUDA_CHECK(cudaMemPrefetchAsync(mgw.transitions, trans_size, device));
        // CUDA_CHECK(cudaMemPrefetchAsync(mgw.rewards, reward_size, device));
        // CUDA_CHECK(cudaMemPrefetchAsync(mgw.is_terminal, flag_size, device));
        // CUDA_CHECK(cudaMemPrefetchAsync(mgw.is_obstacle, flag_size, device));
        // CUDA_CHECK(cudaMemPrefetchAsync(V_curr, value_size, device));
        // CUDA_CHECK(cudaMemPrefetchAsync(V_next, value_size, device));
        // CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemPrefetchAsync(mgw.transitions, trans_size, PREFETCH_DEST));
        CUDA_CHECK(cudaMemPrefetchAsync(mgw.rewards, reward_size, PREFETCH_DEST));
        CUDA_CHECK(cudaMemPrefetchAsync(mgw.is_terminal, flag_size, PREFETCH_DEST));
        CUDA_CHECK(cudaMemPrefetchAsync(mgw.is_obstacle, flag_size, PREFETCH_DEST));
        CUDA_CHECK(cudaMemPrefetchAsync(V_curr, value_size, PREFETCH_DEST));
        CUDA_CHECK(cudaMemPrefetchAsync(V_next, value_size, PREFETCH_DEST));
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("Prefetch complete.\n\n");
    }
#endif

    /* --------------------------------------------------------
     * Value iteration loop
     * -------------------------------------------------------- */
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    cudaEvent_t loop_start, loop_end;
    CUDA_CHECK(cudaEventCreate(&loop_start));
    CUDA_CHECK(cudaEventCreate(&loop_end));

    printf("Running value iteration...\n");
    CUDA_CHECK(cudaEventRecord(loop_start));

    int iter;
    float max_delta;
    float *V_final = V_next;

    for (iter = 0; iter < max_iters; iter++) {

        bellman_update_kernel<<<blocks, threads_per_block>>>(
            mgw.transitions, mgw.rewards,
            mgw.is_terminal, mgw.is_obstacle,
            V_curr, V_next,
            deltas, policy,
            N, gamma_val);
        CUDA_CHECK(cudaGetLastError());

        max_delta = gpu_max_reduce(deltas, reduce_workspace, N);

        if ((iter + 1) % 100 == 0 || iter == 0) {
            printf("  Iteration %d: max_delta = %.8f\n",
                   iter + 1, max_delta);
        }

        if (max_delta < threshold) {
            printf("  Converged at iteration %d (max_delta = %.8f)\n",
                   iter + 1, max_delta);
            V_final = V_next;
            break;
        }

        float *tmp = V_curr;
        V_curr = V_next;
        V_next = tmp;
        V_final = V_curr;
    }

    CUDA_CHECK(cudaEventRecord(loop_end));
    CUDA_CHECK(cudaEventSynchronize(loop_end));

    float loop_ms;
    CUDA_CHECK(cudaEventElapsedTime(&loop_ms, loop_start, loop_end));

    if (iter == max_iters) {
        printf("  Warning: did not converge after %d iterations\n",
               max_iters);
    }

    int total_iters = (iter < max_iters) ? iter + 1 : max_iters;

    /* --------------------------------------------------------
     * Results are already accessible on the CPU — no copy needed.
     * Just synchronize to ensure GPU kernels are done.
     * -------------------------------------------------------- */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* --------------------------------------------------------
     * Print results
     * -------------------------------------------------------- */
    printf("\n=== Results ===\n");
    printf("Iterations: %d\n", total_iters);
    printf("Build time: %.3f ms\n", build_ms);
    printf("Iteration loop: %.2f ms\n", loop_ms);
    printf("Avg per iteration: %.3f ms\n", loop_ms / total_iters);
    printf("Total: %.2f ms\n", build_ms + loop_ms);
    printf("(No explicit copy to/from GPU)\n");

    /* Print policy near goal */
    const char *arrows = "^v<>";
    int preview = (rows < 20) ? rows : 20;
    int pcols   = (cols < 20) ? cols : 20;

    printf("\nPolicy (bottom-right corner near goal):\n");
    int br0 = (rows > preview) ? rows - preview : 0;
    int bc0 = (cols > pcols)   ? cols - pcols   : 0;
    printf("     ");
    for (int c = bc0; c < cols; c++) {
        if ((c - bc0) % 5 == 0) printf("%-5d", c);
    }
    printf("\n");
    for (int r = br0; r < rows; r++) {
        printf("%4d ", r);
        for (int c = bc0; c < cols; c++) {
            int s = r * cols + c;
            if (mgw.is_terminal[s])       printf("G");
            else if (mgw.is_obstacle[s])  printf("#");
            else                          printf("%c", arrows[policy[s]]);
        }
        printf("\n");
    }

    /* Print values near goal */
    printf("\nValues (bottom-right corner near goal):\n");
    int vr = (rows < 8) ? rows : 8;
    int vc = (cols < 8) ? cols : 8;
    for (int r = rows - vr; r < rows; r++) {
        printf("%4d |", r);
        for (int c = cols - vc; c < cols; c++) {
            int s = r * cols + c;
            if (mgw.is_obstacle[s])
                printf("  ####  ");
            else
                printf(" %6.1f ", V_final[s]);
        }
        printf("\n");
    }

    /* Save value function for comparison */
    char filename[256];
    snprintf(filename, sizeof(filename),
             "unified_values_%dx%d.bin", rows, cols);
    FILE *fp = fopen(filename, "wb");
    if (fp) {
        fwrite(V_final, sizeof(float), N, fp);
        fclose(fp);
        printf("\nValue function saved to %s\n", filename);
    }

    /* --------------------------------------------------------
     * Cleanup
     * -------------------------------------------------------- */
    CUDA_CHECK(cudaFree(mgw.transitions));
    CUDA_CHECK(cudaFree(mgw.rewards));
    CUDA_CHECK(cudaFree(mgw.is_terminal));
    CUDA_CHECK(cudaFree(mgw.is_obstacle));
    CUDA_CHECK(cudaFree(V_curr));
    CUDA_CHECK(cudaFree(V_next));
    CUDA_CHECK(cudaFree(deltas));
    CUDA_CHECK(cudaFree(reduce_workspace));
    CUDA_CHECK(cudaFree(policy));

    CUDA_CHECK(cudaEventDestroy(build_start));
    CUDA_CHECK(cudaEventDestroy(build_end));
    CUDA_CHECK(cudaEventDestroy(loop_start));
    CUDA_CHECK(cudaEventDestroy(loop_end));

    printf("Done.\n");
    return 0;
}
