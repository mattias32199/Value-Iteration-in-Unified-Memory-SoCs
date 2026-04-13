// discrete_vi.cu
/*
 * Discrete-memory CUDA value iteration (baseline).
 *
 * Standard GPU-accelerated approach:
 *   - Host allocates with malloc, device allocates with cudaMalloc
 *   - Explicit cudaMemcpy to transfer data between CPU and GPU
 *   - Bellman update kernel: one thread per state
 *   - Parallel reduction kernel: computes max |V_new - V_old|
 *   - Convergence check on CPU after each sweep
 *
 * This is the conventional approach that the unified memory
 * version will be compared against.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

/* Include the grid world as C code */
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
 * Bellman update kernel
 *
 * Each thread processes one state. Computes the best Q-value
 * across all actions and writes the result to V_next.
 * Also computes |V_next[s] - V_curr[s]| into a delta array.
 * ---------------------------------------------------------------- */
__global__ void bellman_update_kernel(
    const int *transitions,     /* [num_states * NUM_ACTIONS] */
    const float *rewards,       /* [num_states * NUM_ACTIONS] */
    const int8_t *is_terminal,
    const int8_t *is_obstacle,
    const float *V_curr,
    float *V_next,
    float *deltas,              /* per-state delta for reduction */
    int *policy,
    int num_states,
    float gamma)
{
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_states) return;

    /* Terminal and obstacle states stay at 0 */
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
 * Parallel max reduction kernel
 *
 * Reduces the deltas array to find the maximum value.
 * Uses shared memory and sequential addressing for efficiency.
 * Each block reduces its portion, then we reduce across blocks.
 * ---------------------------------------------------------------- */
__global__ void max_reduce_kernel(
    const float *input,
    float *output,
    int n)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    /* Load two elements per thread into shared memory */
    float val = 0.0f;
    if (i < n) val = input[i];
    if (i + blockDim.x < n) val = fmaxf(val, input[i + blockDim.x]);
    sdata[tid] = val;
    __syncthreads();

    /* Reduction in shared memory */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    /* Write block result */
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

/* ----------------------------------------------------------------
 * Host-side max reduction
 *
 * Launches reduction kernels iteratively until we have a single
 * value, then copies it back to the host.
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

        /* Swap src and dst for next iteration */
        float *tmp = src;
        src = dst;
        dst = tmp;
    }

    /* Copy the single result back to host */
    CUDA_CHECK(cudaMemcpy(&h_result, src, sizeof(float),
                          cudaMemcpyDeviceToHost));
    return h_result;
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

    printf("=== Discrete Memory CUDA Value Iteration ===\n");
    printf("Grid: %dx%d (%d states)\n", rows, cols, N);
    printf("Gamma: %.4f\n", gamma_val);
    printf("Threshold: %.1e\n", threshold);
    printf("Max iterations: %d\n\n", max_iters);

    /* Print GPU info */
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Global memory: %.0f MB\n\n",
           prop.totalGlobalMem / (1024.0 * 1024.0));

    /* --------------------------------------------------------
     * Build grid world on the CPU
     * -------------------------------------------------------- */
    printf("Building grid world...\n");
    GridWorld *gw = gridworld_create(rows, cols, -1.0f, 0.0f);
    if (!gw) return 1;
    gridworld_default_layout(gw);
    gridworld_build_transitions(gw);

    int n_obstacles = 0, n_terminals = 0;
    for (int i = 0; i < N; i++) {
        n_obstacles += gw->is_obstacle[i];
        n_terminals += gw->is_terminal[i];
    }
    printf("Grid world ready (%d obstacles, %d terminals)\n\n",
           n_obstacles, n_terminals);

    /* --------------------------------------------------------
     * Allocate host memory
     * -------------------------------------------------------- */
    float *h_values = (float *)calloc(N, sizeof(float));
    int *h_policy   = (int *)calloc(N, sizeof(int));
    if (!h_values || !h_policy) {
        fprintf(stderr, "Host allocation failed\n");
        free(h_values);
        free(h_policy);
        gridworld_free(gw);
        return 1;
    }

    /* --------------------------------------------------------
     * Allocate device memory
     * -------------------------------------------------------- */
    int *d_transitions;
    float *d_rewards;
    int8_t *d_is_terminal, *d_is_obstacle;
    float *d_V_curr, *d_V_next;
    float *d_deltas, *d_reduce_workspace;
    int *d_policy;

    size_t trans_size   = (size_t)N * NUM_ACTIONS * sizeof(int);
    size_t reward_size  = (size_t)N * NUM_ACTIONS * sizeof(float);
    size_t flag_size    = (size_t)N * sizeof(int8_t);
    size_t value_size   = (size_t)N * sizeof(float);
    size_t policy_size  = (size_t)N * sizeof(int);

    /* Workspace for reduction: needs at most N/2 floats */
    int reduce_blocks = (N + 511) / 512;
    size_t reduce_size = reduce_blocks * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_transitions, trans_size));
    CUDA_CHECK(cudaMalloc(&d_rewards, reward_size));
    CUDA_CHECK(cudaMalloc(&d_is_terminal, flag_size));
    CUDA_CHECK(cudaMalloc(&d_is_obstacle, flag_size));
    CUDA_CHECK(cudaMalloc(&d_V_curr, value_size));
    CUDA_CHECK(cudaMalloc(&d_V_next, value_size));
    CUDA_CHECK(cudaMalloc(&d_deltas, value_size));
    CUDA_CHECK(cudaMalloc(&d_reduce_workspace, reduce_size));
    CUDA_CHECK(cudaMalloc(&d_policy, policy_size));

    printf("Device memory allocated: %.2f MB\n",
           (trans_size + reward_size + 2 * flag_size +
            3 * value_size + reduce_size + policy_size) / (1024.0 * 1024.0));

    /* --------------------------------------------------------
     * Copy data from host to device (initial transfer)
     * -------------------------------------------------------- */
    cudaEvent_t copy_start, copy_end;
    CUDA_CHECK(cudaEventCreate(&copy_start));
    CUDA_CHECK(cudaEventCreate(&copy_end));

    CUDA_CHECK(cudaEventRecord(copy_start));

    CUDA_CHECK(cudaMemcpy(d_transitions, gw->transitions, trans_size,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rewards, gw->rewards, reward_size,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_is_terminal, gw->is_terminal, flag_size,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_is_obstacle, gw->is_obstacle, flag_size,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V_curr, h_values, value_size,
                          cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(copy_end));
    CUDA_CHECK(cudaEventSynchronize(copy_end));

    float copy_to_gpu_ms;
    CUDA_CHECK(cudaEventElapsedTime(&copy_to_gpu_ms, copy_start, copy_end));
    printf("Initial copy to GPU: %.3f ms\n\n", copy_to_gpu_ms);

    /* --------------------------------------------------------
     * Value iteration loop
     * -------------------------------------------------------- */
    int threads_per_block = 256;
    int blocks = (N + threads_per_block - 1) / threads_per_block;

    cudaEvent_t iter_start, iter_end;
    CUDA_CHECK(cudaEventCreate(&iter_start));
    CUDA_CHECK(cudaEventCreate(&iter_end));

    /* Time the entire iteration loop */
    cudaEvent_t loop_start, loop_end;
    CUDA_CHECK(cudaEventCreate(&loop_start));
    CUDA_CHECK(cudaEventCreate(&loop_end));

    printf("Running value iteration...\n");
    CUDA_CHECK(cudaEventRecord(loop_start));

    int iter;
    float max_delta;
    float *d_V_final = d_V_next; /* tracks where the latest values are */

    for (iter = 0; iter < max_iters; iter++) {

        /* Launch Bellman update kernel */
        bellman_update_kernel<<<blocks, threads_per_block>>>(
            d_transitions, d_rewards,
            d_is_terminal, d_is_obstacle,
            d_V_curr, d_V_next,
            d_deltas, d_policy,
            N, gamma_val);
        CUDA_CHECK(cudaGetLastError());

        /* Compute max delta via parallel reduction */
        max_delta = gpu_max_reduce(d_deltas, d_reduce_workspace, N);

        /* Print progress */
        if ((iter + 1) % 100 == 0 || iter == 0) {
            printf("  Iteration %d: max_delta = %.8f\n",
                   iter + 1, max_delta);
        }

        /* Check convergence */
        if (max_delta < threshold) {
            printf("  Converged at iteration %d (max_delta = %.8f)\n",
                   iter + 1, max_delta);
            d_V_final = d_V_next; /* kernel just wrote here */
            break;
        }

        /* Swap V_curr and V_next pointers */
        float *tmp = d_V_curr;
        d_V_curr = d_V_next;
        d_V_next = tmp;
        d_V_final = d_V_curr; /* after swap, latest is in d_V_curr */
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
     * Copy results back to host
     * -------------------------------------------------------- */
    cudaEvent_t copyback_start, copyback_end;
    CUDA_CHECK(cudaEventCreate(&copyback_start));
    CUDA_CHECK(cudaEventCreate(&copyback_end));

    CUDA_CHECK(cudaEventRecord(copyback_start));

    /*
     * d_V_final tracks where the latest values are:
     *   - If converged: d_V_next (kernel wrote there, no swap)
     *   - If max_iters hit: d_V_curr (last swap moved it there)
     */
    CUDA_CHECK(cudaMemcpy(h_values, d_V_final, value_size,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_policy, d_policy, policy_size,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(copyback_end));
    CUDA_CHECK(cudaEventSynchronize(copyback_end));

    float copy_from_gpu_ms;
    CUDA_CHECK(cudaEventElapsedTime(&copy_from_gpu_ms,
                                    copyback_start, copyback_end));

    /* --------------------------------------------------------
     * Print results
     * -------------------------------------------------------- */
    printf("\n=== Results ===\n");
    printf("Iterations: %d\n", total_iters);
    printf("Copy to GPU: %.3f ms\n", copy_to_gpu_ms);
    printf("Iteration loop: %.2f ms\n", loop_ms);
    printf("Avg per iteration: %.3f ms\n", loop_ms / total_iters);
    printf("Copy from GPU: %.3f ms\n", copy_from_gpu_ms);
    printf("Total (copy + iterate + copy): %.2f ms\n",
           copy_to_gpu_ms + loop_ms + copy_from_gpu_ms);

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
            if (gw->is_terminal[s])       printf("G");
            else if (gw->is_obstacle[s])  printf("#");
            else                          printf("%c", arrows[h_policy[s]]);
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
            if (gw->is_obstacle[s])
                printf("  ####  ");
            else
                printf(" %6.1f ", h_values[s]);
        }
        printf("\n");
    }

    /* Save value function for comparison */
    char filename[256];
    snprintf(filename, sizeof(filename),
             "discrete_values_%dx%d.bin", rows, cols);
    FILE *fp = fopen(filename, "wb");
    if (fp) {
        fwrite(h_values, sizeof(float), N, fp);
        fclose(fp);
        printf("\nValue function saved to %s\n", filename);
    }

    /* --------------------------------------------------------
     * Cleanup
     * -------------------------------------------------------- */
    CUDA_CHECK(cudaFree(d_transitions));
    CUDA_CHECK(cudaFree(d_rewards));
    CUDA_CHECK(cudaFree(d_is_terminal));
    CUDA_CHECK(cudaFree(d_is_obstacle));
    CUDA_CHECK(cudaFree(d_V_curr));
    CUDA_CHECK(cudaFree(d_V_next));
    CUDA_CHECK(cudaFree(d_deltas));
    CUDA_CHECK(cudaFree(d_reduce_workspace));
    CUDA_CHECK(cudaFree(d_policy));

    CUDA_CHECK(cudaEventDestroy(copy_start));
    CUDA_CHECK(cudaEventDestroy(copy_end));
    CUDA_CHECK(cudaEventDestroy(iter_start));
    CUDA_CHECK(cudaEventDestroy(iter_end));
    CUDA_CHECK(cudaEventDestroy(loop_start));
    CUDA_CHECK(cudaEventDestroy(loop_end));
    CUDA_CHECK(cudaEventDestroy(copyback_start));
    CUDA_CHECK(cudaEventDestroy(copyback_end));

    free(h_values);
    free(h_policy);
    gridworld_free(gw);

    printf("Done.\n");
    return 0;
}
