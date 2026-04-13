// cpu_vi.c
#define _POSIX_C_SOURCE 199309L
#include "gridworld.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/*
 * CPU-only value iteration.
 *
 * Performs synchronous Bellman updates over all states until
 * the maximum value change falls below the convergence threshold.
 *
 * This serves as the correctness reference for the GPU implementations.
 */

typedef struct {
    float *values;          /* value function: V[s] */
    int *policy;            /* optimal action per state: pi[s] */
    int iterations;         /* number of sweeps to converge */
    double total_time_ms;   /* total wall-clock time */
    double avg_iter_ms;     /* average time per iteration */
} VIResult;

/*
 * Run value iteration on the CPU.
 *
 * Parameters:
 *   gw        - the grid world environment (transitions already built)
 *   gamma     - discount factor (e.g., 0.99)
 *   threshold - convergence threshold on max |V_new - V_old|
 *   max_iters - safety cap on iteration count
 *
 * Returns:
 *   A VIResult struct with the converged value function and policy.
 *   Caller is responsible for freeing values and policy arrays.
 */
VIResult cpu_value_iteration(const GridWorld *gw,
                             float gamma,
                             float threshold,
                             int max_iters) {
    VIResult result;
    int N = gw->num_states;

    /* Allocate value function arrays (current and next) */
    float *V_curr = (float *)calloc(N, sizeof(float));
    float *V_next = (float *)calloc(N, sizeof(float));
    int *policy   = (int *)calloc(N, sizeof(int));

    if (!V_curr || !V_next || !policy) {
        fprintf(stderr, "Failed to allocate value iteration arrays\n");
        free(V_curr);
        free(V_next);
        free(policy);
        result.values = NULL;
        result.policy = NULL;
        return result;
    }

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    int iter;
    for (iter = 0; iter < max_iters; iter++) {
        float max_delta = 0.0f;

        for (int s = 0; s < N; s++) {
            /* Terminal states have value 0, skip them */
            if (gw->is_terminal[s]) {
                V_next[s] = 0.0f;
                policy[s] = 0;
                continue;
            }

            /* Obstacle states shouldn't be occupied, but handle them */
            if (gw->is_obstacle[s]) {
                V_next[s] = 0.0f;
                policy[s] = 0;
                continue;
            }

            float best_value = -INFINITY;
            int best_action = 0;

            for (int a = 0; a < NUM_ACTIONS; a++) {
                int idx = s * NUM_ACTIONS + a;
                int next_state = gw->transitions[idx];
                float reward = gw->rewards[idx];

                float q_value = reward + gamma * V_curr[next_state];

                if (q_value > best_value) {
                    best_value = q_value;
                    best_action = a;
                }
            }

            V_next[s] = best_value;
            policy[s] = best_action;

            float delta = fabsf(V_next[s] - V_curr[s]);
            if (delta > max_delta) {
                max_delta = delta;
            }
        }

        /* Swap current and next value arrays */
        float *tmp = V_curr;
        V_curr = V_next;
        V_next = tmp;

        /* Print progress every 100 iterations */
        if ((iter + 1) % 100 == 0 || iter == 0) {
            printf("  Iteration %d: max_delta = %.8f\n", iter + 1, max_delta);
        }

        /* Check convergence */
        if (max_delta < threshold) {
            printf("  Converged at iteration %d (max_delta = %.8f)\n",
                   iter + 1, max_delta);
            break;
        }
    }

    if (iter == max_iters) {
        printf("  Warning: did not converge after %d iterations\n", max_iters);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);

    double elapsed_ms = (t_end.tv_sec - t_start.tv_sec) * 1000.0
                      + (t_end.tv_nsec - t_start.tv_nsec) / 1e6;

    /* V_curr holds the final converged values (due to swapping) */
    result.values = V_curr;
    result.policy = policy;
    result.iterations = (iter < max_iters) ? iter + 1 : max_iters;
    result.total_time_ms = elapsed_ms;
    result.avg_iter_ms = elapsed_ms / result.iterations;

    /* Free the extra buffer */
    free(V_next);

    return result;
}

/*
 * Print a small region of the policy as directional arrows.
 */
void print_policy_region(const GridWorld *gw, const int *policy,
                         int r0, int c0, int r1, int c1) {
    const char *arrows = "^v<>";

    if (r0 < 0) r0 = 0;
    if (c0 < 0) c0 = 0;
    if (r1 > gw->rows) r1 = gw->rows;
    if (c1 > gw->cols) c1 = gw->cols;

    printf("     ");
    for (int c = c0; c < c1; c++) {
        if ((c - c0) % 5 == 0) printf("%-5d", c);
    }
    printf("\n");

    for (int r = r0; r < r1; r++) {
        printf("%4d ", r);
        for (int c = c0; c < c1; c++) {
            int s = r * gw->cols + c;
            if (gw->is_terminal[s])       printf("G");
            else if (gw->is_obstacle[s])  printf("#");
            else                          printf("%c", arrows[policy[s]]);
        }
        printf("\n");
    }
}

/*
 * Print a small region of the value function.
 */
void print_value_region(const GridWorld *gw, const float *values,
                        int r0, int c0, int r1, int c1) {
    if (r0 < 0) r0 = 0;
    if (c0 < 0) c0 = 0;
    if (r1 > gw->rows) r1 = gw->rows;
    if (c1 > gw->cols) c1 = gw->cols;

    for (int r = r0; r < r1; r++) {
        printf("%4d |", r);
        for (int c = c0; c < c1; c++) {
            int s = r * gw->cols + c;
            if (gw->is_obstacle[s])
                printf("  ####  ");
            else
                printf(" %6.1f ", values[s]);
        }
        printf("\n");
    }
}

int main(int argc, char **argv) {
    int rows = 1000;
    int cols = 1000;
    float gamma = 0.99f;
    float threshold = 1e-4f;
    int max_iters = 10000;

    /* Parse optional arguments */
    if (argc >= 3) {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
    }
    if (argc >= 4) gamma = atof(argv[3]);
    if (argc >= 5) threshold = atof(argv[4]);

    printf("=== CPU Value Iteration ===\n");
    printf("Grid: %dx%d (%d states)\n", rows, cols, rows * cols);
    printf("Gamma: %.4f\n", gamma);
    printf("Threshold: %.1e\n", threshold);
    printf("Max iterations: %d\n\n", max_iters);

    /* Build the grid world */
    printf("Building grid world...\n");
    GridWorld *gw = gridworld_create(rows, cols, -1.0f, 0.0f);
    if (!gw) return 1;

    gridworld_default_layout(gw);
    gridworld_build_transitions(gw);
    int n_obstacles = 0, n_terminals = 0;
    for (int i = 0; i < gw->num_states; i++) {
        n_obstacles += gw->is_obstacle[i];
        n_terminals += gw->is_terminal[i];
    }
    printf("Grid world ready (%d obstacles, %d terminals)\n\n",
           n_obstacles, n_terminals);

    /* Run value iteration */
    printf("Running value iteration...\n");
    VIResult result = cpu_value_iteration(gw, gamma, threshold, max_iters);

    if (!result.values) {
        gridworld_free(gw);
        return 1;
    }

    /* Print timing results */
    printf("\n=== Results ===\n");
    printf("Iterations: %d\n", result.iterations);
    printf("Total time: %.2f ms\n", result.total_time_ms);
    printf("Avg per iteration: %.3f ms\n", result.avg_iter_ms);

    /* Print policy and values for a small region to visually verify */
    int preview_r = (rows < 20) ? rows : 20;
    int preview_c = (cols < 20) ? cols : 20;

    printf("\nPolicy (top-left corner):\n");
    print_policy_region(gw, result.policy, 0, 0, preview_r, preview_c);

    /* Show bottom-right corner where the goal is */
    printf("\nPolicy (bottom-right corner near goal):\n");
    int br0 = (rows > preview_r) ? rows - preview_r : 0;
    int bc0 = (cols > preview_c) ? cols - preview_c : 0;
    print_policy_region(gw, result.policy, br0, bc0, rows, cols);

    printf("\nValues (bottom-right corner near goal):\n");
    int vr = (rows < 8) ? rows : 8;
    int vc = (cols < 8) ? cols : 8;
    print_value_region(gw, result.values,
                       rows - vr, cols - vc, rows, cols);

    /* Save value function to file for later comparison */
    char filename[256];
    snprintf(filename, sizeof(filename),
             "cpu_values_%dx%d.bin", rows, cols);
    FILE *fp = fopen(filename, "wb");
    if (fp) {
        fwrite(result.values, sizeof(float), gw->num_states, fp);
        fclose(fp);
        printf("\nValue function saved to %s\n", filename);
    }

    /* Cleanup */
    free(result.values);
    free(result.policy);
    gridworld_free(gw);

    printf("Done.\n");
    return 0;
}
