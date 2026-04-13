// gridworld.c
#include "gridworld.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

GridWorld *gridworld_create(int rows, int cols,
                            float step_reward,
                            float goal_reward) {
    GridWorld *gw = (GridWorld *)malloc(sizeof(GridWorld));
    if (!gw) {
        fprintf(stderr, "Failed to allocate GridWorld struct\n");
        return NULL;
    }

    gw->rows = rows;
    gw->cols = cols;
    gw->num_states = rows * cols;
    gw->step_reward = step_reward;
    gw->goal_reward = goal_reward;

    /* Allocate flat arrays */
    gw->transitions = (int *)malloc(
        (size_t)gw->num_states * NUM_ACTIONS * sizeof(int));
    gw->rewards = (float *)malloc(
        (size_t)gw->num_states * NUM_ACTIONS * sizeof(float));
    gw->is_terminal = (int8_t *)calloc(gw->num_states, sizeof(int8_t));
    gw->is_obstacle = (int8_t *)calloc(gw->num_states, sizeof(int8_t));

    if (!gw->transitions || !gw->rewards ||
        !gw->is_terminal || !gw->is_obstacle) {
        fprintf(stderr, "Failed to allocate grid arrays for %d states\n",
                gw->num_states);
        gridworld_free(gw);
        return NULL;
    }

    return gw;
}

void gridworld_add_obstacle_rect(GridWorld *gw,
                                 int row_start, int col_start,
                                 int row_end, int col_end) {
    for (int r = row_start; r <= row_end && r < gw->rows; r++) {
        for (int c = col_start; c <= col_end && c < gw->cols; c++) {
            gw->is_obstacle[r * gw->cols + c] = 1;
        }
    }
}

void gridworld_add_terminal(GridWorld *gw, int row, int col) {
    if (row >= 0 && row < gw->rows && col >= 0 && col < gw->cols) {
        gw->is_terminal[row * gw->cols + col] = 1;
    }
}

void gridworld_default_layout(GridWorld *gw) {
    /*
     * Generate a reproducible obstacle pattern for large grids.
     * Strategy: place horizontal and vertical wall segments that
     * create corridors, plus some scattered rectangular blocks.
     * The terminal state is in the bottom-right corner.
     *
     * For a 1000x1000 grid, this creates a non-trivial navigation
     * problem with multiple paths of varying length.
     */
    int r = gw->rows;
    int c = gw->cols;

    /* Scale obstacle placement to grid size */
    int wall_thickness = (r > 100) ? r / 200 : 1;

    /*
     * Horizontal walls at roughly 20%, 40%, 60%, 80% of grid height.
     * Each wall has a gap for passage.
     */
    int wall_positions[4] = { r / 5, 2 * r / 5, 3 * r / 5, 4 * r / 5 };
    int gap_width = c / 10;  /* 10% of grid width */

    for (int i = 0; i < 4; i++) {
        int wr = wall_positions[i];
        /* Gap position alternates between left and right side */
        int gap_start = (i % 2 == 0) ? c / 10 : c - c / 10 - gap_width;

        /* Left portion of wall (before gap) */
        if (gap_start > 0) {
            gridworld_add_obstacle_rect(gw,
                wr, 0,
                wr + wall_thickness - 1, gap_start - 1);
        }
        /* Right portion of wall (after gap) */
        if (gap_start + gap_width < c) {
            gridworld_add_obstacle_rect(gw,
                wr, gap_start + gap_width,
                wr + wall_thickness - 1, c - 1);
        }
    }

    /* Vertical walls at 30% and 70% of grid width */
    int vwall_positions[2] = { 3 * c / 10, 7 * c / 10 };
    int vgap_height = r / 10;

    for (int i = 0; i < 2; i++) {
        int wc = vwall_positions[i];
        int vgap_start = (i % 2 == 0) ? 3 * r / 10 : 6 * r / 10;

        /* Top portion */
        if (vgap_start > 0) {
            gridworld_add_obstacle_rect(gw,
                0, wc,
                vgap_start - 1, wc + wall_thickness - 1);
        }
        /* Bottom portion */
        if (vgap_start + vgap_height < r) {
            gridworld_add_obstacle_rect(gw,
                vgap_start + vgap_height, wc,
                r - 1, wc + wall_thickness - 1);
        }
    }

    /* A few scattered rectangular blocks */
    int block_size = r / 20;
    gridworld_add_obstacle_rect(gw,
        r / 8, c / 4,
        r / 8 + block_size, c / 4 + block_size);
    gridworld_add_obstacle_rect(gw,
        5 * r / 8, 3 * c / 8,
        5 * r / 8 + block_size, 3 * c / 8 + block_size);
    gridworld_add_obstacle_rect(gw,
        3 * r / 8, 5 * c / 8,
        3 * r / 8 + block_size, 5 * c / 8 + block_size);

    /* Terminal state: bottom-right corner */
    gw->is_terminal[gw->num_states - 1] = 1;

    /* Make sure terminal is not on an obstacle */
    gw->is_obstacle[gw->num_states - 1] = 0;
}

void gridworld_build_transitions(GridWorld *gw) {
    /*
     * For each state and action, compute the next state and reward.
     *
     * Rules:
     *   - Terminal states: transitions go to self, reward = 0
     *     (value is fixed at 0, no further updates)
     *   - Obstacle states: should never be occupied, but if they are,
     *     transitions go to self with step_reward
     *   - Normal states: move in the chosen direction if the target
     *     cell is in-bounds and not an obstacle; otherwise stay in place
     *   - Reward is goal_reward when entering a terminal state,
     *     step_reward otherwise
     */

    /* Direction offsets: UP, DOWN, LEFT, RIGHT */
    int dr[NUM_ACTIONS] = { -1,  1,  0,  0 };
    int dc[NUM_ACTIONS] = {  0,  0, -1,  1 };

    for (int s = 0; s < gw->num_states; s++) {
        int row = s / gw->cols;
        int col = s % gw->cols;

        for (int a = 0; a < NUM_ACTIONS; a++) {
            int idx = s * NUM_ACTIONS + a;

            /* Terminal states are absorbing */
            if (gw->is_terminal[s]) {
                gw->transitions[idx] = s;
                gw->rewards[idx] = 0.0f;
                continue;
            }

            /* Compute candidate next position */
            int nr = row + dr[a];
            int nc = col + dc[a];

            /* Check bounds and obstacles */
            int next_state;
            if (nr < 0 || nr >= gw->rows ||
                nc < 0 || nc >= gw->cols ||
                gw->is_obstacle[nr * gw->cols + nc]) {
                /* Blocked: stay in place */
                next_state = s;
            } else {
                next_state = nr * gw->cols + nc;
            }

            gw->transitions[idx] = next_state;

            /* Reward depends on where we end up */
            if (gw->is_terminal[next_state]) {
                gw->rewards[idx] = gw->goal_reward;
            } else {
                gw->rewards[idx] = gw->step_reward;
            }
        }
    }
}

void gridworld_print_region(const GridWorld *gw,
                            int r0, int c0, int r1, int c1) {
    /* Clamp bounds */
    if (r0 < 0) r0 = 0;
    if (c0 < 0) c0 = 0;
    if (r1 > gw->rows) r1 = gw->rows;
    if (c1 > gw->cols) c1 = gw->cols;

    /* Column header */
    printf("     ");
    for (int c = c0; c < c1; c++) {
        if ((c - c0) % 5 == 0) {
            printf("%-5d", c);
        }
    }
    printf("\n");

    for (int r = r0; r < r1; r++) {
        printf("%4d ", r);
        for (int c = c0; c < c1; c++) {
            int s = r * gw->cols + c;
            if (gw->is_terminal[s])       printf("G");
            else if (gw->is_obstacle[s])  printf("#");
            else                          printf(".");
        }
        printf("\n");
    }
}

void gridworld_free(GridWorld *gw) {
    if (!gw) return;
    free(gw->transitions);
    free(gw->rewards);
    free(gw->is_terminal);
    free(gw->is_obstacle);
    free(gw);
}
