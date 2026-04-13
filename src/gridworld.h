// gridworld.h
#ifndef GRIDWORLD_H
#define GRIDWORLD_H

#include <stdint.h>

/*
 * GridWorld environment for value iteration benchmarking.
 *
 * Layout:
 *   - 2D grid of size rows x cols
 *   - States indexed row-major: state = row * cols + col
 *   - 4 actions: UP=0, DOWN=1, LEFT=2, RIGHT=3
 *   - Obstacle cells are impassable (agent stays in place)
 *   - Terminal cells end the episode (value fixed at 0)
 *   - Moving into a wall or obstacle keeps the agent in place
 *
 * Output arrays (flat, contiguous, GPU-friendly):
 *   transitions[s * NUM_ACTIONS + a] = next_state
 *   rewards[s * NUM_ACTIONS + a]     = immediate reward
 *   is_terminal[s]                   = 1 if terminal, 0 otherwise
 *   is_obstacle[s]                   = 1 if obstacle, 0 otherwise
 */

#define NUM_ACTIONS 4
#define ACTION_UP    0
#define ACTION_DOWN  1
#define ACTION_LEFT  2
#define ACTION_RIGHT 3

typedef struct {
    int rows;
    int cols;
    int num_states;       /* rows * cols */

    int *transitions;     /* size: num_states * NUM_ACTIONS */
    float *rewards;       /* size: num_states * NUM_ACTIONS */
    int8_t *is_terminal;  /* size: num_states */
    int8_t *is_obstacle;  /* size: num_states */

    /* Environment parameters */
    float step_reward;    /* reward per non-terminal step (typically -1) */
    float goal_reward;    /* reward for reaching a terminal state */
} GridWorld;

/*
 * Create a grid world with the given dimensions.
 * Obstacles and terminals are not yet placed; call the
 * setup functions below or add them manually before
 * calling gridworld_build_transitions().
 */
GridWorld *gridworld_create(int rows, int cols,
                            float step_reward,
                            float goal_reward);

/*
 * Mark a rectangular block of cells as obstacles.
 * (row_start, col_start) to (row_end, col_end), inclusive.
 */
void gridworld_add_obstacle_rect(GridWorld *gw,
                                 int row_start, int col_start,
                                 int row_end, int col_end);

/*
 * Mark a single cell as a terminal (goal) state.
 */
void gridworld_add_terminal(GridWorld *gw, int row, int col);

/*
 * Generate a default obstacle pattern suitable for large grids.
 * Places several rectangular obstacle regions and a terminal state
 * in the bottom-right area. Deterministic for reproducibility.
 */
void gridworld_default_layout(GridWorld *gw);

/*
 * Build the transition and reward arrays based on the current
 * obstacle/terminal configuration. Must be called after all
 * obstacles and terminals have been placed.
 */
void gridworld_build_transitions(GridWorld *gw);

/*
 * Print a small region of the grid for visual debugging.
 * Shows obstacles as '#', terminals as 'G', and empty cells as '.'.
 * Prints rows [r0, r1) and cols [c0, c1).
 */
void gridworld_print_region(const GridWorld *gw,
                            int r0, int c0, int r1, int c1);

/*
 * Free all memory associated with the grid world.
 */
void gridworld_free(GridWorld *gw);

/*
 * Utility: convert (row, col) to flat state index.
 */
static inline int gridworld_state(const GridWorld *gw, int row, int col) {
    return row * gw->cols + col;
}

/*
 * Utility: extract row from flat state index.
 */
static inline int gridworld_row(const GridWorld *gw, int state) {
    return state / gw->cols;
}

/*
 * Utility: extract col from flat state index.
 */
static inline int gridworld_col(const GridWorld *gw, int state) {
    return state % gw->cols;
}

#endif /* GRIDWORLD_H */
