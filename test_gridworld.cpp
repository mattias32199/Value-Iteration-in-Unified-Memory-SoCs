#include "gridworld.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    int rows = 1000;
    int cols = 1000;

    /* Allow smaller grids for quick testing */
    if (argc >= 3) {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
    }

    printf("Creating %dx%d grid world (%d states)...\n",
           rows, cols, rows * cols);

    clock_t t0 = clock();

    GridWorld *gw = gridworld_create(rows, cols,
                                     -1.0f,   /* step reward */
                                      0.0f);  /* goal reward */
    if (!gw) {
        fprintf(stderr, "Failed to create grid world\n");
        return 1;
    }

    gridworld_default_layout(gw);

    clock_t t1 = clock();
    printf("Layout created in %.3f ms\n",
           1000.0 * (t1 - t0) / CLOCKS_PER_SEC);

    gridworld_build_transitions(gw);

    clock_t t2 = clock();
    printf("Transitions built in %.3f ms\n",
           1000.0 * (t2 - t1) / CLOCKS_PER_SEC);

    /* Count obstacles and terminals */
    int n_obstacles = 0, n_terminals = 0;
    for (int s = 0; s < gw->num_states; s++) {
        if (gw->is_obstacle[s]) n_obstacles++;
        if (gw->is_terminal[s]) n_terminals++;
    }
    printf("Obstacles: %d (%.2f%%)\n",
           n_obstacles, 100.0 * n_obstacles / gw->num_states);
    printf("Terminals: %d\n", n_terminals);

    /* Print corners of the grid for visual sanity check */
    int preview = (rows < 25) ? rows : 25;
    int pcols   = (cols < 40) ? cols : 40;

    printf("\nTop-left corner:\n");
    gridworld_print_region(gw, 0, 0, preview, pcols);

    printf("\nBottom-right corner:\n");
    int br0 = (rows > preview) ? rows - preview : 0;
    int bc0 = (cols > pcols)   ? cols - pcols   : 0;
    gridworld_print_region(gw, br0, bc0, rows, cols);

    /* Spot-check a few transitions */
    printf("\nTransition spot-checks:\n");
    int test_states[] = { 0, cols / 2, gw->num_states / 2, gw->num_states - 1 };
    const char *action_names[] = { "UP", "DOWN", "LEFT", "RIGHT" };

    for (int i = 0; i < 4; i++) {
        int s = test_states[i];
        int r = gridworld_row(gw, s);
        int c = gridworld_col(gw, s);
        printf("  State %d (%d,%d)%s%s:\n", s, r, c,
               gw->is_obstacle[s] ? " [OBSTACLE]" : "",
               gw->is_terminal[s] ? " [TERMINAL]" : "");
        for (int a = 0; a < NUM_ACTIONS; a++) {
            int idx = s * NUM_ACTIONS + a;
            int ns = gw->transitions[idx];
            printf("    %-5s -> state %d (%d,%d)  r=%.1f\n",
                   action_names[a], ns,
                   gridworld_row(gw, ns),
                   gridworld_col(gw, ns),
                   gw->rewards[idx]);
        }
    }

    /* Memory usage estimate */
    size_t mem = (size_t)gw->num_states * NUM_ACTIONS * sizeof(int)    /* transitions */
               + (size_t)gw->num_states * NUM_ACTIONS * sizeof(float)  /* rewards */
               + (size_t)gw->num_states * 2 * sizeof(int8_t);         /* flags */
    printf("\nGrid world memory: %.2f MB\n", mem / (1024.0 * 1024.0));

    gridworld_free(gw);
    printf("Done.\n");
    return 0;
}
