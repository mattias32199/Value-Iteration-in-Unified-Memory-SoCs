// verify.c
#include "verify.h"
#include <stdio.h>
#include <math.h>

int verify_values(const float *v_ref, const float *v_test,
                  int num_states, float tolerance,
                  const char *ref_name, const char *test_name) {
    int mismatches = 0;
    float max_diff = 0.0f;
    int max_diff_state = -1;

    for (int s = 0; s < num_states; s++) {
        float diff = fabsf(v_ref[s] - v_test[s]);
        if (diff > tolerance) {
            if (mismatches < 5) {
                printf("  Mismatch at state %d: %s=%.6f, %s=%.6f (diff=%.6f)\n",
                       s, ref_name, v_ref[s], test_name, v_test[s], diff);
            }
            mismatches++;
        }
        if (diff > max_diff) {
            max_diff = diff;
            max_diff_state = s;
        }
    }

    if (mismatches == 0) {
        printf("  PASS: %s vs %s match within tolerance %.1e (max diff=%.1e)\n",
               ref_name, test_name, tolerance, max_diff);
        return 1;
    } else {
        printf("  FAIL: %d / %d states differ beyond tolerance %.1e\n",
               mismatches, num_states, tolerance);
        printf("  Worst mismatch at state %d: diff=%.6f\n",
               max_diff_state, max_diff);
        return 0;
    }
}

int verify_policy(const int *pi_ref, const int *pi_test,
                  int num_states,
                  const char *ref_name, const char *test_name) {
    int mismatches = 0;

    for (int s = 0; s < num_states; s++) {
        if (pi_ref[s] != pi_test[s]) {
            if (mismatches < 5) {
                printf("  Policy mismatch at state %d: %s=%d, %s=%d\n",
                       s, ref_name, pi_ref[s], test_name, pi_test[s]);
            }
            mismatches++;
        }
    }

    if (mismatches == 0) {
        printf("  PASS: policies match across all %d states\n", num_states);
        return 1;
    } else {
        printf("  NOTE: %d / %d policy mismatches (may be valid ties)\n",
               mismatches, num_states);
        return 0;
    }
}
