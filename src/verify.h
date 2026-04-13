// verify.h
#ifndef VERIFY_H
#define VERIFY_H

/*
 * Compare two value function arrays element-wise.
 * Returns 1 if all values match within tolerance, 0 otherwise.
 * Prints the first few mismatches if any are found.
 */
int verify_values(const float *v_ref, const float *v_test,
                  int num_states, float tolerance,
                  const char *ref_name, const char *test_name);

/*
 * Compare two policy arrays (integer actions per state).
 * Returns 1 if all match, 0 otherwise.
 * Note: ties in the value function can produce different but
 * equally valid policies, so mismatches here are not always errors.
 */
int verify_policy(const int *pi_ref, const int *pi_test,
                  int num_states,
                  const char *ref_name, const char *test_name);

#endif /* VERIFY_H */
