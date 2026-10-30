#include "lms.h"

#include <stdlib.h>

struct LmsState {
  size_t n_ref;
  size_t n_err;
  size_t order;
  double step;
  double *w;
  double *x_state;
};

void LmsDestroy(LmsState *state) {
  if (state == NULL) {
    return;
  }
  free(state->w);
  free(state->x_state);
  free(state);
};

LmsState *LmsCreate(size_t n_ref, size_t n_err, size_t order, double step) {
  if (n_ref == 0 || n_err == 0 || order == 0 || !(step > 0.0)) {
    return NULL;
  }

  LmsState *state = (LmsState *)calloc(1, sizeof(*state));
  if (state == NULL) {
    return NULL;
  }
  state->n_ref = n_ref;
  state->n_err = n_err;
  state->order = order;
  state->step = step;
  state->w = (double *)calloc(n_err * n_ref * order, 
        sizeof(double));
  state->x_state = (double *)calloc(n_ref * order, sizeof(double));
  if (state->w == NULL || state->x_state == NULL) {
    LmsDestroy(state);
    return NULL;
  }
  return state;
};

int LmsStep(LmsState *state, const double *x, const double *d, double *e) {
  if (state == NULL || x == NULL || d == NULL || e == NULL) {
    return -1;
  }

  size_t n_ref = state->n_ref;
  size_t n_err = state->n_err;
  size_t order = state->order;

  for (size_t p = 0; p < n_ref; p++) {
    double *line = state->x_state + p * order;
    for (size_t k = order; k-- > 1;) {
      line[k] = line[k - 1];
    }
    line[0] = x[p];
  }

  for (size_t q = 0; q < n_err; q++) {
    double y = 0.0;
    for (size_t p = 0; p < n_ref; p++) {
      const double *line = state->x_state + p * order;
      const double *coeff = state->w + (q * n_ref + p) * order;
      for (size_t k = 0; k < order; k++) {
        y += coeff[k] * line[k];
      }
    }
    e[q] = d[q] - y;
  }

  for (size_t q = 0; q < n_err; q++) {
    double gain = state->step * e[q];
    for (size_t p = 0; p < n_ref; p++) {
      double *coeff = state->w + (q * n_ref + p) * order;
      const double *line = state->x_state + p * order;
      for (size_t k = 0; k < order; k++) {
        coeff[k] += gain * line[k];
      }
    }
  }
  return 0;
};

int LmsCopyWeights(const LmsState *state, double *w) {
  if (state == NULL || w == NULL) {
    return -1;
  }
  size_t n = state->n_err * state->n_ref * state->order;
  for (size_t i = 0; i < n; i++) {
    w[i] = state->w[i];
  }
  return 0;
};
