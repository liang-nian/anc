#include "fxlms.h"

#include <stdlib.h>

struct FxlmsState {
  size_t n_ref;
  size_t n_spk;
  size_t n_err;
  size_t order;
  size_t sec_order;
  size_t x_len;
  double step;
  double *w;
  double *s;
  double *x_state;
  double *fx_state;
};

void FxlmsDestroy(FxlmsState *state) {
  if (state == NULL) {
    return;
  }
  free(state->w);
  free(state->s);
  free(state->x_state);
  free(state->fx_state);
  free(state);
};

FxlmsState *FxlmsCreate(size_t n_ref, size_t n_spk, size_t n_err, size_t order,
                        size_t sec_order, double step,
                        const double *secondary) {
  if (n_ref == 0 || n_spk == 0 || n_err == 0 || order == 0 || sec_order == 0 ||
      !(step > 0.0) || secondary == NULL) {
    return NULL;
  }

  FxlmsState *state = (FxlmsState *)calloc(1, sizeof(*state));
  if (state == NULL) {
    return NULL;
  }
  state->n_ref = n_ref;
  state->n_spk = n_spk;
  state->n_err = n_err;
  state->order = order;
  state->sec_order = sec_order;
  state->x_len = order > sec_order ? order : sec_order;
  state->step = step;

  size_t n_w = n_spk * n_ref * order;
  size_t n_s = n_err * n_spk * sec_order;
  size_t n_x = n_ref * state->x_len;
  size_t n_fx = n_err * n_spk * n_ref * order;
  state->w = (double *)calloc(n_w, sizeof(double));
  state->s = (double *)calloc(n_s, sizeof(double));
  state->x_state = (double *)calloc(n_x, sizeof(double));
  state->fx_state = (double *)calloc(n_fx, sizeof(double));
  if (state->w == NULL || state->s == NULL || state->x_state == NULL ||
      state->fx_state == NULL) {
    FxlmsDestroy(state);
    return NULL;
  }
  for (size_t i = 0; i < n_s; i++) {
    state->s[i] = secondary[i];
  }
  return state;
};

int FxlmsPrefilter(FxlmsState *state, const double *x) {
  if (state == NULL || x == NULL) {
    return -1;
  }

  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t n_err = state->n_err;
  size_t order = state->order;
  size_t sec_order = state->sec_order;
  size_t x_len = state->x_len;

  for (size_t p = 0; p < n_ref; p++) {
    double *line = state->x_state + p * x_len;
    for (size_t k = x_len; k-- > 1;) {
      line[k] = line[k - 1];
    }
    line[0] = x[p];
  }

  for (size_t q = 0; q < n_err; q++) {
    for (size_t l = 0; l < n_spk; l++) {
      const double *sline = state->s + (q * n_spk + l) * sec_order;
      for (size_t p = 0; p < n_ref; p++) {
        const double *xline = state->x_state + p * x_len;
        double fx = 0.0;
        for (size_t j = 0; j < sec_order; j++) {
          fx += sline[j] * xline[j];
        }
        double *fline =
            state->fx_state + ((q * n_spk + l) * n_ref + p) * order;
        for (size_t k = order; k-- > 1;) {
          fline[k] = fline[k - 1];
        }
        fline[0] = fx;
      }
    }
  }
  return 0;
};

int FxlmsUpdate(FxlmsState *state, const double *e) {
  if (state == NULL || e == NULL) {
    return -1;
  }

  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t n_err = state->n_err;
  size_t order = state->order;

  for (size_t l = 0; l < n_spk; l++) {
    for (size_t p = 0; p < n_ref; p++) {
      double *coeff = state->w + (l * n_ref + p) * order;
      for (size_t k = 0; k < order; k++) {
        double acc = 0.0;
        for (size_t q = 0; q < n_err; q++) {
          const double *fline =
              state->fx_state + ((q * n_spk + l) * n_ref + p) * order;
          acc += e[q] * fline[k];
        }
        coeff[k] += state->step * acc;
      }
    }
  }
  return 0;
};

int FxlmsOutput(FxlmsState *state, double *y) {
  if (state == NULL || y == NULL) {
    return -1;
  }

  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t order = state->order;
  size_t x_len = state->x_len;

  for (size_t l = 0; l < n_spk; l++) {
    double acc = 0.0;
    for (size_t p = 0; p < n_ref; p++) {
      const double *line = state->x_state + p * x_len;
      const double *coeff = state->w + (l * n_ref + p) * order;
      for (size_t k = 0; k < order; k++) {
        acc += coeff[k] * line[k];
      }
    }
    y[l] = acc;
  }
  return 0;
};

int FxlmsCore(FxlmsState *state, const double *x, const double *e, double *y) {
  if (state == NULL || x == NULL || e == NULL || y == NULL) {
    return -1;
  }
  if (FxlmsPrefilter(state, x) != 0) {
    return -1;
  }
  if (FxlmsUpdate(state, e) != 0) {
    return -1;
  }
  return FxlmsOutput(state, y);
};

int FxlmsCopyWeights(const FxlmsState *state, double *w) {
  if (state == NULL || w == NULL) {
    return -1;
  }
  size_t n = state->n_spk * state->n_ref * state->order;
  for (size_t i = 0; i < n; i++) {
    w[i] = state->w[i];
  }
  return 0;
};
