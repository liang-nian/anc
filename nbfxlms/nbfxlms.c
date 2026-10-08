#include "nbfxlms.h"

#include <math.h>
#include <stdlib.h>

struct NbfxlmsState {
  size_t n_harm;
  size_t n_spk;
  size_t n_err;
  size_t sec_order;
  double step;
  double *w;
  double *s;
  double *order;
  double *xc;
  double *xs;
  double *x_now_c;
  double *x_now_s;
  double *x_state_c;
  double *x_state_s;
  double *fx_c;
  double *fx_s;
};

void NbfxlmsDestroy(NbfxlmsState *state) {
  if (state == NULL) {
    return;
  }
  free(state->w);
  free(state->s);
  free(state->order);
  free(state->xc);
  free(state->xs);
  free(state->x_now_c);
  free(state->x_now_s);
  free(state->x_state_c);
  free(state->x_state_s);
  free(state->fx_c);
  free(state->fx_s);
  free(state);
};

NbfxlmsState *NbfxlmsCreate(size_t n_harm, size_t n_spk, size_t n_err,
                            size_t sec_order, double step,
                            const double *secondary, const double *order) {
  if (n_harm == 0 || n_spk == 0 || n_err == 0 || sec_order == 0 ||
      !(step > 0.0) || secondary == NULL || order == NULL) {
    return NULL;
  }

  NbfxlmsState *state = (NbfxlmsState *)calloc(1, sizeof(*state));
  if (state == NULL) {
    return NULL;
  }
  state->n_harm = n_harm;
  state->n_spk = n_spk;
  state->n_err = n_err;
  state->sec_order = sec_order;
  state->step = step;

  size_t n_w = n_spk * n_harm * 2;
  size_t n_s = n_err * n_spk * sec_order;
  size_t n_x = n_harm * sec_order;
  size_t n_fx = n_err * n_spk * n_harm;
  state->w = (double *)calloc(n_w, sizeof(double));
  state->s = (double *)calloc(n_s, sizeof(double));
  state->order = (double *)calloc(n_harm, sizeof(double));
  state->xc = (double *)calloc(n_harm, sizeof(double));
  state->xs = (double *)calloc(n_harm, sizeof(double));
  state->x_now_c = (double *)calloc(n_harm, sizeof(double));
  state->x_now_s = (double *)calloc(n_harm, sizeof(double));
  state->x_state_c = (double *)calloc(n_x, sizeof(double));
  state->x_state_s = (double *)calloc(n_x, sizeof(double));
  state->fx_c = (double *)calloc(n_fx, sizeof(double));
  state->fx_s = (double *)calloc(n_fx, sizeof(double));
  if (state->w == NULL || state->s == NULL || state->order == NULL ||
      state->xc == NULL || state->xs == NULL || state->x_now_c == NULL ||
      state->x_now_s == NULL || state->x_state_c == NULL ||
      state->x_state_s == NULL || state->fx_c == NULL ||
      state->fx_s == NULL) {
    NbfxlmsDestroy(state);
    return NULL;
  }
  for (size_t i = 0; i < n_s; i++) {
    state->s[i] = secondary[i];
  }
  for (size_t i = 0; i < n_harm; i++) {
    if (!(order[i] > 0.0) || !isfinite(order[i])) {
      NbfxlmsDestroy(state);
      return NULL;
    }
    state->order[i] = order[i];
    state->xc[i] = 1.0;
    state->xs[i] = 0.0;
  }
  return state;
};

int NbfxlmsPrefilter(NbfxlmsState *state, double omega0) {
  if (state == NULL || !isfinite(omega0)) {
    return -1;
  }

  size_t n_harm = state->n_harm;
  size_t n_spk = state->n_spk;
  size_t n_err = state->n_err;
  size_t sec_order = state->sec_order;

  for (size_t i = 0; i < n_harm; i++) {
    double xc = state->xc[i];
    double xs = state->xs[i];
    state->x_now_c[i] = xc;
    state->x_now_s[i] = xs;

    double *cline = state->x_state_c + i * sec_order;
    double *sline = state->x_state_s + i * sec_order;
    for (size_t k = sec_order; k-- > 1;) {
      cline[k] = cline[k - 1];
      sline[k] = sline[k - 1];
    }
    cline[0] = xc;
    sline[0] = xs;

    double wi = state->order[i] * omega0;
    double rot_c = cos(wi);
    double rot_s = sin(wi);
    state->xc[i] = rot_c * xc - rot_s * xs;
    state->xs[i] = rot_s * xc + rot_c * xs;
  }

  for (size_t q = 0; q < n_err; q++) {
    for (size_t l = 0; l < n_spk; l++) {
      const double *path = state->s + (q * n_spk + l) * sec_order;
      for (size_t i = 0; i < n_harm; i++) {
        const double *cline = state->x_state_c + i * sec_order;
        const double *sline = state->x_state_s + i * sec_order;
        double fc = 0.0;
        double fs = 0.0;
        for (size_t j = 0; j < sec_order; j++) {
          fc += path[j] * cline[j];
          fs += path[j] * sline[j];
        }
        size_t fx = (q * n_spk + l) * n_harm + i;
        state->fx_c[fx] = fc;
        state->fx_s[fx] = fs;
      }
    }
  }
  return 0;
};

int NbfxlmsUpdate(NbfxlmsState *state, const double *e) {
  if (state == NULL || e == NULL) {
    return -1;
  }

  size_t n_harm = state->n_harm;
  size_t n_spk = state->n_spk;
  size_t n_err = state->n_err;

  for (size_t l = 0; l < n_spk; l++) {
    for (size_t i = 0; i < n_harm; i++) {
      double acc_c = 0.0;
      double acc_s = 0.0;
      for (size_t q = 0; q < n_err; q++) {
        size_t fx = (q * n_spk + l) * n_harm + i;
        acc_c += e[q] * state->fx_c[fx];
        acc_s += e[q] * state->fx_s[fx];
      }
      double *coeff = state->w + (l * n_harm + i) * 2;
      coeff[0] += state->step * acc_c;
      coeff[1] += state->step * acc_s;
    }
  }
  return 0;
};

int NbfxlmsOutput(NbfxlmsState *state, double *y) {
  if (state == NULL || y == NULL) {
    return -1;
  }

  size_t n_harm = state->n_harm;
  size_t n_spk = state->n_spk;

  for (size_t l = 0; l < n_spk; l++) {
    double acc = 0.0;
    for (size_t i = 0; i < n_harm; i++) {
      const double *coeff = state->w + (l * n_harm + i) * 2;
      acc += coeff[0] * state->x_now_c[i];
      acc += coeff[1] * state->x_now_s[i];
    }
    y[l] = acc;
  }
  return 0;
};

int NbfxlmsCore(NbfxlmsState *state, double omega0, const double *e,
                double *y) {
  if (state == NULL || e == NULL || y == NULL) {
    return -1;
  }
  if (NbfxlmsPrefilter(state, omega0) != 0) {
    return -1;
  }
  if (NbfxlmsUpdate(state, e) != 0) {
    return -1;
  }
  return NbfxlmsOutput(state, y);
};

int NbfxlmsCopyWeights(const NbfxlmsState *state, double *w) {
  if (state == NULL || w == NULL) {
    return -1;
  }
  size_t n = state->n_spk * state->n_harm * 2;
  for (size_t i = 0; i < n; i++) {
    w[i] = state->w[i];
  }
  return 0;
};
