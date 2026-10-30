#include "frls.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct FrlsState {
  size_t n_ref;
  size_t n_spk;
  size_t n_err;
  size_t order;
  size_t sec_order;
  size_t nstate;
  double scale;
  double meas;
  double *s_fx;
  double *w;
  double *ref;
  double *fx;
  double *fx_mem;
  double *g;
  double *p;
  double *pc;
  double *gram;
  double *inv;
  double *aug;
  double *ue;
};

static int MulSize(size_t a, size_t b, size_t *out) {
  if (a != 0 && b > SIZE_MAX / a) {
    return -1;
  }
  *out = a * b;
  return 0;
};

static double Dot(const double *a, const double *b, size_t n) {
  double sum = 0.0;
  for (size_t i = 0; i < n; i++) {
    sum += a[i] * b[i];
  }
  return sum;
};

static int FiniteVec(const double *a, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (!isfinite(a[i])) {
      return 0;
    }
  }
  return 1;
};

static int Invert(double *aug, const double *a, double *inv, size_t n) {
  size_t cols = n * 2;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      aug[i * cols + j] = a[i * n + j];
      aug[i * cols + n + j] = (i == j) ? 1.0 : 0.0;
    }
  }
  for (size_t k = 0; k < n; k++) {
    size_t piv = k;
    double best = fabs(aug[k * cols + k]);
    for (size_t i = k + 1; i < n; i++) {
      double v = fabs(aug[i * cols + k]);
      if (v > best) {
        best = v;
        piv = i;
      }
    }
    if (!(best > 1e-12)) {
      return -1;
    }
    if (piv != k) {
      for (size_t j = 0; j < cols; j++) {
        double tmp = aug[k * cols + j];
        aug[k * cols + j] = aug[piv * cols + j];
        aug[piv * cols + j] = tmp;
      }
    }
    double diag = aug[k * cols + k];
    for (size_t j = 0; j < cols; j++) {
      aug[k * cols + j] /= diag;
    }
    for (size_t i = 0; i < n; i++) {
      if (i == k) {
        continue;
      }
      double factor = aug[i * cols + k];
      for (size_t j = 0; j < cols; j++) {
        aug[i * cols + j] -= factor * aug[k * cols + j];
      }
    }
  }
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      inv[i * n + j] = aug[i * cols + n + j];
    }
  }
  if (!FiniteVec(inv, n * n)) {
    return -1;
  }
  return 0;
};

void FrlsDestroy(FrlsState *state) {
  if (state == NULL) {
    return;
  }
  free(state->s_fx);
  free(state->w);
  free(state->ref);
  free(state->fx);
  free(state->fx_mem);
  free(state->g);
  free(state->p);
  free(state->pc);
  free(state->gram);
  free(state->inv);
  free(state->aug);
  free(state->ue);
  free(state);
};

FrlsState *FrlsCreate(size_t n_ref, size_t n_spk, size_t n_err, size_t order,
                      size_t sec_order, double lambda, double meas_var,
                      double weight_var, const double *secondary) {
  if (n_ref == 0 || n_spk == 0 || n_err == 0 || order == 0 || sec_order == 0 ||
      !(lambda > 0.0) || !(lambda <= 1.0) || !(meas_var > 0.0) ||
      !(weight_var > 0.0) || secondary == NULL || n_err > SIZE_MAX / 2) {
    return NULL;
  }
  size_t nblk = 0;
  size_t nstate = 0;
  size_t np = 0;
  size_t nfx = 0;
  size_t nsec_path = 0;
  size_t nsec = 0;
  size_t nfx_mem = 0;
  size_t ng = 0;
  size_t ngram = 0;
  size_t naug = 0;
  if (MulSize(n_ref, order, &nblk) != 0 || MulSize(n_spk, nblk, &nstate) != 0 ||
      MulSize(nstate, nstate, &np) != 0 || MulSize(n_err, nstate, &nfx) != 0 ||
      MulSize(n_err, n_spk, &nsec_path) != 0 ||
      MulSize(nsec_path, sec_order, &nsec) != 0 ||
      MulSize(n_ref, sec_order, &nfx_mem) != 0 ||
      MulSize(nstate, n_err, &ng) != 0 || MulSize(n_err, n_err, &ngram) != 0 ||
      MulSize(n_err, n_err * 2, &naug) != 0) {
    return NULL;
  }

  FrlsState *state = (FrlsState *)calloc(1, sizeof(*state));
  if (state == NULL) {
    return NULL;
  }
  state->n_ref = n_ref;
  state->n_spk = n_spk;
  state->n_err = n_err;
  state->order = order;
  state->sec_order = sec_order;
  state->nstate = nstate;
  state->scale = 1.0 / sqrt(lambda);
  state->meas = meas_var;
  state->s_fx = (double *)calloc(nsec, sizeof(double));
  state->w = (double *)calloc(nstate, sizeof(double));
  state->ref = (double *)calloc(nblk, sizeof(double));
  state->fx = (double *)calloc(nfx, sizeof(double));
  state->fx_mem = (double *)calloc(nfx_mem, sizeof(double));
  state->g = (double *)calloc(ng, sizeof(double));
  state->p = (double *)calloc(np, sizeof(double));
  state->pc = (double *)calloc(ng, sizeof(double));
  state->gram = (double *)calloc(ngram, sizeof(double));
  state->inv = (double *)calloc(ngram, sizeof(double));
  state->aug = (double *)calloc(naug, sizeof(double));
  state->ue = (double *)calloc(nstate, sizeof(double));
  if (state->s_fx == NULL || state->w == NULL || state->ref == NULL ||
      state->fx == NULL || state->fx_mem == NULL || state->g == NULL ||
      state->p == NULL || state->pc == NULL || state->gram == NULL ||
      state->inv == NULL || state->aug == NULL || state->ue == NULL) {
    FrlsDestroy(state);
    return NULL;
  }
  double ls = state->scale;
  for (size_t q = 0; q < n_err; q++) {
    for (size_t l = 0; l < n_spk; l++) {
      const double *src = secondary + (q * n_spk + l) * sec_order;
      double *dst = state->s_fx + (q * n_spk + l) * sec_order;
      double gain = 1.0;
      for (size_t k = 0; k < sec_order; k++) {
        dst[k] = src[k] * gain;
        gain *= ls;
      }
    }
  }
  for (size_t i = 0; i < nstate; i++) {
    state->p[i * nstate + i] = weight_var;
  }
  return state;
};

static void BuildFiltered(FrlsState *state, const double *x) {
  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t n_err = state->n_err;
  size_t order = state->order;
  size_t sec_order = state->sec_order;

  for (size_t p = 0; p < n_ref; p++) {
    double *line = state->ref + p * order;
    for (size_t k = order; k-- > 1;) {
      line[k] = line[k - 1];
    }
    line[0] = x[p];

    double *mem = state->fx_mem + p * sec_order;
    for (size_t k = sec_order; k-- > 1;) {
      mem[k] = mem[k - 1];
    }
    mem[0] = x[p];
  }

  for (size_t q = 0; q < n_err; q++) {
    for (size_t l = 0; l < n_spk; l++) {
      const double *sline = state->s_fx + (q * n_spk + l) * sec_order;
      for (size_t p = 0; p < n_ref; p++) {
        double fx = Dot(sline, state->fx_mem + p * sec_order, sec_order);
        double *fh =
            state->fx + ((q * n_spk + l) * n_ref + p) * order;
        for (size_t k = order; k-- > 1;) {
          fh[k] = fh[k - 1];
        }
        fh[0] = fx;
      }
    }
  }
};

static void BuildRegressor(FrlsState *state) {
  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t n_err = state->n_err;
  size_t order = state->order;
  size_t n = state->nstate;
  for (size_t i = 0; i < n; i++) {
    for (size_t q = 0; q < n_err; q++) {
      size_t l = i / (n_ref * order);
      size_t rem = i % (n_ref * order);
      size_t p = rem / order;
      size_t k = rem % order;
      const double *fh =
          state->fx + ((q * n_spk + l) * n_ref + p) * order;
      state->g[i * n_err + q] = fh[k];
    }
  }
};

static int UpdateJoint(FrlsState *state, const double *e) {
  size_t n = state->nstate;
  size_t qn = state->n_err;
  double *p = state->p;
  double *g = state->g;
  double *pc = state->pc;
  double *gram = state->gram;
  double *inv = state->inv;
  double *ue = state->ue;
  double scale = state->scale;
  double ls2 = scale * scale;

  if (!FiniteVec(e, qn)) {
    return -1;
  }
  for (size_t i = 0; i < n; i++) {
    const double *row = p + i * n;
    for (size_t q = 0; q < qn; q++) {
      double sum = 0.0;
      for (size_t j = 0; j < n; j++) {
        sum += row[j] * g[j * qn + q];
      }
      pc[i * qn + q] = sum;
    }
  }
  for (size_t a = 0; a < qn; a++) {
    for (size_t b = 0; b < qn; b++) {
      double sum = (a == b) ? state->meas : 0.0;
      for (size_t i = 0; i < n; i++) {
        sum += g[i * qn + a] * pc[i * qn + b];
      }
      gram[a * qn + b] = sum;
    }
  }
  if (Invert(state->aug, gram, inv, qn) != 0) {
    return -1;
  }
  for (size_t i = 0; i < n; i++) {
    for (size_t q = 0; q < qn; q++) {
      double sum = 0.0;
      for (size_t b = 0; b < qn; b++) {
        sum += pc[i * qn + b] * inv[b * qn + q];
      }
      g[i * qn + q] = sum;
    }
  }
  for (size_t i = 0; i < n; i++) {
    double sum = 0.0;
    for (size_t q = 0; q < qn; q++) {
      sum += g[i * qn + q] * e[q];
    }
    ue[i] = sum;
    state->w[i] = scale * state->w[i] + scale * sum;
  }
  if (!FiniteVec(state->w, n)) {
    return -1;
  }
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      double dot = 0.0;
      for (size_t q = 0; q < qn; q++) {
        dot += g[i * qn + q] * pc[j * qn + q];
      }
      p[i * n + j] = ls2 * (p[i * n + j] - dot);
    }
  }
  if (!FiniteVec(p, n * n)) {
    return -1;
  }
  return 0;
};

int FrlsCore(FrlsState *state, const double *x, const double *e, double *y) {
  if (state == NULL || x == NULL || e == NULL || y == NULL) {
    return -1;
  }
  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t order = state->order;

  BuildFiltered(state, x);
  BuildRegressor(state);
  if (UpdateJoint(state, e) != 0) {
    return -1;
  }
  for (size_t l = 0; l < n_spk; l++) {
    double acc = 0.0;
    for (size_t p = 0; p < n_ref; p++) {
      const double *w = state->w + (l * n_ref + p) * order;
      const double *line = state->ref + p * order;
      acc += Dot(w, line, order);
    }
    if (!isfinite(acc)) {
      return -1;
    }
    y[l] = acc;
  }
  return 0;
};

int FrlsCopyWeights(const FrlsState *state, double *w) {
  if (state == NULL || w == NULL) {
    return -1;
  }
  size_t n = state->nstate;
  for (size_t i = 0; i < n; i++) {
    w[i] = state->w[i];
  }
  return 0;
};
