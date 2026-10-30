#include "rls.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct RlsState {
  size_t n_ref;
  size_t n_err;
  size_t order;
  size_t nstate;
  double lambda;
  double *w;
  double *line;
  double *stack;
  double *p;
  double *pc;
  double *kw;
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

static void MatVec(const double *a, const double *x, double *y, size_t n) {
  for (size_t i = 0; i < n; i++) {
    double sum = 0.0;
    const double *row = a + i * n;
    for (size_t j = 0; j < n; j++) {
      sum += row[j] * x[j];
    }
    y[i] = sum;
  }
};

static int FiniteVec(const double *a, size_t n) {
  for (size_t i = 0; i < n; i++) {
    if (!isfinite(a[i])) {
      return 0;
    }
  }
  return 1;
};

void RlsDestroy(RlsState *state) {
  if (state == NULL) {
    return;
  }
  free(state->w);
  free(state->line);
  free(state->stack);
  free(state->p);
  free(state->pc);
  free(state->kw);
  free(state);
};

RlsState *RlsCreate(size_t n_ref, size_t n_err, size_t order, double lambda,
                    double delta) {
  if (n_ref == 0 || n_err == 0 || order == 0 || !(lambda > 0.0) ||
      !(lambda <= 1.0) || !(delta > 0.0)) {
    return NULL;
  }
  size_t nstate = 0;
  size_t nw = 0;
  size_t nn = 0;
  size_t np = 0;
  if (MulSize(n_ref, order, &nstate) != 0 || MulSize(n_err, nstate, &nw) != 0 ||
      MulSize(nstate, nstate, &nn) != 0 || MulSize(n_err, nn, &np) != 0) {
    return NULL;
  }

  RlsState *state = (RlsState *)calloc(1, sizeof(*state));
  if (state == NULL) {
    return NULL;
  }
  state->n_ref = n_ref;
  state->n_err = n_err;
  state->order = order;
  state->nstate = nstate;
  state->lambda = lambda;
  state->w = (double *)calloc(nw, sizeof(double));
  state->line = (double *)calloc(nw / n_err, sizeof(double));
  state->stack = (double *)calloc(nstate, sizeof(double));
  state->p = (double *)calloc(np, sizeof(double));
  state->pc = (double *)calloc(nstate, sizeof(double));
  state->kw = (double *)calloc(nstate, sizeof(double));
  if (state->w == NULL || state->line == NULL || state->stack == NULL ||
      state->p == NULL || state->pc == NULL || state->kw == NULL) {
    RlsDestroy(state);
    return NULL;
  }
  for (size_t q = 0; q < n_err; q++) {
    double *pq = state->p + q * nn;
    for (size_t i = 0; i < nstate; i++) {
      pq[i * nstate + i] = delta;
    }
  }
  return state;
};

static int UpdateOne(RlsState *state, size_t q, double eps) {
  size_t n = state->nstate;
  double *w = state->w + q * n;
  double *p = state->p + q * n * n;
  double *pc = state->pc;
  double *kw = state->kw;
  const double *x = state->stack;
  double lambda = state->lambda;

  MatVec(p, x, pc, n);
  double den = lambda + Dot(x, pc, n);
  if (!(den > 0.0) || !isfinite(den) || !isfinite(eps)) {
    return -1;
  }
  double inv = 1.0 / den;
  for (size_t i = 0; i < n; i++) {
    kw[i] = pc[i] * inv;
    w[i] += kw[i] * eps;
  }
  if (!FiniteVec(w, n)) {
    return -1;
  }
  double inv_lambda = 1.0 / lambda;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      p[i * n + j] = (p[i * n + j] - pc[i] * pc[j] * inv) * inv_lambda;
    }
  }
  return 0;
};

int RlsCore(RlsState *state, const double *x, const double *d, double *y) {
  if (state == NULL || x == NULL || d == NULL || y == NULL) {
    return -1;
  }
  size_t n_ref = state->n_ref;
  size_t n_err = state->n_err;
  size_t order = state->order;
  size_t n = state->nstate;

  for (size_t p = 0; p < n_ref; p++) {
    double *line = state->line + p * order;
    for (size_t k = order; k-- > 1;) {
      line[k] = line[k - 1];
    }
    line[0] = x[p];
    memcpy(state->stack + p * order, line, order * sizeof(double));
  }

  for (size_t q = 0; q < n_err; q++) {
    const double *w = state->w + q * n;
    double pred = Dot(w, state->stack, n);
    if (!isfinite(pred)) {
      return -1;
    }
    y[q] = pred;
    if (UpdateOne(state, q, d[q] - pred) != 0) {
      return -1;
    }
  }
  return 0;
};

int RlsCopyWeights(const RlsState *state, double *w) {
  if (state == NULL || w == NULL) {
    return -1;
  }
  size_t n = state->n_err * state->nstate;
  for (size_t i = 0; i < n; i++) {
    w[i] = state->w[i];
  }
  return 0;
};
