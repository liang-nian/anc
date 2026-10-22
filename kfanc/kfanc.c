#include "kfanc.h"

#include <math.h>
#include <stdlib.h>

struct KfancState {
  size_t order;
  size_t sec_order;
  size_t ns;
  size_t nstate;
  double scale;
  double meas;
  double *s;
  double *w;
  double *psi;
  double *r;
  double *p;
  double *ck;
  double *pc;
  double *gain;
  double *ak;
  double *pfilt;
  double *work;
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

static void MatMul(const double *a, const double *b, double *c, size_t n) {
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      double sum = 0.0;
      for (size_t k = 0; k < n; k++) {
        sum += a[i * n + k] * b[k * n + j];
      }
      c[i * n + j] = sum;
    }
  }
};

void KfancDestroy(KfancState *state) {
  if (state == NULL) {
    return;
  }
  free(state->s);
  free(state->w);
  free(state->psi);
  free(state->r);
  free(state->p);
  free(state->ck);
  free(state->pc);
  free(state->gain);
  free(state->ak);
  free(state->pfilt);
  free(state->work);
  free(state);
};

KfancState *KfancCreate(size_t order, size_t sec_order, double lambda,
                        double meas_var, double weight_var,
                        const double *secondary) {
  if (order == 0 || sec_order == 0 || !(lambda > 0.0) || !(lambda <= 1.0) ||
      !(meas_var > 0.0) || !(weight_var > 0.0) || secondary == NULL) {
    return NULL;
  }
  size_t ns = sec_order - 1;
  if (order > SIZE_MAX - ns) {
    return NULL;
  }
  size_t nstate = order + ns;
  size_t nn = 0;
  if (MulSize(nstate, nstate, &nn) != 0) {
    return NULL;
  }

  KfancState *state = (KfancState *)calloc(1, sizeof(*state));
  if (state == NULL) {
    return NULL;
  }
  state->order = order;
  state->sec_order = sec_order;
  state->ns = ns;
  state->nstate = nstate;
  state->scale = 1.0 / sqrt(lambda);
  state->meas = meas_var;

  state->s = (double *)calloc(sec_order, sizeof(double));
  state->w = (double *)calloc(order, sizeof(double));
  state->r = (double *)calloc(order, sizeof(double));
  state->p = (double *)calloc(nn, sizeof(double));
  state->ck = (double *)calloc(nstate, sizeof(double));
  state->pc = (double *)calloc(nstate, sizeof(double));
  state->gain = (double *)calloc(nstate, sizeof(double));
  state->ak = (double *)calloc(nn, sizeof(double));
  state->pfilt = (double *)calloc(nn, sizeof(double));
  state->work = (double *)calloc(nn, sizeof(double));
  if (ns > 0) {
    state->psi = (double *)calloc(ns, sizeof(double));
  }
  if (state->s == NULL || state->w == NULL || state->r == NULL ||
      state->p == NULL || state->ck == NULL || state->pc == NULL ||
      state->gain == NULL || state->ak == NULL || state->pfilt == NULL ||
      state->work == NULL || (ns > 0 && state->psi == NULL)) {
    KfancDestroy(state);
    return NULL;
  }
  for (size_t i = 0; i < sec_order; i++) {
    state->s[i] = secondary[i];
  }
  for (size_t i = 0; i < order; i++) {
    state->p[i * nstate + i] = weight_var;
  }
  return state;
};

int KfancCore(KfancState *state, double x, double e, double *y) {
  if (state == NULL || y == NULL) {
    return -1;
  }

  size_t order = state->order;
  size_t ns = state->ns;
  size_t n = state->nstate;
  double scale = state->scale;
  double *r = state->r;
  double *w = state->w;
  double *psi = state->psi;
  double *s = state->s;
  double *p = state->p;
  double *ck = state->ck;
  double *pc = state->pc;
  double *gain = state->gain;
  double *ak = state->ak;
  double *pfilt = state->pfilt;
  double *work = state->work;

  for (size_t i = order; i-- > 1;) {
    r[i] = r[i - 1];
  }
  r[0] = x;
  *y = Dot(w, r, order);

  double eps = e;
  for (size_t i = 0; i < ns; i++) {
    eps -= s[i + 1] * psi[i];
  }

  for (size_t i = 0; i < n; i++) {
    ck[i] = 0.0;
  }
  if (s[0] != 0.0) {
    for (size_t j = 0; j < order; j++) {
      ck[j] = s[0] * r[j];
    }
  }
  for (size_t i = 0; i < ns; i++) {
    ck[order + i] = s[i + 1];
  }

  for (size_t i = 0; i < n * n; i++) {
    ak[i] = 0.0;
  }
  for (size_t i = 0; i < order; i++) {
    ak[i * n + i] = scale;
  }
  if (ns > 0) {
    for (size_t j = 0; j < order; j++) {
      ak[order * n + j] = r[j];
    }
    for (size_t i = 1; i < ns; i++) {
      ak[(order + i) * n + (order + i - 1)] = 1.0;
    }
  }

  MatVec(p, ck, pc, n);
  double re = state->meas + Dot(ck, pc, n);
  if (!(re > 0.0) || !isfinite(re)) {
    return -1;
  }

  double inv = 1.0 / re;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      pfilt[i * n + j] = p[i * n + j] - pc[i] * pc[j] * inv;
    }
  }
  MatMul(ak, pfilt, work, n);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      double sum = 0.0;
      for (size_t k = 0; k < n; k++) {
        sum += work[i * n + k] * ak[j * n + k];
      }
      p[i * n + j] = sum;
    }
  }
  for (size_t i = 0; i < n; i++) {
    for (size_t j = i + 1; j < n; j++) {
      double avg = 0.5 * (p[i * n + j] + p[j * n + i]);
      p[i * n + j] = avg;
      p[j * n + i] = avg;
    }
  }

  MatVec(ak, pc, gain, n);
  for (size_t i = 0; i < n; i++) {
    gain[i] *= inv;
  }
  for (size_t i = 0; i < order; i++) {
    w[i] = scale * w[i] + gain[i] * eps;
  }
  if (ns > 0) {
    for (size_t i = ns; i-- > 1;) {
      psi[i] = psi[i - 1] + gain[order + i] * eps;
    }
    psi[0] = gain[order] * eps;
  }
  if (!isfinite(*y) || !isfinite(Dot(w, w, order))) {
    return -1;
  }
  return 0;
};

int KfancCopyWeights(const KfancState *state, double *w) {
  if (state == NULL || w == NULL) {
    return -1;
  }
  for (size_t i = 0; i < state->order; i++) {
    w[i] = state->w[i];
  }
  return 0;
};
