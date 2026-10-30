#include "farls.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct FarlsState {
  size_t n_ref;
  size_t n_spk;
  size_t n_err;
  size_t order;
  size_t n1;
  size_t sec_order;
  size_t alpha;
  size_t nrows;
  size_t cols;
  double scale;
  double *s_fx;
  double *w;
  double *ref;
  double *fx;
  double *fx_mem;
  double *gen;
  double *kbar;
  double *rsqrt;
  double *pre;
  double *aug;
  double *rhs;
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

static void GivensAt(double *pre, size_t nprows, size_t prow, size_t cx,
                     size_t cy, size_t cols) {
  double x = pre[prow * cols + cx];
  double y = pre[prow * cols + cy];
  double rad = hypot(x, y);
  if (!(rad > 0.0)) {
    return;
  }
  double c = x / rad;
  double s = y / rad;
  for (size_t row = 0; row < nprows; row++) {
    double *rp = pre + row * cols;
    double xi = rp[cx];
    double yi = rp[cy];
    rp[cx] = c * xi + s * yi;
    rp[cy] = -s * xi + c * yi;
  }
};

static int HypAt(double *pre, size_t nprows, size_t prow, size_t cx,
                 size_t cy, size_t cols) {
  double x = pre[prow * cols + cx];
  double y = pre[prow * cols + cy];
  if (y == 0.0) {
    return 0;
  }
  double disc = x * x - y * y;
  if (!(disc > 0.0)) {
    return -1;
  }
  double ch = fabs(x) / sqrt(disc);
  double sh = (y / x) * ch;
  for (size_t row = 0; row < nprows; row++) {
    double *rp = pre + row * cols;
    double xi = rp[cx];
    double yi = rp[cy];
    rp[cx] = ch * xi - sh * yi;
    rp[cy] = -sh * xi + ch * yi;
  }
  return 0;
};

static int Solve(const double *a, const double *b, double *x, size_t n,
                 double *aug) {
  size_t cols = n + 1;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      aug[i * cols + j] = a[i * n + j];
    }
    aug[i * cols + n] = b[i];
  }
  for (size_t c = 0; c < n; c++) {
    size_t piv = c;
    double best = fabs(aug[c * cols + c]);
    for (size_t r = c + 1; r < n; r++) {
      double v = fabs(aug[r * cols + c]);
      if (v > best) {
        best = v;
        piv = r;
      }
    }
    if (!(best > 1e-12)) {
      return -1;
    }
    if (piv != c) {
      for (size_t j = c; j < cols; j++) {
        double tmp = aug[c * cols + j];
        aug[c * cols + j] = aug[piv * cols + j];
        aug[piv * cols + j] = tmp;
      }
    }
    double div = aug[c * cols + c];
    for (size_t j = c; j < cols; j++) {
      aug[c * cols + j] /= div;
    }
    for (size_t r = 0; r < n; r++) {
      if (r == c) {
        continue;
      }
      double f = aug[r * cols + c];
      for (size_t j = c; j < cols; j++) {
        aug[r * cols + j] -= f * aug[c * cols + j];
      }
    }
  }
  for (size_t i = 0; i < n; i++) {
    x[i] = aug[i * cols + n];
  }
  return 0;
};

void FarlsDestroy(FarlsState *state) {
  if (state == NULL) {
    return;
  }
  free(state->s_fx);
  free(state->w);
  free(state->ref);
  free(state->fx);
  free(state->fx_mem);
  free(state->gen);
  free(state->kbar);
  free(state->rsqrt);
  free(state->pre);
  free(state->aug);
  free(state->rhs);
  free(state);
};

static void InitGen(FarlsState *state, double weight_var, double lambda,
                    double meas) {
  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t n_err = state->n_err;
  size_t n = state->order;
  size_t n1 = state->n1;
  size_t alpha = state->alpha;
  size_t nlines = n_spk * n_ref;
  memset(state->gen, 0, state->nrows * alpha * sizeof(double));
  memset(state->kbar, 0, state->nrows * n_err * sizeof(double));
  memset(state->rsqrt, 0, n_err * n_err * sizeof(double));
  double root = sqrt(meas);
  for (size_t q = 0; q < n_err; q++) {
    state->rsqrt[q * n_err + q] = root;
  }
  double sd = sqrt(weight_var);
  double tail = sd * pow(lambda, 0.5 * (double)n);
  for (size_t t = 0; t < nlines; t++) {
    state->gen[(t * n1) * alpha + (2 * t)] = sd;
    state->gen[(t * n1 + n) * alpha + (2 * t + 1)] = tail;
  }
};

FarlsState *FarlsCreate(size_t n_ref, size_t n_spk, size_t n_err,
                        size_t order, size_t sec_order, double lambda,
                        double meas_var, double weight_var,
                        const double *secondary) {
  if (n_ref == 0 || n_spk == 0 || n_err == 0 || order == 0 ||
      sec_order == 0 || !(lambda > 0.0) || !(lambda <= 1.0) ||
      !(meas_var > 0.0) || !(weight_var > 0.0) || secondary == NULL) {
    return NULL;
  }
  if (order > SIZE_MAX - 1) {
    return NULL;
  }
  size_t n1 = order + 1;
  size_t nlines = 0;
  size_t alpha = 0;
  size_t taps = 0;
  size_t nrows = 0;
  size_t nw = 0;
  size_t nsfx = 0;
  size_t nfx = 0;
  size_t ngen = 0;
  size_t nkb = 0;
  size_t npre = 0;
  size_t nfx_mem = 0;
  size_t naug = 0;
  size_t nrs = 0;
  if (MulSize(n_spk, n_ref, &nlines) != 0 || nlines > SIZE_MAX / 2) {
    return NULL;
  }
  alpha = 2 * nlines;
  size_t cols = n_err + alpha;
  size_t nprows = 0;
  if (MulSize(n_ref, order, &taps) != 0 || MulSize(nlines, n1, &nrows) != 0 ||
      MulSize(n_spk, taps, &nw) != 0 || MulSize(n_err, n_spk, &nsfx) != 0 ||
      MulSize(nsfx, sec_order, &nsfx) != 0 ||
      MulSize(n_err, nrows, &nfx) != 0 ||
      MulSize(nrows, alpha, &ngen) != 0 || MulSize(nrows, n_err, &nkb) != 0 ||
      MulSize(n_err, n_err, &nrs) != 0 ||
      MulSize(n_err, n_err + 1, &naug) != 0 ||
      MulSize(n_ref, sec_order, &nfx_mem) != 0) {
    return NULL;
  }
  if (n_err > SIZE_MAX - nrows) {
    return NULL;
  }
  nprows = n_err + nrows;
  if (MulSize(nprows, cols, &npre) != 0) {
    return NULL;
  }

  FarlsState *state = (FarlsState *)calloc(1, sizeof(*state));
  if (state == NULL) {
    return NULL;
  }
  state->n_ref = n_ref;
  state->n_spk = n_spk;
  state->n_err = n_err;
  state->order = order;
  state->n1 = n1;
  state->sec_order = sec_order;
  state->alpha = alpha;
  state->nrows = nrows;
  state->cols = cols;
  state->scale = 1.0 / sqrt(lambda);
  state->s_fx = (double *)calloc(nsfx, sizeof(double));
  state->w = (double *)calloc(nw, sizeof(double));
  state->ref = (double *)calloc(taps, sizeof(double));
  state->fx = (double *)calloc(nfx, sizeof(double));
  state->fx_mem = (double *)calloc(nfx_mem, sizeof(double));
  state->gen = (double *)calloc(ngen, sizeof(double));
  state->kbar = (double *)calloc(nkb, sizeof(double));
  state->rsqrt = (double *)calloc(nrs, sizeof(double));
  state->pre = (double *)calloc(npre, sizeof(double));
  state->aug = (double *)calloc(naug, sizeof(double));
  state->rhs = (double *)calloc(n_err, sizeof(double));
  if (state->s_fx == NULL || state->w == NULL || state->ref == NULL ||
      state->fx == NULL || state->fx_mem == NULL || state->gen == NULL ||
      state->kbar == NULL || state->rsqrt == NULL || state->pre == NULL ||
      state->aug == NULL || state->rhs == NULL) {
    FarlsDestroy(state);
    return NULL;
  }
  double ls = state->scale;
  for (size_t q = 0; q < n_err; q++) {
    for (size_t l = 0; l < n_spk; l++) {
      double gain = 1.0;
      double *sf = state->s_fx + (q * n_spk + l) * sec_order;
      const double *src = secondary + (q * n_spk + l) * sec_order;
      for (size_t k = 0; k < sec_order; k++) {
        sf[k] = src[k] * gain;
        gain *= ls;
      }
    }
  }
  InitGen(state, weight_var, lambda, meas_var);
  return state;
};

static int UpdateJoint(FarlsState *state, const double *e) {
  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t n_err = state->n_err;
  size_t n = state->order;
  size_t n1 = state->n1;
  size_t alpha = state->alpha;
  size_t nrows = state->nrows;
  size_t cols = state->cols;
  size_t nprows = n_err + nrows;
  size_t nlines = n_spk * n_ref;
  double *pre = state->pre;
  double *gen = state->gen;
  double *kb = state->kbar;
  double ls = state->scale;

  for (size_t q = 0; q < n_err; q++) {
    if (!isfinite(e[q])) {
      return -1;
    }
  }
  memset(pre, 0, nprows * cols * sizeof(double));
  for (size_t i = 0; i < n_err; i++) {
    for (size_t j = 0; j < n_err; j++) {
      pre[i * cols + j] = state->rsqrt[i * n_err + j];
    }
  }
  for (size_t a = 0; a < alpha; a++) {
    for (size_t q = 0; q < n_err; q++) {
      double dot = 0.0;
      for (size_t t = 0; t < nlines; t++) {
        size_t spk = t / n_ref;
        size_t pref = t % n_ref;
        const double *hist =
            state->fx + ((q * n_spk + spk) * n_ref + pref) * n1;
        for (size_t i = 0; i < n1; i++) {
          dot += hist[i] * gen[(t * n1 + i) * alpha + a];
        }
      }
      pre[q * cols + n_err + a] = dot;
    }
  }
  for (size_t t = 0; t < nlines; t++) {
    for (size_t i = 1; i < n1; i++) {
      size_t row = n_err + t * n1 + i;
      size_t src = (t * n1 + i - 1) * n_err;
      for (size_t q = 0; q < n_err; q++) {
        pre[row * cols + q] = kb[src + q];
      }
    }
    for (size_t i = 0; i < n1; i++) {
      size_t row = n_err + t * n1 + i;
      for (size_t a = 0; a < alpha; a++) {
        double src = gen[(t * n1 + i) * alpha + a];
        pre[row * cols + n_err + a] = ls * src;
      }
    }
  }
  for (size_t a = 0; a < alpha; a++) {
    size_t cy = n_err + a;
    int hyp = (a % 2) == 1;
    for (size_t q = 0; q < n_err; q++) {
      if (hyp) {
        if (HypAt(pre, nprows, q, q, cy, cols) != 0) {
          return -1;
        }
      } else {
        GivensAt(pre, nprows, q, q, cy, cols);
      }
    }
  }
  for (size_t i = 0; i < n_err; i++) {
    for (size_t j = 0; j < n_err; j++) {
      state->rsqrt[i * n_err + j] = pre[i * cols + j];
    }
  }
  if (!FiniteVec(state->rsqrt, n_err * n_err)) {
    return -1;
  }
  for (size_t i = 0; i < nrows; i++) {
    for (size_t q = 0; q < n_err; q++) {
      kb[i * n_err + q] = pre[(n_err + i) * cols + q];
    }
    for (size_t a = 0; a < alpha; a++) {
      gen[i * alpha + a] = pre[(n_err + i) * cols + n_err + a];
    }
  }
  if (Solve(state->rsqrt, e, state->rhs, n_err, state->aug) != 0) {
    return -1;
  }
  for (size_t t = 0; t < nlines; t++) {
    size_t spk = t / n_ref;
    size_t pref = t % n_ref;
    double *w = state->w + (spk * n_ref + pref) * n;
    for (size_t i = 0; i < n; i++) {
      double add = 0.0;
      size_t row = (t * n1 + i) * n_err;
      for (size_t q = 0; q < n_err; q++) {
        add += kb[row + q] * state->rhs[q];
      }
      w[i] = ls * w[i] + add;
    }
  }
  if (!FiniteVec(state->w, nlines * n)) {
    return -1;
  }
  return 0;
};

int FarlsCore(FarlsState *state, const double *x, const double *e, double *y) {
  if (state == NULL || x == NULL || e == NULL || y == NULL) {
    return -1;
  }
  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t n_err = state->n_err;
  size_t order = state->order;
  size_t n1 = state->n1;
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
      const double *sf = state->s_fx + (q * n_spk + l) * sec_order;
      for (size_t p = 0; p < n_ref; p++) {
        double fx = Dot(sf, state->fx_mem + p * sec_order, sec_order);
        double *fh = state->fx + ((q * n_spk + l) * n_ref + p) * n1;
        for (size_t k = n1; k-- > 1;) {
          fh[k] = fh[k - 1];
        }
        fh[0] = fx;
      }
    }
  }
  if (UpdateJoint(state, e) != 0) {
    return -1;
  }
  for (size_t s = 0; s < n_spk; s++) {
    const double *w = state->w + s * n_ref * order;
    double acc = 0.0;
    for (size_t p = 0; p < n_ref; p++) {
      acc += Dot(w + p * order, state->ref + p * order, order);
    }
    if (!isfinite(acc)) {
      return -1;
    }
    y[s] = acc;
  }
  return 0;
};

int FarlsCopyWeights(const FarlsState *state, double *w) {
  if (state == NULL || w == NULL) {
    return -1;
  }
  size_t n = state->n_spk * state->n_ref * state->order;
  for (size_t i = 0; i < n; i++) {
    w[i] = state->w[i];
  }
  return 0;
};
