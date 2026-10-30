#include "fakf.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct FakfState {
  size_t n_ref;
  size_t n_spk;
  size_t n_err;
  size_t order;
  size_t n1;
  size_t sec_order;
  size_t ns;
  size_t alpha;
  size_t n1tot;
  size_t nrows;
  size_t cols;
  double scale;
  double *s;
  double *w;
  double *ref;
  double *psi;
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

static int SolveDare(const double *cs, size_t ns, double meas, double proc,
                     double *kpsi, double *rsqrt) {
  size_t nn = 0;
  if (MulSize(ns, ns, &nn) != 0) {
    return -1;
  }
  double *p = (double *)calloc(nn, sizeof(double));
  double *nx = (double *)calloc(nn, sizeof(double));
  double *v = (double *)calloc(ns, sizeof(double));
  double *av = (double *)calloc(ns, sizeof(double));
  if (p == NULL || nx == NULL || v == NULL || av == NULL) {
    free(p);
    free(nx);
    free(v);
    free(av);
    return -1;
  }
  for (size_t iter = 0; iter < 80; iter++) {
    for (size_t i = 0; i < ns; i++) {
      double acc = 0.0;
      for (size_t j = 0; j < ns; j++) {
        acc += p[i * ns + j] * cs[j];
      }
      v[i] = acc;
    }
    double svar = meas;
    for (size_t i = 0; i < ns; i++) {
      svar += cs[i] * v[i];
    }
    if (!(svar > 0.0)) {
      free(p);
      free(nx);
      free(v);
      free(av);
      return -1;
    }
    av[0] = 0.0;
    for (size_t i = 1; i < ns; i++) {
      av[i] = v[i - 1];
    }
    double diff = 0.0;
    for (size_t i = 0; i < ns; i++) {
      for (size_t j = 0; j < ns; j++) {
        double aij = 0.0;
        if (i > 0 && j > 0) {
          aij = p[(i - 1) * ns + (j - 1)];
        }
        aij -= av[i] * av[j] / svar;
        if (i == 0 && j == 0) {
          aij += proc;
        }
        diff += fabs(aij - p[i * ns + j]);
        nx[i * ns + j] = aij;
      }
    }
    memcpy(p, nx, nn * sizeof(double));
    if (iter > 2 && diff < 1.0e-12) {
      break;
    }
  }
  for (size_t i = 0; i < ns; i++) {
    double acc = 0.0;
    for (size_t j = 0; j < ns; j++) {
      acc += p[i * ns + j] * cs[j];
    }
    v[i] = acc;
  }
  double svar = meas;
  for (size_t i = 0; i < ns; i++) {
    svar += cs[i] * v[i];
  }
  if (!(svar > 0.0)) {
    free(p);
    free(nx);
    free(v);
    free(av);
    return -1;
  }
  *rsqrt = sqrt(svar);
  kpsi[0] = 0.0;
  for (size_t i = 1; i < ns; i++) {
    kpsi[i] = v[i - 1] / (*rsqrt);
  }
  free(p);
  free(nx);
  free(v);
  free(av);
  return 0;
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

static int HypAt(double *pre, size_t nprows, size_t prow, size_t cx, size_t cy,
                 size_t cols) {
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

void FakfDestroy(FakfState *state) {
  if (state == NULL) {
    return;
  }
  free(state->s);
  free(state->w);
  free(state->ref);
  free(state->psi);
  free(state->gen);
  free(state->kbar);
  free(state->rsqrt);
  free(state->pre);
  free(state->aug);
  free(state->rhs);
  free(state);
};

static int InitGen(FakfState *state, double weight_var, double lambda,
                   double meas, double proc) {
  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t n_err = state->n_err;
  size_t n = state->order;
  size_t n1 = state->n1;
  size_t ns = state->ns;
  size_t alpha = state->alpha;
  size_t n1tot = state->n1tot;
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
  if (!(proc > 0.0) || ns == 0) {
    return 0;
  }
  for (size_t spk = 0; spk < n_spk; spk++) {
    double *kpsi = (double *)calloc(ns, sizeof(double));
    double rsqrt = 0.0;
    const double *sec = state->s + spk * state->sec_order;
    if (kpsi == NULL ||
        SolveDare(sec + 1, ns, meas, proc, kpsi, &rsqrt) != 0) {
      free(kpsi);
      return -1;
    }
    if (n_err == 1 && n_spk == 1 && rsqrt > 0.0 && isfinite(rsqrt)) {
      state->rsqrt[0] = rsqrt;
    }
    for (size_t j = 0; j < ns; j++) {
      state->kbar[(n1tot + spk * ns + j) * n_err] = kpsi[j];
    }
    free(kpsi);
  }
  return 0;
};

FakfState *FakfCreate(size_t n_ref, size_t n_spk, size_t n_err, size_t order,
                      size_t sec_order, double lambda, double meas_var,
                      double proc_var, double weight_var,
                      const double *secondary) {
  if (n_ref == 0 || n_spk == 0 || n_err == 0 || order == 0 || sec_order == 0 ||
      !(lambda > 0.0) || !(lambda <= 1.0) || !(meas_var > 0.0) ||
      !(proc_var >= 0.0) || !(weight_var > 0.0) || secondary == NULL) {
    return NULL;
  }
  if (order > SIZE_MAX - 1) {
    return NULL;
  }
  size_t n1 = order + 1;
  size_t ns = sec_order - 1;
  size_t nlines = 0;
  size_t alpha = 0;
  size_t n1tot = 0;
  size_t nrows = 0;
  size_t taps = 0;
  size_t nref = 0;
  size_t nsec = 0;
  size_t nw = 0;
  size_t npsi = 0;
  size_t ngen = 0;
  size_t nkb = 0;
  size_t npre = 0;
  size_t nrs = 0;
  size_t naug = 0;
  if (MulSize(n_spk, n_ref, &nlines) != 0 || nlines > SIZE_MAX / 2) {
    return NULL;
  }
  alpha = 2 * nlines;
  if (MulSize(nlines, n1, &n1tot) != 0) {
    return NULL;
  }
  size_t npsi_rows = 0;
  if (MulSize(n_spk, ns == 0 ? 0 : ns, &npsi_rows) != 0 ||
      n1tot > SIZE_MAX - npsi_rows) {
    return NULL;
  }
  nrows = n1tot + npsi_rows;
  size_t cols = n_err + alpha;
  size_t nprows = 0;
  if (n_err > SIZE_MAX - nrows) {
    return NULL;
  }
  nprows = n_err + nrows;
  if (MulSize(n_ref, order, &taps) != 0 || MulSize(n_spk, taps, &nw) != 0 ||
      MulSize(n_ref, n1, &nref) != 0 || MulSize(n_err, n_spk, &nsec) != 0 ||
      MulSize(nsec, sec_order, &nsec) != 0 ||
      MulSize(n_spk, ns == 0 ? 1 : ns, &npsi) != 0 ||
      MulSize(nrows, alpha, &ngen) != 0 || MulSize(nrows, n_err, &nkb) != 0 ||
      MulSize(nprows, cols, &npre) != 0 || MulSize(n_err, n_err, &nrs) != 0 ||
      MulSize(n_err, n_err + 1, &naug) != 0) {
    return NULL;
  }

  FakfState *state = (FakfState *)calloc(1, sizeof(*state));
  if (state == NULL) {
    return NULL;
  }
  state->n_ref = n_ref;
  state->n_spk = n_spk;
  state->n_err = n_err;
  state->order = order;
  state->n1 = n1;
  state->sec_order = sec_order;
  state->ns = ns;
  state->alpha = alpha;
  state->n1tot = n1tot;
  state->nrows = nrows;
  state->cols = cols;
  state->scale = 1.0 / sqrt(lambda);
  state->s = (double *)calloc(nsec, sizeof(double));
  state->w = (double *)calloc(nw, sizeof(double));
  state->ref = (double *)calloc(nref, sizeof(double));
  state->psi = (double *)calloc(npsi, sizeof(double));
  state->gen = (double *)calloc(ngen, sizeof(double));
  state->kbar = (double *)calloc(nkb, sizeof(double));
  state->rsqrt = (double *)calloc(nrs, sizeof(double));
  state->pre = (double *)calloc(npre, sizeof(double));
  state->aug = (double *)calloc(naug, sizeof(double));
  state->rhs = (double *)calloc(n_err, sizeof(double));
  if (state->s == NULL || state->w == NULL || state->ref == NULL ||
      state->psi == NULL || state->gen == NULL || state->kbar == NULL ||
      state->rsqrt == NULL || state->pre == NULL || state->aug == NULL ||
      state->rhs == NULL) {
    FakfDestroy(state);
    return NULL;
  }
  memcpy(state->s, secondary, nsec * sizeof(double));
  if (InitGen(state, weight_var, lambda, meas_var, proc_var) != 0) {
    FakfDestroy(state);
    return NULL;
  }
  return state;
};

static int UpdateJoint(FakfState *state, const double *e) {
  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t n_err = state->n_err;
  size_t n = state->order;
  size_t n1 = state->n1;
  size_t ns = state->ns;
  size_t alpha = state->alpha;
  size_t n1tot = state->n1tot;
  size_t nrows = state->nrows;
  size_t cols = state->cols;
  size_t nprows = n_err + nrows;
  size_t nlines = n_spk * n_ref;
  size_t nw = n_ref * n;
  double *pre = state->pre;
  double *gen = state->gen;
  double *kb = state->kbar;
  double ls = state->scale;

  for (size_t q = 0; q < n_err; q++) {
    if (!isfinite(e[q])) {
      return -1;
    }
    double eps = e[q];
    for (size_t spk = 0; spk < n_spk; spk++) {
      const double *sec = state->s + (q * n_spk + spk) * state->sec_order;
      const double *ps = state->psi + spk * ns;
      for (size_t j = 0; j < ns; j++) {
        eps -= sec[j + 1] * ps[j];
      }
    }
    state->rhs[q] = eps;
  }
  memset(pre, 0, nprows * cols * sizeof(double));
  for (size_t i = 0; i < n_err; i++) {
    for (size_t j = 0; j < n_err; j++) {
      pre[i * cols + j] = state->rsqrt[i * n_err + j];
    }
  }
  for (size_t a = 0; a < alpha; a++) {
    for (size_t q = 0; q < n_err; q++) {
      double acc = 0.0;
      for (size_t spk = 0; spk < n_spk; spk++) {
        const double *sec = state->s + (q * n_spk + spk) * state->sec_order;
        double dot = 0.0;
        for (size_t pref = 0; pref < n_ref; pref++) {
          size_t t = spk * n_ref + pref;
          const double *hist = state->ref + pref * n1;
          for (size_t i = 0; i < n1; i++) {
            dot += hist[i] * gen[(t * n1 + i) * alpha + a];
          }
        }
        acc += sec[0] * dot;
        for (size_t j = 0; j < ns; j++) {
          acc += sec[j + 1] * gen[(n1tot + spk * ns + j) * alpha + a];
        }
      }
      pre[q * cols + n_err + a] = acc;
    }
  }
  for (size_t t = 0; t < nlines; t++) {
    for (size_t i = 1; i < n1; i++) {
      size_t row = n_err + t * n1 + i;
      for (size_t q = 0; q < n_err; q++) {
        pre[row * cols + q] = kb[(t * n1 + i - 1) * n_err + q];
      }
    }
    for (size_t i = 0; i < n1; i++) {
      size_t row = n_err + t * n1 + i;
      for (size_t a = 0; a < alpha; a++) {
        pre[row * cols + n_err + a] = ls * gen[(t * n1 + i) * alpha + a];
      }
    }
  }
  if (ns > 0) {
    for (size_t spk = 0; spk < n_spk; spk++) {
      size_t base = n1tot + spk * ns;
      for (size_t j = 0; j < ns; j++) {
        size_t row = n_err + base + j;
        for (size_t q = 0; q < n_err; q++) {
          pre[row * cols + q] = kb[(base + j) * n_err + q];
        }
      }
      for (size_t a = 0; a < alpha; a++) {
        double dot = 0.0;
        for (size_t pref = 0; pref < n_ref; pref++) {
          size_t t = spk * n_ref + pref;
          const double *hist = state->ref + pref * n1;
          for (size_t i = 0; i < n1; i++) {
            dot += hist[i] * gen[(t * n1 + i) * alpha + a];
          }
        }
        pre[(n_err + base) * cols + n_err + a] = dot;
        for (size_t j = 1; j < ns; j++) {
          double src = gen[(base + j - 1) * alpha + a];
          pre[(n_err + base + j) * cols + n_err + a] = src;
        }
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
  for (size_t i = 0; i < nrows; i++) {
    for (size_t q = 0; q < n_err; q++) {
      kb[i * n_err + q] = pre[(n_err + i) * cols + q];
    }
    for (size_t a = 0; a < alpha; a++) {
      gen[i * alpha + a] = pre[(n_err + i) * cols + n_err + a];
    }
  }
  if (Solve(state->rsqrt, state->rhs, state->rhs, n_err, state->aug) != 0) {
    return -1;
  }
  for (size_t spk = 0; spk < n_spk; spk++) {
    double *w = state->w + spk * nw;
    for (size_t pref = 0; pref < n_ref; pref++) {
      size_t t = spk * n_ref + pref;
      for (size_t i = 0; i < n; i++) {
        double add = 0.0;
        size_t row = (t * n1 + i) * n_err;
        for (size_t q = 0; q < n_err; q++) {
          add += kb[row + q] * state->rhs[q];
        }
        w[pref * n + i] = ls * w[pref * n + i] + add;
      }
    }
    if (ns > 0) {
      double *ps = state->psi + spk * ns;
      size_t base = n1tot + spk * ns;
      for (size_t j = ns; j-- > 1;) {
        double add = 0.0;
        for (size_t q = 0; q < n_err; q++) {
          add += kb[(base + j) * n_err + q] * state->rhs[q];
        }
        ps[j] = ps[j - 1] + add;
      }
      double add0 = 0.0;
      for (size_t q = 0; q < n_err; q++) {
        add0 += kb[base * n_err + q] * state->rhs[q];
      }
      ps[0] = add0;
    }
  }
  return 0;
};

int FakfCore(FakfState *state, const double *x, const double *e, double *y) {
  if (state == NULL || x == NULL || e == NULL || y == NULL) {
    return -1;
  }
  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t order = state->order;
  size_t n1 = state->n1;
  for (size_t p = 0; p < n_ref; p++) {
    double *hist = state->ref + p * n1;
    for (size_t k = n1; k-- > 1;) {
      hist[k] = hist[k - 1];
    }
    hist[0] = x[p];
  }
  if (UpdateJoint(state, e) != 0) {
    return -1;
  }
  for (size_t s = 0; s < n_spk; s++) {
    const double *w = state->w + s * n_ref * order;
    double acc = 0.0;
    for (size_t p = 0; p < n_ref; p++) {
      acc += Dot(w + p * order, state->ref + p * n1, order);
    }
    if (!isfinite(acc)) {
      return -1;
    }
    y[s] = acc;
  }
  return 0;
};

int FakfCopyWeights(const FakfState *state, double *w) {
  if (state == NULL || w == NULL) {
    return -1;
  }
  size_t n = state->n_spk * state->n_ref * state->order;
  for (size_t i = 0; i < n; i++) {
    w[i] = state->w[i];
  }
  return 0;
};
