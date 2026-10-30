#include "kfanc.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

struct KfancState {
  size_t n_ref;
  size_t n_spk;
  size_t n_err;
  size_t order;
  size_t sec_order;
  size_t ns;
  size_t nb;
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
  double *apc;
  double *gain;
  double *ak;
  double *pfilt;
  double *work;
  double *gram;
  double *invg;
  double *eps;
  double *aug;
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

static int Invert(const double *a, double *inv, size_t n, double *aug) {
  size_t cols = n * 2;
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      aug[i * cols + j] = a[i * n + j];
      aug[i * cols + n + j] = (i == j) ? 1.0 : 0.0;
    }
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
      for (size_t j = 0; j < cols; j++) {
        double tmp = aug[c * cols + j];
        aug[c * cols + j] = aug[piv * cols + j];
        aug[piv * cols + j] = tmp;
      }
    }
    double div = aug[c * cols + c];
    for (size_t j = 0; j < cols; j++) {
      aug[c * cols + j] /= div;
    }
    for (size_t r = 0; r < n; r++) {
      if (r == c) {
        continue;
      }
      double f = aug[r * cols + c];
      for (size_t j = 0; j < cols; j++) {
        aug[r * cols + j] -= f * aug[c * cols + j];
      }
    }
  }
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      inv[i * n + j] = aug[i * cols + n + j];
    }
  }
  return 0;
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
  free(state->apc);
  free(state->gain);
  free(state->ak);
  free(state->pfilt);
  free(state->work);
  free(state->gram);
  free(state->invg);
  free(state->eps);
  free(state->aug);
  free(state);
};

KfancState *KfancCreate(size_t n_ref, size_t n_spk, size_t n_err, size_t order,
                        size_t sec_order, double lambda, double meas_var,
                        double weight_var, const double *secondary) {
  if (n_ref == 0 || n_spk == 0 || n_err == 0 || order == 0 || sec_order == 0 ||
      !(lambda > 0.0) || !(lambda <= 1.0) || !(meas_var > 0.0) ||
      !(weight_var > 0.0) || secondary == NULL) {
    return NULL;
  }
  size_t ns = sec_order - 1;
  size_t nw = 0;
  size_t nb = 0;
  size_t nstate = 0;
  size_t nn = 0;
  size_t nsec = 0;
  size_t nck = 0;
  size_t ngram = 0;
  size_t naug = 0;
  size_t npsi = 0;
  size_t nr = 0;
  if (MulSize(n_ref, order, &nw) != 0 || nw > SIZE_MAX - ns) {
    return NULL;
  }
  nb = nw + ns;
  if (MulSize(n_spk, nb, &nstate) != 0 || MulSize(nstate, nstate, &nn) != 0 ||
      MulSize(n_err, n_spk, &nsec) != 0 ||
      MulSize(nsec, sec_order, &nsec) != 0 ||
      MulSize(nstate, n_err, &nck) != 0 || MulSize(n_err, n_err, &ngram) != 0 ||
      MulSize(n_err, n_err * 2, &naug) != 0 ||
      MulSize(n_spk, ns == 0 ? 1 : ns, &npsi) != 0 ||
      MulSize(n_ref, order, &nr) != 0) {
    return NULL;
  }

  KfancState *state = (KfancState *)calloc(1, sizeof(*state));
  if (state == NULL) {
    return NULL;
  }
  state->n_ref = n_ref;
  state->n_spk = n_spk;
  state->n_err = n_err;
  state->order = order;
  state->sec_order = sec_order;
  state->ns = ns;
  state->nb = nb;
  state->nstate = nstate;
  state->scale = 1.0 / sqrt(lambda);
  state->meas = meas_var;
  state->s = (double *)calloc(nsec, sizeof(double));
  state->w = (double *)calloc(n_spk * nw, sizeof(double));
  state->psi = (double *)calloc(npsi, sizeof(double));
  state->r = (double *)calloc(nr, sizeof(double));
  state->p = (double *)calloc(nn, sizeof(double));
  state->ck = (double *)calloc(nck, sizeof(double));
  state->pc = (double *)calloc(nck, sizeof(double));
  state->apc = (double *)calloc(nck, sizeof(double));
  state->gain = (double *)calloc(nck, sizeof(double));
  state->ak = (double *)calloc(nn, sizeof(double));
  state->pfilt = (double *)calloc(nn, sizeof(double));
  state->work = (double *)calloc(nn, sizeof(double));
  state->gram = (double *)calloc(ngram, sizeof(double));
  state->invg = (double *)calloc(ngram, sizeof(double));
  state->eps = (double *)calloc(n_err, sizeof(double));
  state->aug = (double *)calloc(naug, sizeof(double));
  if (state->s == NULL || state->w == NULL || state->psi == NULL ||
      state->r == NULL || state->p == NULL || state->ck == NULL ||
      state->pc == NULL || state->apc == NULL || state->gain == NULL ||
      state->ak == NULL || state->pfilt == NULL || state->work == NULL ||
      state->gram == NULL || state->invg == NULL || state->eps == NULL ||
      state->aug == NULL) {
    KfancDestroy(state);
    return NULL;
  }
  memcpy(state->s, secondary, nsec * sizeof(double));
  for (size_t l = 0; l < n_spk; l++) {
    for (size_t i = 0; i < nw; i++) {
      state->p[(l * nb + i) * nstate + (l * nb + i)] = weight_var;
    }
  }
  return state;
};

int KfancCore(KfancState *state, const double *x, const double *e, double *y) {
  if (state == NULL || x == NULL || e == NULL || y == NULL) {
    return -1;
  }
  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t n_err = state->n_err;
  size_t order = state->order;
  size_t ns = state->ns;
  size_t nb = state->nb;
  size_t n = state->nstate;
  size_t nw = n_ref * order;
  double scale = state->scale;
  double *r = state->r;
  double *w = state->w;
  double *psi = state->psi;
  double *s = state->s;
  double *p = state->p;
  double *ck = state->ck;
  double *pc = state->pc;
  double *apc = state->apc;
  double *gain = state->gain;
  double *ak = state->ak;
  double *pfilt = state->pfilt;
  double *work = state->work;

  for (size_t pref = 0; pref < n_ref; pref++) {
    double *line = r + pref * order;
    for (size_t k = order; k-- > 1;) {
      line[k] = line[k - 1];
    }
    line[0] = x[pref];
  }
  for (size_t spk = 0; spk < n_spk; spk++) {
    double acc = 0.0;
    const double *ws = w + spk * nw;
    for (size_t pref = 0; pref < n_ref; pref++) {
      acc += Dot(ws + pref * order, r + pref * order, order);
    }
    if (!isfinite(acc)) {
      return -1;
    }
    y[spk] = acc;
  }

  for (size_t q = 0; q < n_err; q++) {
    if (!isfinite(e[q])) {
      return -1;
    }
    double eps = e[q];
    for (size_t spk = 0; spk < n_spk; spk++) {
      const double *sec = s + (q * n_spk + spk) * state->sec_order;
      const double *ps = psi + spk * ns;
      for (size_t j = 0; j < ns; j++) {
        eps -= sec[j + 1] * ps[j];
      }
    }
    state->eps[q] = eps;
  }

  memset(ck, 0, n * n_err * sizeof(double));
  memset(ak, 0, n * n * sizeof(double));
  for (size_t spk = 0; spk < n_spk; spk++) {
    size_t base = spk * nb;
    for (size_t i = 0; i < nw; i++) {
      ak[(base + i) * n + (base + i)] = scale;
    }
    if (ns > 0) {
      for (size_t pref = 0; pref < n_ref; pref++) {
        for (size_t k = 0; k < order; k++) {
          ak[(base + nw) * n + (base + pref * order + k)] = r[pref * order + k];
        }
      }
      for (size_t j = 1; j < ns; j++) {
        ak[(base + nw + j) * n + (base + nw + j - 1)] = 1.0;
      }
    }
    for (size_t q = 0; q < n_err; q++) {
      const double *sec = s + (q * n_spk + spk) * state->sec_order;
      if (sec[0] != 0.0) {
        for (size_t pref = 0; pref < n_ref; pref++) {
          for (size_t k = 0; k < order; k++) {
            ck[(base + pref * order + k) * n_err + q] =
                sec[0] * r[pref * order + k];
          }
        }
      }
      for (size_t j = 0; j < ns; j++) {
        ck[(base + nw + j) * n_err + q] = sec[j + 1];
      }
    }
  }

  for (size_t i = 0; i < n; i++) {
    for (size_t q = 0; q < n_err; q++) {
      double sum = 0.0;
      for (size_t k = 0; k < n; k++) {
        sum += p[i * n + k] * ck[k * n_err + q];
      }
      pc[i * n_err + q] = sum;
    }
  }
  for (size_t q = 0; q < n_err; q++) {
    for (size_t rcol = 0; rcol < n_err; rcol++) {
      double sum = (q == rcol) ? state->meas : 0.0;
      for (size_t i = 0; i < n; i++) {
        sum += ck[i * n_err + q] * pc[i * n_err + rcol];
      }
      state->gram[q * n_err + rcol] = sum;
    }
  }
  if (Invert(state->gram, state->invg, n_err, state->aug) != 0) {
    return -1;
  }

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      double corr = 0.0;
      for (size_t q = 0; q < n_err; q++) {
        for (size_t rcol = 0; rcol < n_err; rcol++) {
          corr += pc[i * n_err + q] * state->invg[q * n_err + rcol] *
                  pc[j * n_err + rcol];
        }
      }
      pfilt[i * n + j] = p[i * n + j] - corr;
    }
  }
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      double sum = 0.0;
      for (size_t k = 0; k < n; k++) {
        sum += ak[i * n + k] * pfilt[k * n + j];
      }
      work[i * n + j] = sum;
    }
  }
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

  for (size_t i = 0; i < n; i++) {
    for (size_t q = 0; q < n_err; q++) {
      double sum = 0.0;
      for (size_t k = 0; k < n; k++) {
        sum += ak[i * n + k] * pc[k * n_err + q];
      }
      apc[i * n_err + q] = sum;
    }
  }
  for (size_t i = 0; i < n; i++) {
    for (size_t q = 0; q < n_err; q++) {
      double sum = 0.0;
      for (size_t rcol = 0; rcol < n_err; rcol++) {
        sum += apc[i * n_err + rcol] * state->invg[rcol * n_err + q];
      }
      gain[i * n_err + q] = sum;
    }
  }

  for (size_t spk = 0; spk < n_spk; spk++) {
    size_t base = spk * nb;
    double *ws = w + spk * nw;
    for (size_t i = 0; i < nw; i++) {
      double add = 0.0;
      for (size_t q = 0; q < n_err; q++) {
        add += gain[(base + i) * n_err + q] * state->eps[q];
      }
      ws[i] = scale * ws[i] + add;
    }
    if (ns > 0) {
      double *ps = psi + spk * ns;
      for (size_t j = ns; j-- > 1;) {
        double add = 0.0;
        for (size_t q = 0; q < n_err; q++) {
          add += gain[(base + nw + j) * n_err + q] * state->eps[q];
        }
        ps[j] = ps[j - 1] + add;
      }
      double add0 = 0.0;
      for (size_t q = 0; q < n_err; q++) {
        add0 += gain[(base + nw) * n_err + q] * state->eps[q];
      }
      ps[0] = add0;
    }
  }
  if (!isfinite(Dot(w, w, n_spk * nw))) {
    return -1;
  }
  return 0;
};

int KfancCopyWeights(const KfancState *state, double *w) {
  if (state == NULL || w == NULL) {
    return -1;
  }
  size_t n = state->n_spk * state->n_ref * state->order;
  for (size_t i = 0; i < n; i++) {
    w[i] = state->w[i];
  }
  return 0;
};
