#include "fdfxlms.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

struct FdfxlmsState {
  size_t n_ref;
  size_t n_spk;
  size_t n_err;
  size_t block;
  size_t nfft;
  size_t sec_order;
  size_t sec_nfft;
  double step;
  double *w_re;
  double *w_im;
  double *s_re;
  double *s_im;
  double *x_hist;
  double *xp_hist;
  double *x_re;
  double *x_im;
  double *e_re;
  double *e_im;
  double *re;
  double *im;
};

static void Fft(double *re, double *im, size_t n, int invert) {
  for (size_t i = 1, j = 0; i < n; i++) {
    size_t bit = n >> 1;
    for (; (j & bit) != 0; bit >>= 1) {
      j ^= bit;
    }
    j ^= bit;
    if (i < j) {
      double tr = re[i];
      double ti = im[i];
      re[i] = re[j];
      im[i] = im[j];
      re[j] = tr;
      im[j] = ti;
    }
  }

  for (size_t len = 2; len <= n; len <<= 1) {
    double ang = 6.28318530717958647692 / (double)len;
    if (!invert) {
      ang = -ang;
    }
    double wlen_re = cos(ang);
    double wlen_im = sin(ang);
    size_t half = len >> 1;
    for (size_t i = 0; i < n; i += len) {
      double wr = 1.0;
      double wi = 0.0;
      for (size_t k = 0; k < half; k++) {
        size_t u = i + k;
        size_t v = u + half;
        double vr = re[v] * wr - im[v] * wi;
        double vi = re[v] * wi + im[v] * wr;
        re[v] = re[u] - vr;
        im[v] = im[u] - vi;
        re[u] += vr;
        im[u] += vi;
        double nwr = wr * wlen_re - wi * wlen_im;
        wi = wr * wlen_im + wi * wlen_re;
        wr = nwr;
      }
    }
  }

  if (invert) {
    double inv = 1.0 / (double)n;
    for (size_t i = 0; i < n; i++) {
      re[i] *= inv;
      im[i] *= inv;
    }
  }
};

static size_t NextPow2(size_t n) {
  size_t p = 1;
  while (p < n) {
    if (p > (SIZE_MAX >> 1)) {
      return 0;
    }
    p <<= 1;
  }
  return p;
};

static void ShiftAppend(double *line, size_t n, const double *src,
                        size_t block) {
  for (size_t i = 0; i + block < n; i++) {
    line[i] = line[i + block];
  }
  for (size_t i = 0; i < block; i++) {
    line[n - block + i] = src[i];
  }
};

void FdfxlmsDestroy(FdfxlmsState *state) {
  if (state == NULL) {
    return;
  }
  free(state->w_re);
  free(state->w_im);
  free(state->s_re);
  free(state->s_im);
  free(state->x_hist);
  free(state->xp_hist);
  free(state->x_re);
  free(state->x_im);
  free(state->e_re);
  free(state->e_im);
  free(state->re);
  free(state->im);
  free(state);
};

FdfxlmsState *FdfxlmsCreate(size_t n_ref, size_t n_spk, size_t n_err,
                            size_t block, size_t sec_order, double step,
                            const double *secondary) {
  int power_of_two = block > 0 && (block & (block - 1)) == 0;
  if (n_ref == 0 || n_spk == 0 || n_err == 0 || !power_of_two ||
      sec_order == 0 || block > SIZE_MAX / 2 ||
      sec_order - 1 > SIZE_MAX - block || !(step > 0.0) ||
      secondary == NULL) {
    return NULL;
  }

  size_t nfft = block * 2;
  size_t sec_nfft = NextPow2(block + sec_order - 1);
  if (sec_nfft == 0) {
    return NULL;
  }
  if (sec_nfft < nfft) {
    sec_nfft = nfft;
  }

  FdfxlmsState *state = (FdfxlmsState *)calloc(1, sizeof(*state));
  if (state == NULL) {
    return NULL;
  }
  state->n_ref = n_ref;
  state->n_spk = n_spk;
  state->n_err = n_err;
  state->block = block;
  state->nfft = nfft;
  state->sec_order = sec_order;
  state->sec_nfft = sec_nfft;
  state->step = step;

  size_t n_w = n_spk * n_ref * nfft;
  size_t n_s = n_err * n_spk * sec_nfft;
  size_t n_x = n_ref * sec_nfft;
  size_t n_xp = n_err * n_spk * n_ref * nfft;
  size_t n_e = n_err * nfft;
  state->w_re = (double *)calloc(n_w, sizeof(double));
  state->w_im = (double *)calloc(n_w, sizeof(double));
  state->s_re = (double *)calloc(n_s, sizeof(double));
  state->s_im = (double *)calloc(n_s, sizeof(double));
  state->x_hist = (double *)calloc(n_x, sizeof(double));
  state->xp_hist = (double *)calloc(n_xp, sizeof(double));
  state->x_re = (double *)calloc(n_x, sizeof(double));
  state->x_im = (double *)calloc(n_x, sizeof(double));
  state->e_re = (double *)calloc(n_e, sizeof(double));
  state->e_im = (double *)calloc(n_e, sizeof(double));
  state->re = (double *)calloc(sec_nfft, sizeof(double));
  state->im = (double *)calloc(sec_nfft, sizeof(double));
  if (state->w_re == NULL || state->w_im == NULL || state->s_re == NULL ||
      state->s_im == NULL || state->x_hist == NULL ||
      state->xp_hist == NULL || state->x_re == NULL ||
      state->x_im == NULL || state->e_re == NULL || state->e_im == NULL ||
      state->re == NULL || state->im == NULL) {
    FdfxlmsDestroy(state);
    return NULL;
  }

  for (size_t q = 0; q < n_err; q++) {
    for (size_t l = 0; l < n_spk; l++) {
      const double *src = secondary + (q * n_spk + l) * sec_order;
      for (size_t j = 0; j < sec_order; j++) {
        state->re[j] = src[j];
      }
      for (size_t j = sec_order; j < sec_nfft; j++) {
        state->re[j] = 0.0;
      }
      for (size_t j = 0; j < sec_nfft; j++) {
        state->im[j] = 0.0;
      }
      Fft(state->re, state->im, sec_nfft, 0);
      double *sr = state->s_re + (q * n_spk + l) * sec_nfft;
      double *si = state->s_im + (q * n_spk + l) * sec_nfft;
      for (size_t k = 0; k < sec_nfft; k++) {
        sr[k] = state->re[k];
        si[k] = state->im[k];
      }
    }
  }
  return state;
};

int FdfxlmsPrefilter(FdfxlmsState *state, const double *x) {
  if (state == NULL || x == NULL) {
    return -1;
  }

  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t n_err = state->n_err;
  size_t block = state->block;
  size_t nfft = state->nfft;
  size_t sec_nfft = state->sec_nfft;

  for (size_t p = 0; p < n_ref; p++) {
    ShiftAppend(state->x_hist + p * sec_nfft, sec_nfft, x + p * block,
                block);
  }

  for (size_t p = 0; p < n_ref; p++) {
    double *re = state->x_re + p * sec_nfft;
    double *im = state->x_im + p * sec_nfft;
    const double *hist = state->x_hist + p * sec_nfft;
    for (size_t t = 0; t < sec_nfft; t++) {
      re[t] = hist[t];
      im[t] = 0.0;
    }
    Fft(re, im, sec_nfft, 0);
  }

  for (size_t q = 0; q < n_err; q++) {
    for (size_t l = 0; l < n_spk; l++) {
      const double *sr = state->s_re + (q * n_spk + l) * sec_nfft;
      const double *si = state->s_im + (q * n_spk + l) * sec_nfft;
      for (size_t p = 0; p < n_ref; p++) {
        const double *xr = state->x_re + p * sec_nfft;
        const double *xi = state->x_im + p * sec_nfft;
        for (size_t k = 0; k < sec_nfft; k++) {
          state->re[k] = sr[k] * xr[k] - si[k] * xi[k];
          state->im[k] = sr[k] * xi[k] + si[k] * xr[k];
        }
        Fft(state->re, state->im, sec_nfft, 1);
        double *xp =
            state->xp_hist + ((q * n_spk + l) * n_ref + p) * nfft;
        ShiftAppend(xp, nfft, state->re + (sec_nfft - block), block);
      }
    }
  }
  return 0;
};

int FdfxlmsUpdate(FdfxlmsState *state, const double *e) {
  if (state == NULL || e == NULL) {
    return -1;
  }

  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t n_err = state->n_err;
  size_t block = state->block;
  size_t nfft = state->nfft;

  for (size_t q = 0; q < n_err; q++) {
    double *re = state->e_re + q * nfft;
    double *im = state->e_im + q * nfft;
    for (size_t t = 0; t < block; t++) {
      re[t] = 0.0;
      im[t] = 0.0;
    }
    const double *err = e + q * block;
    for (size_t t = 0; t < block; t++) {
      re[block + t] = err[t];
      im[block + t] = 0.0;
    }
    Fft(re, im, nfft, 0);
  }

  for (size_t l = 0; l < n_spk; l++) {
    for (size_t p = 0; p < n_ref; p++) {
      for (size_t k = 0; k < nfft; k++) {
        state->re[k] = 0.0;
        state->im[k] = 0.0;
      }
      for (size_t q = 0; q < n_err; q++) {
        const double *xp =
            state->xp_hist + ((q * n_spk + l) * n_ref + p) * nfft;
        double *xr = state->x_re;
        double *xi = state->x_im;
        for (size_t t = 0; t < nfft; t++) {
          xr[t] = xp[t];
          xi[t] = 0.0;
        }
        Fft(xr, xi, nfft, 0);
        const double *er = state->e_re + q * nfft;
        const double *ei = state->e_im + q * nfft;
        for (size_t k = 0; k < nfft; k++) {
          state->re[k] += er[k] * xr[k] + ei[k] * xi[k];
          state->im[k] += ei[k] * xr[k] - er[k] * xi[k];
        }
      }

      Fft(state->re, state->im, nfft, 1);
      for (size_t k = 0; k < block; k++) {
        state->im[k] = 0.0;
      }
      for (size_t k = block; k < nfft; k++) {
        state->re[k] = 0.0;
        state->im[k] = 0.0;
      }
      Fft(state->re, state->im, nfft, 0);

      double *wr = state->w_re + (l * n_ref + p) * nfft;
      double *wi = state->w_im + (l * n_ref + p) * nfft;
      for (size_t k = 0; k < nfft; k++) {
        wr[k] += state->step * state->re[k];
        wi[k] += state->step * state->im[k];
      }
    }
  }
  return 0;
};

int FdfxlmsOutput(FdfxlmsState *state, double *y) {
  if (state == NULL || y == NULL) {
    return -1;
  }

  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t block = state->block;
  size_t nfft = state->nfft;
  size_t sec_nfft = state->sec_nfft;

  for (size_t p = 0; p < n_ref; p++) {
    const double *hist =
        state->x_hist + p * sec_nfft + (sec_nfft - nfft);
    double *re = state->x_re + p * sec_nfft;
    double *im = state->x_im + p * sec_nfft;
    for (size_t t = 0; t < nfft; t++) {
      re[t] = hist[t];
      im[t] = 0.0;
    }
    Fft(re, im, nfft, 0);
  }

  for (size_t l = 0; l < n_spk; l++) {
    for (size_t k = 0; k < nfft; k++) {
      state->re[k] = 0.0;
      state->im[k] = 0.0;
    }
    for (size_t p = 0; p < n_ref; p++) {
      const double *wr = state->w_re + (l * n_ref + p) * nfft;
      const double *wi = state->w_im + (l * n_ref + p) * nfft;
      const double *xr = state->x_re + p * sec_nfft;
      const double *xi = state->x_im + p * sec_nfft;
      for (size_t k = 0; k < nfft; k++) {
        state->re[k] += wr[k] * xr[k] - wi[k] * xi[k];
        state->im[k] += wr[k] * xi[k] + wi[k] * xr[k];
      }
    }
    Fft(state->re, state->im, nfft, 1);
    double *out = y + l * block;
    for (size_t i = 0; i < block; i++) {
      out[i] = state->re[nfft - block + i];
    }
  }
  return 0;
};

int FdfxlmsCore(FdfxlmsState *state, const double *x, const double *e,
                double *y) {
  if (state == NULL || x == NULL || e == NULL || y == NULL) {
    return -1;
  }
  if (FdfxlmsPrefilter(state, x) != 0) {
    return -1;
  }
  if (FdfxlmsUpdate(state, e) != 0) {
    return -1;
  }
  return FdfxlmsOutput(state, y);
};

int FdfxlmsCopyWeights(const FdfxlmsState *state, double *w) {
  if (state == NULL || w == NULL) {
    return -1;
  }

  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t block = state->block;
  size_t nfft = state->nfft;
  double *re = (double *)malloc(nfft * sizeof(double));
  double *im = (double *)malloc(nfft * sizeof(double));
  if (re == NULL || im == NULL) {
    free(re);
    free(im);
    return -1;
  }
  for (size_t l = 0; l < n_spk; l++) {
    for (size_t p = 0; p < n_ref; p++) {
      const double *wr = state->w_re + (l * n_ref + p) * nfft;
      const double *wi = state->w_im + (l * n_ref + p) * nfft;
      for (size_t k = 0; k < nfft; k++) {
        re[k] = wr[k];
        im[k] = wi[k];
      }
      Fft(re, im, nfft, 1);
      double *dst = w + (l * n_ref + p) * block;
      for (size_t m = 0; m < block; m++) {
        dst[m] = re[m];
      }
    }
  }
  free(re);
  free(im);
  return 0;
};
