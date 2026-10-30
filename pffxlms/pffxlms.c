#include "pffxlms.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

struct PffxlmsState {
  size_t n_ref;
  size_t n_spk;
  size_t n_err;
  size_t block;
  size_t nfft;
  size_t sec_order;
  size_t ctrl_order;
  size_t sec_parts;
  size_t ctrl_parts;
  size_t ref_parts;
  double step;
  double *w_re;
  double *w_im;
  double *s_re;
  double *s_im;
  double *x_re;
  double *x_im;
  double *x_time;
  double *xp_re;
  double *xp_im;
  double *xp_time;
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

static int MulSize(size_t a, size_t b, size_t *out) {
  if (a != 0 && b > SIZE_MAX / a) {
    return -1;
  }
  *out = a * b;
  return 0;
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

static void ShiftParts(double *re, double *im, size_t nparts,
                       size_t nfft) {
  if (nparts < 2) {
    return;
  }
  for (size_t b = nparts - 1; b > 0; b--) {
    double *dst_re = re + b * nfft;
    double *src_re = re + (b - 1) * nfft;
    double *dst_im = im + b * nfft;
    double *src_im = im + (b - 1) * nfft;
    for (size_t k = 0; k < nfft; k++) {
      dst_re[k] = src_re[k];
      dst_im[k] = src_im[k];
    }
  }
};

static void FrameToPart(const double *time, double *re, double *im,
                        size_t nfft) {
  for (size_t t = 0; t < nfft; t++) {
    re[t] = time[t];
    im[t] = 0.0;
  }
  Fft(re, im, nfft, 0);
};

void PffxlmsDestroy(PffxlmsState *state) {
  if (state == NULL) {
    return;
  }
  free(state->w_re);
  free(state->w_im);
  free(state->s_re);
  free(state->s_im);
  free(state->x_re);
  free(state->x_im);
  free(state->x_time);
  free(state->xp_re);
  free(state->xp_im);
  free(state->xp_time);
  free(state->e_re);
  free(state->e_im);
  free(state->re);
  free(state->im);
  free(state);
};

PffxlmsState *PffxlmsCreate(size_t n_ref, size_t n_spk, size_t n_err,
                            size_t block, size_t sec_order,
                            size_t ctrl_order, double step,
                            const double *secondary) {
  int power_of_two = block > 0 && (block & (block - 1)) == 0;
  int ctrl_fit = ctrl_order >= block && ctrl_order % block == 0;
  if (n_ref == 0 || n_spk == 0 || n_err == 0 || !power_of_two ||
      sec_order == 0 || !ctrl_fit || block > SIZE_MAX / 2 ||
      sec_order > SIZE_MAX - (block - 1) || !(step > 0.0) ||
      secondary == NULL) {
    return NULL;
  }

  size_t nfft = block * 2;
  size_t sec_parts = (sec_order + block - 1) / block;
  size_t ctrl_parts = ctrl_order / block;
  size_t ref_parts = sec_parts > ctrl_parts ? sec_parts : ctrl_parts;

  size_t n_w = 0;
  size_t n_s = 0;
  size_t n_x = 0;
  size_t n_xt = 0;
  size_t n_xp = 0;
  size_t n_xpt = 0;
  size_t n_e = 0;
  size_t qlp = 0;
  size_t ql = 0;
  if (MulSize(n_spk, n_ref, &n_w) != 0 ||
      MulSize(n_w, ctrl_parts, &n_w) != 0 || MulSize(n_w, nfft, &n_w) != 0 ||
      MulSize(n_err, n_spk, &ql) != 0 || MulSize(ql, sec_parts, &n_s) != 0 ||
      MulSize(n_s, nfft, &n_s) != 0 || MulSize(n_ref, ref_parts, &n_x) != 0 ||
      MulSize(n_x, nfft, &n_x) != 0 || MulSize(n_ref, nfft, &n_xt) != 0 ||
      MulSize(ql, n_ref, &qlp) != 0 || MulSize(qlp, ctrl_parts, &n_xp) != 0 ||
      MulSize(n_xp, nfft, &n_xp) != 0 || MulSize(qlp, nfft, &n_xpt) != 0 ||
      MulSize(n_err, nfft, &n_e) != 0) {
    return NULL;
  }

  PffxlmsState *state = (PffxlmsState *)calloc(1, sizeof(*state));
  if (state == NULL) {
    return NULL;
  }
  state->n_ref = n_ref;
  state->n_spk = n_spk;
  state->n_err = n_err;
  state->block = block;
  state->nfft = nfft;
  state->sec_order = sec_order;
  state->ctrl_order = ctrl_order;
  state->sec_parts = sec_parts;
  state->ctrl_parts = ctrl_parts;
  state->ref_parts = ref_parts;
  state->step = step;

  state->w_re = (double *)calloc(n_w, sizeof(double));
  state->w_im = (double *)calloc(n_w, sizeof(double));
  state->s_re = (double *)calloc(n_s, sizeof(double));
  state->s_im = (double *)calloc(n_s, sizeof(double));
  state->x_re = (double *)calloc(n_x, sizeof(double));
  state->x_im = (double *)calloc(n_x, sizeof(double));
  state->x_time = (double *)calloc(n_xt, sizeof(double));
  state->xp_re = (double *)calloc(n_xp, sizeof(double));
  state->xp_im = (double *)calloc(n_xp, sizeof(double));
  state->xp_time = (double *)calloc(n_xpt, sizeof(double));
  state->e_re = (double *)calloc(n_e, sizeof(double));
  state->e_im = (double *)calloc(n_e, sizeof(double));
  state->re = (double *)calloc(nfft, sizeof(double));
  state->im = (double *)calloc(nfft, sizeof(double));
  if (state->w_re == NULL || state->w_im == NULL || state->s_re == NULL ||
      state->s_im == NULL || state->x_re == NULL || state->x_im == NULL ||
      state->x_time == NULL || state->xp_re == NULL ||
      state->xp_im == NULL || state->xp_time == NULL ||
      state->e_re == NULL || state->e_im == NULL || state->re == NULL ||
      state->im == NULL) {
    PffxlmsDestroy(state);
    return NULL;
  }

  for (size_t q = 0; q < n_err; q++) {
    for (size_t l = 0; l < n_spk; l++) {
      const double *src = secondary + (q * n_spk + l) * sec_order;
      for (size_t b = 0; b < sec_parts; b++) {
        for (size_t j = 0; j < nfft; j++) {
          state->re[j] = 0.0;
          state->im[j] = 0.0;
        }
        size_t start = b * block;
        size_t n_tap = 0;
        if (start < sec_order) {
          n_tap = sec_order - start;
          if (n_tap > block) {
            n_tap = block;
          }
        }
        for (size_t j = 0; j < n_tap; j++) {
          state->re[j] = src[start + j];
        }
        Fft(state->re, state->im, nfft, 0);
        size_t idx = ((q * n_spk + l) * sec_parts + b) * nfft;
        double *sr = state->s_re + idx;
        double *si = state->s_im + idx;
        for (size_t k = 0; k < nfft; k++) {
          sr[k] = state->re[k];
          si[k] = state->im[k];
        }
      }
    }
  }
  return state;
};

int PffxlmsPrefilter(PffxlmsState *state, const double *x) {
  if (state == NULL || x == NULL) {
    return -1;
  }

  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t n_err = state->n_err;
  size_t block = state->block;
  size_t nfft = state->nfft;
  size_t sec_parts = state->sec_parts;
  size_t ctrl_parts = state->ctrl_parts;
  size_t ref_parts = state->ref_parts;

  for (size_t p = 0; p < n_ref; p++) {
    double *time = state->x_time + p * nfft;
    ShiftAppend(time, nfft, x + p * block, block);
    double *re = state->x_re + p * ref_parts * nfft;
    double *im = state->x_im + p * ref_parts * nfft;
    ShiftParts(re, im, ref_parts, nfft);
    FrameToPart(time, re, im, nfft);
  }

  for (size_t q = 0; q < n_err; q++) {
    for (size_t l = 0; l < n_spk; l++) {
      for (size_t p = 0; p < n_ref; p++) {
        for (size_t k = 0; k < nfft; k++) {
          state->re[k] = 0.0;
          state->im[k] = 0.0;
        }
        for (size_t b = 0; b < sec_parts; b++) {
          size_t s_idx = ((q * n_spk + l) * sec_parts + b) * nfft;
          size_t x_idx = (p * ref_parts + b) * nfft;
          const double *sr = state->s_re + s_idx;
          const double *si = state->s_im + s_idx;
          const double *xr = state->x_re + x_idx;
          const double *xi = state->x_im + x_idx;
          for (size_t k = 0; k < nfft; k++) {
            state->re[k] += sr[k] * xr[k] - si[k] * xi[k];
            state->im[k] += sr[k] * xi[k] + si[k] * xr[k];
          }
        }
        Fft(state->re, state->im, nfft, 1);
        size_t xp_ch = (q * n_spk + l) * n_ref + p;
        double *xp_time = state->xp_time + xp_ch * nfft;
        ShiftAppend(xp_time, nfft, state->re + block, block);
        double *xp_re = state->xp_re + xp_ch * ctrl_parts * nfft;
        double *xp_im = state->xp_im + xp_ch * ctrl_parts * nfft;
        ShiftParts(xp_re, xp_im, ctrl_parts, nfft);
        FrameToPart(xp_time, xp_re, xp_im, nfft);
      }
    }
  }
  return 0;
};

int PffxlmsUpdate(PffxlmsState *state, const double *e) {
  if (state == NULL || e == NULL) {
    return -1;
  }

  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t n_err = state->n_err;
  size_t block = state->block;
  size_t nfft = state->nfft;
  size_t ctrl_parts = state->ctrl_parts;

  for (size_t q = 0; q < n_err; q++) {
    double *re = state->e_re + q * nfft;
    double *im = state->e_im + q * nfft;
    for (size_t t = 0; t < block; t++) {
      re[t] = 0.0;
      im[t] = 0.0;
      re[block + t] = e[q * block + t];
      im[block + t] = 0.0;
    }
    Fft(re, im, nfft, 0);
  }

  for (size_t l = 0; l < n_spk; l++) {
    for (size_t p = 0; p < n_ref; p++) {
      for (size_t b = 0; b < ctrl_parts; b++) {
        for (size_t k = 0; k < nfft; k++) {
          state->re[k] = 0.0;
          state->im[k] = 0.0;
        }
        for (size_t q = 0; q < n_err; q++) {
          size_t xp_ch = (q * n_spk + l) * n_ref + p;
          size_t xp_idx = (xp_ch * ctrl_parts + b) * nfft;
          const double *xr = state->xp_re + xp_idx;
          const double *xi = state->xp_im + xp_idx;
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
        size_t w_idx = ((l * n_ref + p) * ctrl_parts + b) * nfft;
        double *wr = state->w_re + w_idx;
        double *wi = state->w_im + w_idx;
        for (size_t k = 0; k < nfft; k++) {
          wr[k] += state->step * state->re[k];
          wi[k] += state->step * state->im[k];
        }
      }
    }
  }
  return 0;
};

int PffxlmsOutput(PffxlmsState *state, double *y) {
  if (state == NULL || y == NULL) {
    return -1;
  }

  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t block = state->block;
  size_t nfft = state->nfft;
  size_t ctrl_parts = state->ctrl_parts;
  size_t ref_parts = state->ref_parts;

  for (size_t l = 0; l < n_spk; l++) {
    for (size_t k = 0; k < nfft; k++) {
      state->re[k] = 0.0;
      state->im[k] = 0.0;
    }
    for (size_t p = 0; p < n_ref; p++) {
      for (size_t b = 0; b < ctrl_parts; b++) {
        size_t w_idx = ((l * n_ref + p) * ctrl_parts + b) * nfft;
        size_t x_idx = (p * ref_parts + b) * nfft;
        const double *wr = state->w_re + w_idx;
        const double *wi = state->w_im + w_idx;
        const double *xr = state->x_re + x_idx;
        const double *xi = state->x_im + x_idx;
        for (size_t k = 0; k < nfft; k++) {
          state->re[k] += wr[k] * xr[k] - wi[k] * xi[k];
          state->im[k] += wr[k] * xi[k] + wi[k] * xr[k];
        }
      }
    }
    Fft(state->re, state->im, nfft, 1);
    double *out = y + l * block;
    for (size_t i = 0; i < block; i++) {
      out[i] = state->re[block + i];
    }
  }
  return 0;
};

int PffxlmsCore(PffxlmsState *state, const double *x, const double *e,
                double *y) {
  if (state == NULL || x == NULL || e == NULL || y == NULL) {
    return -1;
  }
  if (PffxlmsPrefilter(state, x) != 0) {
    return -1;
  }
  if (PffxlmsUpdate(state, e) != 0) {
    return -1;
  }
  return PffxlmsOutput(state, y);
};

int PffxlmsCopyWeights(const PffxlmsState *state, double *w) {
  if (state == NULL || w == NULL) {
    return -1;
  }

  size_t n_ref = state->n_ref;
  size_t n_spk = state->n_spk;
  size_t block = state->block;
  size_t nfft = state->nfft;
  size_t ctrl_parts = state->ctrl_parts;
  size_t ctrl_order = state->ctrl_order;
  double *re = (double *)malloc(nfft * sizeof(double));
  double *im = (double *)malloc(nfft * sizeof(double));
  if (re == NULL || im == NULL) {
    free(re);
    free(im);
    return -1;
  }
  for (size_t l = 0; l < n_spk; l++) {
    for (size_t p = 0; p < n_ref; p++) {
      double *dst = w + (l * n_ref + p) * ctrl_order;
      for (size_t b = 0; b < ctrl_parts; b++) {
        size_t w_idx = ((l * n_ref + p) * ctrl_parts + b) * nfft;
        const double *wr = state->w_re + w_idx;
        const double *wi = state->w_im + w_idx;
        for (size_t k = 0; k < nfft; k++) {
          re[k] = wr[k];
          im[k] = wi[k];
        }
        Fft(re, im, nfft, 1);
        for (size_t m = 0; m < block; m++) {
          dst[b * block + m] = re[m];
        }
      }
    }
  }
  free(re);
  free(im);
  return 0;
};
