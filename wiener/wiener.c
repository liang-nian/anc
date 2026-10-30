#include "wiener.h"

#include <math.h>
#include <stdlib.h>

static int CholeskySolve(double *R, const double *p, double *w, size_t m) {
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j <= i; j++) {
      double sum = R[i * m + j];
      for (size_t k = 0; k < j; k++) {
        sum -= R[i * m + k] * R[j * m + k];
      }
      if (i == j) {
        if (!(sum > 0.0)) {
          return -1;
        }
        R[i * m + j] = sqrt(sum);
      } else {
        R[i * m + j] = sum / R[j * m + j];
      }
    }
  }

  for (size_t i = 0; i < m; i++) {
    double sum = p[i];
    for (size_t k = 0; k < i; k++) {
      sum -= R[i * m + k] * w[k];
    }
    w[i] = sum / R[i * m + i];
  }

  for (size_t i = m; i-- > 0;) {
    double sum = w[i];
    for (size_t k = i + 1; k < m; k++) {
      sum -= R[k * m + i] * w[k];
    }
    w[i] = sum / R[i * m + i];
  }
  return 0;
};

static double FirDot(const double *x, size_t n_index, const double *w,
                     size_t order) {
  double acc = 0.0;
  size_t kmax = order;
  if (n_index + 1 < order) {
    kmax = n_index + 1;
  }
  for (size_t k = 0; k < kmax; k++) {
    acc += w[k] * x[n_index - k];
  }
  return acc;
};

int WienerFirDesign(const double *x, const double *d, size_t n, size_t order,
                    double *w, double *mse) {
  if (x == NULL || d == NULL || w == NULL || order == 0 || order > n) {
    return -1;
  }

  size_t start = order - 1;
  size_t count = n - start;
  double *R = (double *)calloc(order * order, sizeof(double));
  double *p = (double *)calloc(order, sizeof(double));
  if (R == NULL || p == NULL) {
    free(R);
    free(p);
    return -3;
  }

  for (size_t i = 0; i < order; i++) {
    for (size_t j = 0; j <= i; j++) {
      double acc = 0.0;
      for (size_t t = start; t < n; t++) {
        acc += x[t - i] * x[t - j];
      }
      acc /= (double)count;
      R[i * order + j] = acc;
      R[j * order + i] = acc;
    }
    double acc = 0.0;
    for (size_t t = start; t < n; t++) {
      acc += d[t] * x[t - i];
    }
    p[i] = acc / (double)count;
  }

  int rc = CholeskySolve(R, p, w, order);
  if (rc != 0) {
    free(R);
    free(p);
    return -2;
  }

  if (mse != NULL) {
    double err = 0.0;
    for (size_t t = start; t < n; t++) {
      double e = d[t] - FirDot(x, t, w, order);
      err += e * e;
    }
    *mse = err / (double)count;
  }

  free(R);
  free(p);
  return 0;
};

int WienerFirApply(const double *x, size_t n, const double *w, size_t order,
                   double *y) {
  if (x == NULL || w == NULL || y == NULL || order == 0) {
    return -1;
  }
  for (size_t t = 0; t < n; t++) {
    y[t] = FirDot(x, t, w, order);
  }
  return 0;
};
