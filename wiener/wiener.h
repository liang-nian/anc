#ifndef WIENER_H
#define WIENER_H

#include <stddef.h>

int WienerFirDesign(const double *x, const double *d, size_t n, size_t order,
                    double *w, double *mse);

int WienerFirApply(const double *x, size_t n, const double *w, size_t order,
                   double *y);

#endif
