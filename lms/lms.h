#ifndef LMS_H
#define LMS_H

#include <stddef.h>

typedef struct LmsState LmsState;

LmsState *LmsCreate(size_t n_ref, size_t n_err, size_t order, double step);

void LmsDestroy(LmsState *state);

int LmsStep(LmsState *state, const double *x, const double *d, double *e);

int LmsCopyWeights(const LmsState *state, double *w);

#endif
