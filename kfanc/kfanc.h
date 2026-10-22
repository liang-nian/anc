#ifndef KFANC_H
#define KFANC_H

#include <stddef.h>

typedef struct KfancState KfancState;

KfancState *KfancCreate(size_t order, size_t sec_order, double lambda,
                        double meas_var, double weight_var,
                        const double *secondary);

void KfancDestroy(KfancState *state);

int KfancCore(KfancState *state, double x, double e, double *y);

int KfancCopyWeights(const KfancState *state, double *w);

#endif
