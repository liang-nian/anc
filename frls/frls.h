#ifndef FRLS_H
#define FRLS_H

#include <stddef.h>

typedef struct FrlsState FrlsState;

/*
 * 多通道滤波 RLS。参考、扬声器、误差麦克风分开计数。
 * 次级路径按麦克风、扬声器排列。抽头用全部误差更新。
 */
FrlsState *FrlsCreate(size_t n_ref, size_t n_spk, size_t n_err, size_t order,
                      size_t sec_order, double lambda, double meas_var,
                      double weight_var, const double *secondary);

void FrlsDestroy(FrlsState *state);

int FrlsCore(FrlsState *state, const double *x, const double *e, double *y);

int FrlsCopyWeights(const FrlsState *state, double *w);

#endif
