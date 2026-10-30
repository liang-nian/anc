#ifndef FARLS_H
#define FARLS_H

#include <stddef.h>

typedef struct FarlsState FarlsState;

/*
 * 多通道快速阵列 filtered-RLS。次级路径按麦克风、扬声器排列。
 * 每只扬声器的参考抽头用全部误差麦克风更新。全体扬声器共用一套
 * 生成元，左上角是误差麦克风维数的平方根因子，每路参考秩 2。
 */
FarlsState *FarlsCreate(size_t n_ref, size_t n_spk, size_t n_err,
                        size_t order, size_t sec_order, double lambda,
                        double meas_var, double weight_var,
                        const double *secondary);

void FarlsDestroy(FarlsState *state);

int FarlsCore(FarlsState *state, const double *x, const double *e, double *y);

int FarlsCopyWeights(const FarlsState *state, double *w);

#endif
