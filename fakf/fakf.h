#ifndef FAKF_H
#define FAKF_H

#include <stddef.h>

typedef struct FakfState FakfState;

/*
 * 多通道快速阵列卡尔曼。次级路径按麦克风、扬声器排列。
 * 全体扬声器共用一套生成元，观测是全部误差麦克风。
 * 每路参考秩 2。proc_var 是次级状态过程噪声方差，只进初始方程。
 */
FakfState *FakfCreate(size_t n_ref, size_t n_spk, size_t n_err, size_t order,
                      size_t sec_order, double lambda, double meas_var,
                      double proc_var, double weight_var,
                      const double *secondary);

void FakfDestroy(FakfState *state);

int FakfCore(FakfState *state, const double *x, const double *e, double *y);

int FakfCopyWeights(const FakfState *state, double *w);

#endif
