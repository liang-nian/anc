#ifndef KFANC_H
#define KFANC_H

#include <stddef.h>

typedef struct KfancState KfancState;

/*
 * 多通道卡尔曼主动降噪。次级路径按麦克风、扬声器排列。
 * 状态把各扬声器的权值和次级延迟线接在一起，观测是全部误差麦克风。
 * 返回的 y 是更新前的扬声器。
 */
KfancState *KfancCreate(size_t n_ref, size_t n_spk, size_t n_err,
                        size_t order, size_t sec_order, double lambda,
                        double meas_var, double weight_var,
                        const double *secondary);

void KfancDestroy(KfancState *state);

int KfancCore(KfancState *state, const double *x, const double *e, double *y);

int KfancCopyWeights(const KfancState *state, double *w);

#endif
