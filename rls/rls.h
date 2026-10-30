#ifndef RLS_H
#define RLS_H

#include <stddef.h>

typedef struct RlsState RlsState;

/* 无次级路径。每个误差通道各有一套权值和逆相关矩阵。 */
RlsState *RlsCreate(size_t n_ref, size_t n_err, size_t order, double lambda,
                    double delta);

void RlsDestroy(RlsState *state);

/* y 是更新前的预测。内部用 d - y 做先验误差。 */
int RlsCore(RlsState *state, const double *x, const double *d, double *y);

int RlsCopyWeights(const RlsState *state, double *w);

#endif
