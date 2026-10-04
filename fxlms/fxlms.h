#ifndef FXLMS_H
#define FXLMS_H

#include <stddef.h>

typedef struct FxlmsState FxlmsState;

FxlmsState *FxlmsCreate(size_t n_ref, size_t n_spk, size_t n_err, size_t order,
                        size_t sec_order, double step,
                        const double *secondary);

void FxlmsDestroy(FxlmsState *state);

int FxlmsPrefilter(FxlmsState *state, const double *x);

int FxlmsUpdate(FxlmsState *state, const double *e);

int FxlmsOutput(FxlmsState *state, double *y);

int FxlmsCore(FxlmsState *state, const double *x, const double *e, double *y);

int FxlmsCopyWeights(const FxlmsState *state, double *w);

#endif
