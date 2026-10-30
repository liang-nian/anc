#ifndef FDFXLMS_H
#define FDFXLMS_H

#include <stddef.h>

typedef struct FdfxlmsState FdfxlmsState;

FdfxlmsState *FdfxlmsCreate(size_t n_ref, size_t n_spk, size_t n_err,
                            size_t block, size_t sec_order, double step,
                            const double *secondary);

void FdfxlmsDestroy(FdfxlmsState *state);

int FdfxlmsPrefilter(FdfxlmsState *state, const double *x);

int FdfxlmsUpdate(FdfxlmsState *state, const double *e);

int FdfxlmsOutput(FdfxlmsState *state, double *y);

int FdfxlmsCore(FdfxlmsState *state, const double *x, const double *e,
                double *y);

int FdfxlmsCopyWeights(const FdfxlmsState *state, double *w);

#endif
