#ifndef NBFXLMS_H
#define NBFXLMS_H

#include <stddef.h>

typedef struct NbfxlmsState NbfxlmsState;

NbfxlmsState *NbfxlmsCreate(size_t n_harm, size_t n_spk, size_t n_err,
                            size_t sec_order, double step,
                            const double *secondary, const double *order);

void NbfxlmsDestroy(NbfxlmsState *state);

int NbfxlmsPrefilter(NbfxlmsState *state, double omega0);

int NbfxlmsUpdate(NbfxlmsState *state, const double *e);

int NbfxlmsOutput(NbfxlmsState *state, double *y);

int NbfxlmsCore(NbfxlmsState *state, double omega0, const double *e,
                double *y);

int NbfxlmsCopyWeights(const NbfxlmsState *state, double *w);

#endif
