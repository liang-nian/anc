#ifndef PFFXLMS_H
#define PFFXLMS_H

#include <stddef.h>

typedef struct PffxlmsState PffxlmsState;

PffxlmsState *PffxlmsCreate(size_t n_ref, size_t n_spk, size_t n_err,
                            size_t block, size_t sec_order,
                            size_t ctrl_order, double step,
                            const double *secondary);

void PffxlmsDestroy(PffxlmsState *state);

int PffxlmsPrefilter(PffxlmsState *state, const double *x);

int PffxlmsUpdate(PffxlmsState *state, const double *e);

int PffxlmsOutput(PffxlmsState *state, double *y);

int PffxlmsCore(PffxlmsState *state, const double *x, const double *e,
                double *y);

int PffxlmsCopyWeights(const PffxlmsState *state, double *w);

#endif
