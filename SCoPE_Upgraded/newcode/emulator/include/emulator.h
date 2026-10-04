#ifndef EMULATOR_H
#define EMULATOR_H

#include <stddef.h>

typedef struct Emulator Emulator;

typedef struct {
    int n_params;
    int buffer_capacity;
    int max_pca_modes;
    int pca_interval;
    int gp_interval;
    int min_train_points;
    int use_emulator_after;
    const char *model_dir;
    int emulator_type;   // 0 = GP, 1 = Polynomial
    int poly_degree;     // polynomial degree (only used if emulator_type == 1)
} EmulatorConfig;

Emulator* emulator_init(const EmulatorConfig *config);

/* Feed one accepted sample to the emulator.
 *
 * Ownership contract for cl_tt / cl_te / cl_ee / cl_bb:
 *   returns 1 -> the internal training buffer took the pointers.
 *                The caller must NOT free them; the buffer frees them
 *                only when the circular slot is overwritten, or on
 *                emulator_free().
 *   returns 0 -> the buffer did NOT take them. The caller still owns
 *                them and must free them itself.
 * params is always copied internally; the caller keeps ownership of it. */
int emulator_update(Emulator *emu, int step,
                    const double *params,
                    const double *cl_tt, const double *cl_te,
                    const double *cl_ee, const double *cl_bb,
                    double loglike);

void emulator_train(Emulator *emu);

int emulator_predict(Emulator *emu, const double *params,
                     double *cl_tt, double *cl_te,
                     double *cl_ee, double *cl_bb,
                     double *uncertainty);

int emulator_is_ready(const Emulator *emu);
void emulator_free(Emulator *emu);
int emulator_buffer_size(const Emulator *emu);

#endif
