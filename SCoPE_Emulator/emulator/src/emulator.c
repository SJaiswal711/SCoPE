#include "emulator.h"
#include "training_buffer.h"
#include "pca_online.h"
#include "gp_online.h"
#include "transforms.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define N_SPECTRA 2601

struct Emulator {
    EmulatorConfig cfg;
    TrainingBuffer *buffer;
    PCAModel *pca_tt;
    PCAModel *pca_ee;
    PCAModel *pca_bb;
    PCAModel *pca_rho;
    GPModel **gp_tt;
    int n_modes;
    int is_trained;
};

Emulator* emulator_init(const EmulatorConfig *config) {
    Emulator *emu = calloc(1, sizeof(Emulator));
    if (!emu) return NULL;
    emu->cfg = *config;
    emu->buffer = training_buffer_create(config->buffer_capacity, config->n_params);
    if (!emu->buffer) { free(emu); return NULL; }
    emu->is_trained = 0;
    return emu;
}

void emulator_update(Emulator *emu, int step,
                     const double *params,
                     const double *cl_tt, const double *cl_te,
                     const double *cl_ee, const double *cl_bb,
                     double loglike) {
    if (!emu) return;
    training_buffer_add(emu->buffer, (double*)params,
                        (double*)cl_tt, (double*)cl_te,
                        (double*)cl_ee, (double*)cl_bb,
                        loglike, step);
}

void emulator_train(Emulator *emu) {
    /* same as before – unchanged */
    if (!emu) return;
    int N = training_buffer_size(emu->buffer);
    if (N < emu->cfg.min_train_points) return;

    double *params, *cl_tt, *cl_te, *cl_ee, *cl_bb;
    N = training_buffer_copy_all(emu->buffer, &params, &cl_tt, &cl_te, &cl_ee, &cl_bb);
    if (N == 0) return;

    int D = N_SPECTRA;
    int n_params = emu->cfg.n_params;

    double *tt_log = malloc(N * D * sizeof(double));
    double *ee_log = malloc(N * D * sizeof(double));
    double *bb_log = malloc(N * D * sizeof(double));
    double *rho = malloc(N * D * sizeof(double));

    memcpy(tt_log, cl_tt, N * D * sizeof(double));
    memcpy(ee_log, cl_ee, N * D * sizeof(double));
    memcpy(bb_log, cl_bb, N * D * sizeof(double));
    transform_log10(tt_log, N * D, 1e-12);
    transform_log10(ee_log, N * D, 1e-12);
    transform_log10(bb_log, N * D, 1e-12);
    transform_atanh_rho(cl_tt, cl_ee, cl_te, rho, N, D, 1e-12);

    PCAModel *new_pca_tt = pca_train_from_spectra(tt_log, N, D, emu->cfg.max_pca_modes, 0);
    PCAModel *new_pca_ee = pca_train_from_spectra(ee_log, N, D, emu->cfg.max_pca_modes, 0);
    PCAModel *new_pca_bb = pca_train_from_spectra(bb_log, N, D, emu->cfg.max_pca_modes, 0);
    PCAModel *new_pca_rho = pca_train_from_spectra(rho, N, D, emu->cfg.max_pca_modes, 0);

    if (!new_pca_tt || !new_pca_ee || !new_pca_bb || !new_pca_rho) {
        fprintf(stderr, "PCA training failed\n");
        goto cleanup;
    }

    int n_modes = new_pca_tt->n_modes;
    if (n_modes == 0) goto cleanup;

    double *coeffs = malloc(N * n_modes * sizeof(double));
    for (int i = 0; i < N; i++) {
        double *spectrum = &tt_log[i * D];
        double *coeff_row = &coeffs[i * n_modes];
        pca_project(new_pca_tt, spectrum, coeff_row);
    }

    GPModel **new_gp_tt = malloc(n_modes * sizeof(GPModel*));
    for (int m = 0; m < n_modes; m++) {
        double *y = malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) y[i] = coeffs[i * n_modes + m];
        new_gp_tt[m] = online_gp_init(n_params);
        online_gp_train(new_gp_tt[m], params, y, N);
        free(y);
    }

    if (emu->pca_tt) pca_free(emu->pca_tt);
    if (emu->pca_ee) pca_free(emu->pca_ee);
    if (emu->pca_bb) pca_free(emu->pca_bb);
    if (emu->pca_rho) pca_free(emu->pca_rho);
    if (emu->gp_tt) {
        for (int m = 0; m < emu->n_modes; m++) online_gp_free(emu->gp_tt[m]);
        free(emu->gp_tt);
    }
    emu->pca_tt = new_pca_tt;
    emu->pca_ee = new_pca_ee;
    emu->pca_bb = new_pca_bb;
    emu->pca_rho = new_pca_rho;
    emu->gp_tt = new_gp_tt;
    emu->n_modes = n_modes;
    emu->is_trained = 1;

    printf("Emulator trained on %d points, %d PCA modes\n", N, n_modes);

cleanup:
    free(tt_log); free(ee_log); free(bb_log); free(rho);
    free(params); free(cl_tt); free(cl_te); free(cl_ee); free(cl_bb);
    if (coeffs) free(coeffs);
}

/* NEW: emulator_predict with uncertainty */
int emulator_predict(Emulator *emu, const double *params,
                     double *cl_tt, double *cl_te,
                     double *cl_ee, double *cl_bb,
                     double *uncertainty) {
    if (!emu || !emu->is_trained) return 0;
    int n_modes = emu->n_modes;
    int D = N_SPECTRA;

    double *coeffs = malloc(n_modes * sizeof(double));
    double *coeff_var = malloc(n_modes * sizeof(double));
    double total_var = 0.0;

    for (int m = 0; m < n_modes; m++) {
        double mean, var;
        online_gp_predict(emu->gp_tt[m], params, &mean, &var);
        coeffs[m] = mean;
        coeff_var[m] = var;
        total_var += var;
    }

    /* Uncertainty = RMS of predictive variances */
    *uncertainty = sqrt(total_var / n_modes);

    double *tt_log = malloc(D * sizeof(double));
    pca_reconstruct(emu->pca_tt, coeffs, tt_log);

    /* Use mean spectra for EE, BB, RHO (can be extended) */
    memcpy(cl_tt, emu->pca_tt->mean, D * sizeof(double));
    memcpy(cl_ee, emu->pca_ee->mean, D * sizeof(double));
    memcpy(cl_bb, emu->pca_bb->mean, D * sizeof(double));
    memcpy(cl_te, emu->pca_rho->mean, D * sizeof(double));

    inverse_log10(cl_tt, D);
    inverse_log10(cl_ee, D);
    inverse_log10(cl_bb, D);
    inverse_atanh_te(cl_te, cl_te, cl_tt, cl_ee, D);

    free(coeffs);
    free(coeff_var);
    free(tt_log);
    return 1;
}

int emulator_is_ready(const Emulator *emu) {
    return emu && emu->is_trained;
}
int emulator_buffer_size(const Emulator *emu) {
    return emu ? training_buffer_size(emu->buffer) : 0;
}
void emulator_free(Emulator *emu) {
    if (!emu) return;
    training_buffer_destroy(emu->buffer);
    pca_free(emu->pca_tt);
    pca_free(emu->pca_ee);
    pca_free(emu->pca_bb);
    pca_free(emu->pca_rho);
    if (emu->gp_tt) {
        for (int m = 0; m < emu->n_modes; m++) online_gp_free(emu->gp_tt[m]);
        free(emu->gp_tt);
    }
    free(emu);
}