#include "emulator.h"
#include "training_buffer.h"
#include "pca_online.h"
#include "gp_online.h"
#include "poly_online.h"
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
    int n_modes_tt, n_modes_ee, n_modes_bb, n_modes_rho;
    int is_trained;
    int pca_trained;
    int points_since_pca;
    int points_since_retrain;

    GPModel **gp_tt;
    GPModel **gp_ee;
    GPModel **gp_bb;
    GPModel **gp_rho;

    PolyModel **poly_tt;
    PolyModel **poly_ee;
    PolyModel **poly_bb;
    PolyModel **poly_rho;
};

/* --- Forward declarations --- */
static void emulator_train_full(Emulator *emu);
static void emulator_train_emulator_only(Emulator *emu);

Emulator* emulator_init(const EmulatorConfig *config) {
    Emulator *emu = calloc(1, sizeof(Emulator));
    if (!emu) return NULL;
    emu->cfg = *config;
    emu->buffer = training_buffer_create(config->buffer_capacity, config->n_params);
    if (!emu->buffer) { free(emu); return NULL; }

    emu->gp_tt = emu->gp_ee = emu->gp_bb = emu->gp_rho = NULL;
    emu->poly_tt = emu->poly_ee = emu->poly_bb = emu->poly_rho = NULL;
    emu->n_modes_tt = emu->n_modes_ee = emu->n_modes_bb = emu->n_modes_rho = 0;
    emu->is_trained = 0;
    emu->pca_trained = 0;
    emu->points_since_pca = 0;
    emu->points_since_retrain = 0;

    return emu;
}

int emulator_update(Emulator *emu, int step,
                    const double *params,
                    const double *cl_tt, const double *cl_te,
                    const double *cl_ee, const double *cl_bb,
                    double loglike) {
    if (!emu) return 0;

    int taken = training_buffer_add(emu->buffer, (double*)params,
                                    (double*)cl_tt, (double*)cl_te,
                                    (double*)cl_ee, (double*)cl_bb,
                                    loglike, step);
    if (!taken) return 0;

    int N = training_buffer_size(emu->buffer);

    if (!emu->is_trained && N >= emu->cfg.min_train_points) {
        emulator_train_full(emu);
        return 1;
    }

    if (emu->is_trained) {
        emu->points_since_pca++;
        emu->points_since_retrain++;

        if (emu->points_since_pca >= emu->cfg.pca_interval) {
            emulator_train_full(emu);
        } else if (emu->points_since_retrain >= emu->cfg.gp_interval) {
            emulator_train_emulator_only(emu);
            emu->points_since_retrain = 0;
        }
    }
    return 1;
}

void emulator_train(Emulator *emu) {
    emulator_train_full(emu);
}

static int coefficients_are_valid(double *coeffs, int N, int n_modes) {
    if (!coeffs) return 0;
    for (int i = 0; i < N * n_modes; i++) {
        if (!isfinite(coeffs[i])) return 0;
    }
    return 1;
}

/* =====================================================================
   FULL RETRAIN — unchanged
   ===================================================================== */
static void emulator_train_full(Emulator *emu) {
    if (!emu) return;
    int N = training_buffer_size(emu->buffer);
    if (N < emu->cfg.min_train_points) return;

    double *params = NULL, *cl_tt = NULL, *cl_te = NULL, *cl_ee = NULL, *cl_bb = NULL;
    double *tt_log = NULL, *ee_log = NULL, *bb_log = NULL, *rho = NULL;

    PCAModel *new_pca_tt = NULL, *new_pca_ee = NULL, *new_pca_bb = NULL, *new_pca_rho = NULL;
    double *coeffs_tt = NULL, *coeffs_ee = NULL, *coeffs_bb = NULL, *coeffs_rho = NULL;

    GPModel   **new_gp_tt   = NULL, **new_gp_ee   = NULL, **new_gp_bb   = NULL, **new_gp_rho   = NULL;
    int n_gp_tt = 0, n_gp_ee = 0, n_gp_bb = 0, n_gp_rho = 0;

    PolyModel **new_poly_tt = NULL, **new_poly_ee = NULL, **new_poly_bb = NULL, **new_poly_rho = NULL;
    int n_poly_tt = 0, n_poly_ee = 0, n_poly_bb = 0, n_poly_rho = 0;

    int success = 0;

    int D = N_SPECTRA;
    int n_params = emu->cfg.n_params;

    N = training_buffer_copy_all(emu->buffer, &params, &cl_tt, &cl_te, &cl_ee, &cl_bb);
    if (N == 0) goto cleanup;

    tt_log = malloc((size_t)N * D * sizeof(double));
    ee_log = malloc((size_t)N * D * sizeof(double));
    bb_log = malloc((size_t)N * D * sizeof(double));
    rho    = malloc((size_t)N * D * sizeof(double));
    if (!tt_log || !ee_log || !bb_log || !rho) {
        fprintf(stderr, "Memory allocation failed in emulator_train_full\n");
        goto cleanup;
    }

    memcpy(tt_log, cl_tt, (size_t)N * D * sizeof(double));
    memcpy(ee_log, cl_ee, (size_t)N * D * sizeof(double));
    memcpy(bb_log, cl_bb, (size_t)N * D * sizeof(double));
    transform_log10(tt_log, (size_t)N * D, 1e-12);
    transform_log10(ee_log, (size_t)N * D, 1e-12);
    transform_log10(bb_log, (size_t)N * D, 1e-12);
    transform_atanh_rho(cl_tt, cl_ee, cl_te, rho, N, D, 1e-12);

    new_pca_tt  = pca_train_from_spectra(tt_log, N, D, emu->cfg.max_pca_modes, 0);
    new_pca_ee  = pca_train_from_spectra(ee_log, N, D, emu->cfg.max_pca_modes, 0);
    new_pca_bb  = pca_train_from_spectra(bb_log, N, D, emu->cfg.max_pca_modes, 0);
    new_pca_rho = pca_train_from_spectra(rho,    N, D, emu->cfg.max_pca_modes, 0);

    if (!new_pca_tt || !new_pca_ee || !new_pca_bb || !new_pca_rho) {
        fprintf(stderr, "PCA training failed\n");
        goto cleanup;
    }

    int n_modes_tt  = new_pca_tt->n_modes;
    int n_modes_ee  = new_pca_ee->n_modes;
    int n_modes_bb  = new_pca_bb->n_modes;
    int n_modes_rho = new_pca_rho->n_modes;

    if (n_modes_tt == 0 || n_modes_ee == 0 || n_modes_bb == 0 || n_modes_rho == 0) {
        fprintf(stderr, "Zero PCA modes detected\n");
        goto cleanup;
    }

    coeffs_tt  = malloc((size_t)N * n_modes_tt  * sizeof(double));
    coeffs_ee  = malloc((size_t)N * n_modes_ee  * sizeof(double));
    coeffs_bb  = malloc((size_t)N * n_modes_bb  * sizeof(double));
    coeffs_rho = malloc((size_t)N * n_modes_rho * sizeof(double));
    if (!coeffs_tt || !coeffs_ee || !coeffs_bb || !coeffs_rho) {
        fprintf(stderr, "Memory allocation for coefficients failed\n");
        goto cleanup;
    }

    for (int i = 0; i < N; i++) {
        pca_project(new_pca_tt,  &tt_log[i*D],  &coeffs_tt [i*n_modes_tt]);
        pca_project(new_pca_ee,  &ee_log[i*D],  &coeffs_ee [i*n_modes_ee]);
        pca_project(new_pca_bb,  &bb_log[i*D],  &coeffs_bb [i*n_modes_bb]);
        pca_project(new_pca_rho, &rho[i*D],     &coeffs_rho[i*n_modes_rho]);
    }

    if (!coefficients_are_valid(coeffs_tt, N, n_modes_tt) ||
        !coefficients_are_valid(coeffs_ee, N, n_modes_ee) ||
        !coefficients_are_valid(coeffs_bb, N, n_modes_bb) ||
        !coefficients_are_valid(coeffs_rho, N, n_modes_rho)) {
        fprintf(stderr, "NaN/Inf in PCA coefficients; skipping full retrain.\n");
        goto cleanup;
    }

    if (emu->cfg.emulator_type == 0) {
        new_gp_tt = malloc((size_t)n_modes_tt * sizeof(GPModel*));
        if (!new_gp_tt) { fprintf(stderr, "Alloc failed for GP models\n"); goto cleanup; }
        for (int m = 0; m < n_modes_tt; m++) new_gp_tt[m] = NULL;
        n_gp_tt = n_modes_tt;
        for (int m = 0; m < n_modes_tt; m++) {
            double *y = malloc((size_t)N * sizeof(double));
            if (!y) continue;
            for (int i = 0; i < N; i++) y[i] = coeffs_tt[i * n_modes_tt + m];
            new_gp_tt[m] = online_gp_init(n_params);
            if (new_gp_tt[m]) online_gp_train(new_gp_tt[m], params, y, N);
            free(y);
        }

        new_gp_ee = malloc((size_t)n_modes_ee * sizeof(GPModel*));
        if (!new_gp_ee) { fprintf(stderr, "Alloc failed for GP models\n"); goto cleanup; }
        for (int m = 0; m < n_modes_ee; m++) new_gp_ee[m] = NULL;
        n_gp_ee = n_modes_ee;
        for (int m = 0; m < n_modes_ee; m++) {
            double *y = malloc((size_t)N * sizeof(double));
            if (!y) continue;
            for (int i = 0; i < N; i++) y[i] = coeffs_ee[i * n_modes_ee + m];
            new_gp_ee[m] = online_gp_init(n_params);
            if (new_gp_ee[m]) online_gp_train(new_gp_ee[m], params, y, N);
            free(y);
        }

        new_gp_bb = malloc((size_t)n_modes_bb * sizeof(GPModel*));
        if (!new_gp_bb) { fprintf(stderr, "Alloc failed for GP models\n"); goto cleanup; }
        for (int m = 0; m < n_modes_bb; m++) new_gp_bb[m] = NULL;
        n_gp_bb = n_modes_bb;
        for (int m = 0; m < n_modes_bb; m++) {
            double *y = malloc((size_t)N * sizeof(double));
            if (!y) continue;
            for (int i = 0; i < N; i++) y[i] = coeffs_bb[i * n_modes_bb + m];
            new_gp_bb[m] = online_gp_init(n_params);
            if (new_gp_bb[m]) online_gp_train(new_gp_bb[m], params, y, N);
            free(y);
        }

        new_gp_rho = malloc((size_t)n_modes_rho * sizeof(GPModel*));
        if (!new_gp_rho) { fprintf(stderr, "Alloc failed for GP models\n"); goto cleanup; }
        for (int m = 0; m < n_modes_rho; m++) new_gp_rho[m] = NULL;
        n_gp_rho = n_modes_rho;
        for (int m = 0; m < n_modes_rho; m++) {
            double *y = malloc((size_t)N * sizeof(double));
            if (!y) continue;
            for (int i = 0; i < N; i++) y[i] = coeffs_rho[i * n_modes_rho + m];
            new_gp_rho[m] = online_gp_init(n_params);
            if (new_gp_rho[m]) online_gp_train(new_gp_rho[m], params, y, N);
            free(y);
        }

        int all_ok = 1;
        for (int m = 0; m < n_modes_tt;  m++) if (!new_gp_tt[m]  || new_gp_tt[m]->n_train  == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_ee;  m++) if (!new_gp_ee[m]  || new_gp_ee[m]->n_train  == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_bb;  m++) if (!new_gp_bb[m]  || new_gp_bb[m]->n_train  == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_rho; m++) if (!new_gp_rho[m] || new_gp_rho[m]->n_train == 0) { all_ok = 0; break; }

        if (!all_ok) {
            fprintf(stderr, "GP training failed for some modes. Emulator will NOT be trained.\n");
            goto cleanup;
        }

        if (emu->gp_tt)  { for (int m = 0; m < emu->n_modes_tt;  m++) online_gp_free(emu->gp_tt[m]);  free(emu->gp_tt);  emu->gp_tt  = NULL; }
        if (emu->gp_ee)  { for (int m = 0; m < emu->n_modes_ee;  m++) online_gp_free(emu->gp_ee[m]);  free(emu->gp_ee);  emu->gp_ee  = NULL; }
        if (emu->gp_bb)  { for (int m = 0; m < emu->n_modes_bb;  m++) online_gp_free(emu->gp_bb[m]);  free(emu->gp_bb);  emu->gp_bb  = NULL; }
        if (emu->gp_rho) { for (int m = 0; m < emu->n_modes_rho; m++) online_gp_free(emu->gp_rho[m]); free(emu->gp_rho); emu->gp_rho = NULL; }

        emu->gp_tt  = new_gp_tt;   new_gp_tt  = NULL;
        emu->gp_ee  = new_gp_ee;   new_gp_ee  = NULL;
        emu->gp_bb  = new_gp_bb;   new_gp_bb  = NULL;
        emu->gp_rho = new_gp_rho;  new_gp_rho = NULL;
        emu->is_trained = 1;
        success = 1;

    } else {
        int degree = emu->cfg.poly_degree;

        new_poly_tt = malloc((size_t)n_modes_tt * sizeof(PolyModel*));
        if (!new_poly_tt) { fprintf(stderr, "Alloc failed for poly models\n"); goto cleanup; }
        for (int m = 0; m < n_modes_tt; m++) new_poly_tt[m] = NULL;
        n_poly_tt = n_modes_tt;
        for (int m = 0; m < n_modes_tt; m++) {
            double *y = malloc((size_t)N * sizeof(double));
            if (!y) continue;
            for (int i = 0; i < N; i++) y[i] = coeffs_tt[i * n_modes_tt + m];
            new_poly_tt[m] = online_poly_init(n_params, degree);
            if (new_poly_tt[m]) online_poly_train(new_poly_tt[m], params, y, N);
            free(y);
        }

        new_poly_ee = malloc((size_t)n_modes_ee * sizeof(PolyModel*));
        if (!new_poly_ee) { fprintf(stderr, "Alloc failed for poly models\n"); goto cleanup; }
        for (int m = 0; m < n_modes_ee; m++) new_poly_ee[m] = NULL;
        n_poly_ee = n_modes_ee;
        for (int m = 0; m < n_modes_ee; m++) {
            double *y = malloc((size_t)N * sizeof(double));
            if (!y) continue;
            for (int i = 0; i < N; i++) y[i] = coeffs_ee[i * n_modes_ee + m];
            new_poly_ee[m] = online_poly_init(n_params, degree);
            if (new_poly_ee[m]) online_poly_train(new_poly_ee[m], params, y, N);
            free(y);
        }

        new_poly_bb = malloc((size_t)n_modes_bb * sizeof(PolyModel*));
        if (!new_poly_bb) { fprintf(stderr, "Alloc failed for poly models\n"); goto cleanup; }
        for (int m = 0; m < n_modes_bb; m++) new_poly_bb[m] = NULL;
        n_poly_bb = n_modes_bb;
        for (int m = 0; m < n_modes_bb; m++) {
            double *y = malloc((size_t)N * sizeof(double));
            if (!y) continue;
            for (int i = 0; i < N; i++) y[i] = coeffs_bb[i * n_modes_bb + m];
            new_poly_bb[m] = online_poly_init(n_params, degree);
            if (new_poly_bb[m]) online_poly_train(new_poly_bb[m], params, y, N);
            free(y);
        }

        new_poly_rho = malloc((size_t)n_modes_rho * sizeof(PolyModel*));
        if (!new_poly_rho) { fprintf(stderr, "Alloc failed for poly models\n"); goto cleanup; }
        for (int m = 0; m < n_modes_rho; m++) new_poly_rho[m] = NULL;
        n_poly_rho = n_modes_rho;
        for (int m = 0; m < n_modes_rho; m++) {
            double *y = malloc((size_t)N * sizeof(double));
            if (!y) continue;
            for (int i = 0; i < N; i++) y[i] = coeffs_rho[i * n_modes_rho + m];
            new_poly_rho[m] = online_poly_init(n_params, degree);
            if (new_poly_rho[m]) online_poly_train(new_poly_rho[m], params, y, N);
            free(y);
        }

        int all_ok = 1;
        for (int m = 0; m < n_modes_tt;  m++) if (!new_poly_tt[m]  || new_poly_tt[m]->n_train  == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_ee;  m++) if (!new_poly_ee[m]  || new_poly_ee[m]->n_train  == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_bb;  m++) if (!new_poly_bb[m]  || new_poly_bb[m]->n_train  == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_rho; m++) if (!new_poly_rho[m] || new_poly_rho[m]->n_train == 0) { all_ok = 0; break; }

        if (!all_ok) {
            fprintf(stderr, "Polynomial training failed for some modes. Emulator will NOT be trained.\n");
            goto cleanup;
        }

        if (emu->poly_tt)  { for (int m = 0; m < emu->n_modes_tt;  m++) online_poly_free(emu->poly_tt[m]);  free(emu->poly_tt);  emu->poly_tt  = NULL; }
        if (emu->poly_ee)  { for (int m = 0; m < emu->n_modes_ee;  m++) online_poly_free(emu->poly_ee[m]);  free(emu->poly_ee);  emu->poly_ee  = NULL; }
        if (emu->poly_bb)  { for (int m = 0; m < emu->n_modes_bb;  m++) online_poly_free(emu->poly_bb[m]);  free(emu->poly_bb);  emu->poly_bb  = NULL; }
        if (emu->poly_rho) { for (int m = 0; m < emu->n_modes_rho; m++) online_poly_free(emu->poly_rho[m]); free(emu->poly_rho); emu->poly_rho = NULL; }

        emu->poly_tt  = new_poly_tt;   new_poly_tt  = NULL;
        emu->poly_ee  = new_poly_ee;   new_poly_ee  = NULL;
        emu->poly_bb  = new_poly_bb;   new_poly_bb  = NULL;
        emu->poly_rho = new_poly_rho;  new_poly_rho = NULL;
        emu->is_trained = 1;
        success = 1;
    }

    if (success) {
        if (emu->pca_tt)  pca_free(emu->pca_tt);
        if (emu->pca_ee)  pca_free(emu->pca_ee);
        if (emu->pca_bb)  pca_free(emu->pca_bb);
        if (emu->pca_rho) pca_free(emu->pca_rho);

        emu->pca_tt  = new_pca_tt;   new_pca_tt  = NULL;
        emu->pca_ee  = new_pca_ee;   new_pca_ee  = NULL;
        emu->pca_bb  = new_pca_bb;   new_pca_bb  = NULL;
        emu->pca_rho = new_pca_rho;  new_pca_rho = NULL;

        emu->n_modes_tt  = n_modes_tt;
        emu->n_modes_ee  = n_modes_ee;
        emu->n_modes_bb  = n_modes_bb;
        emu->n_modes_rho = n_modes_rho;
        emu->pca_trained = 1;
        emu->points_since_pca = 0;
        emu->points_since_retrain = 0;

        printf("Emulator FULL retrain (%s) on %d points: TT=%d, EE=%d, BB=%d, RHO=%d\n",
               (emu->cfg.emulator_type == 0) ? "GP" : "Polynomial",
               N, n_modes_tt, n_modes_ee, n_modes_bb, n_modes_rho);
    }

cleanup:
    free(coeffs_tt);  free(coeffs_ee);  free(coeffs_bb);  free(coeffs_rho);
    free(tt_log); free(ee_log); free(bb_log); free(rho);
    free(params); free(cl_tt); free(cl_te); free(cl_ee); free(cl_bb);

    pca_free(new_pca_tt);
    pca_free(new_pca_ee);
    pca_free(new_pca_bb);
    pca_free(new_pca_rho);

    if (new_gp_tt)  { for (int m = 0; m < n_gp_tt;  m++) online_gp_free(new_gp_tt[m]);  free(new_gp_tt); }
    if (new_gp_ee)  { for (int m = 0; m < n_gp_ee;  m++) online_gp_free(new_gp_ee[m]);  free(new_gp_ee); }
    if (new_gp_bb)  { for (int m = 0; m < n_gp_bb;  m++) online_gp_free(new_gp_bb[m]);  free(new_gp_bb); }
    if (new_gp_rho) { for (int m = 0; m < n_gp_rho; m++) online_gp_free(new_gp_rho[m]); free(new_gp_rho); }

    if (new_poly_tt)  { for (int m = 0; m < n_poly_tt;  m++) online_poly_free(new_poly_tt[m]);  free(new_poly_tt); }
    if (new_poly_ee)  { for (int m = 0; m < n_poly_ee;  m++) online_poly_free(new_poly_ee[m]);  free(new_poly_ee); }
    if (new_poly_bb)  { for (int m = 0; m < n_poly_bb;  m++) online_poly_free(new_poly_bb[m]);  free(new_poly_bb); }
    if (new_poly_rho) { for (int m = 0; m < n_poly_rho; m++) online_poly_free(new_poly_rho[m]); free(new_poly_rho); }
}

/* =====================================================================
   EMULATOR-ONLY RETRAIN — unchanged
   ===================================================================== */
static void emulator_train_emulator_only(Emulator *emu) {
    if (!emu || !emu->pca_trained) return;
    int N = training_buffer_size(emu->buffer);
    if (N < emu->cfg.min_train_points) return;

    double *params = NULL, *cl_tt = NULL, *cl_te = NULL, *cl_ee = NULL, *cl_bb = NULL;
    double *tt_log = NULL, *ee_log = NULL, *bb_log = NULL, *rho = NULL;
    double *coeffs_tt = NULL, *coeffs_ee = NULL, *coeffs_bb = NULL, *coeffs_rho = NULL;

    GPModel   **new_gp_tt   = NULL, **new_gp_ee   = NULL, **new_gp_bb   = NULL, **new_gp_rho   = NULL;
    int n_gp_tt = 0, n_gp_ee = 0, n_gp_bb = 0, n_gp_rho = 0;

    PolyModel **new_poly_tt = NULL, **new_poly_ee = NULL, **new_poly_bb = NULL, **new_poly_rho = NULL;
    int n_poly_tt = 0, n_poly_ee = 0, n_poly_bb = 0, n_poly_rho = 0;

    int success = 0;
    int D = N_SPECTRA;
    int n_params = emu->cfg.n_params;

    N = training_buffer_copy_all(emu->buffer, &params, &cl_tt, &cl_te, &cl_ee, &cl_bb);
    if (N == 0) goto cleanup;

    tt_log = malloc((size_t)N * D * sizeof(double));
    ee_log = malloc((size_t)N * D * sizeof(double));
    bb_log = malloc((size_t)N * D * sizeof(double));
    rho    = malloc((size_t)N * D * sizeof(double));
    if (!tt_log || !ee_log || !bb_log || !rho) {
        fprintf(stderr, "Memory allocation failed in emulator_train_emulator_only\n");
        goto cleanup;
    }

    memcpy(tt_log, cl_tt, (size_t)N * D * sizeof(double));
    memcpy(ee_log, cl_ee, (size_t)N * D * sizeof(double));
    memcpy(bb_log, cl_bb, (size_t)N * D * sizeof(double));
    transform_log10(tt_log, (size_t)N * D, 1e-12);
    transform_log10(ee_log, (size_t)N * D, 1e-12);
    transform_log10(bb_log, (size_t)N * D, 1e-12);
    transform_atanh_rho(cl_tt, cl_ee, cl_te, rho, N, D, 1e-12);

    int n_modes_tt  = emu->n_modes_tt;
    int n_modes_ee  = emu->n_modes_ee;
    int n_modes_bb  = emu->n_modes_bb;
    int n_modes_rho = emu->n_modes_rho;

    coeffs_tt  = malloc((size_t)N * n_modes_tt  * sizeof(double));
    coeffs_ee  = malloc((size_t)N * n_modes_ee  * sizeof(double));
    coeffs_bb  = malloc((size_t)N * n_modes_bb  * sizeof(double));
    coeffs_rho = malloc((size_t)N * n_modes_rho * sizeof(double));
    if (!coeffs_tt || !coeffs_ee || !coeffs_bb || !coeffs_rho) {
        fprintf(stderr, "Memory allocation for coefficients failed in emulator-only retrain\n");
        goto cleanup;
    }

    for (int i = 0; i < N; i++) {
        pca_project(emu->pca_tt,  &tt_log[i*D],  &coeffs_tt [i*n_modes_tt]);
        pca_project(emu->pca_ee,  &ee_log[i*D],  &coeffs_ee [i*n_modes_ee]);
        pca_project(emu->pca_bb,  &bb_log[i*D],  &coeffs_bb [i*n_modes_bb]);
        pca_project(emu->pca_rho, &rho[i*D],     &coeffs_rho[i*n_modes_rho]);
    }

    if (!coefficients_are_valid(coeffs_tt, N, n_modes_tt) ||
        !coefficients_are_valid(coeffs_ee, N, n_modes_ee) ||
        !coefficients_are_valid(coeffs_bb, N, n_modes_bb) ||
        !coefficients_are_valid(coeffs_rho, N, n_modes_rho)) {
        fprintf(stderr, "NaN/Inf in PCA coefficients; skipping emulator-only retrain.\n");
        goto cleanup;
    }

    if (emu->cfg.emulator_type == 0) {
        new_gp_tt = malloc((size_t)n_modes_tt * sizeof(GPModel*));
        if (!new_gp_tt) { fprintf(stderr, "Alloc failed for GP models\n"); goto cleanup; }
        for (int m = 0; m < n_modes_tt; m++) new_gp_tt[m] = NULL;
        n_gp_tt = n_modes_tt;
        for (int m = 0; m < n_modes_tt; m++) {
            double *y = malloc((size_t)N * sizeof(double));
            if (!y) continue;
            for (int i = 0; i < N; i++) y[i] = coeffs_tt[i * n_modes_tt + m];
            new_gp_tt[m] = online_gp_init(n_params);
            if (new_gp_tt[m]) online_gp_train(new_gp_tt[m], params, y, N);
            free(y);
        }

        new_gp_ee = malloc((size_t)n_modes_ee * sizeof(GPModel*));
        if (!new_gp_ee) { fprintf(stderr, "Alloc failed for GP models\n"); goto cleanup; }
        for (int m = 0; m < n_modes_ee; m++) new_gp_ee[m] = NULL;
        n_gp_ee = n_modes_ee;
        for (int m = 0; m < n_modes_ee; m++) {
            double *y = malloc((size_t)N * sizeof(double));
            if (!y) continue;
            for (int i = 0; i < N; i++) y[i] = coeffs_ee[i * n_modes_ee + m];
            new_gp_ee[m] = online_gp_init(n_params);
            if (new_gp_ee[m]) online_gp_train(new_gp_ee[m], params, y, N);
            free(y);
        }

        new_gp_bb = malloc((size_t)n_modes_bb * sizeof(GPModel*));
        if (!new_gp_bb) { fprintf(stderr, "Alloc failed for GP models\n"); goto cleanup; }
        for (int m = 0; m < n_modes_bb; m++) new_gp_bb[m] = NULL;
        n_gp_bb = n_modes_bb;
        for (int m = 0; m < n_modes_bb; m++) {
            double *y = malloc((size_t)N * sizeof(double));
            if (!y) continue;
            for (int i = 0; i < N; i++) y[i] = coeffs_bb[i * n_modes_bb + m];
            new_gp_bb[m] = online_gp_init(n_params);
            if (new_gp_bb[m]) online_gp_train(new_gp_bb[m], params, y, N);
            free(y);
        }

        new_gp_rho = malloc((size_t)n_modes_rho * sizeof(GPModel*));
        if (!new_gp_rho) { fprintf(stderr, "Alloc failed for GP models\n"); goto cleanup; }
        for (int m = 0; m < n_modes_rho; m++) new_gp_rho[m] = NULL;
        n_gp_rho = n_modes_rho;
        for (int m = 0; m < n_modes_rho; m++) {
            double *y = malloc((size_t)N * sizeof(double));
            if (!y) continue;
            for (int i = 0; i < N; i++) y[i] = coeffs_rho[i * n_modes_rho + m];
            new_gp_rho[m] = online_gp_init(n_params);
            if (new_gp_rho[m]) online_gp_train(new_gp_rho[m], params, y, N);
            free(y);
        }

        int all_ok = 1;
        for (int m = 0; m < n_modes_tt;  m++) if (!new_gp_tt[m]  || new_gp_tt[m]->n_train  == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_ee;  m++) if (!new_gp_ee[m]  || new_gp_ee[m]->n_train  == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_bb;  m++) if (!new_gp_bb[m]  || new_gp_bb[m]->n_train  == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_rho; m++) if (!new_gp_rho[m] || new_gp_rho[m]->n_train == 0) { all_ok = 0; break; }

        if (!all_ok) {
            fprintf(stderr, "GP-only retrain failed. Keeping old models.\n");
            goto cleanup;
        }

        if (emu->gp_tt)  { for (int m = 0; m < emu->n_modes_tt;  m++) online_gp_free(emu->gp_tt[m]);  free(emu->gp_tt);  emu->gp_tt  = NULL; }
        if (emu->gp_ee)  { for (int m = 0; m < emu->n_modes_ee;  m++) online_gp_free(emu->gp_ee[m]);  free(emu->gp_ee);  emu->gp_ee  = NULL; }
        if (emu->gp_bb)  { for (int m = 0; m < emu->n_modes_bb;  m++) online_gp_free(emu->gp_bb[m]);  free(emu->gp_bb);  emu->gp_bb  = NULL; }
        if (emu->gp_rho) { for (int m = 0; m < emu->n_modes_rho; m++) online_gp_free(emu->gp_rho[m]); free(emu->gp_rho); emu->gp_rho = NULL; }

        emu->gp_tt  = new_gp_tt;   new_gp_tt  = NULL;
        emu->gp_ee  = new_gp_ee;   new_gp_ee  = NULL;
        emu->gp_bb  = new_gp_bb;   new_gp_bb  = NULL;
        emu->gp_rho = new_gp_rho;  new_gp_rho = NULL;
        emu->is_trained = 1;
        success = 1;
        printf("Emulator GP-only retrain on %d points\n", N);

    } else {
        int degree = emu->cfg.poly_degree;

        new_poly_tt = malloc((size_t)n_modes_tt * sizeof(PolyModel*));
        if (!new_poly_tt) { fprintf(stderr, "Alloc failed for poly models\n"); goto cleanup; }
        for (int m = 0; m < n_modes_tt; m++) new_poly_tt[m] = NULL;
        n_poly_tt = n_modes_tt;
        for (int m = 0; m < n_modes_tt; m++) {
            double *y = malloc((size_t)N * sizeof(double));
            if (!y) continue;
            for (int i = 0; i < N; i++) y[i] = coeffs_tt[i * n_modes_tt + m];
            new_poly_tt[m] = online_poly_init(n_params, degree);
            if (new_poly_tt[m]) online_poly_train(new_poly_tt[m], params, y, N);
            free(y);
        }

        new_poly_ee = malloc((size_t)n_modes_ee * sizeof(PolyModel*));
        if (!new_poly_ee) { fprintf(stderr, "Alloc failed for poly models\n"); goto cleanup; }
        for (int m = 0; m < n_modes_ee; m++) new_poly_ee[m] = NULL;
        n_poly_ee = n_modes_ee;
        for (int m = 0; m < n_modes_ee; m++) {
            double *y = malloc((size_t)N * sizeof(double));
            if (!y) continue;
            for (int i = 0; i < N; i++) y[i] = coeffs_ee[i * n_modes_ee + m];
            new_poly_ee[m] = online_poly_init(n_params, degree);
            if (new_poly_ee[m]) online_poly_train(new_poly_ee[m], params, y, N);
            free(y);
        }

        new_poly_bb = malloc((size_t)n_modes_bb * sizeof(PolyModel*));
        if (!new_poly_bb) { fprintf(stderr, "Alloc failed for poly models\n"); goto cleanup; }
        for (int m = 0; m < n_modes_bb; m++) new_poly_bb[m] = NULL;
        n_poly_bb = n_modes_bb;
        for (int m = 0; m < n_modes_bb; m++) {
            double *y = malloc((size_t)N * sizeof(double));
            if (!y) continue;
            for (int i = 0; i < N; i++) y[i] = coeffs_bb[i * n_modes_bb + m];
            new_poly_bb[m] = online_poly_init(n_params, degree);
            if (new_poly_bb[m]) online_poly_train(new_poly_bb[m], params, y, N);
            free(y);
        }

        new_poly_rho = malloc((size_t)n_modes_rho * sizeof(PolyModel*));
        if (!new_poly_rho) { fprintf(stderr, "Alloc failed for poly models\n"); goto cleanup; }
        for (int m = 0; m < n_modes_rho; m++) new_poly_rho[m] = NULL;
        n_poly_rho = n_modes_rho;
        for (int m = 0; m < n_modes_rho; m++) {
            double *y = malloc((size_t)N * sizeof(double));
            if (!y) continue;
            for (int i = 0; i < N; i++) y[i] = coeffs_rho[i * n_modes_rho + m];
            new_poly_rho[m] = online_poly_init(n_params, degree);
            if (new_poly_rho[m]) online_poly_train(new_poly_rho[m], params, y, N);
            free(y);
        }

        int all_ok = 1;
        for (int m = 0; m < n_modes_tt;  m++) if (!new_poly_tt[m]  || new_poly_tt[m]->n_train  == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_ee;  m++) if (!new_poly_ee[m]  || new_poly_ee[m]->n_train  == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_bb;  m++) if (!new_poly_bb[m]  || new_poly_bb[m]->n_train  == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_rho; m++) if (!new_poly_rho[m] || new_poly_rho[m]->n_train == 0) { all_ok = 0; break; }

        if (!all_ok) {
            fprintf(stderr, "Polynomial-only retrain failed. Keeping old models.\n");
            goto cleanup;
        }

        if (emu->poly_tt)  { for (int m = 0; m < emu->n_modes_tt;  m++) online_poly_free(emu->poly_tt[m]);  free(emu->poly_tt);  emu->poly_tt  = NULL; }
        if (emu->poly_ee)  { for (int m = 0; m < emu->n_modes_ee;  m++) online_poly_free(emu->poly_ee[m]);  free(emu->poly_ee);  emu->poly_ee  = NULL; }
        if (emu->poly_bb)  { for (int m = 0; m < emu->n_modes_bb;  m++) online_poly_free(emu->poly_bb[m]);  free(emu->poly_bb);  emu->poly_bb  = NULL; }
        if (emu->poly_rho) { for (int m = 0; m < emu->n_modes_rho; m++) online_poly_free(emu->poly_rho[m]); free(emu->poly_rho); emu->poly_rho = NULL; }

        emu->poly_tt  = new_poly_tt;   new_poly_tt  = NULL;
        emu->poly_ee  = new_poly_ee;   new_poly_ee  = NULL;
        emu->poly_bb  = new_poly_bb;   new_poly_bb  = NULL;
        emu->poly_rho = new_poly_rho;  new_poly_rho = NULL;
        emu->is_trained = 1;
        success = 1;
        printf("Emulator Polynomial-only retrain on %d points (degree %d)\n", N, degree);
    }

    (void)success;

cleanup:
    free(coeffs_tt);  free(coeffs_ee);  free(coeffs_bb);  free(coeffs_rho);
    free(tt_log); free(ee_log); free(bb_log); free(rho);
    free(params); free(cl_tt); free(cl_te); free(cl_ee); free(cl_bb);

    if (new_gp_tt)  { for (int m = 0; m < n_gp_tt;  m++) online_gp_free(new_gp_tt[m]);  free(new_gp_tt); }
    if (new_gp_ee)  { for (int m = 0; m < n_gp_ee;  m++) online_gp_free(new_gp_ee[m]);  free(new_gp_ee); }
    if (new_gp_bb)  { for (int m = 0; m < n_gp_bb;  m++) online_gp_free(new_gp_bb[m]);  free(new_gp_bb); }
    if (new_gp_rho) { for (int m = 0; m < n_gp_rho; m++) online_gp_free(new_gp_rho[m]); free(new_gp_rho); }

    if (new_poly_tt)  { for (int m = 0; m < n_poly_tt;  m++) online_poly_free(new_poly_tt[m]);  free(new_poly_tt); }
    if (new_poly_ee)  { for (int m = 0; m < n_poly_ee;  m++) online_poly_free(new_poly_ee[m]);  free(new_poly_ee); }
    if (new_poly_bb)  { for (int m = 0; m < n_poly_bb;  m++) online_poly_free(new_poly_bb[m]);  free(new_poly_bb); }
    if (new_poly_rho) { for (int m = 0; m < n_poly_rho; m++) online_poly_free(new_poly_rho[m]); free(new_poly_rho); }
}

/* =====================================================================
   PREDICTION — signal-thresholded aggregation

   Each spectrum's mean LINEAR Cl is computed from the PCA mean. If a
   spectrum's mean Cl is below SIGNAL_THRESHOLD, that spectrum is
   treated as pure numerical noise and excluded from the aggregate.

   For Profile 1 (r = 0):
     TT_mean_cl  ≈ 1e2     → included
     EE_mean_cl  ≈ 1e0     → included
     BB_mean_cl  ≈ 1e-15   → excluded (numerical noise)
     RHO         included if both TT and EE have signal
                 (TE = rho·sqrt(TT·EE), so if either is noise, rho is
                  meaningless)

   This removes BB's y_std² ≈ 36-100 contribution from the aggregate,
   which had been dominating the weight (TT is ~0.1, EE is ~1-5).
   ===================================================================== */
int emulator_predict_diag(Emulator *emu, const double *params,
                          double *cl_tt, double *cl_te,
                          double *cl_ee, double *cl_bb,
                          double *uncertainty, EmulatorDiag *diag_out) {
    if (!emu || !emu->is_trained) return 0;
    int D = N_SPECTRA;

    double *coeffs_tt, *coeffs_ee, *coeffs_bb, *coeffs_rho;
    double *coeff_var_tt, *coeff_var_ee, *coeff_var_bb, *coeff_var_rho;

    if (emu->cfg.emulator_type == 0) {
        coeffs_tt = malloc(emu->n_modes_tt * sizeof(double));
        coeff_var_tt = malloc(emu->n_modes_tt * sizeof(double));
        if (!coeffs_tt || !coeff_var_tt) { free(coeffs_tt); free(coeff_var_tt); return 0; }
        for (int m = 0; m < emu->n_modes_tt; m++) {
            double mean, var;
            online_gp_predict(emu->gp_tt[m], params, &mean, &var);
            coeffs_tt[m] = mean; coeff_var_tt[m] = var;
        }
        coeffs_ee = malloc(emu->n_modes_ee * sizeof(double));
        coeff_var_ee = malloc(emu->n_modes_ee * sizeof(double));
        if (!coeffs_ee || !coeff_var_ee) { free(coeffs_tt); free(coeff_var_tt); free(coeffs_ee); free(coeff_var_ee); return 0; }
        for (int m = 0; m < emu->n_modes_ee; m++) {
            double mean, var;
            online_gp_predict(emu->gp_ee[m], params, &mean, &var);
            coeffs_ee[m] = mean; coeff_var_ee[m] = var;
        }
        coeffs_bb = malloc(emu->n_modes_bb * sizeof(double));
        coeff_var_bb = malloc(emu->n_modes_bb * sizeof(double));
        if (!coeffs_bb || !coeff_var_bb) { free(coeffs_tt); free(coeff_var_tt); free(coeffs_ee); free(coeff_var_ee); free(coeffs_bb); free(coeff_var_bb); return 0; }
        for (int m = 0; m < emu->n_modes_bb; m++) {
            double mean, var;
            online_gp_predict(emu->gp_bb[m], params, &mean, &var);
            coeffs_bb[m] = mean; coeff_var_bb[m] = var;
        }
        coeffs_rho = malloc(emu->n_modes_rho * sizeof(double));
        coeff_var_rho = malloc(emu->n_modes_rho * sizeof(double));
        if (!coeffs_rho || !coeff_var_rho) { free(coeffs_tt); free(coeff_var_tt); free(coeffs_ee); free(coeff_var_ee); free(coeffs_bb); free(coeff_var_bb); free(coeffs_rho); free(coeff_var_rho); return 0; }
        for (int m = 0; m < emu->n_modes_rho; m++) {
            double mean, var;
            online_gp_predict(emu->gp_rho[m], params, &mean, &var);
            coeffs_rho[m] = mean; coeff_var_rho[m] = var;
        }
    } else {
        coeffs_tt = malloc(emu->n_modes_tt * sizeof(double));
        coeff_var_tt = malloc(emu->n_modes_tt * sizeof(double));
        if (!coeffs_tt || !coeff_var_tt) { free(coeffs_tt); free(coeff_var_tt); return 0; }
        for (int m = 0; m < emu->n_modes_tt; m++) {
            double mean, var;
            online_poly_predict(emu->poly_tt[m], params, &mean, &var);
            coeffs_tt[m] = mean; coeff_var_tt[m] = var;
        }
        coeffs_ee = malloc(emu->n_modes_ee * sizeof(double));
        coeff_var_ee = malloc(emu->n_modes_ee * sizeof(double));
        if (!coeffs_ee || !coeff_var_ee) { free(coeffs_tt); free(coeff_var_tt); free(coeffs_ee); free(coeff_var_ee); return 0; }
        for (int m = 0; m < emu->n_modes_ee; m++) {
            double mean, var;
            online_poly_predict(emu->poly_ee[m], params, &mean, &var);
            coeffs_ee[m] = mean; coeff_var_ee[m] = var;
        }
        coeffs_bb = malloc(emu->n_modes_bb * sizeof(double));
        coeff_var_bb = malloc(emu->n_modes_bb * sizeof(double));
        if (!coeffs_bb || !coeff_var_bb) { free(coeffs_tt); free(coeff_var_tt); free(coeffs_ee); free(coeff_var_ee); free(coeffs_bb); free(coeff_var_bb); return 0; }
        for (int m = 0; m < emu->n_modes_bb; m++) {
            double mean, var;
            online_poly_predict(emu->poly_bb[m], params, &mean, &var);
            coeffs_bb[m] = mean; coeff_var_bb[m] = var;
        }
        coeffs_rho = malloc(emu->n_modes_rho * sizeof(double));
        coeff_var_rho = malloc(emu->n_modes_rho * sizeof(double));
        if (!coeffs_rho || !coeff_var_rho) { free(coeffs_tt); free(coeff_var_tt); free(coeffs_ee); free(coeff_var_ee); free(coeffs_bb); free(coeff_var_bb); free(coeffs_rho); free(coeff_var_rho); return 0; }
        for (int m = 0; m < emu->n_modes_rho; m++) {
            double mean, var;
            online_poly_predict(emu->poly_rho[m], params, &mean, &var);
            coeffs_rho[m] = mean; coeff_var_rho[m] = var;
        }
    }

    /* ============================================================
       SIGNAL-THRESHOLDED AGGREGATION
       ============================================================ */
    double tt_mean_cl = 0.0, ee_mean_cl = 0.0, bb_mean_cl = 0.0;
    for (int l = 0; l < D; l++) {
        tt_mean_cl += pow(10.0, emu->pca_tt->mean[l]);
        ee_mean_cl += pow(10.0, emu->pca_ee->mean[l]);
        bb_mean_cl += pow(10.0, emu->pca_bb->mean[l]);
    }
    tt_mean_cl /= D;
    ee_mean_cl /= D;
    bb_mean_cl /= D;

    const double SIGNAL_THRESHOLD = 1e-8;
    int tt_has_signal  = (tt_mean_cl > SIGNAL_THRESHOLD);
    int ee_has_signal  = (ee_mean_cl > SIGNAL_THRESHOLD);
    int bb_has_signal  = (bb_mean_cl > SIGNAL_THRESHOLD);
    int rho_has_signal = tt_has_signal && ee_has_signal;

    double total_recon_var = 0.0;
    double total_train_var = 0.0;

    #define ACCUM_AGG(MODES_ARRAY, N_MODES, VAR_ARRAY) do { \
        for (int m = 0; m < (N_MODES); m++) { \
            double ys = (MODES_ARRAY)[m]->y_std; \
            if (!(ys >= 1e-30)) ys = 1.0; \
            total_recon_var += (VAR_ARRAY)[m]; \
            total_train_var += ys * ys; \
        } \
    } while (0)

    if (emu->cfg.emulator_type == 0) {
        if (tt_has_signal)  ACCUM_AGG(emu->gp_tt,  emu->n_modes_tt,  coeff_var_tt);
        if (ee_has_signal)  ACCUM_AGG(emu->gp_ee,  emu->n_modes_ee,  coeff_var_ee);
        if (bb_has_signal)  ACCUM_AGG(emu->gp_bb,  emu->n_modes_bb,  coeff_var_bb);
        if (rho_has_signal) ACCUM_AGG(emu->gp_rho, emu->n_modes_rho, coeff_var_rho);
    } else {
        if (tt_has_signal)  ACCUM_AGG(emu->poly_tt,  emu->n_modes_tt,  coeff_var_tt);
        if (ee_has_signal)  ACCUM_AGG(emu->poly_ee,  emu->n_modes_ee,  coeff_var_ee);
        if (bb_has_signal)  ACCUM_AGG(emu->poly_bb,  emu->n_modes_bb,  coeff_var_bb);
        if (rho_has_signal) ACCUM_AGG(emu->poly_rho, emu->n_modes_rho, coeff_var_rho);
    }

    if (total_train_var < 1e-30) {
        if (emu->cfg.emulator_type == 0)
            ACCUM_AGG(emu->gp_tt, emu->n_modes_tt, coeff_var_tt);
        else
            ACCUM_AGG(emu->poly_tt, emu->n_modes_tt, coeff_var_tt);
    }
    if (total_train_var < 1e-30) total_train_var = 1.0;
    *uncertainty = sqrt(total_recon_var / total_train_var);

    /* ---------- Diagnostic fill ---------- */
    if (diag_out) {
        memset(diag_out, 0, sizeof(*diag_out));
        diag_out->uncertainty     = *uncertainty;
        diag_out->abs_log10_error = sqrt(total_recon_var);
        diag_out->signal_log10    = sqrt(total_train_var);
        diag_out->uncertainty_abs = diag_out->abs_log10_error / 0.01;
        diag_out->n_modes_tt  = emu->n_modes_tt;
        diag_out->n_modes_ee  = emu->n_modes_ee;
        diag_out->n_modes_bb  = emu->n_modes_bb;
        diag_out->n_modes_rho = emu->n_modes_rho;

        if (emu->cfg.emulator_type == 1) {
            double tt_sum = 0.0, ee_sum = 0.0, bb_sum = 0.0, rho_sum = 0.0;
            double tt_ysq = 0.0, ee_ysq = 0.0, bb_ysq = 0.0, rho_ysq = 0.0;

            #define PER_SPEC(ARR, N, SPECTRUM) do { \
                for (int m = 0; m < (N); m++) { \
                    double ys = (ARR)[m]->y_std; \
                    if (!(ys >= 1e-30)) ys = 1.0; \
                    SPECTRUM##_sum += coeff_var_##SPECTRUM[m]; \
                    SPECTRUM##_ysq += ys * ys; \
                } \
            } while (0)

            PER_SPEC(emu->poly_tt,  emu->n_modes_tt,  tt);
            PER_SPEC(emu->poly_ee,  emu->n_modes_ee,  ee);
            PER_SPEC(emu->poly_bb,  emu->n_modes_bb,  bb);
            PER_SPEC(emu->poly_rho, emu->n_modes_rho, rho);

            if (getenv("EMU_DIAG_SPEC")) {
                fprintf(stderr,
                    "[DIAG_SPEC] TT[y=%.3f,r=%.3e] EE[y=%.3f,r=%.3e] "
                    "BB[y=%.3f,r=%.3e] RHO[y=%.3f,r=%.3e] "
                    "flags=T%d/E%d/B%d/R%d\n",
                    sqrt(tt_ysq),  sqrt(tt_sum),
                    sqrt(ee_ysq),  sqrt(ee_sum),
                    sqrt(bb_ysq),  sqrt(bb_sum),
                    sqrt(rho_ysq), sqrt(rho_sum),
                    tt_has_signal, ee_has_signal, bb_has_signal, rho_has_signal);
            }

            double rv_w_num = 0.0, rv_w_den = 0.0, rv_max = 0.0;

            #define ACCUM_RV(MODES_ARRAY, N_MODES) do { \
                for (int m = 0; m < (N_MODES); m++) { \
                    if (!(MODES_ARRAY)[m]) continue; \
                    double rv_m = (MODES_ARRAY)[m]->residual_var; \
                    double ys_m = (MODES_ARRAY)[m]->y_std; \
                    if (!(ys_m >= 1e-30)) ys_m = 1.0; \
                    double sq_m = sqrt(rv_m); \
                    rv_w_num += rv_m * ys_m * ys_m; \
                    rv_w_den += ys_m * ys_m; \
                    if (sq_m > rv_max) rv_max = sq_m; \
                } \
            } while (0)

            if (tt_has_signal)  ACCUM_RV(emu->poly_tt,  emu->n_modes_tt);
            if (ee_has_signal)  ACCUM_RV(emu->poly_ee,  emu->n_modes_ee);
            if (bb_has_signal)  ACCUM_RV(emu->poly_bb,  emu->n_modes_bb);
            if (rho_has_signal) ACCUM_RV(emu->poly_rho, emu->n_modes_rho);

            diag_out->sqrt_rv_weighted = (rv_w_den > 0) ? sqrt(rv_w_num / rv_w_den) : 0.0;
            diag_out->sqrt_rv_max      = rv_max;

            if (emu->poly_tt[0] && emu->poly_tt[0]->knn)
                diag_out->dnorm_tt  = knn_query_dnorm(emu->poly_tt[0]->knn,  params);
            if (emu->poly_ee[0] && emu->poly_ee[0]->knn)
                diag_out->dnorm_ee  = knn_query_dnorm(emu->poly_ee[0]->knn,  params);
            if (emu->poly_bb[0] && emu->poly_bb[0]->knn)
                diag_out->dnorm_bb  = knn_query_dnorm(emu->poly_bb[0]->knn,  params);
            if (emu->poly_rho[0] && emu->poly_rho[0]->knn)
                diag_out->dnorm_rho = knn_query_dnorm(emu->poly_rho[0]->knn, params);

            diag_out->whitened = 0;
        } else {
            diag_out->sqrt_rv_weighted = sqrt(total_recon_var / total_train_var);
            diag_out->sqrt_rv_max      = diag_out->sqrt_rv_weighted;
            diag_out->dnorm_tt = diag_out->dnorm_ee = diag_out->dnorm_bb = diag_out->dnorm_rho = -1.0;
            diag_out->whitened = 0;
        }
    }

    /* Reconstruct Cls */
    double *tt_log = malloc(D * sizeof(double));
    double *ee_log = malloc(D * sizeof(double));
    double *bb_log = malloc(D * sizeof(double));
    double *rho_recon = malloc(D * sizeof(double));
    if (!tt_log || !ee_log || !bb_log || !rho_recon) {
        free(tt_log); free(ee_log); free(bb_log); free(rho_recon);
        free(coeffs_tt); free(coeff_var_tt);
        free(coeffs_ee); free(coeff_var_ee);
        free(coeffs_bb); free(coeff_var_bb);
        free(coeffs_rho); free(coeff_var_rho);
        return 0;
    }

    pca_reconstruct(emu->pca_tt,  coeffs_tt,  tt_log);
    pca_reconstruct(emu->pca_ee,  coeffs_ee,  ee_log);
    pca_reconstruct(emu->pca_bb,  coeffs_bb,  bb_log);
    pca_reconstruct(emu->pca_rho, coeffs_rho, rho_recon);

    for (int l = 0; l < D; l++) {
        cl_tt[l] = pow(10.0, tt_log[l]);
        cl_ee[l] = pow(10.0, ee_log[l]);
        cl_bb[l] = pow(10.0, bb_log[l]);
        double rho = tanh(rho_recon[l]);
        if (rho > 0.999) rho = 0.999;
        if (rho < -0.999) rho = -0.999;
        cl_te[l] = rho * sqrt(cl_tt[l] * cl_ee[l]);
    }

    free(coeffs_tt); free(coeff_var_tt);
    free(coeffs_ee); free(coeff_var_ee);
    free(coeffs_bb); free(coeff_var_bb);
    free(coeffs_rho); free(coeff_var_rho);
    free(tt_log); free(ee_log); free(bb_log); free(rho_recon);
    return 1;
}

int emulator_predict(Emulator *emu, const double *params,
                     double *cl_tt, double *cl_te,
                     double *cl_ee, double *cl_bb,
                     double *uncertainty) {
    return emulator_predict_diag(emu, params, cl_tt, cl_te, cl_ee, cl_bb,
                                 uncertainty, NULL);
}

int emulator_is_ready(const Emulator *emu) {
    if (!emu || !emu->is_trained) return 0;
    if (emu->cfg.emulator_type == 0) {
        return (emu->gp_tt && emu->gp_ee && emu->gp_bb && emu->gp_rho);
    } else {
        if (!emu->poly_tt || !emu->poly_ee || !emu->poly_bb || !emu->poly_rho) return 0;
        for (int m = 0; m < emu->n_modes_tt; m++)
            if (!emu->poly_tt[m] || emu->poly_tt[m]->n_train == 0) return 0;
        for (int m = 0; m < emu->n_modes_ee; m++)
            if (!emu->poly_ee[m] || emu->poly_ee[m]->n_train == 0) return 0;
        for (int m = 0; m < emu->n_modes_bb; m++)
            if (!emu->poly_bb[m] || emu->poly_bb[m]->n_train == 0) return 0;
        for (int m = 0; m < emu->n_modes_rho; m++)
            if (!emu->poly_rho[m] || emu->poly_rho[m]->n_train == 0) return 0;
        return 1;
    }
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

    if (emu->gp_tt)  { for (int m = 0; m < emu->n_modes_tt;  m++) online_gp_free(emu->gp_tt[m]);  free(emu->gp_tt); }
    if (emu->gp_ee)  { for (int m = 0; m < emu->n_modes_ee;  m++) online_gp_free(emu->gp_ee[m]);  free(emu->gp_ee); }
    if (emu->gp_bb)  { for (int m = 0; m < emu->n_modes_bb;  m++) online_gp_free(emu->gp_bb[m]);  free(emu->gp_bb); }
    if (emu->gp_rho) { for (int m = 0; m < emu->n_modes_rho; m++) online_gp_free(emu->gp_rho[m]); free(emu->gp_rho); }

    if (emu->poly_tt)  { for (int m = 0; m < emu->n_modes_tt;  m++) online_poly_free(emu->poly_tt[m]);  free(emu->poly_tt); }
    if (emu->poly_ee)  { for (int m = 0; m < emu->n_modes_ee;  m++) online_poly_free(emu->poly_ee[m]);  free(emu->poly_ee); }
    if (emu->poly_bb)  { for (int m = 0; m < emu->n_modes_bb;  m++) online_poly_free(emu->poly_bb[m]);  free(emu->poly_bb); }
    if (emu->poly_rho) { for (int m = 0; m < emu->n_modes_rho; m++) online_poly_free(emu->poly_rho[m]); free(emu->poly_rho); }

    free(emu);
}