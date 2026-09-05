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

    // GP models
    GPModel **gp_tt;
    GPModel **gp_ee;
    GPModel **gp_bb;
    GPModel **gp_rho;

    // Polynomial models
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

    int N = training_buffer_size(emu->buffer);

    if (!emu->is_trained && N >= emu->cfg.min_train_points) {
        emulator_train_full(emu);
        return;
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
}

void emulator_train(Emulator *emu) {
    emulator_train_full(emu);
}

/* ---------- Helper: check coefficients for NaNs/Infs ---------- */
static int coefficients_are_valid(double *coeffs, int N, int n_modes) {
    for (int i = 0; i < N * n_modes; i++) {
        if (!isfinite(coeffs[i])) return 0;
    }
    return 1;
}

/* ---------- Full retrain (PCA + emulator) ---------- */
static void emulator_train_full(Emulator *emu) {
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
    if (!tt_log || !ee_log || !bb_log || !rho) {
        fprintf(stderr, "Memory allocation failed in emulator_train_full\n");
        goto cleanup_free_temps;
    }

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
        goto cleanup_free_new_pcas;
    }

    int n_modes_tt = new_pca_tt->n_modes;
    int n_modes_ee = new_pca_ee->n_modes;
    int n_modes_bb = new_pca_bb->n_modes;
    int n_modes_rho = new_pca_rho->n_modes;

    if (n_modes_tt == 0 || n_modes_ee == 0 || n_modes_bb == 0 || n_modes_rho == 0) {
        fprintf(stderr, "Zero PCA modes detected\n");
        goto cleanup_free_new_pcas;
    }

    double *coeffs_tt = malloc(N * n_modes_tt * sizeof(double));
    double *coeffs_ee = malloc(N * n_modes_ee * sizeof(double));
    double *coeffs_bb = malloc(N * n_modes_bb * sizeof(double));
    double *coeffs_rho = malloc(N * n_modes_rho * sizeof(double));
    if (!coeffs_tt || !coeffs_ee || !coeffs_bb || !coeffs_rho) {
        fprintf(stderr, "Memory allocation for coefficients failed\n");
        free(coeffs_tt); free(coeffs_ee); free(coeffs_bb); free(coeffs_rho);
        goto cleanup_free_new_pcas;
    }

    for (int i = 0; i < N; i++) {
        pca_project(new_pca_tt, &tt_log[i*D], &coeffs_tt[i*n_modes_tt]);
        pca_project(new_pca_ee, &ee_log[i*D], &coeffs_ee[i*n_modes_ee]);
        pca_project(new_pca_bb, &bb_log[i*D], &coeffs_bb[i*n_modes_bb]);
        pca_project(new_pca_rho, &rho[i*D], &coeffs_rho[i*n_modes_rho]);
    }

    // Validate coefficients
    if (!coefficients_are_valid(coeffs_tt, N, n_modes_tt) ||
        !coefficients_are_valid(coeffs_ee, N, n_modes_ee) ||
        !coefficients_are_valid(coeffs_bb, N, n_modes_bb) ||
        !coefficients_are_valid(coeffs_rho, N, n_modes_rho)) {
        fprintf(stderr, "NaN/Inf in PCA coefficients; skipping full retrain.\n");
        free(coeffs_tt); free(coeffs_ee); free(coeffs_bb); free(coeffs_rho);
        goto cleanup_free_new_pcas;
    }

    /* ----- Train the selected emulator type ----- */
    int success = 0;
    if (emu->cfg.emulator_type == 0) {
        // ---------- GP ----------
        GPModel **new_gp_tt = NULL, **new_gp_ee = NULL, **new_gp_bb = NULL, **new_gp_rho = NULL;

        new_gp_tt = malloc(n_modes_tt * sizeof(GPModel*));
        if (!new_gp_tt) goto gp_cleanup;
        for (int m = 0; m < n_modes_tt; m++) {
            double *y = malloc(N * sizeof(double));
            if (!y) { new_gp_tt[m] = NULL; continue; }
            for (int i = 0; i < N; i++) y[i] = coeffs_tt[i * n_modes_tt + m];
            new_gp_tt[m] = online_gp_init(n_params);
            if (!new_gp_tt[m]) { free(y); continue; }
            online_gp_train(new_gp_tt[m], params, y, N);
            free(y);
        }
        new_gp_ee = malloc(n_modes_ee * sizeof(GPModel*));
        if (!new_gp_ee) goto gp_cleanup;
        for (int m = 0; m < n_modes_ee; m++) {
            double *y = malloc(N * sizeof(double));
            if (!y) { new_gp_ee[m] = NULL; continue; }
            for (int i = 0; i < N; i++) y[i] = coeffs_ee[i * n_modes_ee + m];
            new_gp_ee[m] = online_gp_init(n_params);
            if (!new_gp_ee[m]) { free(y); continue; }
            online_gp_train(new_gp_ee[m], params, y, N);
            free(y);
        }
        new_gp_bb = malloc(n_modes_bb * sizeof(GPModel*));
        if (!new_gp_bb) goto gp_cleanup;
        for (int m = 0; m < n_modes_bb; m++) {
            double *y = malloc(N * sizeof(double));
            if (!y) { new_gp_bb[m] = NULL; continue; }
            for (int i = 0; i < N; i++) y[i] = coeffs_bb[i * n_modes_bb + m];
            new_gp_bb[m] = online_gp_init(n_params);
            if (!new_gp_bb[m]) { free(y); continue; }
            online_gp_train(new_gp_bb[m], params, y, N);
            free(y);
        }
        new_gp_rho = malloc(n_modes_rho * sizeof(GPModel*));
        if (!new_gp_rho) goto gp_cleanup;
        for (int m = 0; m < n_modes_rho; m++) {
            double *y = malloc(N * sizeof(double));
            if (!y) { new_gp_rho[m] = NULL; continue; }
            for (int i = 0; i < N; i++) y[i] = coeffs_rho[i * n_modes_rho + m];
            new_gp_rho[m] = online_gp_init(n_params);
            if (!new_gp_rho[m]) { free(y); continue; }
            online_gp_train(new_gp_rho[m], params, y, N);
            free(y);
        }

        // Verify all models were created
        int all_ok = 1;
        for (int m = 0; m < n_modes_tt; m++) if (!new_gp_tt[m] || new_gp_tt[m]->n_train == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_ee; m++) if (!new_gp_ee[m] || new_gp_ee[m]->n_train == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_bb; m++) if (!new_gp_bb[m] || new_gp_bb[m]->n_train == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_rho; m++) if (!new_gp_rho[m] || new_gp_rho[m]->n_train == 0) { all_ok = 0; break; }

        if (!all_ok) {
            fprintf(stderr, "GP training failed for some modes. Emulator will NOT be trained.\n");
            // Free partial allocations
            if (new_gp_tt) { for (int m = 0; m < n_modes_tt; m++) online_gp_free(new_gp_tt[m]); free(new_gp_tt); }
            if (new_gp_ee) { for (int m = 0; m < n_modes_ee; m++) online_gp_free(new_gp_ee[m]); free(new_gp_ee); }
            if (new_gp_bb) { for (int m = 0; m < n_modes_bb; m++) online_gp_free(new_gp_bb[m]); free(new_gp_bb); }
            if (new_gp_rho) { for (int m = 0; m < n_modes_rho; m++) online_gp_free(new_gp_rho[m]); free(new_gp_rho); }
            goto cleanup_new_pcas_no_coeff;
        }

        // Free old GP models
        if (emu->gp_tt) { for (int m = 0; m < emu->n_modes_tt; m++) online_gp_free(emu->gp_tt[m]); free(emu->gp_tt); emu->gp_tt = NULL; }
        if (emu->gp_ee) { for (int m = 0; m < emu->n_modes_ee; m++) online_gp_free(emu->gp_ee[m]); free(emu->gp_ee); emu->gp_ee = NULL; }
        if (emu->gp_bb) { for (int m = 0; m < emu->n_modes_bb; m++) online_gp_free(emu->gp_bb[m]); free(emu->gp_bb); emu->gp_bb = NULL; }
        if (emu->gp_rho) { for (int m = 0; m < emu->n_modes_rho; m++) online_gp_free(emu->gp_rho[m]); free(emu->gp_rho); emu->gp_rho = NULL; }

        emu->gp_tt = new_gp_tt;
        emu->gp_ee = new_gp_ee;
        emu->gp_bb = new_gp_bb;
        emu->gp_rho = new_gp_rho;
        emu->is_trained = 1;
        success = 1;
        goto gp_cleanup_done;

gp_cleanup:
        fprintf(stderr, "Allocation failed for GP models\n");
        if (new_gp_tt) { for (int m = 0; m < n_modes_tt; m++) online_gp_free(new_gp_tt[m]); free(new_gp_tt); }
        if (new_gp_ee) { for (int m = 0; m < n_modes_ee; m++) online_gp_free(new_gp_ee[m]); free(new_gp_ee); }
        if (new_gp_bb) { for (int m = 0; m < n_modes_bb; m++) online_gp_free(new_gp_bb[m]); free(new_gp_bb); }
        if (new_gp_rho) { for (int m = 0; m < n_modes_rho; m++) online_gp_free(new_gp_rho[m]); free(new_gp_rho); }
        // Do not set is_trained
gp_cleanup_done:
        ;
    } else {
        // ---------- Polynomial ----------
        int degree = emu->cfg.poly_degree;
        PolyModel **new_poly_tt = NULL, **new_poly_ee = NULL, **new_poly_bb = NULL, **new_poly_rho = NULL;

        new_poly_tt = malloc(n_modes_tt * sizeof(PolyModel*));
        if (!new_poly_tt) goto poly_cleanup;
        for (int m = 0; m < n_modes_tt; m++) {
            double *y = malloc(N * sizeof(double));
            if (!y) { new_poly_tt[m] = NULL; continue; }
            for (int i = 0; i < N; i++) y[i] = coeffs_tt[i * n_modes_tt + m];
            new_poly_tt[m] = online_poly_init(n_params, degree);
            if (!new_poly_tt[m]) { free(y); continue; }
            online_poly_train(new_poly_tt[m], params, y, N);
            free(y);
        }
        new_poly_ee = malloc(n_modes_ee * sizeof(PolyModel*));
        if (!new_poly_ee) goto poly_cleanup;
        for (int m = 0; m < n_modes_ee; m++) {
            double *y = malloc(N * sizeof(double));
            if (!y) { new_poly_ee[m] = NULL; continue; }
            for (int i = 0; i < N; i++) y[i] = coeffs_ee[i * n_modes_ee + m];
            new_poly_ee[m] = online_poly_init(n_params, degree);
            if (!new_poly_ee[m]) { free(y); continue; }
            online_poly_train(new_poly_ee[m], params, y, N);
            free(y);
        }
        new_poly_bb = malloc(n_modes_bb * sizeof(PolyModel*));
        if (!new_poly_bb) goto poly_cleanup;
        for (int m = 0; m < n_modes_bb; m++) {
            double *y = malloc(N * sizeof(double));
            if (!y) { new_poly_bb[m] = NULL; continue; }
            for (int i = 0; i < N; i++) y[i] = coeffs_bb[i * n_modes_bb + m];
            new_poly_bb[m] = online_poly_init(n_params, degree);
            if (!new_poly_bb[m]) { free(y); continue; }
            online_poly_train(new_poly_bb[m], params, y, N);
            free(y);
        }
        new_poly_rho = malloc(n_modes_rho * sizeof(PolyModel*));
        if (!new_poly_rho) goto poly_cleanup;
        for (int m = 0; m < n_modes_rho; m++) {
            double *y = malloc(N * sizeof(double));
            if (!y) { new_poly_rho[m] = NULL; continue; }
            for (int i = 0; i < N; i++) y[i] = coeffs_rho[i * n_modes_rho + m];
            new_poly_rho[m] = online_poly_init(n_params, degree);
            if (!new_poly_rho[m]) { free(y); continue; }
            online_poly_train(new_poly_rho[m], params, y, N);
            free(y);
        }

        int all_ok = 1;
        for (int m = 0; m < n_modes_tt; m++) if (!new_poly_tt[m] || new_poly_tt[m]->n_train == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_ee; m++) if (!new_poly_ee[m] || new_poly_ee[m]->n_train == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_bb; m++) if (!new_poly_bb[m] || new_poly_bb[m]->n_train == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_rho; m++) if (!new_poly_rho[m] || new_poly_rho[m]->n_train == 0) { all_ok = 0; break; }

        if (!all_ok) {
            fprintf(stderr, "Polynomial training failed for some modes. Emulator will NOT be trained.\n");
            if (new_poly_tt) { for (int m = 0; m < n_modes_tt; m++) online_poly_free(new_poly_tt[m]); free(new_poly_tt); }
            if (new_poly_ee) { for (int m = 0; m < n_modes_ee; m++) online_poly_free(new_poly_ee[m]); free(new_poly_ee); }
            if (new_poly_bb) { for (int m = 0; m < n_modes_bb; m++) online_poly_free(new_poly_bb[m]); free(new_poly_bb); }
            if (new_poly_rho) { for (int m = 0; m < n_modes_rho; m++) online_poly_free(new_poly_rho[m]); free(new_poly_rho); }
            goto cleanup_new_pcas_no_coeff;
        }

        // Free old Poly models
        if (emu->poly_tt) { for (int m = 0; m < emu->n_modes_tt; m++) online_poly_free(emu->poly_tt[m]); free(emu->poly_tt); emu->poly_tt = NULL; }
        if (emu->poly_ee) { for (int m = 0; m < emu->n_modes_ee; m++) online_poly_free(emu->poly_ee[m]); free(emu->poly_ee); emu->poly_ee = NULL; }
        if (emu->poly_bb) { for (int m = 0; m < emu->n_modes_bb; m++) online_poly_free(emu->poly_bb[m]); free(emu->poly_bb); emu->poly_bb = NULL; }
        if (emu->poly_rho) { for (int m = 0; m < emu->n_modes_rho; m++) online_poly_free(emu->poly_rho[m]); free(emu->poly_rho); emu->poly_rho = NULL; }

        emu->poly_tt = new_poly_tt;
        emu->poly_ee = new_poly_ee;
        emu->poly_bb = new_poly_bb;
        emu->poly_rho = new_poly_rho;
        emu->is_trained = 1;
        success = 1;
        goto poly_cleanup_done;

poly_cleanup:
        fprintf(stderr, "Allocation failed for polynomial models\n");
        if (new_poly_tt) { for (int m = 0; m < n_modes_tt; m++) online_poly_free(new_poly_tt[m]); free(new_poly_tt); }
        if (new_poly_ee) { for (int m = 0; m < n_modes_ee; m++) online_poly_free(new_poly_ee[m]); free(new_poly_ee); }
        if (new_poly_bb) { for (int m = 0; m < n_modes_bb; m++) online_poly_free(new_poly_bb[m]); free(new_poly_bb); }
        if (new_poly_rho) { for (int m = 0; m < n_modes_rho; m++) online_poly_free(new_poly_rho[m]); free(new_poly_rho); }
poly_cleanup_done:
        ;
    }

    // If training succeeded, replace PCA models
    if (success) {
        if (emu->pca_tt) pca_free(emu->pca_tt);
        if (emu->pca_ee) pca_free(emu->pca_ee);
        if (emu->pca_bb) pca_free(emu->pca_bb);
        if (emu->pca_rho) pca_free(emu->pca_rho);
        emu->pca_tt = new_pca_tt;
        emu->pca_ee = new_pca_ee;
        emu->pca_bb = new_pca_bb;
        emu->pca_rho = new_pca_rho;
        emu->n_modes_tt = n_modes_tt;
        emu->n_modes_ee = n_modes_ee;
        emu->n_modes_bb = n_modes_bb;
        emu->n_modes_rho = n_modes_rho;
        emu->pca_trained = 1;
        emu->points_since_pca = 0;
        emu->points_since_retrain = 0;

        printf("Emulator FULL retrain (%s) on %d points: TT=%d, EE=%d, BB=%d, RHO=%d\n",
               (emu->cfg.emulator_type == 0) ? "GP" : "Polynomial",
               N, n_modes_tt, n_modes_ee, n_modes_bb, n_modes_rho);
    } else {
        // Training failed – do not change PCA models, keep emu->is_trained as is
        // Free the new PCAs to avoid leak
        pca_free(new_pca_tt);
        pca_free(new_pca_ee);
        pca_free(new_pca_bb);
        pca_free(new_pca_rho);
    }

    // Free coefficient arrays (always done)
    free(coeffs_tt); free(coeffs_ee); free(coeffs_bb); free(coeffs_rho);
    cleanup_new_pcas_no_coeff:
    // Free the new PCAs if they weren't taken (if success=0 they are freed above)
    // But if success=1, we already assigned them to emu->pca_*, so don't free here
    if (!success) {
        // They are already freed in the success==0 branch; avoid double free
    }
    goto cleanup_free_temps;

cleanup_free_new_pcas:
    // Free the new PCAs if they were allocated and not taken
    if (new_pca_tt && new_pca_tt != emu->pca_tt) pca_free(new_pca_tt);
    if (new_pca_ee && new_pca_ee != emu->pca_ee) pca_free(new_pca_ee);
    if (new_pca_bb && new_pca_bb != emu->pca_bb) pca_free(new_pca_bb);
    if (new_pca_rho && new_pca_rho != emu->pca_rho) pca_free(new_pca_rho);
    // coeffs may be partially allocated, free them
    free(coeffs_tt); free(coeffs_ee); free(coeffs_bb); free(coeffs_rho);
cleanup_free_temps:
    free(tt_log); free(ee_log); free(bb_log); free(rho);
    free(params); free(cl_tt); free(cl_te); free(cl_ee); free(cl_bb);
}

/* ---------- Emulator‑only retrain (no new PCA) ---------- */
static void emulator_train_emulator_only(Emulator *emu) {
    if (!emu || !emu->pca_trained) return;
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
    if (!tt_log || !ee_log || !bb_log || !rho) {
        fprintf(stderr, "Memory allocation failed in emulator_train_emulator_only\n");
        goto cleanup;
    }

    memcpy(tt_log, cl_tt, N * D * sizeof(double));
    memcpy(ee_log, cl_ee, N * D * sizeof(double));
    memcpy(bb_log, cl_bb, N * D * sizeof(double));
    transform_log10(tt_log, N * D, 1e-12);
    transform_log10(ee_log, N * D, 1e-12);
    transform_log10(bb_log, N * D, 1e-12);
    transform_atanh_rho(cl_tt, cl_ee, cl_te, rho, N, D, 1e-12);

    int n_modes_tt = emu->n_modes_tt;
    int n_modes_ee = emu->n_modes_ee;
    int n_modes_bb = emu->n_modes_bb;
    int n_modes_rho = emu->n_modes_rho;

    double *coeffs_tt = malloc(N * n_modes_tt * sizeof(double));
    double *coeffs_ee = malloc(N * n_modes_ee * sizeof(double));
    double *coeffs_bb = malloc(N * n_modes_bb * sizeof(double));
    double *coeffs_rho = malloc(N * n_modes_rho * sizeof(double));
    if (!coeffs_tt || !coeffs_ee || !coeffs_bb || !coeffs_rho) {
        fprintf(stderr, "Memory allocation for coefficients failed in emulator-only retrain\n");
        free(coeffs_tt); free(coeffs_ee); free(coeffs_bb); free(coeffs_rho);
        goto cleanup;
    }

    for (int i = 0; i < N; i++) {
        pca_project(emu->pca_tt, &tt_log[i*D], &coeffs_tt[i*n_modes_tt]);
        pca_project(emu->pca_ee, &ee_log[i*D], &coeffs_ee[i*n_modes_ee]);
        pca_project(emu->pca_bb, &bb_log[i*D], &coeffs_bb[i*n_modes_bb]);
        pca_project(emu->pca_rho, &rho[i*D], &coeffs_rho[i*n_modes_rho]);
    }

    if (!coefficients_are_valid(coeffs_tt, N, n_modes_tt) ||
        !coefficients_are_valid(coeffs_ee, N, n_modes_ee) ||
        !coefficients_are_valid(coeffs_bb, N, n_modes_bb) ||
        !coefficients_are_valid(coeffs_rho, N, n_modes_rho)) {
        fprintf(stderr, "NaN/Inf in PCA coefficients; skipping emulator-only retrain.\n");
        free(coeffs_tt); free(coeffs_ee); free(coeffs_bb); free(coeffs_rho);
        goto cleanup;
    }

    if (emu->cfg.emulator_type == 0) {
        // ---------- GP ----------
        // Free old models
        if (emu->gp_tt) { for (int m = 0; m < n_modes_tt; m++) online_gp_free(emu->gp_tt[m]); free(emu->gp_tt); emu->gp_tt = NULL; }
        if (emu->gp_ee) { for (int m = 0; m < n_modes_ee; m++) online_gp_free(emu->gp_ee[m]); free(emu->gp_ee); emu->gp_ee = NULL; }
        if (emu->gp_bb) { for (int m = 0; m < n_modes_bb; m++) online_gp_free(emu->gp_bb[m]); free(emu->gp_bb); emu->gp_bb = NULL; }
        if (emu->gp_rho) { for (int m = 0; m < n_modes_rho; m++) online_gp_free(emu->gp_rho[m]); free(emu->gp_rho); emu->gp_rho = NULL; }

        GPModel **new_gp_tt = malloc(n_modes_tt * sizeof(GPModel*));
        if (!new_gp_tt) { fprintf(stderr, "Alloc failed for GP models\n"); goto cleanup_after_coeff; }
        for (int m = 0; m < n_modes_tt; m++) {
            double *y = malloc(N * sizeof(double));
            if (!y) { new_gp_tt[m] = NULL; continue; }
            for (int i = 0; i < N; i++) y[i] = coeffs_tt[i * n_modes_tt + m];
            new_gp_tt[m] = online_gp_init(n_params);
            if (!new_gp_tt[m]) { free(y); continue; }
            online_gp_train(new_gp_tt[m], params, y, N);
            free(y);
        }
        GPModel **new_gp_ee = malloc(n_modes_ee * sizeof(GPModel*));
        if (!new_gp_ee) { fprintf(stderr, "Alloc failed for GP models\n"); goto gp_cleanup_emu; }
        for (int m = 0; m < n_modes_ee; m++) {
            double *y = malloc(N * sizeof(double));
            if (!y) { new_gp_ee[m] = NULL; continue; }
            for (int i = 0; i < N; i++) y[i] = coeffs_ee[i * n_modes_ee + m];
            new_gp_ee[m] = online_gp_init(n_params);
            if (!new_gp_ee[m]) { free(y); continue; }
            online_gp_train(new_gp_ee[m], params, y, N);
            free(y);
        }
        GPModel **new_gp_bb = malloc(n_modes_bb * sizeof(GPModel*));
        if (!new_gp_bb) { fprintf(stderr, "Alloc failed for GP models\n"); goto gp_cleanup_emu; }
        for (int m = 0; m < n_modes_bb; m++) {
            double *y = malloc(N * sizeof(double));
            if (!y) { new_gp_bb[m] = NULL; continue; }
            for (int i = 0; i < N; i++) y[i] = coeffs_bb[i * n_modes_bb + m];
            new_gp_bb[m] = online_gp_init(n_params);
            if (!new_gp_bb[m]) { free(y); continue; }
            online_gp_train(new_gp_bb[m], params, y, N);
            free(y);
        }
        GPModel **new_gp_rho = malloc(n_modes_rho * sizeof(GPModel*));
        if (!new_gp_rho) { fprintf(stderr, "Alloc failed for GP models\n"); goto gp_cleanup_emu; }
        for (int m = 0; m < n_modes_rho; m++) {
            double *y = malloc(N * sizeof(double));
            if (!y) { new_gp_rho[m] = NULL; continue; }
            for (int i = 0; i < N; i++) y[i] = coeffs_rho[i * n_modes_rho + m];
            new_gp_rho[m] = online_gp_init(n_params);
            if (!new_gp_rho[m]) { free(y); continue; }
            online_gp_train(new_gp_rho[m], params, y, N);
            free(y);
        }

        int all_ok = 1;
        for (int m = 0; m < n_modes_tt; m++) if (!new_gp_tt[m] || new_gp_tt[m]->n_train == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_ee; m++) if (!new_gp_ee[m] || new_gp_ee[m]->n_train == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_bb; m++) if (!new_gp_bb[m] || new_gp_bb[m]->n_train == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_rho; m++) if (!new_gp_rho[m] || new_gp_rho[m]->n_train == 0) { all_ok = 0; break; }

        if (!all_ok) {
            fprintf(stderr, "GP-only retrain failed. Keeping old models.\n");
            // Free new models
            if (new_gp_tt) { for (int m = 0; m < n_modes_tt; m++) online_gp_free(new_gp_tt[m]); free(new_gp_tt); }
            if (new_gp_ee) { for (int m = 0; m < n_modes_ee; m++) online_gp_free(new_gp_ee[m]); free(new_gp_ee); }
            if (new_gp_bb) { for (int m = 0; m < n_modes_bb; m++) online_gp_free(new_gp_bb[m]); free(new_gp_bb); }
            if (new_gp_rho) { for (int m = 0; m < n_modes_rho; m++) online_gp_free(new_gp_rho[m]); free(new_gp_rho); }
            // Restore old models (they were freed earlier, so we must set to NULL)
            emu->gp_tt = emu->gp_ee = emu->gp_bb = emu->gp_rho = NULL;
            emu->is_trained = 0;
            goto cleanup_after_coeff;
        }

        emu->gp_tt = new_gp_tt;
        emu->gp_ee = new_gp_ee;
        emu->gp_bb = new_gp_bb;
        emu->gp_rho = new_gp_rho;
        emu->is_trained = 1;
        printf("Emulator GP-only retrain on %d points\n", N);
        goto gp_cleanup_emu_done;

gp_cleanup_emu:
        if (new_gp_tt) { for (int m = 0; m < n_modes_tt; m++) online_gp_free(new_gp_tt[m]); free(new_gp_tt); }
        if (new_gp_ee) { for (int m = 0; m < n_modes_ee; m++) online_gp_free(new_gp_ee[m]); free(new_gp_ee); }
        if (new_gp_bb) { for (int m = 0; m < n_modes_bb; m++) online_gp_free(new_gp_bb[m]); free(new_gp_bb); }
        if (new_gp_rho) { for (int m = 0; m < n_modes_rho; m++) online_gp_free(new_gp_rho[m]); free(new_gp_rho); }
        emu->gp_tt = emu->gp_ee = emu->gp_bb = emu->gp_rho = NULL;
        emu->is_trained = 0;
gp_cleanup_emu_done:
        ;
    } else {
        // ---------- Polynomial ----------
        int degree = emu->cfg.poly_degree;

        if (emu->poly_tt) { for (int m = 0; m < n_modes_tt; m++) online_poly_free(emu->poly_tt[m]); free(emu->poly_tt); emu->poly_tt = NULL; }
        if (emu->poly_ee) { for (int m = 0; m < n_modes_ee; m++) online_poly_free(emu->poly_ee[m]); free(emu->poly_ee); emu->poly_ee = NULL; }
        if (emu->poly_bb) { for (int m = 0; m < n_modes_bb; m++) online_poly_free(emu->poly_bb[m]); free(emu->poly_bb); emu->poly_bb = NULL; }
        if (emu->poly_rho) { for (int m = 0; m < n_modes_rho; m++) online_poly_free(emu->poly_rho[m]); free(emu->poly_rho); emu->poly_rho = NULL; }

        PolyModel **new_poly_tt = malloc(n_modes_tt * sizeof(PolyModel*));
        if (!new_poly_tt) { fprintf(stderr, "Alloc failed for poly models\n"); goto cleanup_after_coeff; }
        for (int m = 0; m < n_modes_tt; m++) {
            double *y = malloc(N * sizeof(double));
            if (!y) { new_poly_tt[m] = NULL; continue; }
            for (int i = 0; i < N; i++) y[i] = coeffs_tt[i * n_modes_tt + m];
            new_poly_tt[m] = online_poly_init(n_params, degree);
            if (!new_poly_tt[m]) { free(y); continue; }
            online_poly_train(new_poly_tt[m], params, y, N);
            free(y);
        }
        PolyModel **new_poly_ee = malloc(n_modes_ee * sizeof(PolyModel*));
        if (!new_poly_ee) { fprintf(stderr, "Alloc failed for poly models\n"); goto poly_cleanup_emu; }
        for (int m = 0; m < n_modes_ee; m++) {
            double *y = malloc(N * sizeof(double));
            if (!y) { new_poly_ee[m] = NULL; continue; }
            for (int i = 0; i < N; i++) y[i] = coeffs_ee[i * n_modes_ee + m];
            new_poly_ee[m] = online_poly_init(n_params, degree);
            if (!new_poly_ee[m]) { free(y); continue; }
            online_poly_train(new_poly_ee[m], params, y, N);
            free(y);
        }
        PolyModel **new_poly_bb = malloc(n_modes_bb * sizeof(PolyModel*));
        if (!new_poly_bb) { fprintf(stderr, "Alloc failed for poly models\n"); goto poly_cleanup_emu; }
        for (int m = 0; m < n_modes_bb; m++) {
            double *y = malloc(N * sizeof(double));
            if (!y) { new_poly_bb[m] = NULL; continue; }
            for (int i = 0; i < N; i++) y[i] = coeffs_bb[i * n_modes_bb + m];
            new_poly_bb[m] = online_poly_init(n_params, degree);
            if (!new_poly_bb[m]) { free(y); continue; }
            online_poly_train(new_poly_bb[m], params, y, N);
            free(y);
        }
        PolyModel **new_poly_rho = malloc(n_modes_rho * sizeof(PolyModel*));
        if (!new_poly_rho) { fprintf(stderr, "Alloc failed for poly models\n"); goto poly_cleanup_emu; }
        for (int m = 0; m < n_modes_rho; m++) {
            double *y = malloc(N * sizeof(double));
            if (!y) { new_poly_rho[m] = NULL; continue; }
            for (int i = 0; i < N; i++) y[i] = coeffs_rho[i * n_modes_rho + m];
            new_poly_rho[m] = online_poly_init(n_params, degree);
            if (!new_poly_rho[m]) { free(y); continue; }
            online_poly_train(new_poly_rho[m], params, y, N);
            free(y);
        }

        int all_ok = 1;
        for (int m = 0; m < n_modes_tt; m++) if (!new_poly_tt[m] || new_poly_tt[m]->n_train == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_ee; m++) if (!new_poly_ee[m] || new_poly_ee[m]->n_train == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_bb; m++) if (!new_poly_bb[m] || new_poly_bb[m]->n_train == 0) { all_ok = 0; break; }
        if (all_ok) for (int m = 0; m < n_modes_rho; m++) if (!new_poly_rho[m] || new_poly_rho[m]->n_train == 0) { all_ok = 0; break; }

        if (!all_ok) {
            fprintf(stderr, "Polynomial-only retrain failed. Keeping old models.\n");
            if (new_poly_tt) { for (int m = 0; m < n_modes_tt; m++) online_poly_free(new_poly_tt[m]); free(new_poly_tt); }
            if (new_poly_ee) { for (int m = 0; m < n_modes_ee; m++) online_poly_free(new_poly_ee[m]); free(new_poly_ee); }
            if (new_poly_bb) { for (int m = 0; m < n_modes_bb; m++) online_poly_free(new_poly_bb[m]); free(new_poly_bb); }
            if (new_poly_rho) { for (int m = 0; m < n_modes_rho; m++) online_poly_free(new_poly_rho[m]); free(new_poly_rho); }
            emu->poly_tt = emu->poly_ee = emu->poly_bb = emu->poly_rho = NULL;
            emu->is_trained = 0;
            goto cleanup_after_coeff;
        }

        emu->poly_tt = new_poly_tt;
        emu->poly_ee = new_poly_ee;
        emu->poly_bb = new_poly_bb;
        emu->poly_rho = new_poly_rho;
        emu->is_trained = 1;
        printf("Emulator Polynomial-only retrain on %d points (degree %d)\n", N, degree);
        goto poly_cleanup_emu_done;

poly_cleanup_emu:
        if (new_poly_tt) { for (int m = 0; m < n_modes_tt; m++) online_poly_free(new_poly_tt[m]); free(new_poly_tt); }
        if (new_poly_ee) { for (int m = 0; m < n_modes_ee; m++) online_poly_free(new_poly_ee[m]); free(new_poly_ee); }
        if (new_poly_bb) { for (int m = 0; m < n_modes_bb; m++) online_poly_free(new_poly_bb[m]); free(new_poly_bb); }
        if (new_poly_rho) { for (int m = 0; m < n_modes_rho; m++) online_poly_free(new_poly_rho[m]); free(new_poly_rho); }
        emu->poly_tt = emu->poly_ee = emu->poly_bb = emu->poly_rho = NULL;
        emu->is_trained = 0;
poly_cleanup_emu_done:
        ;
    }

cleanup_after_coeff:
    free(coeffs_tt); free(coeffs_ee); free(coeffs_bb); free(coeffs_rho);
cleanup:
    free(tt_log); free(ee_log); free(bb_log); free(rho);
    free(params); free(cl_tt); free(cl_te); free(cl_ee); free(cl_bb);
}

/* ---------- Prediction ---------- */
int emulator_predict(Emulator *emu, const double *params,
                     double *cl_tt, double *cl_te,
                     double *cl_ee, double *cl_bb,
                     double *uncertainty) {
    if (!emu || !emu->is_trained) return 0;
    int D = N_SPECTRA;

    double *coeffs_tt, *coeffs_ee, *coeffs_bb, *coeffs_rho;
    double *coeff_var_tt, *coeff_var_ee, *coeff_var_bb, *coeff_var_rho;

    if (emu->cfg.emulator_type == 0) {
        // ----- GP -----
        coeffs_tt = malloc(emu->n_modes_tt * sizeof(double));
        coeff_var_tt = malloc(emu->n_modes_tt * sizeof(double));
        if (!coeffs_tt || !coeff_var_tt) { free(coeffs_tt); free(coeff_var_tt); return 0; }
        for (int m = 0; m < emu->n_modes_tt; m++) {
            double mean, var;
            online_gp_predict(emu->gp_tt[m], params, &mean, &var);
            coeffs_tt[m] = mean;
            coeff_var_tt[m] = var;
        }
        coeffs_ee = malloc(emu->n_modes_ee * sizeof(double));
        coeff_var_ee = malloc(emu->n_modes_ee * sizeof(double));
        if (!coeffs_ee || !coeff_var_ee) { free(coeffs_tt); free(coeff_var_tt); free(coeffs_ee); free(coeff_var_ee); return 0; }
        for (int m = 0; m < emu->n_modes_ee; m++) {
            double mean, var;
            online_gp_predict(emu->gp_ee[m], params, &mean, &var);
            coeffs_ee[m] = mean;
            coeff_var_ee[m] = var;
        }
        coeffs_bb = malloc(emu->n_modes_bb * sizeof(double));
        coeff_var_bb = malloc(emu->n_modes_bb * sizeof(double));
        if (!coeffs_bb || !coeff_var_bb) { free(coeffs_tt); free(coeff_var_tt); free(coeffs_ee); free(coeff_var_ee); free(coeffs_bb); free(coeff_var_bb); return 0; }
        for (int m = 0; m < emu->n_modes_bb; m++) {
            double mean, var;
            online_gp_predict(emu->gp_bb[m], params, &mean, &var);
            coeffs_bb[m] = mean;
            coeff_var_bb[m] = var;
        }
        coeffs_rho = malloc(emu->n_modes_rho * sizeof(double));
        coeff_var_rho = malloc(emu->n_modes_rho * sizeof(double));
        if (!coeffs_rho || !coeff_var_rho) { free(coeffs_tt); free(coeff_var_tt); free(coeffs_ee); free(coeff_var_ee); free(coeffs_bb); free(coeff_var_bb); free(coeffs_rho); free(coeff_var_rho); return 0; }
        for (int m = 0; m < emu->n_modes_rho; m++) {
            double mean, var;
            online_gp_predict(emu->gp_rho[m], params, &mean, &var);
            coeffs_rho[m] = mean;
            coeff_var_rho[m] = var;
        }
    } else {
        // ----- Polynomial -----
        coeffs_tt = malloc(emu->n_modes_tt * sizeof(double));
        coeff_var_tt = malloc(emu->n_modes_tt * sizeof(double));
        if (!coeffs_tt || !coeff_var_tt) { free(coeffs_tt); free(coeff_var_tt); return 0; }
        for (int m = 0; m < emu->n_modes_tt; m++) {
            double mean, var;
            online_poly_predict(emu->poly_tt[m], params, &mean, &var);
            coeffs_tt[m] = mean;
            coeff_var_tt[m] = var;
        }
        coeffs_ee = malloc(emu->n_modes_ee * sizeof(double));
        coeff_var_ee = malloc(emu->n_modes_ee * sizeof(double));
        if (!coeffs_ee || !coeff_var_ee) { free(coeffs_tt); free(coeff_var_tt); free(coeffs_ee); free(coeff_var_ee); return 0; }
        for (int m = 0; m < emu->n_modes_ee; m++) {
            double mean, var;
            online_poly_predict(emu->poly_ee[m], params, &mean, &var);
            coeffs_ee[m] = mean;
            coeff_var_ee[m] = var;
        }
        coeffs_bb = malloc(emu->n_modes_bb * sizeof(double));
        coeff_var_bb = malloc(emu->n_modes_bb * sizeof(double));
        if (!coeffs_bb || !coeff_var_bb) { free(coeffs_tt); free(coeff_var_tt); free(coeffs_ee); free(coeff_var_ee); free(coeffs_bb); free(coeff_var_bb); return 0; }
        for (int m = 0; m < emu->n_modes_bb; m++) {
            double mean, var;
            online_poly_predict(emu->poly_bb[m], params, &mean, &var);
            coeffs_bb[m] = mean;
            coeff_var_bb[m] = var;
        }
        coeffs_rho = malloc(emu->n_modes_rho * sizeof(double));
        coeff_var_rho = malloc(emu->n_modes_rho * sizeof(double));
        if (!coeffs_rho || !coeff_var_rho) { free(coeffs_tt); free(coeff_var_tt); free(coeffs_ee); free(coeff_var_ee); free(coeffs_bb); free(coeff_var_bb); free(coeffs_rho); free(coeff_var_rho); return 0; }
        for (int m = 0; m < emu->n_modes_rho; m++) {
            double mean, var;
            online_poly_predict(emu->poly_rho[m], params, &mean, &var);
            coeffs_rho[m] = mean;
            coeff_var_rho[m] = var;
        }
    }

    double total_var = 0.0;
    int total_modes = emu->n_modes_tt + emu->n_modes_ee + emu->n_modes_bb + emu->n_modes_rho;
    for (int m = 0; m < emu->n_modes_tt; m++) total_var += coeff_var_tt[m];
    for (int m = 0; m < emu->n_modes_ee; m++) total_var += coeff_var_ee[m];
    for (int m = 0; m < emu->n_modes_bb; m++) total_var += coeff_var_bb[m];
    for (int m = 0; m < emu->n_modes_rho; m++) total_var += coeff_var_rho[m];
    *uncertainty = sqrt(total_var / total_modes);

    /* Reconstruct spectra (same for both) */
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

    pca_reconstruct(emu->pca_tt, coeffs_tt, tt_log);
    pca_reconstruct(emu->pca_ee, coeffs_ee, ee_log);
    pca_reconstruct(emu->pca_bb, coeffs_bb, bb_log);
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

    if (emu->gp_tt) {
        for (int m = 0; m < emu->n_modes_tt; m++) online_gp_free(emu->gp_tt[m]);
        free(emu->gp_tt);
    }
    if (emu->gp_ee) {
        for (int m = 0; m < emu->n_modes_ee; m++) online_gp_free(emu->gp_ee[m]);
        free(emu->gp_ee);
    }
    if (emu->gp_bb) {
        for (int m = 0; m < emu->n_modes_bb; m++) online_gp_free(emu->gp_bb[m]);
        free(emu->gp_bb);
    }
    if (emu->gp_rho) {
        for (int m = 0; m < emu->n_modes_rho; m++) online_gp_free(emu->gp_rho[m]);
        free(emu->gp_rho);
    }

    if (emu->poly_tt) {
        for (int m = 0; m < emu->n_modes_tt; m++) online_poly_free(emu->poly_tt[m]);
        free(emu->poly_tt);
    }
    if (emu->poly_ee) {
        for (int m = 0; m < emu->n_modes_ee; m++) online_poly_free(emu->poly_ee[m]);
        free(emu->poly_ee);
    }
    if (emu->poly_bb) {
        for (int m = 0; m < emu->n_modes_bb; m++) online_poly_free(emu->poly_bb[m]);
        free(emu->poly_bb);
    }
    if (emu->poly_rho) {
        for (int m = 0; m < emu->n_modes_rho; m++) online_poly_free(emu->poly_rho[m]);
        free(emu->poly_rho);
    }

    free(emu);
}