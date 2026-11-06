#include "poly_online.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <lapacke.h>

/* =====================================================================
   UNCERTAINTY METRIC SELECTION
   =====================================================================
   Choose the per-mode predictive variance formula:
   
   1 = B1 (Global rv):        var = rv + (1-rv) * f
   3 = B3 (Pure distance):    var = f
   4 = B4 (Bayesian):         var = rv * (1 + phi^T * (A^T A + lambda I)^-1 * phi)
   5 = B5 (Local-dominant):   var = 0.95 * f + 0.05 * rv
   6 = B6 (Bayesian):         var = rv * (1 + phi^T * (A^T A + lambda I)^-1 * phi)
       (Note: B4 and B6 have identical per-mode formulas; the difference 
        is that B4 uses y_std^2 weighted aggregation in emulator.c, 
        while B6 uses unweighted aggregation.)
   ===================================================================== */
#define UNCERTAINTY_METRIC 4

#define EPS 1e-12
#define RIDGE_LAMBDA 1e-2
#define NOISE_FLOOR  1e-4

/* ---------- internal helpers ---------- */

static int count_terms(int n, int d) {
    if (d == 0) return 1;
    long long res = 1;
    for (int i = 1; i <= d; i++) {
        res = res * (n + i) / i;
    }
    return (int)res;
}

static void fill_monomials(const double *x, int n_inputs, int degree,
                           double *row, int *idx, int dim,
                           int current_deg, double current_val) {
    if (dim == n_inputs) {
        row[(*idx)++] = current_val;
        return;
    }
    int max_e = degree - current_deg;
    double pow_x = 1.0;
    for (int e = 0; e <= max_e; e++) {
        fill_monomials(x, n_inputs, degree, row, idx, dim + 1,
                       current_deg + e, current_val * pow_x);
        pow_x *= x[dim];
    }
}

static double* build_design_matrix(const double *X_norm, int N, int n_inputs,
                                   int degree, int *n_terms) {
    *n_terms = count_terms(n_inputs, degree);
    if (*n_terms <= 0) return NULL;
    double *A = malloc((size_t)N * (*n_terms) * sizeof(double));
    if (!A) return NULL;

    for (int i = 0; i < N; i++) {
        const double *x = &X_norm[i * n_inputs];
        int idx = 0;
        fill_monomials(x, n_inputs, degree, &A[i * (*n_terms)], &idx, 0, 0, 1.0);
    }
    return A;
}

/* ---------- Public functions ---------- */

PolyModel* online_poly_init(int n_inputs, int degree) {
    if (n_inputs <= 0 || degree < 1) return NULL;
    PolyModel *poly = calloc(1, sizeof(PolyModel));
    if (!poly) return NULL;

    poly->n_inputs = n_inputs;
    poly->degree = degree;
    poly->n_train = 0;
    poly->coeff = NULL;
    poly->chol_L = NULL;
    poly->X_mean = malloc(n_inputs * sizeof(double));
    poly->X_std = malloc(n_inputs * sizeof(double));

    /* KNN index capacity — matches typical buffer size; adjust if needed */
    poly->knn = knn_create(n_inputs, 2000, 5);

    if (!poly->X_mean || !poly->X_std || !poly->knn) {
        free(poly->X_mean);
        free(poly->X_std);
        knn_destroy(poly->knn);
        free(poly);
        return NULL;
    }
    poly->y_mean = 0.0;
    poly->y_std = 1.0;
    poly->residual_var = 0.0;
    return poly;
}

void online_poly_train(PolyModel *poly, const double *X, const double *y, int N) {
    if (!poly || N < 2 || !X || !y) {
        if (poly) {
            poly->n_train = 0;
            free(poly->coeff); poly->coeff = NULL;
            free(poly->chol_L); poly->chol_L = NULL;
        }
        return;
    }
    int n_in = poly->n_inputs;
    int deg = poly->degree;

    int n_terms = count_terms(n_in, deg);
    if (N < n_terms) {
        fprintf(stderr, "Warning: N=%d < n_terms=%d, cannot train polynomial.\n", N, n_terms);
        poly->n_train = 0;
        free(poly->coeff); poly->coeff = NULL;
        free(poly->chol_L); poly->chol_L = NULL;
        return;
    }

    /* Free old data */
    free(poly->coeff); poly->coeff = NULL;
    free(poly->chol_L); poly->chol_L = NULL;
    poly->n_train = 0;

    /* Standardise inputs */
    for (int j = 0; j < n_in; j++) {
        double mean = 0.0;
        for (int i = 0; i < N; i++) mean += X[i * n_in + j];
        mean /= N;
        poly->X_mean[j] = mean;

        double var = 0.0;
        for (int i = 0; i < N; i++) {
            double d = X[i * n_in + j] - mean;
            var += d * d;
        }
        var /= N;
        poly->X_std[j] = sqrt(var);
        if (poly->X_std[j] < EPS) poly->X_std[j] = 1.0;
    }

    double *X_norm = malloc(N * n_in * sizeof(double));
    if (!X_norm) return;
    for (int i = 0; i < N; i++)
        for (int j = 0; j < n_in; j++)
            X_norm[i * n_in + j] = (X[i * n_in + j] - poly->X_mean[j]) / poly->X_std[j];

    /* Standardise outputs */
    double y_mean = 0.0;
    for (int i = 0; i < N; i++) y_mean += y[i];
    y_mean /= N;
    poly->y_mean = y_mean;

    double y_var = 0.0;
    for (int i = 0; i < N; i++) {
        double d = y[i] - y_mean;
        y_var += d * d;
    }
    y_var /= N;
    double y_std = sqrt(y_var);
    if (y_std < EPS) y_std = 1.0;
    poly->y_std = y_std;

    double *y_norm = malloc(N * sizeof(double));
    if (!y_norm) { free(X_norm); return; }
    for (int i = 0; i < N; i++) y_norm[i] = (y[i] - y_mean) / y_std;

    /* Build design matrix */
    int n_terms_actual;
    double *A = build_design_matrix(X_norm, N, n_in, deg, &n_terms_actual);
    if (!A || n_terms_actual <= 0) {
        free(X_norm); free(y_norm);
        return;
    }
    poly->n_terms = n_terms_actual;

    /* Normal equations: (A^T A + lambda I) c = A^T y */
    double *ATA = malloc((size_t)n_terms_actual * n_terms_actual * sizeof(double));
    double *ATy = malloc((size_t)n_terms_actual * sizeof(double));
    if (!ATA || !ATy) {
        free(A); free(X_norm); free(y_norm);
        free(ATA); free(ATy);
        return;
    }

    for (int i = 0; i < n_terms_actual; i++) {
        ATy[i] = 0.0;
        for (int j = 0; j < n_terms_actual; j++) ATA[i * n_terms_actual + j] = 0.0;
    }

    for (int k = 0; k < N; k++) {
        for (int i = 0; i < n_terms_actual; i++) {
            double aik = A[k * n_terms_actual + i];
            ATy[i] += aik * y_norm[k];
            for (int j = 0; j < n_terms_actual; j++) {
                ATA[i * n_terms_actual + j] += aik * A[k * n_terms_actual + j];
            }
        }
    }

    /* Add ridge regularisation */
    double lambda = RIDGE_LAMBDA;
    for (int i = 0; i < n_terms_actual; i++) ATA[i * n_terms_actual + i] += lambda;

    /* Solve using Cholesky */
    double *coeff = malloc(n_terms_actual * sizeof(double));
    double *chol_L = calloc((size_t)n_terms_actual * n_terms_actual, sizeof(double));
    if (!coeff || !chol_L) {
        free(A); free(X_norm); free(y_norm); free(ATA); free(ATy); free(coeff); free(chol_L);
        return;
    }
    memcpy(coeff, ATy, n_terms_actual * sizeof(double));

    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n_terms_actual, ATA, n_terms_actual);
    if (info != 0) {
        fprintf(stderr, "Cholesky failed (info=%d)\n", info);
        free(A); free(X_norm); free(y_norm); free(ATA); free(ATy); free(coeff); free(chol_L);
        return;
    }

    /* Save the Cholesky factor (lower triangle) */
    for (int i = 0; i < n_terms_actual; i++)
        for (int j = 0; j <= i; j++)
            chol_L[i * n_terms_actual + j] = ATA[i * n_terms_actual + j];

    info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', n_terms_actual, 1, ATA, n_terms_actual, coeff, 1);
    if (info != 0) {
        fprintf(stderr, "Solve failed (info=%d)\n", info);
        free(A); free(X_norm); free(y_norm); free(ATA); free(ATy); free(coeff); free(chol_L);
        return;
    }

    /* In-sample residual */
    double residual = 0.0;
    for (int i = 0; i < N; i++) {
        double pred = 0.0;
        for (int j = 0; j < n_terms_actual; j++)
            pred += A[i * n_terms_actual + j] * coeff[j];
        double diff = y_norm[i] - pred;
        residual += diff * diff;
    }

    /* ---- Ridge-corrected residual variance ---- */
    double tr_H = (double)n_terms_actual;
    if (n_terms_actual > 1) {
        double *L_inv = malloc((size_t)n_terms_actual * n_terms_actual * sizeof(double));
        if (L_inv) {
            memcpy(L_inv, chol_L, (size_t)n_terms_actual * n_terms_actual * sizeof(double));
            int tri_info = LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'L', 'N',
                                          n_terms_actual, L_inv, n_terms_actual);
            if (tri_info == 0) {
                double fro2 = 0.0;
                for (int i = 0; i < n_terms_actual; i++)
                    for (int j = 0; j <= i; j++) {
                        double v = L_inv[i * n_terms_actual + j];
                        fro2 += v * v;
                    }
                tr_H = (double)n_terms_actual - RIDGE_LAMBDA * fro2;
            }
            free(L_inv);
        }
        if (tr_H < 1.0)                    tr_H = 1.0;
        if (tr_H > (double)n_terms_actual) tr_H = (double)n_terms_actual;
    }

    double df_resid = (double)N - tr_H;
    if (df_resid < 1.0) df_resid = 1.0;

    double res_var = residual / df_resid;
    if (res_var < 0.0 || !isfinite(res_var)) res_var = 0.0;
    if (res_var < NOISE_FLOOR)               res_var = NOISE_FLOOR;
    if (res_var > 1.0)                       res_var = 1.0;
    poly->residual_var = res_var;

    /* Store results */
    poly->coeff = coeff;
    poly->chol_L = chol_L;
    poly->n_train = N;

    /* Rebuild KNN index */
    if (poly->knn) knn_rebuild(poly->knn, X, N);

    if (getenv("EMU_DEBUG_RV")) {
        fprintf(stderr, "[POLY_RV] N=%d n_terms=%d sqrt_rv=%.4f cluster_radius=%.4f\n",
                N, n_terms_actual, sqrt(poly->residual_var),
                poly->knn ? poly->knn->median_nn_dist : -1.0);
    }

    free(A);
    free(ATA);
    free(ATy);
    free(X_norm);
    free(y_norm);
}

double online_poly_predict_mean(const PolyModel *poly, const double *x) {
    if (!poly || poly->n_train == 0 || !poly->coeff || !x)
        return poly->y_mean;

    int n_in = poly->n_inputs;
    double *x_norm = malloc(n_in * sizeof(double));
    if (!x_norm) return poly->y_mean;

    for (int j = 0; j < n_in; j++)
        x_norm[j] = (x[j] - poly->X_mean[j]) / poly->X_std[j];

    double *row = malloc(poly->n_terms * sizeof(double));
    if (!row) { free(x_norm); return poly->y_mean; }

    int idx = 0;
    fill_monomials(x_norm, n_in, poly->degree, row, &idx, 0, 0, 1.0);

    double pred_norm = 0.0;
    for (int i = 0; i < poly->n_terms; i++) pred_norm += row[i] * poly->coeff[i];

    free(x_norm);
    free(row);
    return pred_norm * poly->y_std + poly->y_mean;
}

void online_poly_predict(const PolyModel *poly, const double *x,
                         double *mean, double *var) {
    if (!poly || poly->n_train == 0 || !poly->coeff || !x) {
        *mean = poly->y_mean;
        *var = poly->residual_var * poly->y_std * poly->y_std;
        return;
    }

    int n_in = poly->n_inputs;
    int n_terms = poly->n_terms;

    double *x_norm = malloc(n_in * sizeof(double));
    double *row = malloc(n_terms * sizeof(double));

    if (!x_norm || !row) {
        free(x_norm); free(row);
        *mean = poly->y_mean;
        *var = poly->residual_var * poly->y_std * poly->y_std;
        return;
    }

    /* 1. Standardise and evaluate feature vector */
    for (int j = 0; j < n_in; j++)
        x_norm[j] = (x[j] - poly->X_mean[j]) / poly->X_std[j];

    int idx = 0;
    fill_monomials(x_norm, n_in, poly->degree, row, &idx, 0, 0, 1.0);

    /* 2. Predictive mean */
    double pred_norm = 0.0;
    for (int i = 0; i < n_terms; i++) pred_norm += row[i] * poly->coeff[i];
    *mean = pred_norm * poly->y_std + poly->y_mean;

    /* 3. Predictive variance */
    double rv = poly->residual_var;
    if (rv < 0.0) rv = 0.0;
    if (rv > 1.0) rv = 1.0;

    double var_norm = 0.0;

#if UNCERTAINTY_METRIC == 4 || UNCERTAINTY_METRIC == 6
    /* B4 / B6: Bayesian Predictive Variance
       var = rv * (1 + phi^T * (A^T A + lambda I)^-1 * phi)
       We compute z = L^-1 * phi via forward substitution, then zz = ||z||^2. */
    if (poly->chol_L && n_terms > 0) {
        double *z = malloc(n_terms * sizeof(double));
        if (z) {
            memcpy(z, row, n_terms * sizeof(double));
            for (int i = 0; i < n_terms; i++) {
                double sum = 0.0;
                for (int j = 0; j < i; j++) {
                    sum += poly->chol_L[i * n_terms + j] * z[j];
                }
                z[i] = (z[i] - sum) / poly->chol_L[i * n_terms + i];
            }
            double zz = 0.0;
            for (int i = 0; i < n_terms; i++) zz += z[i] * z[i];
            free(z);
            var_norm = rv * (1.0 + zz);
        } else {
            var_norm = rv;
        }
    } else {
        var_norm = rv;
    }
#else
    /* Compute KNN distance for the distance-based metrics */
    double dnorm = (poly->knn && poly->knn->N >= poly->knn->k)
                 ? knn_query_dnorm(poly->knn, x)
                 : 100.0;
    const double k = 5.0;
    double dnorm_sq = dnorm * dnorm;
    double f = dnorm_sq / (k * k + dnorm_sq);

    #if UNCERTAINTY_METRIC == 1
        /* B1: Global rv */
        var_norm = rv + (1.0 - rv) * f;
    #elif UNCERTAINTY_METRIC == 3
        /* B3: Pure distance */
        var_norm = f;
    #elif UNCERTAINTY_METRIC == 5
        /* B5: Local-dominant */
        const double LOCAL_WEIGHT = 0.95;
        var_norm = LOCAL_WEIGHT * f + (1.0 - LOCAL_WEIGHT) * rv;
    #else
        /* Fallback (should not happen) */
        var_norm = rv;
    #endif
#endif

    *var = var_norm * poly->y_std * poly->y_std;

    if (*var < 0.0 || !isfinite(*var)) *var = 1e10;

    free(x_norm);
    free(row);
}

void online_poly_free(PolyModel *poly) {
    if (poly) {
        free(poly->coeff);
        free(poly->chol_L);
        free(poly->X_mean);
        free(poly->X_std);
        knn_destroy(poly->knn);
        free(poly);
    }
}