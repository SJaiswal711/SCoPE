#include "poly_online.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <lapacke.h>

#define EPS 1e-12
#define RIDGE_LAMBDA 1e-2
#define NOISE_FLOOR  1e-4  /* was 1e-3; restores dynamic range for the gate */

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

    /* ---- Ridge-corrected residual variance ----
       tr(H) = n_terms − λ·||L⁻¹||_F²  (via dtrtri, ~2 ms at n_terms=231) */
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

    /* Rebuild KNN index from the same params array we trained on.
       Uses RAW params (X), not normalised — knn_rebuild handles
       standardisation internally so query normalisation is consistent. */
    if (poly->knn) knn_rebuild(poly->knn, X, N);

    /* Optional diagnostics — enable with EMU_DEBUG_RV=1 */
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

    /* -----------------------------------------------------------------
       Predictive variance — smooth saturation, KNN distance as
       coverage signal.

       var_norm = rv + (1 - rv) · dnorm² / (k² + dnorm²)

       Properties:
         dnorm = 0  (at training data)     → var_norm ≈ rv  (fit floor)
         dnorm = k  (k cluster radii away) → var_norm ≈ rv + (1-rv)/2
         dnorm → ∞  (far extrapolation)    → var_norm → 1.0 (prior)

       With k = 5 and cluster-radius normalisation from knn_rebuild,
       typical MCMC proposals (1σ from center, dnorm ≈ 0.4) give
       f = 0.0067, so uncertainty ≈ sqrt(rv) ≈ 0.05-0.10 — well inside
       the fallback threshold. Proposals 5+ cluster radii away (dnorm
       > 1) see f > 0.14 → uncertainty grows into the block range.
       ----------------------------------------------------------------- */
    double dnorm = (poly->knn && poly->knn->N >= poly->knn->k)
                 ? knn_query_dnorm(poly->knn, x)
                 : 100.0;

    double rv = poly->residual_var;
    if (rv < 0.0) rv = 0.0;
    if (rv > 1.0) rv = 1.0;

    const double k = 5.0;
    double dnorm_sq = dnorm * dnorm;
    double f = dnorm_sq / (k * k + dnorm_sq);
    /* LOCAL-DOMINANT: distance f carries 95% of the dynamic range.
       rv_local becomes a small floor (5%) that prevents confident-
       wrong predictions at training points, but doesn't freeze the
       gate. With rv_local = 0.15:
         at training data (f=0):  var_norm ≈ 0.0075 → unc ≈ 0.09
         at k cluster radii (f=0.5): var_norm ≈ 0.48 → unc ≈ 0.69
         far extrapolation (f=1): var_norm ≈ 0.95 → unc ≈ 0.97
       Dynamic range is 10×, not the 1.8× the old formula gave. */
    const double LOCAL_WEIGHT = 0.95;
    double var_norm = LOCAL_WEIGHT * f + (1.0 - LOCAL_WEIGHT) * rv;
    *var = var_norm * poly->y_std * poly->y_std;
    // double var_norm = rv + (1.0 - rv) * f;
    // *var = var_norm * poly->y_std * poly->y_std;

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