#include "poly_online.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <lapacke.h>

#define EPS 1e-12
#define INF 1e100
#define RIDGE_LAMBDA 1e-4

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
        // Optional: assert(idx == *n_terms) but we trust the recursion
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
    poly->X_mean = malloc(n_inputs * sizeof(double));
    poly->X_std = malloc(n_inputs * sizeof(double));
    poly->X_train_norm = NULL;
    if (!poly->X_mean || !poly->X_std) {
        free(poly->X_mean);
        free(poly->X_std);
        free(poly);
        return NULL;
    }
    poly->y_mean = 0.0;
    poly->y_std = 1.0;
    poly->residual_var = 0.0;
    poly->avg_nn_distance = 0.0;
    return poly;
}

void online_poly_train(PolyModel *poly, const double *X, const double *y, int N) {
    if (!poly || N < 2 || !X || !y) {
        if (poly) {
            poly->n_train = 0;
            if (poly->coeff) { free(poly->coeff); poly->coeff = NULL; }
            if (poly->X_train_norm) { free(poly->X_train_norm); poly->X_train_norm = NULL; }
        }
        return;
    }
    int n_in = poly->n_inputs;
    int deg = poly->degree;

    int n_terms = count_terms(n_in, deg);
    if (N < n_terms) {
        fprintf(stderr, "Warning: N=%d < n_terms=%d, cannot train polynomial.\n", N, n_terms);
        poly->n_train = 0;
        if (poly->coeff) { free(poly->coeff); poly->coeff = NULL; }
        if (poly->X_train_norm) { free(poly->X_train_norm); poly->X_train_norm = NULL; }
        return;
    }

    /* Free old data */
    if (poly->coeff) { free(poly->coeff); poly->coeff = NULL; }
    if (poly->X_train_norm) { free(poly->X_train_norm); poly->X_train_norm = NULL; }
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

    /* ----- Normal equations: (A^T A + λ I) c = A^T y ----- */
    double *ATA = malloc((size_t)n_terms_actual * n_terms_actual * sizeof(double));
    double *ATy = malloc((size_t)n_terms_actual * sizeof(double));
    if (!ATA || !ATy) {
        free(A); free(X_norm); free(y_norm);
        free(ATA); free(ATy);
        return;
    }

    /* Initialise to zero */
    for (int i = 0; i < n_terms_actual; i++) {
        ATy[i] = 0.0;
        for (int j = 0; j < n_terms_actual; j++) {
            ATA[i * n_terms_actual + j] = 0.0;
        }
    }

    /* Compute A^T A and A^T y */
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < n_terms_actual; i++) {
            ATy[i] += A[k * n_terms_actual + i] * y_norm[k];
            for (int j = 0; j < n_terms_actual; j++) {
                ATA[i * n_terms_actual + j] += A[k * n_terms_actual + i] * A[k * n_terms_actual + j];
            }
        }
    }

    /* Add ridge regularisation */
    double lambda = RIDGE_LAMBDA;
    for (int i = 0; i < n_terms_actual; i++) {
        ATA[i * n_terms_actual + i] += lambda;
    }

    /* Solve using Cholesky */
    double *coeff = malloc(n_terms_actual * sizeof(double));
    if (!coeff) {
        free(A); free(X_norm); free(y_norm); free(ATA); free(ATy);
        return;
    }
    memcpy(coeff, ATy, n_terms_actual * sizeof(double));

    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', n_terms_actual, ATA, n_terms_actual);
    if (info != 0) {
        fprintf(stderr, "Cholesky failed (info=%d)\n", info);
        free(A); free(X_norm); free(y_norm); free(ATA); free(ATy); free(coeff);
        return;
    }
    /* FIX: ldb must be 1 for a single right‑hand side (vector) */
    info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', n_terms_actual, 1, ATA, n_terms_actual, coeff, 1);
    if (info != 0) {
        fprintf(stderr, "Solve failed (info=%d)\n", info);
        free(A); free(X_norm); free(y_norm); free(ATA); free(ATy); free(coeff);
        return;
    }

    /* Compute residual variance on original data */
    double residual = 0.0;
    for (int i = 0; i < N; i++) {
        double pred = 0.0;
        for (int j = 0; j < n_terms_actual; j++)
            pred += A[i * n_terms_actual + j] * coeff[j];
        double diff = y_norm[i] - pred;
        residual += diff * diff;
    }
    double res_var = residual / (N - n_terms_actual);
    if (res_var < 0.0 || !isfinite(res_var)) res_var = 0.0;
    poly->residual_var = res_var;

    /* Store coefficients */
    poly->coeff = coeff;
    poly->n_train = N;

    /* Store normalised training inputs for distance‑based uncertainty */
    poly->X_train_norm = X_norm;  // keep it

    /* Average nearest‑neighbour distance */
    double total_dist = 0.0;
    for (int i = 0; i < N; i++) {
        double min_dist = INF;
        for (int j = 0; j < N; j++) {
            if (i == j) continue;
            double dist = 0.0;
            for (int d = 0; d < n_in; d++) {
                double diff = poly->X_train_norm[i * n_in + d] -
                              poly->X_train_norm[j * n_in + d];
                dist += diff * diff;
            }
            dist = sqrt(dist);
            if (dist < min_dist) min_dist = dist;
        }
        total_dist += min_dist;
    }
    poly->avg_nn_distance = (N > 1) ? total_dist / N : 1.0;

    /* Free temporaries (X_norm is now stored in poly) */
    free(A);
    free(ATA);
    free(ATy);
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
    double *x_norm = malloc(n_in * sizeof(double));
    if (!x_norm) {
        *mean = poly->y_mean;
        *var = poly->residual_var * poly->y_std * poly->y_std;
        return;
    }
    for (int j = 0; j < n_in; j++)
        x_norm[j] = (x[j] - poly->X_mean[j]) / poly->X_std[j];

    double *row = malloc(poly->n_terms * sizeof(double));
    if (!row) {
        free(x_norm);
        *mean = poly->y_mean;
        *var = poly->residual_var * poly->y_std * poly->y_std;
        return;
    }
    int idx = 0;
    fill_monomials(x_norm, n_in, poly->degree, row, &idx, 0, 0, 1.0);
    double pred_norm = 0.0;
    for (int i = 0; i < poly->n_terms; i++) pred_norm += row[i] * poly->coeff[i];

    *mean = pred_norm * poly->y_std + poly->y_mean;

    /* Distance‑based scaling – avoid division by zero and cap */
    double min_dist = INF;
    for (int i = 0; i < poly->n_train; i++) {
        double dist = 0.0;
        const double *xi = &poly->X_train_norm[i * n_in];
        for (int d = 0; d < n_in; d++) {
            double diff = x_norm[d] - xi[d];
            dist += diff * diff;
        }
        dist = sqrt(dist);
        if (dist < min_dist) min_dist = dist;
    }
    double avg = (poly->avg_nn_distance > EPS) ? poly->avg_nn_distance : 1.0;
    double scale = 1.0 + (min_dist / avg) * 2.0;
    double var_norm = poly->residual_var * scale;
    *var = var_norm * poly->y_std * poly->y_std;
    if (*var < 0.0 || !isfinite(*var)) *var = 1e10;

    free(x_norm);
    free(row);
}

void online_poly_free(PolyModel *poly) {
    if (poly) {
        free(poly->coeff);
        free(poly->X_mean);
        free(poly->X_std);
        free(poly->X_train_norm);
        free(poly);
    }
}