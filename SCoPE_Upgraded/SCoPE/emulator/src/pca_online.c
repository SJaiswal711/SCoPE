#include "pca_online.h"
#include "utils.h"
#include "transforms.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <lapacke.h>

#define IDX(i, j, ncol) ((i)*(ncol)+(j))
#define EPS 1e-12

static void compute_svd(double *X, int N, int D, double **S, double **U, double **VT) {
    *S = *U = *VT = NULL;

    /* Fail loudly if the input contains NaN/Inf. This catches the case
       where a caller fed us a poisoned matrix before LAPACK would. */
    for (size_t k = 0; k < (size_t)N * D; k++) {
        if (!isfinite(X[k])) {
            fprintf(stderr,
                "compute_svd: non-finite input at flat index %zu (row %zu, col %zu)\n",
                k, k / (size_t)D, k % (size_t)D);
            return;
        }
    }

    int M = (N < D) ? N : D;
    *S  = malloc(M * sizeof(double));
    *U  = malloc((size_t)N * M * sizeof(double));
    *VT = malloc((size_t)M * D * sizeof(double));
    if (!*S || !*U || !*VT) {
        fprintf(stderr, "compute_svd: allocation failed (N=%d, D=%d, M=%d)\n",
                N, D, M);
        free(*S); free(*U); free(*VT);
        *S = *U = *VT = NULL;
        return;
    }

    double *X_copy = malloc((size_t)N * D * sizeof(double));
    if (!X_copy) {
        fprintf(stderr, "compute_svd: allocation failed for X_copy\n");
        free(*S); free(*U); free(*VT);
        *S = *U = *VT = NULL;
        return;
    }
    memcpy(X_copy, X, (size_t)N * D * sizeof(double));

    int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S',
                              N, D, X_copy, D,
                              *S, *U, M, *VT, D);
    free(X_copy);
    if (info) {
        fprintf(stderr, "SVD failed: info=%d (N=%d, D=%d, M=%d)\n",
                info, N, D, M);
        free(*S); free(*U); free(*VT);
        *S = *U = *VT = NULL;
        return;
    }
}

PCAModel* pca_load_model(const char *model_dir, const char *field) {
    char path[512];
    snprintf(path, sizeof(path), "%s/%s/%s_info.txt", model_dir, field, field);
    FILE *fp = fopen(path, "r");
    if (!fp) { perror(path); return NULL; }
    int modes = 0, D = 0;
    char key[256];
    while (fscanf(fp, "%s", key) == 1) {
        if (strcmp(key, "modes") == 0) fscanf(fp, "%d", &modes);
        else if (strcmp(key, "D") == 0) fscanf(fp, "%d", &D);
    }
    fclose(fp);
    if (modes == 0 || D == 0) return NULL;

    PCAModel *pca = calloc(1, sizeof(PCAModel));
    pca->n_features = D;
    pca->n_modes = modes;
    pca->mean = malloc(D * sizeof(double));
    pca->basis = malloc((size_t)modes * D * sizeof(double));
    pca->singular = malloc(modes * sizeof(double));

    snprintf(path, sizeof(path), "%s/%s/%s_mean.bin", model_dir, field, field);
    if (!load_binary_double(path, pca->mean, D)) goto error;

    snprintf(path, sizeof(path), "%s/%s/%s_basis.bin", model_dir, field, field);
    if (!load_binary_double(path, pca->basis, (size_t)modes * D)) goto error;

    // singular values optional
    snprintf(path, sizeof(path), "%s/%s/%s_singular.bin", model_dir, field, field);
    if (!load_binary_double(path, pca->singular, modes)) {
        free(pca->singular);
        pca->singular = NULL;
    }

    // Explained variance cannot be recovered from a truncated set of
    // singular values alone; fall back to a nominal value unless the
    // full spectrum is available in the file.
    pca->explained_var = 0.995;

    return pca;
error:
    pca_free(pca);
    return NULL;
}

PCAModel* pca_train_from_spectra(const double *spectra, int N, int n_features,
                                 int max_modes, int apply_log10) {
    if (N < 2) return NULL;
    int D = n_features;
    double *X = malloc((size_t)N * D * sizeof(double));
    if (!X) return NULL;

    if (apply_log10) {
        for (size_t i = 0; i < (size_t)N * D; i++) {
            double v = spectra[i];
            if (!isfinite(v) || v < 0.0) v = 0.0;
            X[i] = log10(v + EPS);
        }
    } else {
        for (size_t i = 0; i < (size_t)N * D; i++) {
            double v = spectra[i];
            X[i] = isfinite(v) ? v : 0.0;
        }
    }

    // compute mean
    double *mean = calloc(D, sizeof(double));
    if (!mean) { free(X); return NULL; }
    for (int j = 0; j < D; j++) {
        double sum = 0.0;
        for (int i = 0; i < N; i++) sum += X[i * D + j];
        mean[j] = sum / N;
    }

    // center
    double *Y = malloc((size_t)N * D * sizeof(double));
    if (!Y) { free(X); free(mean); return NULL; }
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            Y[i * D + j] = X[i * D + j] - mean[j];

    double *S, *U, *VT;
    compute_svd(Y, N, D, &S, &U, &VT);
    free(Y);
    if (!S || !U || !VT) {
        fprintf(stderr, "SVD computation failed, PCA training aborted\n");
        free(X); free(mean);
        return NULL;
    }

    // select modes
    double total_var = 0.0;
    int M = (N < D) ? N : D;
    for (int i = 0; i < M; i++) total_var += S[i] * S[i];
    double cum = 0.0;
    int modes = M;
    for (int i = 0; i < M; i++) {
        cum += S[i] * S[i];
        if (total_var > 0.0 && cum / total_var >= 0.9999) {
            modes = i + 1;
            break;
        }
    }
    if (modes > max_modes) modes = max_modes;

    PCAModel *pca = calloc(1, sizeof(PCAModel));
    if (!pca) {
        free(S); free(U); free(VT); free(X); free(mean);
        return NULL;
    }
    pca->n_features = D;
    pca->n_modes = modes;
    pca->mean = mean;
    pca->basis = malloc((size_t)modes * D * sizeof(double));
    if (!pca->basis) {
        free(S); free(U); free(VT); free(X);
        pca_free(pca);
        return NULL;
    }
    for (int m = 0; m < modes; m++)
        for (int j = 0; j < D; j++)
            pca->basis[m * D + j] = VT[m * D + j];

    pca->singular = malloc(modes * sizeof(double));
    if (pca->singular) {
        for (int m = 0; m < modes; m++) pca->singular[m] = S[m];
    }

    if (total_var > 0.0) {
        double cum_selected = 0.0;
        for (int m = 0; m < modes; m++) cum_selected += S[m] * S[m];
        pca->explained_var = cum_selected / total_var;
    } else {
        pca->explained_var = 0.0;
    }

    free(S); free(U); free(VT);
    free(X);
    return pca;
}

void pca_reconstruct(const PCAModel *pca, const double *coeffs, double *spectrum) {
    int D = pca->n_features;
    memcpy(spectrum, pca->mean, D * sizeof(double));
    for (int m = 0; m < pca->n_modes; m++) {
        double c = coeffs[m];
        if (c == 0.0) continue;
        const double *basis = pca->basis + m * D;
        for (int i = 0; i < D; i++)
            spectrum[i] += c * basis[i];
    }
}

void pca_project(const PCAModel *pca, const double *spectrum, double *coeffs) {
    int D = pca->n_features;
    for (int m = 0; m < pca->n_modes; m++) {
        double sum = 0.0;
        const double *basis = pca->basis + m * D;
        for (int i = 0; i < D; i++) {
            double diff = spectrum[i] - pca->mean[i];
            sum += diff * basis[i];
        }
        coeffs[m] = sum;
    }
}

void pca_free(PCAModel *pca) {
    if (pca) {
        free(pca->mean);
        free(pca->basis);
        free(pca->singular);
        free(pca);
    }
}