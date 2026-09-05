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
    int M = (N < D) ? N : D;
    *S = malloc(M * sizeof(double));
    *U = malloc(N * M * sizeof(double));
    *VT = malloc(M * D * sizeof(double));
    double *X_copy = malloc(N * D * sizeof(double));
    memcpy(X_copy, X, N * D * sizeof(double));
    int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S',
                              N, D, X_copy, D,
                              *S, *U, M, *VT, D);
    free(X_copy);
    if (info) {
        fprintf(stderr, "SVD failed\n");
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
    pca->basis = malloc(modes * D * sizeof(double));
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

    // explained variance not stored; compute approximate from singular values
    if (pca->singular) {
        double total = 0.0;
        for (int i = 0; i < modes; i++) total += pca->singular[i] * pca->singular[i];
        double cum = 0.0;
        for (int i = 0; i < modes; i++) cum += pca->singular[i] * pca->singular[i];
        pca->explained_var = cum / total;
    } else {
        pca->explained_var = 0.995;
    }

    return pca;
error:
    pca_free(pca);
    return NULL;
}

PCAModel* pca_train_from_spectra(const double *spectra, int N, int n_features,
                                 int max_modes, int apply_log10) {
    if (N < 2) return NULL;
    int D = n_features;
    double *X = malloc(N * D * sizeof(double));
    if (apply_log10) {
        for (int i = 0; i < N * D; i++)
            X[i] = log10(spectra[i] + EPS);
    } else {
        memcpy(X, spectra, N * D * sizeof(double));
    }

    // compute mean
    double *mean = calloc(D, sizeof(double));
    for (int j = 0; j < D; j++) {
        double sum = 0.0;
        for (int i = 0; i < N; i++) sum += X[i * D + j];
        mean[j] = sum / N;
    }

    // center
    double *Y = malloc(N * D * sizeof(double));
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
    for (int i = 0; i < (N<D?N:D); i++) total_var += S[i] * S[i];
    double cum = 0.0;
    int modes = (N<D?N:D);
    for (int i = 0; i < modes; i++) {
        cum += S[i] * S[i];
        if (cum / total_var >= 0.9999) {
            modes = i + 1;
            break;
        }
    }
    if (modes > max_modes) modes = max_modes;

    PCAModel *pca = calloc(1, sizeof(PCAModel));
    pca->n_features = D;
    pca->n_modes = modes;
    pca->mean = mean;
    pca->basis = malloc(modes * D * sizeof(double));
    for (int m = 0; m < modes; m++)
        for (int j = 0; j < D; j++)
            pca->basis[m * D + j] = VT[m * D + j];
    pca->singular = malloc(modes * sizeof(double));
    for (int m = 0; m < modes; m++) pca->singular[m] = S[m];
    pca->explained_var = cum / total_var;

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