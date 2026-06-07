#ifndef PCA_ONLINE_H
#define PCA_ONLINE_H

#include <stddef.h>

/* PCA model (can be loaded from file or trained from buffer) */
typedef struct {
    int n_features;          /* Spectrum length (e.g., 2601) */
    int n_modes;             /* Number of PCA modes */
    double *mean;            /* Mean spectrum (n_features) */
    double *basis;           /* Basis vectors: n_modes × n_features (row-major) */
    double *singular;        /* Singular values (n_modes) - optional */
    double explained_var;    /* Fraction of variance explained */
} PCAModel;

/* Load PCA model from binary files (produced by pca_train.c) */
PCAModel* pca_load_model(const char *model_dir, const char *field);

/* Train PCA from a set of spectra (N points, each of length n_features) */
/* If apply_log10 = 1, apply log10 transform before training */
PCAModel* pca_train_from_spectra(const double *spectra, int N, int n_features,
                                 int max_modes, int apply_log10);

/* Reconstruct spectrum from PCA coefficients */
void pca_reconstruct(const PCAModel *pca, const double *coeffs, double *spectrum);

/* Project spectrum to PCA coefficients */
void pca_project(const PCAModel *pca, const double *spectrum, double *coeffs);

/* Free PCA model */
void pca_free(PCAModel *pca);

#endif /* PCA_ONLINE_H */