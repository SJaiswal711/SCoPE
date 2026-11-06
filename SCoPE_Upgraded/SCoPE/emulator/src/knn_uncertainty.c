#include "knn_uncertainty.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

KNNUncertainty* knn_create(int n_params, int capacity, int k) {
    KNNUncertainty *knn = calloc(1, sizeof(KNNUncertainty));
    if (!knn) return NULL;
    knn->n_params = n_params;
    knn->k = (k > 0) ? k : 5;
    knn->N = 0;
    knn->X_norm = malloc((size_t)capacity * n_params * sizeof(double));
    knn->X_mean = malloc(n_params * sizeof(double));
    knn->X_std  = malloc(n_params * sizeof(double));
    if (!knn->X_norm || !knn->X_mean || !knn->X_std) {
        knn_destroy(knn);
        return NULL;
    }
    knn->median_nn_dist = 1.0;
    return knn;
}

void knn_destroy(KNNUncertainty *knn) {
    if (!knn) return;
    free(knn->X_norm);
    free(knn->X_mean);
    free(knn->X_std);
    free(knn);
}

void knn_rebuild(KNNUncertainty *knn, const double *params, int N) {
    if (!knn || !params || N <= 0) return;
    int np = knn->n_params;

    /* Per-parameter mean and std from the training set */
    for (int j = 0; j < np; j++) {
        double mean = 0.0;
        for (int i = 0; i < N; i++) mean += params[i * np + j];
        mean /= N;
        knn->X_mean[j] = mean;

        double var = 0.0;
        for (int i = 0; i < N; i++) {
            double d = params[i * np + j] - mean;
            var += d * d;
        }
        var /= N;
        knn->X_std[j] = (var > 1e-30) ? sqrt(var) : 1.0;
    }

    /* Normalise and store */
    for (int i = 0; i < N; i++)
        for (int j = 0; j < np; j++)
            knn->X_norm[i * np + j] =
                (params[i * np + j] - knn->X_mean[j]) / knn->X_std[j];
    knn->N = N;

    /* -----------------------------------------------------------------
       NORMALISER: RMS distance from centroid = cluster radius.

       This is the important fix. The previous code used the median
       nearest-neighbour distance, which measures data DENSITY
       (~cluster_radius / sqrt(N)). At convergence with N=400 in 6D,
       that's ~0.05 in normalised units. Typical MCMC proposals then
       have dnorm = 1.0 / 0.05 = 20 -- permanently way outside the
       gate regardless of where the proposal actually is.

       Using RMS distance from centroid instead gives ~sqrt(np) for a
       roughly Gaussian cluster (≈ 2.45 for 6D). Typical proposals
       then have dnorm ≈ 0.4. Combined with k=5 saturation in
       online_poly_predict, this produces uncertainty ≈ sqrt(rv) for
       interpolation and → 1.0 for extrapolation.

       The field name `median_nn_dist` is preserved for API
       compatibility with knn_uncertainty.h; it now stores the RMS
       cluster radius.
       ----------------------------------------------------------------- */
    double rms_sq = 0.0;
    for (int i = 0; i < N; i++) {
        double s = 0.0;
        for (int p = 0; p < np; p++) {
            double d = knn->X_norm[i * np + p];   /* already centred */
            s += d * d;
        }
        rms_sq += s;
    }
    knn->median_nn_dist = sqrt(rms_sq / N);
    if (knn->median_nn_dist < 1e-6) knn->median_nn_dist = 1e-6;
}

double knn_query_dnorm(const KNNUncertainty *knn, const double *theta) {
    if (!knn || knn->N < knn->k) return 100.0;
    int np = knn->n_params;
    int k  = knn->k;

    /* Fixed-size sorted array of k smallest squared distances (k ≤ 16) */
    double kbest[16];
    for (int i = 0; i < k; i++) kbest[i] = 1e300;
    int nfilled = 0;

    for (int i = 0; i < knn->N; i++) {
        double s = 0.0;
        for (int p = 0; p < np; p++) {
            double xp = (theta[p] - knn->X_mean[p]) / knn->X_std[p];
            double d  = xp - knn->X_norm[i * np + p];
            s += d * d;
        }
        if (nfilled < k) {
            kbest[nfilled++] = s;
            for (int a = nfilled - 1; a > 0 && kbest[a] < kbest[a - 1]; a--) {
                double t = kbest[a]; kbest[a] = kbest[a - 1]; kbest[a - 1] = t;
            }
        } else if (s < kbest[k - 1]) {
            kbest[k - 1] = s;
            for (int a = k - 1; a > 0 && kbest[a] < kbest[a - 1]; a--) {
                double t = kbest[a]; kbest[a] = kbest[a - 1]; kbest[a - 1] = t;
            }
        }
    }

    double avg_d2 = 0.0;
    for (int i = 0; i < k; i++) avg_d2 += kbest[i];
    avg_d2 /= k;

    return sqrt(avg_d2) / knn->median_nn_dist;
}