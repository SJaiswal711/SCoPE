#ifndef KNN_UNCERTAINTY_H
#define KNN_UNCERTAINTY_H

/*
 * KNN-distance uncertainty for polynomial emulators.
 *
 * Measures how far a query point is from the nearest training points,
 * normalised by the typical within-training-set spacing. This is a
 * structural measure of interpolation vs. extrapolation that is robust
 * against the "clustering at MCMC convergence" failure mode of ridge
 * leverage.
 *
 * Returns dnorm where:
 *   dnorm ≈ 0     → at a training point
 *   dnorm ≈ 1     → at the typical training spacing (between points)
 *   dnorm >> 1    → far outside the training hull
 *
 * Consumed by online_poly_predict via:
 *   var_norm = rv + (1-rv) · dnorm²/(1+dnorm²)
 */

typedef struct {
    int      n_params;
    int      N;
    int      k;                /* k nearest neighbours (typically 5) */
    double  *X_norm;           /* [N × n_params] normalised training params */
    double  *X_mean;           /* [n_params] */
    double  *X_std;            /* [n_params] */
    double   median_nn_dist;   /* normaliser for query distances */
} KNNUncertainty;

KNNUncertainty* knn_create(int n_params, int capacity, int k);
void            knn_destroy(KNNUncertainty *knn);
void            knn_rebuild(KNNUncertainty *knn, const double *params, int N);
double          knn_query_dnorm(const KNNUncertainty *knn, const double *theta);

#endif