#ifndef POLY_ONLINE_H
#define POLY_ONLINE_H

#include <stddef.h>
#include "knn_uncertainty.h"
typedef struct {
    int n_inputs;
    int degree;
    int n_terms;
    
    double *coeff;               // Posterior mean of the coefficients (size n_terms)
    double *chol_L;              // Lower Cholesky factor of (A^T A + lambda I), 
                                 // row-major (n_terms x n_terms). Used to compute 
                                 // the exact predictive variance: z = L^-1 phi(x_*)
                                 // var = sigma^2 * (1 + z^T z).

    double *X_mean;              // input means (size n_inputs)
    double *X_std;               // input stds (size n_inputs)
    
    double y_mean;               // output mean (scalar)
    double y_std;                // output std (scalar)
    double residual_var;         // residual variance (normalised output space)
    
    int n_train;
    KNNUncertainty *knn;
} PolyModel;

PolyModel* online_poly_init(int n_inputs, int degree);
void online_poly_train(PolyModel *poly, const double *X, const double *y, int N);
double online_poly_predict_mean(const PolyModel *poly, const double *x);
void online_poly_predict(const PolyModel *poly, const double *x, double *mean, double *var);
void online_poly_free(PolyModel *poly);

#endif