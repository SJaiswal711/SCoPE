#ifndef POLY_ONLINE_H
#define POLY_ONLINE_H

#include <stddef.h>

typedef struct {
    int n_inputs;
    int degree;
    int n_terms;
    double *coeff;               // coefficients of polynomial
    double *X_mean;              // input means (size n_inputs)
    double *X_std;               // input stds (size n_inputs)
    double y_mean;               // output mean (scalar)
    double y_std;                // output std (scalar)
    double residual_var;         // residual variance (normalised output space)
    double *X_train_norm;        // normalised training inputs (n_train × n_inputs)
    int n_train;
    double avg_nn_distance;      // average nearest-neighbour distance
} PolyModel;

PolyModel* online_poly_init(int n_inputs, int degree);
void online_poly_train(PolyModel *poly, const double *X, const double *y, int N);
double online_poly_predict_mean(const PolyModel *poly, const double *x);
void online_poly_predict(const PolyModel *poly, const double *x, double *mean, double *var);
void online_poly_free(PolyModel *poly);

#endif