#ifndef GP_ONLINE_H
#define GP_ONLINE_H

#include <stddef.h>

typedef struct {
    int n_train;
    int n_inputs;
    double *X_norm;        /* normalized inputs (n_train × n_inputs) */
    double *y_norm;        /* normalized targets (n_train) */
    double *alpha;
    double *chol_L;
    double *ell;
    double sigma_f;
    double sigma_n;
    double y_mean, y_std;
    double *X_mean, *X_std;
} GPModel;

GPModel* online_gp_init(int n_inputs);
void online_gp_train(GPModel *gp, const double *X, const double *y, int N);
double online_gp_predict_mean(const GPModel *gp, const double *x);
void online_gp_predict(const GPModel *gp, const double *x, double *mean, double *var);
void online_gp_free(GPModel *gp);

#endif