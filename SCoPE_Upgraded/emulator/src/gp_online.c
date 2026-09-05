#include "gp_online.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <lapacke.h>

#define IDX(i, j, ncol) ((i) * (ncol) + (j))
#define EPS 1e-12
#define M_PI 3.14159265358979323846

/* ---------- kernel and covariance builders ---------- */
static double kernel_ard_rbf(const double *x1, const double *x2,
                             double sigma_f, const double *ell, int n)
{
    double r2 = 0.0;
    for (int d = 0; d < n; d++)
    {
        double diff = (x1[d] - x2[d]) / ell[d];
        r2 += diff * diff;
    }
    return sigma_f * sigma_f * exp(-0.5 * r2);
}

static void build_covariance(const double *X, int N, int n_inputs,
                             double sigma_f, const double *ell,
                             double sigma_n, double *K)
{
    double sf2 = sigma_f * sigma_f;
    double sn2 = sigma_n * sigma_n;
    for (int i = 0; i < N; i++)
    {
        for (int j = i; j < N; j++)
        {
            double k = kernel_ard_rbf(&X[i * n_inputs], &X[j * n_inputs],
                                      sigma_f, ell, n_inputs);
            if (i == j)
                k += sn2;
            K[i * N + j] = k;
            K[j * N + i] = k;
        }
    }
}

/* ---------- log marginal likelihood (reuses workspace) ---------- */
static double log_marginal_likelihood(const double *X, const double *y, int N,
                                      int n_inputs,
                                      double sigma_f, const double *ell,
                                      double sigma_n)
{
    double *K = malloc(N * N * sizeof(double));
    build_covariance(X, N, n_inputs, sigma_f, ell, sigma_n, K);

    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', N, K, N);
    if (info != 0)
    {
        free(K);
        return -1e300;
    }

    double *alpha = malloc(N * sizeof(double));
    memcpy(alpha, y, N * sizeof(double));
    info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', N, 1, K, N, alpha, 1);
    if (info != 0)
    {
        free(K);
        free(alpha);
        return -1e300;
    }

    double logdet = 0.0;
    for (int i = 0; i < N; i++)
        logdet += 2.0 * log(K[i * N + i]);

    double quad = 0.0;
    for (int i = 0; i < N; i++)
        quad += y[i] * alpha[i];

    free(K);
    free(alpha);
    return -0.5 * quad - 0.5 * logdet - 0.5 * N * log(2.0 * M_PI);
}

/* ---------- simple coordinate ascent hyperparameter optimisation ---------- */
static void optimize_hyperparams(const double *X_norm, const double *y_norm,
                                 int N, int n_inputs,
                                 double *sigma_f, double *sigma_n,
                                 double *ell,
                                 int max_iter)
{
    /* Work in log space */
    double log_sf = log(*sigma_f);
    double log_sn = log(*sigma_n);
    double *log_ell = malloc(n_inputs * sizeof(double));
    for (int i = 0; i < n_inputs; i++)
        log_ell[i] = log(ell[i]);

    double step = 0.5;
    double best_lml = log_marginal_likelihood(X_norm, y_norm, N, n_inputs, *sigma_f, ell, *sigma_n);

    /* Temporary array for exponentiated length scales */
    double *tmp_ell = malloc(n_inputs * sizeof(double));

    for (int iter = 0; iter < max_iter; iter++)
    {
        int improved = 0;

        /* Optimise log_sigma_f */
        double cand_sf = exp(log_sf + step);
        double cand_lml = log_marginal_likelihood(X_norm, y_norm, N, n_inputs, cand_sf, ell, *sigma_n);
        if (cand_lml > best_lml)
        {
            log_sf += step;
            best_lml = cand_lml;
            improved = 1;
        }
        else
        {
            cand_sf = exp(log_sf - step);
            cand_lml = log_marginal_likelihood(X_norm, y_norm, N, n_inputs, cand_sf, ell, *sigma_n);
            if (cand_lml > best_lml)
            {
                log_sf -= step;
                best_lml = cand_lml;
                improved = 1;
            }
        }

        /* Optimise each length scale */
        for (int d = 0; d < n_inputs; d++)
        {
            memcpy(tmp_ell, ell, n_inputs * sizeof(double));
            tmp_ell[d] = exp(log_ell[d] + step);
            cand_lml = log_marginal_likelihood(X_norm, y_norm, N, n_inputs, exp(log_sf), tmp_ell, *sigma_n);
            if (cand_lml > best_lml)
            {
                best_lml = cand_lml;
                improved = 1;
                log_ell[d] += step;
                ell[d] = tmp_ell[d];
            }
            else
            {
                tmp_ell[d] = exp(log_ell[d] - step);
                cand_lml = log_marginal_likelihood(X_norm, y_norm, N, n_inputs, exp(log_sf), tmp_ell, *sigma_n);
                if (cand_lml > best_lml)
                {
                    best_lml = cand_lml;
                    improved = 1;
                    log_ell[d] -= step;
                    ell[d] = tmp_ell[d];
                }
            }
        }

        /* Optimise log_sigma_n */
        double cand_sn = exp(log_sn + step);
        cand_lml = log_marginal_likelihood(X_norm, y_norm, N, n_inputs, exp(log_sf), ell, cand_sn);
        if (cand_lml > best_lml)
        {
            log_sn += step;
            best_lml = cand_lml;
            improved = 1;
        }
        else
        {
            cand_sn = exp(log_sn - step);
            cand_lml = log_marginal_likelihood(X_norm, y_norm, N, n_inputs, exp(log_sf), ell, cand_sn);
            if (cand_lml > best_lml)
            {
                log_sn -= step;
                best_lml = cand_lml;
                improved = 1;
            }
        }

        if (!improved)
            step *= 0.5;
        if (step < 1e-3)
            break;
    }

    *sigma_f = exp(log_sf);
    *sigma_n = exp(log_sn);
    /* ell already updated in the loop */

    free(log_ell);
    free(tmp_ell);
}

/* ---------- Public functions ---------- */
GPModel *online_gp_init(int n_inputs)
{
    GPModel *gp = calloc(1, sizeof(GPModel));
    if (!gp)
        return NULL;
    gp->n_inputs = n_inputs;
    gp->n_train = 0;
    gp->X_norm = NULL;
    gp->y_norm = NULL;
    gp->alpha = NULL;
    gp->chol_L = NULL;
    gp->ell = malloc(n_inputs * sizeof(double));
    gp->X_mean = malloc(n_inputs * sizeof(double));
    gp->X_std = malloc(n_inputs * sizeof(double));
    gp->sigma_f = 1.0;
    gp->sigma_n = 0.01;
    for (int i = 0; i < n_inputs; i++)
        gp->ell[i] = 1.0;
    gp->y_mean = 0.0;
    gp->y_std = 1.0;
    return gp;
}

void online_gp_train(GPModel *gp, const double *X, const double *y, int N)
{
    if (N < 3 || !gp)
        return;
    int n_in = gp->n_inputs;

    /* Standardise inputs */
    for (int j = 0; j < n_in; j++)
    {
        double mean = 0.0;
        for (int i = 0; i < N; i++)
            mean += X[i * n_in + j];
        mean /= N;
        gp->X_mean[j] = mean;
        double var = 0.0;
        for (int i = 0; i < N; i++)
        {
            double d = X[i * n_in + j] - mean;
            var += d * d;
        }
        var /= N;
        gp->X_std[j] = sqrt(var);
        if (gp->X_std[j] < EPS)
            gp->X_std[j] = 1.0;
    }

    double *X_norm = malloc(N * n_in * sizeof(double));
    for (int i = 0; i < N; i++)
        for (int j = 0; j < n_in; j++)
            X_norm[i * n_in + j] = (X[i * n_in + j] - gp->X_mean[j]) / gp->X_std[j];

    /* Standardise outputs */
    double y_mean = 0.0;
    for (int i = 0; i < N; i++)
        y_mean += y[i];
    y_mean /= N;
    double y_var = 0.0;
    for (int i = 0; i < N; i++)
    {
        double d = y[i] - y_mean;
        y_var += d * d;
    }
    y_var /= N;
    double y_std = sqrt(y_var);
    if (y_std < EPS)
        y_std = 1.0;
    gp->y_mean = y_mean;
    gp->y_std = y_std;

    double *y_norm = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++)
        y_norm[i] = (y[i] - y_mean) / y_std;

    /* Initial hyperparameters (will be optimised) */
    double sigma_f = y_std;        // signal standard deviation = output std dev
    double sigma_n = 0.01 * y_std; // noise std dev = 1% of signal
    double *ell = malloc(n_in * sizeof(double));
    for (int i = 0; i < n_in; i++)
        ell[i] = 1.0;

    /* Optimise hyperparameters (20 iterations is enough) */
    optimize_hyperparams(X_norm, y_norm, N, n_in, &sigma_f, &sigma_n, ell, 120);

    /* Store optimised hyperparameters */
    gp->sigma_f = sigma_f;
    gp->sigma_n = sigma_n;
    for (int i = 0; i < n_in; i++)
        gp->ell[i] = ell[i];

    /* Build covariance with optimised hyperparameters */
    double *K = malloc(N * N * sizeof(double));
    build_covariance(X_norm, N, n_in, sigma_f, ell, sigma_n, K);

    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', N, K, N);
    if (info != 0)
    {
        fprintf(stderr, "GP Cholesky failed\n");
        free(X_norm);
        free(y_norm);
        free(K);
        free(ell);
        return;
    }

    double *alpha = malloc(N * sizeof(double));
    memcpy(alpha, y_norm, N * sizeof(double));
    info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', N, 1, K, N, alpha, 1);
    if (info != 0)
    {
        fprintf(stderr, "GP solve failed\n");
        free(alpha);
        alpha = NULL;
    }

    /* Free old data and assign new */
    if (gp->X_norm)
        free(gp->X_norm);
    if (gp->y_norm)
        free(gp->y_norm);
    if (gp->alpha)
        free(gp->alpha);
    if (gp->chol_L)
        free(gp->chol_L);
    gp->X_norm = X_norm;
    gp->y_norm = y_norm;
    gp->alpha = alpha;
    gp->chol_L = K;
    gp->n_train = N;

    free(ell);
}

double online_gp_predict_mean(const GPModel *gp, const double *x)
{
    if (!gp || gp->n_train == 0 || !gp->alpha)
        return gp->y_mean;
    int n_in = gp->n_inputs;
    double *x_norm = malloc(n_in * sizeof(double));
    for (int j = 0; j < n_in; j++)
        x_norm[j] = (x[j] - gp->X_mean[j]) / gp->X_std[j];

    double *kstar = malloc(gp->n_train * sizeof(double));
    for (int i = 0; i < gp->n_train; i++)
    {
        kstar[i] = kernel_ard_rbf(x_norm, &gp->X_norm[i * n_in],
                                  gp->sigma_f, gp->ell, n_in);
    }
    double mean_norm = 0.0;
    for (int i = 0; i < gp->n_train; i++)
        mean_norm += kstar[i] * gp->alpha[i];
    free(x_norm);
    free(kstar);
    return mean_norm * gp->y_std + gp->y_mean;
}

void online_gp_predict(const GPModel *gp, const double *x, double *mean, double *var)
{
    if (!gp || gp->n_train == 0 || !gp->alpha)
    {
        *mean = gp->y_mean;
        *var = gp->sigma_n * gp->sigma_n;
        return;
    }
    int n_in = gp->n_inputs;
    double *x_norm = malloc(n_in * sizeof(double));
    for (int j = 0; j < n_in; j++)
        x_norm[j] = (x[j] - gp->X_mean[j]) / gp->X_std[j];

    double *kstar = malloc(gp->n_train * sizeof(double));
    for (int i = 0; i < gp->n_train; i++)
    {
        kstar[i] = kernel_ard_rbf(x_norm, &gp->X_norm[i * n_in],
                                  gp->sigma_f, gp->ell, n_in);
    }

    double mean_norm = 0.0;
    for (int i = 0; i < gp->n_train; i++)
        mean_norm += kstar[i] * gp->alpha[i];

    /* variance: K(x,x) - kstar^T * K^{-1} * kstar */
    double var_norm = gp->sigma_f * gp->sigma_f + gp->sigma_n * gp->sigma_n;
    double *v = malloc(gp->n_train * sizeof(double));
    memcpy(v, kstar, gp->n_train * sizeof(double));
    /* Solve L * v = kstar (forward substitution) */
    for (int i = 0; i < gp->n_train; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < i; j++)
            sum += gp->chol_L[i * gp->n_train + j] * v[j];
        v[i] = (v[i] - sum) / gp->chol_L[i * gp->n_train + i];
    }
    double vv = 0.0;
    for (int i = 0; i < gp->n_train; i++)
        vv += v[i] * v[i];
    var_norm -= vv;
    if (var_norm < 0.0)
        var_norm = 0.0;

    *mean = mean_norm * gp->y_std + gp->y_mean;
    *var = var_norm * gp->y_std * gp->y_std;

    free(x_norm);
    free(kstar);
    free(v);
}

void online_gp_free(GPModel *gp)
{
    if (gp)
    {
        free(gp->X_norm);
        free(gp->y_norm);
        free(gp->alpha);
        free(gp->chol_L);
        free(gp->ell);
        free(gp->X_mean);
        free(gp->X_std);
        free(gp);
    }
}