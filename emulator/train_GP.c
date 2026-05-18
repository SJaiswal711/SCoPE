/*
 * ============================================================
 * GP TRAINING IN C (with hyperparameter optimization)
 * ============================================================
 *
 * Kernel:
 *   ConstantKernel * RBF(ARD) + WhiteKernel
 *
 * Training:
 *   maximize log marginal likelihood using multi-start
 *   coordinate search in log-parameter space
 *
 * Output:
 *   cGP_outputs/TT/
 *
 * Compile:
 *   gcc -O3 -march=native train_gp.c \
 *       -llapacke -llapack -lblas -lm \
 *       -o train_gp
 * ============================================================
 */

#define _USE_MATH_DEFINES

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <sys/stat.h>
#include <time.h>

#include <lapacke.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define IDX(i, j, ncol) ((i) * (ncol) + (j))
#define EPS 1e-12

// ============================================================
// SETTINGS
// ============================================================

static const char *PCA_DIR  = "Ppca_outputs";
static const char *DATA_DIR = "fisher_data_all4";
static const char *OUTDIR   = "cGP_outputs";
static const char *FIELD    = "TT";

#define NPARAMS 11
#define NRESTARTS 3

// log-bounds corresponding to sklearn bounds
#define LOG_SF_LO   (-6.907755278982137)   // log(1e-3)
#define LOG_SF_HI   ( 6.907755278982137)   // log(1e3)
#define LOG_ELL_LO  (-6.907755278982137)   // log(1e-3)
#define LOG_ELL_HI  ( 6.907755278982137)   // log(1e3)
#define LOG_SN_LO   (-23.025850929940457)  // log(1e-10)
#define LOG_SN_HI   ( 2.302585092994046)   // log(1e1)

// ============================================================
// HELPERS
// ============================================================

static double *alloc_double(long n)
{
    double *x = (double *)calloc((size_t)n, sizeof(double));
    if (!x)
    {
        printf("Allocation failed\n");
        exit(1);
    }
    return x;
}

static int64_t *alloc_i64(long n)
{
    int64_t *x = (int64_t *)calloc((size_t)n, sizeof(int64_t));
    if (!x)
    {
        printf("Allocation failed\n");
        exit(1);
    }
    return x;
}

static void load_binary(const char *fname, void *ptr, long n, size_t elem_size)
{
    FILE *fp = fopen(fname, "rb");
    if (!fp)
    {
        printf("Cannot open %s\n", fname);
        exit(1);
    }

    size_t nr = fread(ptr, elem_size, (size_t)n, fp);
    fclose(fp);

    if (nr != (size_t)n)
    {
        printf("Read error %s expected %ld got %zu\n", fname, n, nr);
        exit(1);
    }
}

static void save_binary(const char *fname, const void *ptr, long n, size_t elem_size)
{
    FILE *fp = fopen(fname, "wb");
    if (!fp)
    {
        printf("Cannot write %s\n", fname);
        exit(1);
    }

    size_t nw = fwrite(ptr, elem_size, (size_t)n, fp);
    fclose(fp);

    if (nw != (size_t)n)
    {
        printf("Write error %s expected %ld got %zu\n", fname, n, nw);
        exit(1);
    }
}

static double clamp(double x, double lo, double hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static double urand01(void)
{
    return (double)rand() / (double)RAND_MAX;
}

static double urand(double lo, double hi)
{
    return lo + (hi - lo) * urand01();
}

// ============================================================
// STANDARDIZATION (same as sklearn StandardScaler, ddof=0)
// ============================================================

typedef struct
{
    double mean[NPARAMS];
    double std[NPARAMS];
} Scaler;

static void compute_scaler(const double *X, long N, Scaler *S)
{
    for (int j = 0; j < NPARAMS; j++)
    {
        double mean = 0.0;
        for (long i = 0; i < N; i++)
            mean += X[IDX(i, j, NPARAMS)];
        mean /= (double)N;
        S->mean[j] = mean;

        double var = 0.0;
        for (long i = 0; i < N; i++)
        {
            double d = X[IDX(i, j, NPARAMS)] - mean;
            var += d * d;
        }
        var /= (double)N;
        S->std[j] = sqrt(var);
        if (S->std[j] < EPS) S->std[j] = 1.0;
    }
}

static void apply_scaler(const double *X, double *Y, long N, const Scaler *S)
{
    for (long i = 0; i < N; i++)
    {
        for (int j = 0; j < NPARAMS; j++)
        {
            Y[IDX(i, j, NPARAMS)] =
                (X[IDX(i, j, NPARAMS)] - S->mean[j]) / S->std[j];
        }
    }
}

// ============================================================
// GP HYPERPARAMETERS (log-space)
// theta = [log_sigma_f, log_ell[0..10], log_sigma_n]
// ============================================================

#define NTHETA (1 + NPARAMS + 1)

typedef struct
{
    double log_sigma_f;
    double log_ell[NPARAMS];
    double log_sigma_n;
} Theta;

static void theta_copy(Theta *dst, const Theta *src)
{
    memcpy(dst, src, sizeof(Theta));
}

static void theta_to_values(const Theta *th, double *sigma_f, double ell[NPARAMS], double *sigma_n)
{
    *sigma_f = exp(th->log_sigma_f);
    *sigma_n = exp(th->log_sigma_n);
    for (int d = 0; d < NPARAMS; d++)
        ell[d] = exp(th->log_ell[d]);
}

static double theta_get(const Theta *th, int p)
{
    if (p == 0) return th->log_sigma_f;
    if (p >= 1 && p <= NPARAMS) return th->log_ell[p - 1];
    return th->log_sigma_n;
}

static void theta_set(Theta *th, int p, double v)
{
    if (p == 0) th->log_sigma_f = v;
    else if (p >= 1 && p <= NPARAMS) th->log_ell[p - 1] = v;
    else th->log_sigma_n = v;
}

static void theta_clip(Theta *th)
{
    th->log_sigma_f = clamp(th->log_sigma_f, LOG_SF_LO, LOG_SF_HI);
    for (int d = 0; d < NPARAMS; d++)
        th->log_ell[d] = clamp(th->log_ell[d], LOG_ELL_LO, LOG_ELL_HI);
    th->log_sigma_n = clamp(th->log_sigma_n, LOG_SN_LO, LOG_SN_HI);
}

// ============================================================
// ARD RBF + WHITE KERNEL
// K_ij = sf^2 * exp(-0.5 * sum_d ((xi-xj)/ell_d)^2 ) + sn^2 delta_ij
// ============================================================

static double ard_rbf_kernel(const double *x1, const double *x2, double sigma_f, const double *ell)
{
    double r2 = 0.0;
    for (int d = 0; d < NPARAMS; d++)
    {
        double diff = (x1[d] - x2[d]) / ell[d];
        r2 += diff * diff;
    }
    return sigma_f * sigma_f * exp(-0.5 * r2);
}

static void build_covariance(const double *X, long N, const Theta *th, double *K)
{
    double sigma_f, ell[NPARAMS], sigma_n;
    theta_to_values(th, &sigma_f, ell, &sigma_n);

    for (long i = 0; i < N; i++)
    {
        for (long j = i; j < N; j++)
        {
            double v = ard_rbf_kernel(&X[IDX(i, 0, NPARAMS)], &X[IDX(j, 0, NPARAMS)], sigma_f, ell);
            if (i == j)
                v += sigma_n * sigma_n + 1e-10;
            K[IDX(i, j, N)] = v;
            K[IDX(j, i, N)] = v;
        }
    }
}

// ============================================================
// LOG MARGINAL LIKELIHOOD
// ============================================================

static double log_marginal_likelihood(const double *X, const double *y, long N, const Theta *th)
{
    double *K = alloc_double(N * N);
    build_covariance(X, N, th, K);

    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', (lapack_int)N, K, (lapack_int)N);
    if (info != 0)
    {
        free(K);
        return -1e300;
    }

    double *alpha = alloc_double(N);
    memcpy(alpha, y, (size_t)N * sizeof(double));

    info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', (lapack_int)N, 1, K, (lapack_int)N, alpha, 1);
    if (info != 0)
    {
        free(K);
        free(alpha);
        return -1e300;
    }

    double logdet = 0.0;
    for (long i = 0; i < N; i++)
        logdet += 2.0 * log(K[IDX(i, i, N)]);

    double quad = 0.0;
    for (long i = 0; i < N; i++)
        quad += y[i] * alpha[i];

    free(K);
    free(alpha);

    return -0.5 * quad - 0.5 * logdet - 0.5 * (double)N * log(2.0 * M_PI);
}

// ============================================================
// HYPERPARAMETER OPTIMIZATION
// multi-start coordinate search in log-space
// ============================================================

static double optimize_theta(const double *X, const double *y, long N, Theta *best)
{
    Theta starts[NRESTARTS];
    Theta cand, current;
    double best_lml = -1e300;

    // Start 0: sklearn-like initial guess
    starts[0].log_sigma_f = 0.0;
    for (int d = 0; d < NPARAMS; d++)
        starts[0].log_ell[d] = 0.0;
    starts[0].log_sigma_n = log(1e-5);

    // Random restarts
    for (int r = 1; r < NRESTARTS; r++)
    {
        starts[r].log_sigma_f = urand(LOG_SF_LO, LOG_SF_HI);
        for (int d = 0; d < NPARAMS; d++)
            starts[r].log_ell[d] = urand(LOG_ELL_LO, LOG_ELL_HI);
        starts[r].log_sigma_n = urand(LOG_SN_LO, LOG_SN_HI);
    }

    for (int r = 0; r < NRESTARTS; r++)
    {
        theta_copy(&current, &starts[r]);
        theta_clip(&current);

        double cur_lml = log_marginal_likelihood(X, y, N, &current);

        double step = 1.0;

        for (int iter = 0; iter < 60; iter++)
        {
            int improved = 0;

            for (int p = 0; p < NTHETA; p++)
            {
                double base = theta_get(&current, p);

                // try +step
                theta_copy(&cand, &current);
                theta_set(&cand, p, base + step);
                theta_clip(&cand);
                double lml_plus = log_marginal_likelihood(X, y, N, &cand);

                // try -step
                theta_copy(&cand, &current);
                theta_set(&cand, p, base - step);
                theta_clip(&cand);
                double lml_minus = log_marginal_likelihood(X, y, N, &cand);

                if (lml_plus > cur_lml && lml_plus >= lml_minus)
                {
                    theta_copy(&current, &cand);
                    theta_set(&current, p, base + step);
                    theta_clip(&current);
                    cur_lml = lml_plus;
                    improved = 1;
                }
                else if (lml_minus > cur_lml)
                {
                    theta_copy(&current, &cand);
                    theta_set(&current, p, base - step);
                    theta_clip(&current);
                    cur_lml = lml_minus;
                    improved = 1;
                }
            }

            if (!improved)
                step *= 0.5;

            if (step < 1e-3)
                break;
        }

        if (cur_lml > best_lml)
        {
            best_lml = cur_lml;
            theta_copy(best, &current);
        }
    }

    return best_lml;
}

// ============================================================
// GP PREDICTION
// ============================================================

static void gp_predict(
    const double *Xtrain,
    const double *Xtest,
    const double *y_train_norm,
    long Ntrain,
    long Ntest,
    const Theta *th,
    double *pred_mean_norm,
    double *pred_std_norm
)
{
    double *K = alloc_double(Ntrain * Ntrain);
    build_covariance(Xtrain, Ntrain, th, K);

    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', (lapack_int)Ntrain, K, (lapack_int)Ntrain);
    if (info != 0)
    {
        printf("Cholesky failed in prediction\n");
        exit(1);
    }

    double *alpha = alloc_double(Ntrain);
    memcpy(alpha, y_train_norm, (size_t)Ntrain * sizeof(double));

    info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', (lapack_int)Ntrain, 1, K, (lapack_int)Ntrain, alpha, 1);
    if (info != 0)
    {
        printf("Solve failed in prediction\n");
        exit(1);
    }

    double sigma_f, ell[NPARAMS], sigma_n;
    theta_to_values(th, &sigma_f, ell, &sigma_n);
    double sf2 = sigma_f * sigma_f;
    double sn2 = sigma_n * sigma_n;

    for (long t = 0; t < Ntest; t++)
    {
        double *kstar = alloc_double(Ntrain);

        for (long i = 0; i < Ntrain; i++)
        {
            kstar[i] = ard_rbf_kernel(
                &Xtest[IDX(t, 0, NPARAMS)],
                &Xtrain[IDX(i, 0, NPARAMS)],
                sigma_f,
                ell
            );
        }

        double mean = 0.0;
        for (long i = 0; i < Ntrain; i++)
            mean += kstar[i] * alpha[i];

        double *v = alloc_double(Ntrain);
        memcpy(v, kstar, (size_t)Ntrain * sizeof(double));

        info = LAPACKE_dtrtrs(LAPACK_ROW_MAJOR, 'L', 'N', 'N',
                              (lapack_int)Ntrain, 1,
                              K, (lapack_int)Ntrain,
                              v, 1);
        if (info != 0)
        {
            printf("Triangular solve failed in prediction\n");
            exit(1);
        }

        double vv = 0.0;
        for (long i = 0; i < Ntrain; i++)
            vv += v[i] * v[i];

        double var = sf2 + sn2 - vv;
        if (var < 0.0) var = 0.0;

        pred_mean_norm[t] = mean;
        pred_std_norm[t] = sqrt(var);

        free(kstar);
        free(v);
    }

    free(K);
    free(alpha);
}

// ============================================================
// MAIN
// ============================================================

int main(void)
{
    srand(42);

    mkdir(OUTDIR, 0777);

    char outfield[512];
    snprintf(outfield, sizeof(outfield), "%s/%s", OUTDIR, FIELD);
    mkdir(outfield, 0777);

    char path[512];

    // --------------------------------------------------------
    // LOAD PCA INFO
    // --------------------------------------------------------
    snprintf(path, sizeof(path), "%s/%s_info.txt", PCA_DIR, FIELD);
    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        printf("Cannot open %s\n", path);
        return 1;
    }

    long Ntrain = 0, Ntest = 0;
    int nmodes = 0;

    char key[256], val[256];
    while (fscanf(fp, "%s %s", key, val) == 2)
    {
        if (!strcmp(key, "Ntrain")) Ntrain = atol(val);
        else if (!strcmp(key, "Ntest")) Ntest = atol(val);
        else if (!strcmp(key, "modes")) nmodes = atoi(val);
    }
    fclose(fp);

    printf("\n================================================\n");
    printf("GP TRAINING\n");
    printf("================================================\n");
    printf("Field  : %s\n", FIELD);
    printf("Ntrain : %ld\n", Ntrain);
    printf("Ntest  : %ld\n", Ntest);
    printf("Modes  : %d\n", nmodes);

    // --------------------------------------------------------
    // LOAD PARAMS
    // --------------------------------------------------------
    snprintf(path, sizeof(path), "%s/params_4999.bin", PCA_DIR);
    FILE *fp_params = fopen(path, "rb");
    if (!fp_params)
    {
        printf("Cannot open %s\n", path);
        return 1;
    }
    fseek(fp_params, 0, SEEK_END);
    long param_bytes = ftell(fp_params);
    fclose(fp_params);

    long Ntotal = param_bytes / ((long)NPARAMS * (long)sizeof(double));
    printf("Total samples : %ld\n", Ntotal);

    double *params = alloc_double(Ntotal * NPARAMS);
    load_binary(path, params, Ntotal * NPARAMS, sizeof(double));

    // --------------------------------------------------------
    // LOAD INDICES
    // --------------------------------------------------------
    int64_t *train_idx = alloc_i64(Ntrain);
    int64_t *test_idx  = alloc_i64(Ntest);

    snprintf(path, sizeof(path), "%s/train_idx.bin", DATA_DIR);
    load_binary(path, train_idx, Ntrain, sizeof(int64_t));

    snprintf(path, sizeof(path), "%s/test_idx.bin", DATA_DIR);
    load_binary(path, test_idx, Ntest, sizeof(int64_t));

    // --------------------------------------------------------
    // BUILD Xtrain/Xtest
    // --------------------------------------------------------
    double *Xtrain = alloc_double(Ntrain * NPARAMS);
    double *Xtest  = alloc_double(Ntest * NPARAMS);

    for (long i = 0; i < Ntrain; i++)
    {
        memcpy(&Xtrain[IDX(i, 0, NPARAMS)],
               &params[IDX(train_idx[i], 0, NPARAMS)],
               NPARAMS * sizeof(double));
    }

    for (long i = 0; i < Ntest; i++)
    {
        memcpy(&Xtest[IDX(i, 0, NPARAMS)],
               &params[IDX(test_idx[i], 0, NPARAMS)],
               NPARAMS * sizeof(double));
    }

    // --------------------------------------------------------
    // NORMALIZE X
    // --------------------------------------------------------
    Scaler xscaler;
    compute_scaler(Xtrain, Ntrain, &xscaler);

    double *Xtrain_norm = alloc_double(Ntrain * NPARAMS);
    double *Xtest_norm  = alloc_double(Ntest * NPARAMS);

    apply_scaler(Xtrain, Xtrain_norm, Ntrain, &xscaler);
    apply_scaler(Xtest,  Xtest_norm,  Ntest,  &xscaler);

    snprintf(path, sizeof(path), "%s/%s/X_mean.bin", OUTDIR, FIELD);
    save_binary(path, xscaler.mean, NPARAMS, sizeof(double));

    snprintf(path, sizeof(path), "%s/%s/X_std.bin", OUTDIR, FIELD);
    save_binary(path, xscaler.std, NPARAMS, sizeof(double));

    // --------------------------------------------------------
    // LOAD PCA COEFFS
    // --------------------------------------------------------
    double *Ytrain = alloc_double(Ntrain * nmodes);
    double *Ytest  = alloc_double(Ntest * nmodes);

    snprintf(path, sizeof(path), "%s/%s_coeff_train.bin", PCA_DIR, FIELD);
    load_binary(path, Ytrain, Ntrain * nmodes, sizeof(double));

    snprintf(path, sizeof(path), "%s/%s_coeff_test.bin", PCA_DIR, FIELD);
    load_binary(path, Ytest, Ntest * nmodes, sizeof(double));

    // --------------------------------------------------------
    // STORAGE
    // --------------------------------------------------------
    double *pred_mean = alloc_double(Ntest * nmodes);
    double *pred_std  = alloc_double(Ntest * nmodes);

    // --------------------------------------------------------
    // TRAIN + PREDICT EACH MODE
    // --------------------------------------------------------
    for (int mode = 0; mode < nmodes; mode++)
    {
        printf("\n------------------------------------------------\n");
        printf("MODE %d\n", mode);
        printf("------------------------------------------------\n");

        // y for this mode
        double *y = alloc_double(Ntrain);
        double *y_test = alloc_double(Ntest);

        for (long i = 0; i < Ntrain; i++)
            y[i] = Ytrain[IDX(i, mode, nmodes)];
        for (long i = 0; i < Ntest; i++)
            y_test[i] = Ytest[IDX(i, mode, nmodes)];

        // normalize y with ddof=0
        double ymean = 0.0;
        for (long i = 0; i < Ntrain; i++)
            ymean += y[i];
        ymean /= (double)Ntrain;

        double yvar = 0.0;
        for (long i = 0; i < Ntrain; i++)
        {
            double d = y[i] - ymean;
            yvar += d * d;
        }
        yvar /= (double)Ntrain;

        double ystd = sqrt(yvar);
        if (ystd < EPS) ystd = 1.0;

        double *y_train_norm = alloc_double(Ntrain);
        for (long i = 0; i < Ntrain; i++)
            y_train_norm[i] = (y[i] - ymean) / ystd;

        snprintf(path, sizeof(path), "%s/%s/ymean_mode_%d.bin", OUTDIR, FIELD, mode);
        save_binary(path, &ymean, 1, sizeof(double));

        snprintf(path, sizeof(path), "%s/%s/ystd_mode_%d.bin", OUTDIR, FIELD, mode);
        save_binary(path, &ystd, 1, sizeof(double));

        // ----------------------------------------------------
        // Hyperparameter training
        // ----------------------------------------------------
        Theta best_theta;
        double best_lml = optimize_theta(Xtrain_norm, y_train_norm, Ntrain, &best_theta);

        double sigma_f, ell[NPARAMS], sigma_n;
        theta_to_values(&best_theta, &sigma_f, ell, &sigma_n);

        printf("Optimized hyperparameters:\n");
        printf("  sigma_f = %.8e\n", sigma_f);
        printf("  sigma_n = %.8e\n", sigma_n);
        for (int d = 0; d < NPARAMS; d++)
            printf("  ell[%d]   = %.8e\n", d, ell[d]);
        printf("  logML    = %.8e\n", best_lml);

        // save hyperparameters
        double hyper[1 + NPARAMS + 1];
        hyper[0] = sigma_f;
        for (int d = 0; d < NPARAMS; d++)
            hyper[1 + d] = ell[d];
        hyper[1 + NPARAMS] = sigma_n;

        snprintf(path, sizeof(path), "%s/%s/gp_mode_%d_params.bin", OUTDIR, FIELD, mode);
        save_binary(path, hyper, 1 + NPARAMS + 1, sizeof(double));

        // ----------------------------------------------------
        // Predict on test set
        // ----------------------------------------------------
        double *pred_mean_norm = alloc_double(Ntest);
        double *pred_std_norm  = alloc_double(Ntest);

        gp_predict(Xtrain_norm, Xtest_norm, y_train_norm, Ntrain, Ntest,
                   &best_theta, pred_mean_norm, pred_std_norm);

        for (long i = 0; i < Ntest; i++)
        {
            pred_mean[IDX(i, mode, nmodes)] = pred_mean_norm[i] * ystd + ymean;
            pred_std[IDX(i, mode, nmodes)]  = pred_std_norm[i] * ystd;
        }

        // ----------------------------------------------------
        // Metrics
        // ----------------------------------------------------
        double mae = 0.0, mse = 0.0, maxerr = 0.0;
        for (long i = 0; i < Ntest; i++)
        {
            double diff = pred_mean[IDX(i, mode, nmodes)] - Ytest[IDX(i, mode, nmodes)];
            mae += fabs(diff);
            mse += diff * diff;
            if (fabs(diff) > maxerr) maxerr = fabs(diff);
        }
        mae /= (double)Ntest;
        mse /= (double)Ntest;

        printf("\nErrors:\n");
        printf("MAE     = %.6e\n", mae);
        printf("RMSE    = %.6e\n", sqrt(mse));
        printf("MAX ERR = %.6e\n", maxerr);

        free(y);
        free(y_test);
        free(y_train_norm);
        free(pred_mean_norm);
        free(pred_std_norm);
    }

    // --------------------------------------------------------
    // SAVE OUTPUTS
    // --------------------------------------------------------
    snprintf(path, sizeof(path), "%s/%s/pred_coeff_mean.bin", OUTDIR, FIELD);
    save_binary(path, pred_mean, Ntest * nmodes, sizeof(double));

    snprintf(path, sizeof(path), "%s/%s/pred_coeff_std.bin", OUTDIR, FIELD);
    save_binary(path, pred_std, Ntest * nmodes, sizeof(double));

    snprintf(path, sizeof(path), "%s/%s/true_coeff_test.bin", OUTDIR, FIELD);
    save_binary(path, Ytest, Ntest * nmodes, sizeof(double));

    // info
    snprintf(path, sizeof(path), "%s/%s/gp_info.txt", OUTDIR, FIELD);
    fp = fopen(path, "w");
    if (!fp)
    {
        printf("Cannot write %s\n", path);
        return 1;
    }
    fprintf(fp, "FIELD %s\n", FIELD);
    fprintf(fp, "Ntrain %ld\n", Ntrain);
    fprintf(fp, "Ntest %ld\n", Ntest);
    fprintf(fp, "nmodes %d\n", nmodes);
    fclose(fp);

    // --------------------------------------------------------
    // CLEANUP
    // --------------------------------------------------------
    free(params);
    free(train_idx);
    free(test_idx);
    free(Xtrain);
    free(Xtest);
    free(Xtrain_norm);
    free(Xtest_norm);
    free(Ytrain);
    free(Ytest);
    free(pred_mean);
    free(pred_std);

    printf("\n================================================\n");
    printf("DONE\n");
    printf("================================================\n");

    return 0;
}