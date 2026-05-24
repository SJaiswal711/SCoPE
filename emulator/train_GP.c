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

// ========== CONFIGURATION ==========
static const char *PCA_DIR  = "Ppca_outputs";
static const char *DATA_DIR = "fisher_data_all4";
static const char *OUTDIR   = "cGP_outputs";
static const char *FIELD    = "TT";

#define NPARAMS 11
#define NRESTARTS 3
#define MAX_ITER 60

// log-bounds (same as sklearn)
#define LOG_SF_LO   (-6.907755278982137)
#define LOG_SF_HI   ( 6.907755278982137)
#define LOG_ELL_LO  (-6.907755278982137)
#define LOG_ELL_HI  ( 6.907755278982137)
#define LOG_SN_LO   (-23.025850929940457)
#define LOG_SN_HI   ( 2.302585092994046)

// ========== GLOBAL WORKSPACE (reused for training) ==========
static double *ws_K = NULL;
static double *ws_alpha = NULL;
static double *ws_kstar = NULL;
static double *ws_v = NULL;
static long ws_N = 0;

static void ensure_workspace(long N) {
    if (ws_N >= N) return;
    free(ws_K); free(ws_alpha); free(ws_kstar); free(ws_v);
    ws_K = calloc(N * N, sizeof(double));
    ws_alpha = calloc(N, sizeof(double));
    ws_kstar = calloc(N, sizeof(double));
    ws_v = calloc(N, sizeof(double));
    ws_N = N;
    if (!ws_K || !ws_alpha || !ws_kstar || !ws_v) {
        fprintf(stderr, "Workspace allocation failed\n");
        exit(1);
    }
}

static void free_workspace(void) {
    free(ws_K); free(ws_alpha); free(ws_kstar); free(ws_v);
    ws_K = NULL; ws_alpha = NULL; ws_kstar = NULL; ws_v = NULL;
    ws_N = 0;
}

// ========== HELPER FUNCTIONS ==========
static void* alloc_check(size_t n, size_t size) {
    void *ptr = calloc(n, size);
    if (!ptr) { fprintf(stderr, "Allocation failed\n"); exit(1); }
    return ptr;
}

static void save_binary(const char *fname, const void *data, size_t n, size_t elem_size) {
    FILE *fp = fopen(fname, "wb");
    if (!fp) { perror(fname); exit(1); }
    size_t w = fwrite(data, elem_size, n, fp);
    fclose(fp);
    if (w != n) { fprintf(stderr, "Write error %s\n", fname); exit(1); }
}

static void load_binary(const char *fname, void *data, size_t n, size_t elem_size) {
    FILE *fp = fopen(fname, "rb");
    if (!fp) { perror(fname); exit(1); }
    size_t r = fread(data, elem_size, n, fp);
    fclose(fp);
    if (r != n) { fprintf(stderr, "Read error %s\n", fname); exit(1); }
}

static double urand(double lo, double hi) {
    return lo + (hi - lo) * ((double)rand() / RAND_MAX);
}

// ========== STANDARDIZATION ==========
typedef struct { double mean[NPARAMS]; double std[NPARAMS]; } Scaler;

static void compute_scaler(const double *X, long N, Scaler *S) {
    for (int j = 0; j < NPARAMS; j++) {
        double s = 0.0;
        for (long i = 0; i < N; i++) s += X[i * NPARAMS + j];
        S->mean[j] = s / N;
        double var = 0.0;
        for (long i = 0; i < N; i++) {
            double d = X[i * NPARAMS + j] - S->mean[j];
            var += d * d;
        }
        var /= N;
        S->std[j] = sqrt(var);
        if (S->std[j] < EPS) S->std[j] = 1.0;
    }
}

static void apply_scaler(const double *X, double *Y, long N, const Scaler *S) {
    for (long i = 0; i < N; i++)
        for (int j = 0; j < NPARAMS; j++)
            Y[i * NPARAMS + j] = (X[i * NPARAMS + j] - S->mean[j]) / S->std[j];
}

// ========== KERNEL & COVARIANCE ==========
static double ard_rbf_kernel(const double *x1, const double *x2, double sigma_f, const double *ell) {
    double r2 = 0.0;
    for (int d = 0; d < NPARAMS; d++) {
        double diff = (x1[d] - x2[d]) / ell[d];
        r2 += diff * diff;
    }
    return sigma_f * sigma_f * exp(-0.5 * r2);
}

static void build_covariance(const double *X, long N, const double *ell,
                             double sigma_f, double sigma_n, double *K) {
    double sf2 = sigma_f * sigma_f;
    double sn2 = sigma_n * sigma_n;
    for (long i = 0; i < N; i++) {
        for (long j = i; j < N; j++) {
            double v = ard_rbf_kernel(&X[i * NPARAMS], &X[j * NPARAMS], sigma_f, ell);
            if (i == j) v += sn2;
            K[i * N + j] = v;
            K[j * N + i] = v;
        }
    }
}

// ========== LOG MARGINAL LIKELIHOOD (reuses workspace) ==========
static double log_marginal_likelihood(const double *X, const double *y, long N,
                                      const double *ell, double sigma_f, double sigma_n) {
    ensure_workspace(N);
    build_covariance(X, N, ell, sigma_f, sigma_n, ws_K);
    
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', (lapack_int)N, ws_K, (lapack_int)N);
    if (info != 0) return -1e300;
    
    memcpy(ws_alpha, y, N * sizeof(double));
    info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', (lapack_int)N, 1, ws_K, (lapack_int)N, ws_alpha, 1);
    if (info != 0) return -1e300;
    
    double logdet = 0.0;
    for (long i = 0; i < N; i++) logdet += 2.0 * log(ws_K[i * N + i]);
    
    double quad = 0.0;
    for (long i = 0; i < N; i++) quad += y[i] * ws_alpha[i];
    
    return -0.5 * quad - 0.5 * logdet - 0.5 * N * log(2.0 * M_PI);
}

// ========== HYPERPARAMETER OPTIMIZATION ==========
static void optimize_theta(const double *X, const double *y, long N,
                           double *best_sf, double *best_sn, double *best_ell) {
    double best_lml = -1e300;
    
    for (int restart = 0; restart < NRESTARTS; restart++) {
        double sf, sn, ell[NPARAMS];
        if (restart == 0) {
            sf = 1.0;
            sn = 1e-5;
            for (int d = 0; d < NPARAMS; d++) ell[d] = 1.0;
        } else {
            sf = exp(urand(LOG_SF_LO, LOG_SF_HI));
            sn = exp(urand(LOG_SN_LO, LOG_SN_HI));
            for (int d = 0; d < NPARAMS; d++)
                ell[d] = exp(urand(LOG_ELL_LO, LOG_ELL_HI));
        }
        
        double cur_lml = log_marginal_likelihood(X, y, N, ell, sf, sn);
        double step = 1.0;
        
        for (int iter = 0; iter < MAX_ITER; iter++) {
            int improved = 0;
            
            // Optimize log(sigma_f)
            double log_sf = log(sf);
            double lml_p = log_marginal_likelihood(X, y, N, ell, exp(log_sf + step), sn);
            double lml_m = log_marginal_likelihood(X, y, N, ell, exp(log_sf - step), sn);
            if (lml_p > cur_lml && lml_p >= lml_m) {
                sf = exp(log_sf + step); cur_lml = lml_p; improved = 1;
            } else if (lml_m > cur_lml) {
                sf = exp(log_sf - step); cur_lml = lml_m; improved = 1;
            }
            
            // Optimize each length scale
            for (int d = 0; d < NPARAMS; d++) {
                double log_ell = log(ell[d]);
                double ell_plus = exp(log_ell + step), ell_minus = exp(log_ell - step);
                double ell_tmp[NPARAMS];
                memcpy(ell_tmp, ell, sizeof(ell));
                ell_tmp[d] = ell_plus;
                lml_p = log_marginal_likelihood(X, y, N, ell_tmp, sf, sn);
                ell_tmp[d] = ell_minus;
                lml_m = log_marginal_likelihood(X, y, N, ell_tmp, sf, sn);
                if (lml_p > cur_lml && lml_p >= lml_m) {
                    ell[d] = ell_plus; cur_lml = lml_p; improved = 1;
                } else if (lml_m > cur_lml) {
                    ell[d] = ell_minus; cur_lml = lml_m; improved = 1;
                }
            }
            
            // Optimize log(sigma_n)
            double log_sn = log(sn);
            lml_p = log_marginal_likelihood(X, y, N, ell, sf, exp(log_sn + step));
            lml_m = log_marginal_likelihood(X, y, N, ell, sf, exp(log_sn - step));
            if (lml_p > cur_lml && lml_p >= lml_m) {
                sn = exp(log_sn + step); cur_lml = lml_p; improved = 1;
            } else if (lml_m > cur_lml) {
                sn = exp(log_sn - step); cur_lml = lml_m; improved = 1;
            }
            
            if (!improved) step *= 0.5;
            if (step < 1e-4) break;
        }
        
        if (cur_lml > best_lml) {
            best_lml = cur_lml;
            *best_sf = sf;
            *best_sn = sn;
            memcpy(best_ell, ell, NPARAMS * sizeof(double));
        }
    }
}

// ========== PREDICT ON TEST SET (independent, no global workspace reuse) ==========
static void gp_predict_test(const double *Xtrain, const double *ytrain_norm, long Ntrain,
                            const double *Xtest, long Ntest,
                            const double *ell, double sf, double sn,
                            double *pred_mean_norm, double *pred_std_norm) {
    // Build covariance matrix of training data
    double *K = malloc(Ntrain * Ntrain * sizeof(double));
    build_covariance(Xtrain, Ntrain, ell, sf, sn, K);
    
    // Cholesky decomposition
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', (lapack_int)Ntrain, K, (lapack_int)Ntrain);
    if (info) { fprintf(stderr, "Cholesky failed in prediction\n"); exit(1); }
    
    // Compute alpha = K_inv * ytrain_norm
    double *alpha = malloc(Ntrain * sizeof(double));
    memcpy(alpha, ytrain_norm, Ntrain * sizeof(double));
    info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', (lapack_int)Ntrain, 1, K, (lapack_int)Ntrain, alpha, 1);
    if (info) { fprintf(stderr, "Solve failed in prediction\n"); exit(1); }
    
    double sf2 = sf * sf;
    double sn2 = sn * sn;
    
    for (long t = 0; t < Ntest; t++) {
        const double *xt = &Xtest[t * NPARAMS];
        
        // Compute k_star
        double *kstar = malloc(Ntrain * sizeof(double));
        for (long i = 0; i < Ntrain; i++) {
            kstar[i] = ard_rbf_kernel(xt, &Xtrain[i * NPARAMS], sf, ell);
        }
        
        // Mean
        double mean = 0.0;
        for (long i = 0; i < Ntrain; i++) mean += kstar[i] * alpha[i];
        
        // Variance: solve L * v = kstar
        double *v = malloc(Ntrain * sizeof(double));
        memcpy(v, kstar, Ntrain * sizeof(double));
        for (long i = 0; i < Ntrain; i++) {
            double sum = 0.0;
            for (long j = 0; j < i; j++) sum += K[i * Ntrain + j] * v[j];
            v[i] = (v[i] - sum) / K[i * Ntrain + i];
        }
        double var = sf2 + sn2;
        for (long i = 0; i < Ntrain; i++) var -= v[i] * v[i];
        if (var < 0.0) var = 0.0;
        
        pred_mean_norm[t] = mean;
        pred_std_norm[t] = sqrt(var);
        
        free(kstar);
        free(v);
    }
    free(K);
    free(alpha);
}

// ========== PRECOMPUTE ALPHA AND CHOLESKY FOR FAST MCMC PREDICTION ==========
static void save_precomputed(const double *X, const double *y_norm, long N,
                             const double *ell, double sf, double sn,
                             const char *outdir, const char *field, int mode) {
    ensure_workspace(N);
    build_covariance(X, N, ell, sf, sn, ws_K);
    
    int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', (lapack_int)N, ws_K, (lapack_int)N);
    if (info) { fprintf(stderr, "Cholesky failed for mode %d\n", mode); return; }
    
    char path[512];
    snprintf(path, sizeof(path), "%s/%s/chol_mode_%d.bin", outdir, field, mode);
    save_binary(path, ws_K, N * N, sizeof(double));
    
    double *alpha = malloc(N * sizeof(double));
    memcpy(alpha, y_norm, N * sizeof(double));
    info = LAPACKE_dpotrs(LAPACK_ROW_MAJOR, 'L', (lapack_int)N, 1, ws_K, (lapack_int)N, alpha, 1);
    if (info) { fprintf(stderr, "Solve failed for mode %d\n", mode); free(alpha); return; }
    
    snprintf(path, sizeof(path), "%s/%s/alpha_mode_%d.bin", outdir, field, mode);
    save_binary(path, alpha, N, sizeof(double));
    free(alpha);
}

// ========== MAIN ==========
int main(int argc, char **argv) {
    if (argc > 1) FIELD = argv[1];
    srand(42);
    mkdir(OUTDIR, 0777);
    char outfield[512]; snprintf(outfield, sizeof(outfield), "%s/%s", OUTDIR, FIELD);
    mkdir(outfield, 0777);
    
    // Load PCA info (Ntrain, Ntest, nmodes)
    char path[512];
    snprintf(path, sizeof(path), "%s/%s_info.txt", PCA_DIR, FIELD);
    FILE *fp = fopen(path, "r");
    if (!fp) { perror(path); return 1; }
    long Ntrain = 0, Ntest = 0;
    int nmodes = 0;
    char key[256];
    while (fscanf(fp, "%s", key) == 1) {
        if (strcmp(key, "Ntrain") == 0) fscanf(fp, "%ld", &Ntrain);
        else if (strcmp(key, "Ntest") == 0) fscanf(fp, "%ld", &Ntest);
        else if (strcmp(key, "modes") == 0) fscanf(fp, "%d", &nmodes);
    }
    fclose(fp);
    printf("GP training for %s: Ntrain=%ld, Ntest=%ld, nmodes=%d\n", FIELD, Ntrain, Ntest, nmodes);
    
    // Load parameters (full set)
    snprintf(path, sizeof(path), "%s/params_4999.bin", PCA_DIR);
    long file_size; FILE *f = fopen(path, "rb"); fseek(f, 0, SEEK_END); file_size = ftell(f); fclose(f);
    long Ntotal = file_size / (NPARAMS * sizeof(double));
    double *params = malloc(Ntotal * NPARAMS * sizeof(double));
    load_binary(path, params, Ntotal * NPARAMS, sizeof(double));
    
    // Load indices (train and test)
    int64_t *train_idx = malloc(Ntrain * sizeof(int64_t));
    int64_t *test_idx  = malloc(Ntest * sizeof(int64_t));
    snprintf(path, sizeof(path), "%s/train_idx.bin", DATA_DIR);
    load_binary(path, train_idx, Ntrain, sizeof(int64_t));
    snprintf(path, sizeof(path), "%s/test_idx.bin", DATA_DIR);
    load_binary(path, test_idx, Ntest, sizeof(int64_t));
    
    // Build Xtrain and Xtest
    double *Xtrain = malloc(Ntrain * NPARAMS * sizeof(double));
    double *Xtest  = malloc(Ntest  * NPARAMS * sizeof(double));
    for (long i = 0; i < Ntrain; i++)
        memcpy(&Xtrain[i * NPARAMS], &params[train_idx[i] * NPARAMS], NPARAMS * sizeof(double));
    for (long i = 0; i < Ntest; i++)
        memcpy(&Xtest[i * NPARAMS], &params[test_idx[i] * NPARAMS], NPARAMS * sizeof(double));
    
    // Standardize using training statistics
    Scaler scaler;
    compute_scaler(Xtrain, Ntrain, &scaler);
    double *Xtrain_norm = malloc(Ntrain * NPARAMS * sizeof(double));
    double *Xtest_norm  = malloc(Ntest  * NPARAMS * sizeof(double));
    apply_scaler(Xtrain, Xtrain_norm, Ntrain, &scaler);
    apply_scaler(Xtest,  Xtest_norm,  Ntest,  &scaler);
    
    // Save scaler
    snprintf(path, sizeof(path), "%s/%s/X_mean.bin", OUTDIR, FIELD);
    save_binary(path, scaler.mean, NPARAMS, sizeof(double));
    snprintf(path, sizeof(path), "%s/%s/X_std.bin", OUTDIR, FIELD);
    save_binary(path, scaler.std, NPARAMS, sizeof(double));
    snprintf(path, sizeof(path), "%s/%s/X_train_norm.bin", OUTDIR, FIELD);
    save_binary(path, Xtrain_norm, Ntrain * NPARAMS, sizeof(double));
    
    // Load PCA coefficients (train and test)
    double *Ytrain = malloc(Ntrain * nmodes * sizeof(double));
    double *Ytest  = malloc(Ntest  * nmodes * sizeof(double));
    snprintf(path, sizeof(path), "%s/%s_coeff_train.bin", PCA_DIR, FIELD);
    load_binary(path, Ytrain, Ntrain * nmodes, sizeof(double));
    snprintf(path, sizeof(path), "%s/%s_coeff_test.bin", PCA_DIR, FIELD);
    load_binary(path, Ytest, Ntest * nmodes, sizeof(double));
    
    // Save true test coefficients (for Python comparison)
    snprintf(path, sizeof(path), "%s/%s/true_coeff_test.bin", OUTDIR, FIELD);
    save_binary(path, Ytest, Ntest * nmodes, sizeof(double));
    
    // Storage for predictions (row-major: Ntest × nmodes)
    double *pred_mean = malloc(Ntest * nmodes * sizeof(double));
    double *pred_std  = malloc(Ntest * nmodes * sizeof(double));
    
    // Train GP for each mode
    for (int mode = 0; mode < nmodes; mode++) {
        printf("Mode %d/%d\n", mode+1, nmodes);
        
        // Extract y for training and test
        double *y_train = malloc(Ntrain * sizeof(double));
        double *y_test  = malloc(Ntest  * sizeof(double));
        for (long i = 0; i < Ntrain; i++) y_train[i] = Ytrain[i * nmodes + mode];
        for (long i = 0; i < Ntest;  i++) y_test[i]  = Ytest[i * nmodes + mode];
        
        // Normalize y_train
        double ymean = 0.0;
        for (long i = 0; i < Ntrain; i++) ymean += y_train[i];
        ymean /= Ntrain;
        double yvar = 0.0;
        for (long i = 0; i < Ntrain; i++) { double d = y_train[i] - ymean; yvar += d * d; }
        double ystd = sqrt(yvar / Ntrain);
        if (ystd < EPS) ystd = 1.0;
        double *y_train_norm = malloc(Ntrain * sizeof(double));
        for (long i = 0; i < Ntrain; i++) y_train_norm[i] = (y_train[i] - ymean) / ystd;
        
        // Optimize hyperparameters
        double best_sf, best_sn, best_ell[NPARAMS];
        optimize_theta(Xtrain_norm, y_train_norm, Ntrain, &best_sf, &best_sn, best_ell);
        
        // Save hyperparameters
        double hyper[1 + NPARAMS + 1];
        hyper[0] = best_sf;
        for (int d = 0; d < NPARAMS; d++) hyper[1+d] = best_ell[d];
        hyper[1+NPARAMS] = best_sn;
        snprintf(path, sizeof(path), "%s/%s/gp_mode_%d_params.bin", OUTDIR, FIELD, mode);
        save_binary(path, hyper, 1 + NPARAMS + 1, sizeof(double));
        
        // Save y normalization
        snprintf(path, sizeof(path), "%s/%s/ymean_mode_%d.bin", OUTDIR, FIELD, mode);
        save_binary(path, &ymean, 1, sizeof(double));
        snprintf(path, sizeof(path), "%s/%s/ystd_mode_%d.bin", OUTDIR, FIELD, mode);
        save_binary(path, &ystd, 1, sizeof(double));
        
        // Precompute alpha and Cholesky for fast MCMC prediction (optional)
        save_precomputed(Xtrain_norm, y_train_norm, Ntrain, best_ell, best_sf, best_sn, OUTDIR, FIELD, mode);
        
        // Predict on test set
        double *pred_mean_norm = malloc(Ntest * sizeof(double));
        double *pred_std_norm  = malloc(Ntest * sizeof(double));
        gp_predict_test(Xtrain_norm, y_train_norm, Ntrain,
                        Xtest_norm, Ntest,
                        best_ell, best_sf, best_sn,
                        pred_mean_norm, pred_std_norm);
        
        // Denormalize predictions and store (we'll later transpose to (Ntest, nmodes))
        for (long i = 0; i < Ntest; i++) {
            pred_mean[i * nmodes + mode] = pred_mean_norm[i] * ystd + ymean;
            pred_std[i * nmodes + mode]  = pred_std_norm[i] * ystd;
        }
        
        double final_lml = log_marginal_likelihood(Xtrain_norm, y_train_norm, Ntrain, best_ell, best_sf, best_sn);
        printf("  logML = %.2f, sf=%.3f, sn=%.5f\n", final_lml, best_sf, best_sn);
        
        free(y_train); free(y_test); free(y_train_norm);
        free(pred_mean_norm); free(pred_std_norm);
    }
    
    // Save prediction outputs (already row-major: Ntest × nmodes)
    snprintf(path, sizeof(path), "%s/%s/pred_coeff_mean.bin", OUTDIR, FIELD);
    save_binary(path, pred_mean, Ntest * nmodes, sizeof(double));
    snprintf(path, sizeof(path), "%s/%s/pred_coeff_std.bin", OUTDIR, FIELD);
    save_binary(path, pred_std, Ntest * nmodes, sizeof(double));
    
    // Save info file
    snprintf(path, sizeof(path), "%s/%s/gp_info.txt", OUTDIR, FIELD);
    fp = fopen(path, "w");
    if (!fp) { perror(path); exit(1); }
    fprintf(fp, "FIELD %s\n", FIELD);
    fprintf(fp, "Ntrain %ld\n", Ntrain);
    fprintf(fp, "Ntest %ld\n", Ntest);
    fprintf(fp, "nmodes %d\n", nmodes);
    fclose(fp);
    printf("Saved info to %s\n", path);
    
    // Cleanup
    free(params); free(train_idx); free(test_idx);
    free(Xtrain); free(Xtest); free(Xtrain_norm); free(Xtest_norm);
    free(Ytrain); free(Ytest); free(pred_mean); free(pred_std);
    free_workspace();
    printf("GP training complete for %s\n", FIELD);
    return 0;
}
