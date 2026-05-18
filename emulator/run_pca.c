/*
 * gcc -O3 -march=native pca_full.c -llapacke -llapack -lblas -lm -o pca_full
 * ./pca_full
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdint.h>
#include <errno.h>
#include <lapacke.h>

#define NPARAMS 11
#define EPS 1e-12
#define VAR_CUT 0.995
#define IDX(i, j, ncol) ((i) * (ncol) + (j))

static const char *DATA_DIR = "fisher_data_all4";
static const char *OUTDIR   = "Cpca_outputs";

static double *alloc_matrix(long nrow, long ncol)
{
    double *ptr = (double *)calloc((size_t)(nrow * ncol), sizeof(double));
    if (!ptr)
    {
        printf("Allocation failed\n");
        exit(1);
    }
    return ptr;
}

static void save_binary(const char *fname, const double *X, long size)
{
    FILE *fp = fopen(fname, "wb");
    if (!fp)
    {
        printf("Cannot write %s\n", fname);
        exit(1);
    }

    size_t nwrote = fwrite(X, sizeof(double), (size_t)size, fp);
    fclose(fp);

    if (nwrote != (size_t)size)
    {
        printf("Write error: %s (expected %ld, wrote %zu)\n", fname, size, nwrote);
        exit(1);
    }
}

static void save_int64_binary(const char *fname, const int64_t *X, long size)
{
    FILE *fp = fopen(fname, "wb");
    if (!fp)
    {
        printf("Cannot write %s\n", fname);
        exit(1);
    }

    size_t nwrote = fwrite(X, sizeof(int64_t), (size_t)size, fp);
    fclose(fp);

    if (nwrote != (size_t)size)
    {
        printf("Write error: %s (expected %ld, wrote %zu)\n", fname, size, nwrote);
        exit(1);
    }
}

static void load_binary(const char *fname, double *X, long size)
{
    FILE *fp = fopen(fname, "rb");
    if (!fp)
    {
        printf("Cannot open %s\n", fname);
        exit(1);
    }

    size_t nread = fread(X, sizeof(double), (size_t)size, fp);
    fclose(fp);

    if (nread != (size_t)size)
    {
        printf("Read error: %s (expected %ld, got %zu)\n", fname, size, nread);
        exit(1);
    }
}

static void load_int64_binary(const char *fname, int64_t *X, long size)
{
    FILE *fp = fopen(fname, "rb");
    if (!fp)
    {
        printf("Cannot open %s\n", fname);
        exit(1);
    }

    size_t nread = fread(X, sizeof(int64_t), (size_t)size, fp);
    fclose(fp);

    if (nread != (size_t)size)
    {
        printf("Read error: %s (expected %ld, got %zu)\n", fname, size, nread);
        exit(1);
    }
}

static void print_stats(const char *name, const double *X, long size)
{
    double minv = X[0];
    double maxv = X[0];
    double mean = 0.0;

    for (long i = 0; i < size; i++)
    {
        if (X[i] < minv)
            minv = X[i];
        if (X[i] > maxv)
            maxv = X[i];
        mean += X[i];
    }
    mean /= (double)size;

    double var = 0.0;
    for (long i = 0; i < size; i++)
    {
        double d = X[i] - mean;
        var += d * d;
    }
    var /= (double)size;

    printf("\n%s\n", name);
    printf("min  = %.12e\n", minv);
    printf("max  = %.12e\n", maxv);
    printf("mean = %.12e\n", mean);
    printf("std  = %.12e\n", sqrt(var));
}

static void print_mode_stats(const char *name, const double *A, long N, int modes)
{
    printf("\n%s per mode\n", name);
    for (int m = 0; m < modes; m++)
    {
        double minv = A[IDX(0, m, modes)];
        double maxv = A[IDX(0, m, modes)];
        double mean = 0.0;

        for (long i = 0; i < N; i++)
        {
            double v = A[IDX(i, m, modes)];
            if (v < minv)
                minv = v;
            if (v > maxv)
                maxv = v;
            mean += v;
        }
        mean /= (double)N;

        double var = 0.0;
        for (long i = 0; i < N; i++)
        {
            double d = A[IDX(i, m, modes)] - mean;
            var += d * d;
        }
        var /= (double)N;

        printf("mode %2d: min=% .12e  max=% .12e  mean=% .12e  std=%.12e\n",
               m, minv, maxv, mean, sqrt(var));
    }
}

static void log10_transform(double *X, long size)
{
    for (long i = 0; i < size; i++)
        X[i] = log10(X[i] + EPS);
}

static void rho_transform(const double *TT, const double *EE, const double *TE,
                          double *rho_out, long N, int D)
{
    for (long i = 0; i < N; i++)
    {
        for (int l = 0; l < D; l++)
        {
            double tt = TT[IDX(i, l, D)];
            double ee = EE[IDX(i, l, D)];
            double te = TE[IDX(i, l, D)];

            double denom = sqrt((tt + EPS) * (ee + EPS));
            double rho = te / denom;

            if (rho > 0.9999)
                rho = 0.9999;
            if (rho < -0.9999)
                rho = -0.9999;

            rho_out[IDX(i, l, D)] = atanh(rho);
        }
    }
}

static void subset_rows(const double *X, const int64_t *idx, long nsub, int D, double *Y)
{
    for (long i = 0; i < nsub; i++)
    {
        long src = (long)idx[i];
        memcpy(&Y[IDX(i, 0, D)], &X[IDX(src, 0, D)], (size_t)D * sizeof(double));
    }
}

static void save_pca_outputs(const char *name,
                             const double *Y_train_trans,
                             const double *Y_test_trans,
                             const double *mean,
                             const double *S,
                             const double *V_M,
                             const double *A_train,
                             const double *A_test,
                             const double *X_rec_train,
                             const double *X_rec_test,
                             long Ntrain,
                             long Ntest,
                             int D,
                             int modes,
                             double var_loss,
                             double mse_train,
                             double mse_test,
                             double explained_var)
{
    mkdir(OUTDIR, 0777);

    char fname[512];
    FILE *fp = NULL;

    snprintf(fname, sizeof(fname), "%s/%s_train_transformed.bin", OUTDIR, name);
    save_binary(fname, Y_train_trans, (long)Ntrain * D);

    snprintf(fname, sizeof(fname), "%s/%s_test_transformed.bin", OUTDIR, name);
    save_binary(fname, Y_test_trans, (long)Ntest * D);

    snprintf(fname, sizeof(fname), "%s/%s_mean.bin", OUTDIR, name);
    save_binary(fname, mean, D);

    snprintf(fname, sizeof(fname), "%s/%s_singular.bin", OUTDIR, name);
    save_binary(fname, S, modes);

    snprintf(fname, sizeof(fname), "%s/%s_basis.bin", OUTDIR, name);
    save_binary(fname, V_M, (long)D * modes);

    snprintf(fname, sizeof(fname), "%s/%s_coeff_train.bin", OUTDIR, name);
    save_binary(fname, A_train, (long)Ntrain * modes);

    snprintf(fname, sizeof(fname), "%s/%s_coeff_test.bin", OUTDIR, name);
    save_binary(fname, A_test, (long)Ntest * modes);

    snprintf(fname, sizeof(fname), "%s/%s_recon_train.bin", OUTDIR, name);
    save_binary(fname, X_rec_train, (long)Ntrain * D);

    snprintf(fname, sizeof(fname), "%s/%s_recon_test.bin", OUTDIR, name);
    save_binary(fname, X_rec_test, (long)Ntest * D);

    snprintf(fname, sizeof(fname), "%s/%s_info.txt", OUTDIR, name);
    fp = fopen(fname, "w");
    if (!fp)
    {
        printf("Cannot write %s\n", fname);
        exit(1);
    }

    fprintf(fp, "Ntrain %ld\n", Ntrain);
    fprintf(fp, "Ntest %ld\n", Ntest);
    fprintf(fp, "D %d\n", D);
    fprintf(fp, "modes %d\n", modes);
    fprintf(fp, "variance_cut %.15e\n", (double)VAR_CUT);
    fprintf(fp, "variance_loss %.15e\n", var_loss);
    fprintf(fp, "explained_variance %.15e\n", explained_var);
    fprintf(fp, "mse_train %.15e\n", mse_train);
    fprintf(fp, "mse_test %.15e\n", mse_test);
    fclose(fp);

    printf("Saved PCA outputs for %s in %s/\n", name, OUTDIR);
}

static void run_pca_train_test(const double *Y_train_in,
                               const double *Y_test_in,
                               long Ntrain,
                               long Ntest,
                               int D,
                               const char *name)
{
    printf("\n=================================\n%s\n=================================\n", name);
    printf("Train shape: (%ld, %d)\n", Ntrain, D);
    printf("Test  shape: (%ld, %d)\n", Ntest, D);

    // Mean-center using train only
    double *mean = alloc_matrix(1, D);
    for (int j = 0; j < D; j++)
    {
        double s = 0.0;
        for (long i = 0; i < Ntrain; i++)
            s += Y_train_in[IDX(i, j, D)];
        mean[j] = s / (double)Ntrain;
    }

    double *Y_train = alloc_matrix(Ntrain, D);
    double *Y_test = alloc_matrix(Ntest, D);

    for (long i = 0; i < Ntrain; i++)
    {
        for (int j = 0; j < D; j++)
            Y_train[IDX(i, j, D)] = Y_train_in[IDX(i, j, D)] - mean[j];
    }

    for (long i = 0; i < Ntest; i++)
    {
        for (int j = 0; j < D; j++)
            Y_test[IDX(i, j, D)] = Y_test_in[IDX(i, j, D)] - mean[j];
    }

    // IMPORTANT:
    // dgesdd destroys the input matrix, so keep a copy for later projections.
    double *Y_train_svd = alloc_matrix(Ntrain, D);
    memcpy(Y_train_svd, Y_train, (size_t)Ntrain * D * sizeof(double));

    // Thin SVD on train only
    int M = (Ntrain < D) ? (int)Ntrain : D;
    double *S = alloc_matrix(1, M);
    double *U = alloc_matrix(Ntrain, M);
    double *VT = alloc_matrix(M, D);

    int info = LAPACKE_dgesdd(LAPACK_ROW_MAJOR, 'S',
                              (lapack_int)Ntrain, (lapack_int)D,
                              Y_train_svd, (lapack_int)D,
                              S, U, (lapack_int)M, VT, (lapack_int)D);
    if (info)
    {
        printf("SVD failed, info=%d\n", info);
        exit(1);
    }

    // Select number of modes
    double total_var = 0.0;
    for (int i = 0; i < M; i++)
        total_var += S[i] * S[i];

    double cum = 0.0;
    int modes = M;
    for (int i = 0; i < M; i++)
    {
        cum += S[i] * S[i];
        if (cum / total_var >= VAR_CUT)
        {
            modes = i + 1;
            break;
        }
    }

    // Basis V_M : D x modes
    double *V_M = alloc_matrix(D, modes);
    for (int i = 0; i < D; i++)
    {
        for (int j = 0; j < modes; j++)
            V_M[IDX(i, j, modes)] = VT[IDX(j, i, D)];
    }

    // Project train and test onto basis
    double *A_train = alloc_matrix(Ntrain, modes);
    double *A_test = alloc_matrix(Ntest, modes);

    for (long i = 0; i < Ntrain; i++)
    {
        for (int j = 0; j < modes; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < D; k++)
                sum += Y_train[IDX(i, k, D)] * V_M[IDX(k, j, modes)];
            A_train[IDX(i, j, modes)] = sum;
        }
    }

    for (long i = 0; i < Ntest; i++)
    {
        for (int j = 0; j < modes; j++)
        {
            double sum = 0.0;
            for (int k = 0; k < D; k++)
                sum += Y_test[IDX(i, k, D)] * V_M[IDX(k, j, modes)];
            A_test[IDX(i, j, modes)] = sum;
        }
    }

    // Reconstruction
    double *X_rec_train = alloc_matrix(Ntrain, D);
    double *X_rec_test = alloc_matrix(Ntest, D);

    for (long i = 0; i < Ntrain; i++)
    {
        for (int j = 0; j < D; j++)
        {
            double recon = 0.0;
            for (int k = 0; k < modes; k++)
                recon += A_train[IDX(i, k, modes)] * V_M[IDX(j, k, modes)];
            X_rec_train[IDX(i, j, D)] = recon + mean[j];
        }
    }

    for (long i = 0; i < Ntest; i++)
    {
        for (int j = 0; j < D; j++)
        {
            double recon = 0.0;
            for (int k = 0; k < modes; k++)
                recon += A_test[IDX(i, k, modes)] * V_M[IDX(j, k, modes)];
            X_rec_test[IDX(i, j, D)] = recon + mean[j];
        }
    }

    // MSE on transformed scale
    double mse_train = 0.0;
    double mse_test = 0.0;

    for (long i = 0; i < Ntrain; i++)
    {
        for (int j = 0; j < D; j++)
        {
            double diff = Y_train_in[IDX(i, j, D)] - X_rec_train[IDX(i, j, D)];
            mse_train += diff * diff;
        }
    }

    for (long i = 0; i < Ntest; i++)
    {
        for (int j = 0; j < D; j++)
        {
            double diff = Y_test_in[IDX(i, j, D)] - X_rec_test[IDX(i, j, D)];
            mse_test += diff * diff;
        }
    }

    mse_train /= (double)(Ntrain * D);
    mse_test /= (double)(Ntest * D);

    // Variance loss
    double var_loss = 0.0;
    for (int i = modes; i < M; i++)
        var_loss += S[i] * S[i];
    var_loss /= total_var;

    double explained_var = 1.0 - var_loss;

    // Print summary
    printf("\nTop 10 singular values:\n");
    for (int i = 0; i < 10 && i < M; i++)
        printf("%2d  %.12e\n", i, S[i]);

    printf("\nResults:\n");
    printf("Modes: %d\n", modes);
    printf("Variance loss: %.12e\n", var_loss);
    printf("Explained variance: %.12e\n", explained_var);
    printf("Train reconstruction MSE: %.12e\n", mse_train);
    printf("Test  reconstruction MSE: %.12e\n", mse_test);

    print_mode_stats("Train coeff stats", A_train, Ntrain, modes);
    print_mode_stats("Test coeff stats", A_test, Ntest, modes);

    save_pca_outputs(name,
                     Y_train_in,
                     Y_test_in,
                     mean,
                     S,
                     V_M,
                     A_train,
                     A_test,
                     X_rec_train,
                     X_rec_test,
                     Ntrain,
                     Ntest,
                     D,
                     modes,
                     var_loss,
                     mse_train,
                     mse_test,
                     explained_var);

    free(mean);
    free(Y_train);
    free(Y_test);
    free(Y_train_svd);
    free(S);
    free(U);
    free(VT);
    free(V_M);
    free(A_train);
    free(A_test);
    free(X_rec_train);
    free(X_rec_test);
}

int main(void)
{
    const int D = 4999;
    char fname[512];

    mkdir(OUTDIR, 0777);

    // -------------------------------------------------
    // Load train/test indices from DATA_DIR
    // -------------------------------------------------
    snprintf(fname, sizeof(fname), "%s/train_idx.bin", DATA_DIR);
    FILE *fp = fopen(fname, "rb");
    if (!fp)
    {
        printf("Cannot open %s\n", fname);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    long train_bytes = ftell(fp);
    fclose(fp);

    long Ntrain = train_bytes / (long)sizeof(int64_t);

    snprintf(fname, sizeof(fname), "%s/test_idx.bin", DATA_DIR);
    fp = fopen(fname, "rb");
    if (!fp)
    {
        printf("Cannot open %s\n", fname);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    long test_bytes = ftell(fp);
    fclose(fp);

    long Ntest = test_bytes / (long)sizeof(int64_t);

    int64_t *train_idx = (int64_t *)malloc((size_t)Ntrain * sizeof(int64_t));
    int64_t *test_idx = (int64_t *)malloc((size_t)Ntest * sizeof(int64_t));

    if (!train_idx || !test_idx)
    {
        printf("Index allocation failed\n");
        return 1;
    }

    snprintf(fname, sizeof(fname), "%s/train_idx.bin", DATA_DIR);
    load_int64_binary(fname, train_idx, Ntrain);

    snprintf(fname, sizeof(fname), "%s/test_idx.bin", DATA_DIR);
    load_int64_binary(fname, test_idx, Ntest);

    printf("Loaded train/test indices.\n");
    printf("Train count = %ld\n", Ntrain);
    printf("Test  count = %ld\n", Ntest);

    // -------------------------------------------------
    // Load params from raw dataset
    // -------------------------------------------------
    snprintf(fname, sizeof(fname), "%s/params_4999.bin", DATA_DIR);
    fp = fopen(fname, "rb");
    if (!fp)
    {
        printf("Cannot open %s\n", fname);
        return 1;
    }
    fseek(fp, 0, SEEK_END);
    long file_size = ftell(fp);
    fclose(fp);

    long N = file_size / ((long)NPARAMS * (long)sizeof(double));
    printf("\nparams shape = (%ld, %d)\n", N, NPARAMS);

    double *params = alloc_matrix(N, NPARAMS);
    load_binary(fname, params, N * NPARAMS);

    // Save exact copy for comparison with Python outputs
    snprintf(fname, sizeof(fname), "%s/params_4999.bin", OUTDIR);
    save_binary(fname, params, N * NPARAMS);

    int64_t *ells = (int64_t *)malloc((size_t)D * sizeof(int64_t));
    if (!ells)
    {
        printf("ells allocation failed\n");
        return 1;
    }
    for (int i = 0; i < D; i++)
        ells[i] = (int64_t)(i + 2);

    snprintf(fname, sizeof(fname), "%s/ells_4999.bin", OUTDIR);
    save_int64_binary(fname, ells, D);

    // -------------------------------------------------
    // Load spectra from raw dataset
    // -------------------------------------------------
    double *TT_raw = alloc_matrix(N, D);
    double *EE_raw = alloc_matrix(N, D);
    double *BB_raw = alloc_matrix(N, D);
    double *TE_raw = alloc_matrix(N, D);

    snprintf(fname, sizeof(fname), "%s/TT_4999.bin", DATA_DIR);
    load_binary(fname, TT_raw, N * D);

    snprintf(fname, sizeof(fname), "%s/EE_4999.bin", DATA_DIR);
    load_binary(fname, EE_raw, N * D);

    snprintf(fname, sizeof(fname), "%s/BB_4999.bin", DATA_DIR);
    load_binary(fname, BB_raw, N * D);

    snprintf(fname, sizeof(fname), "%s/TE_4999.bin", DATA_DIR);
    load_binary(fname, TE_raw, N * D);

    printf("Loaded spectra shapes:\n");
    printf("TT: (%ld, %d)\n", N, D);
    printf("EE: (%ld, %d)\n", N, D);
    printf("BB: (%ld, %d)\n", N, D);
    printf("TE: (%ld, %d)\n", N, D);

    print_stats("Raw TT stats", TT_raw, N * D);
    print_stats("Raw EE stats", EE_raw, N * D);
    print_stats("Raw BB stats", BB_raw, N * D);
    print_stats("Raw TE stats", TE_raw, N * D);

    // -------------------------------------------------
    // TT
    // -------------------------------------------------
    {
        double *TT_train_raw = alloc_matrix(Ntrain, D);
        double *TT_test_raw = alloc_matrix(Ntest, D);

        subset_rows(TT_raw, train_idx, Ntrain, D, TT_train_raw);
        subset_rows(TT_raw, test_idx, Ntest, D, TT_test_raw);

        log10_transform(TT_train_raw, Ntrain * D);
        log10_transform(TT_test_raw, Ntest * D);

        print_stats("TT transformed train stats", TT_train_raw, Ntrain * D);
        print_stats("TT transformed test stats", TT_test_raw, Ntest * D);

        run_pca_train_test(TT_train_raw, TT_test_raw, Ntrain, Ntest, D, "TT");

        free(TT_train_raw);
        free(TT_test_raw);
    }

    // -------------------------------------------------
    // EE
    // -------------------------------------------------
    {
        double *EE_train_raw = alloc_matrix(Ntrain, D);
        double *EE_test_raw = alloc_matrix(Ntest, D);

        subset_rows(EE_raw, train_idx, Ntrain, D, EE_train_raw);
        subset_rows(EE_raw, test_idx, Ntest, D, EE_test_raw);

        log10_transform(EE_train_raw, Ntrain * D);
        log10_transform(EE_test_raw, Ntest * D);

        print_stats("EE transformed train stats", EE_train_raw, Ntrain * D);
        print_stats("EE transformed test stats", EE_test_raw, Ntest * D);

        run_pca_train_test(EE_train_raw, EE_test_raw, Ntrain, Ntest, D, "EE");

        free(EE_train_raw);
        free(EE_test_raw);
    }

    // -------------------------------------------------
    // BB
    // -------------------------------------------------
    {
        double *BB_train_raw = alloc_matrix(Ntrain, D);
        double *BB_test_raw = alloc_matrix(Ntest, D);

        subset_rows(BB_raw, train_idx, Ntrain, D, BB_train_raw);
        subset_rows(BB_raw, test_idx, Ntest, D, BB_test_raw);

        log10_transform(BB_train_raw, Ntrain * D);
        log10_transform(BB_test_raw, Ntest * D);

        print_stats("BB transformed train stats", BB_train_raw, Ntrain * D);
        print_stats("BB transformed test stats", BB_test_raw, Ntest * D);

        run_pca_train_test(BB_train_raw, BB_test_raw, Ntrain, Ntest, D, "BB");

        free(BB_train_raw);
        free(BB_test_raw);
    }

    // -------------------------------------------------
    // RHO
    // -------------------------------------------------
    {
        double *TT_train_raw = alloc_matrix(Ntrain, D);
        double *TT_test_raw = alloc_matrix(Ntest, D);
        double *EE_train_raw = alloc_matrix(Ntrain, D);
        double *EE_test_raw = alloc_matrix(Ntest, D);
        double *TE_train_raw = alloc_matrix(Ntrain, D);
        double *TE_test_raw = alloc_matrix(Ntest, D);

        subset_rows(TT_raw, train_idx, Ntrain, D, TT_train_raw);
        subset_rows(TT_raw, test_idx, Ntest, D, TT_test_raw);

        subset_rows(EE_raw, train_idx, Ntrain, D, EE_train_raw);
        subset_rows(EE_raw, test_idx, Ntest, D, EE_test_raw);

        subset_rows(TE_raw, train_idx, Ntrain, D, TE_train_raw);
        subset_rows(TE_raw, test_idx, Ntest, D, TE_test_raw);

        double *rho_train = alloc_matrix(Ntrain, D);
        double *rho_test = alloc_matrix(Ntest, D);

        rho_transform(TT_train_raw, EE_train_raw, TE_train_raw, rho_train, Ntrain, D);
        rho_transform(TT_test_raw, EE_test_raw, TE_test_raw, rho_test, Ntest, D);

        print_stats("RHO transformed train stats", rho_train, Ntrain * D);
        print_stats("RHO transformed test stats", rho_test, Ntest * D);

        run_pca_train_test(rho_train, rho_test, Ntrain, Ntest, D, "RHO");

        free(TT_train_raw);
        free(TT_test_raw);
        free(EE_train_raw);
        free(EE_test_raw);
        free(TE_train_raw);
        free(TE_test_raw);
        free(rho_train);
        free(rho_test);
    }

    // Cleanup
    free(train_idx);
    free(test_idx);
    free(ells);
    free(params);
    free(TT_raw);
    free(EE_raw);
    free(BB_raw);
    free(TE_raw);

    return 0;
}