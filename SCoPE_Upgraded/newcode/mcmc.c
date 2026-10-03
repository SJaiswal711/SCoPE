#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <string.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>
#include "param_config_reader.h"
#include "newpar_reader.h"
double get_time(void);
#include "nrunplc_reader.h"

#include "emulator/include/emulator.h"

#define EMULATOR_TYPE 0
#define POLY_DEGREE 2
#define FALLBACK_THRESHOLD 0.05

int USE_EMULATOR = 1;
int USE_DRAGGING = 1;
int VERBOSE      = 1;

#define DEBUG_PRINT(...) do { if (VERBOSE) { printf(__VA_ARGS__); fflush(stdout); } } while (0)

static Emulator *g_emulator = NULL;
static int local_step  = 0;
static int g_n_cosmo   = 0;
static int mapping_done = 0;
static char outdir_buf[512];
static const char *LOAD_COV_PATH = NULL;

#define NOPARAM 26
#define MAX_TASKARRAY_SIZE 150
#define NR_END 1
#define FREE_ARG char *

#define CHAINS 2
static short int Astier = 0;
short int PARAMETERS = NOPARAM;

double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

#define TAG_UPDATE_COV 400
#define N_DRAG 3
#define MAX_FAST 60
#define MAX_FAST_PACKED (1 + MAX_FAST + MAX_FAST*MAX_FAST)

static const char *SLOW_PARAM_NAMES[] = {
    "Omega_m_h2", "Omega_b_h2", "h", "tau", "n_s", "A_s"
};
#define N_SLOW_NAMES ((int)(sizeof(SLOW_PARAM_NAMES)/sizeof(SLOW_PARAM_NAMES[0])))

int slow_idx[NOPARAM], fast_idx[NOPARAM];
int N_SLOW = 0, N_FAST = 0;

unsigned short int CURRENTSTATEPOS;

double fast_eval[MAX_FAST];
double fast_evec[MAX_FAST][MAX_FAST];
double fast_EntireFactor = 0.5;
int    fast_cov_ready = 0;
double fast_cov_buffer[MAX_FAST_PACKED];

Config global_config;

unsigned short int TASKARRAY_SIZE;
unsigned short int MULTIPURPOSEPOS;
unsigned short int TAKEPOS, PROBPOS;
unsigned short int ADAPTIVEPOS;
unsigned int LOGLIKEPOS;
int RANDPOS;
unsigned short int DRAGPOS;
unsigned short int FASTFACTORPOS;
int STEP_POS;

static unsigned int MULTIPURPOSE_REQUIRED = 1;

static short int WMAP7 = 1;
static short int BOOMERANG03 = 0;
static short int VSA = 0;
static short int ACBAR = 0;
static short int CBI = 0;
static short int Riess06 = 0;
short int AstierParameterPos = 0;
short int WMAP5ParameterPos = 0;
short int SDSSLRGParameterPos = 0;

static unsigned int BEGINCOVUPDATE = 120;
static unsigned int BEGINDRAGGING  = 150;   /* FIX #3: restored */
static unsigned int MAXCHAINLENGTH = 50000;

static int GIVEMETASK = 1;
static int TAKERESULT = 2;
static int TAKETASK = 3;
#define TAG_STOP 4

#define SLAVEPARCHAIN 1

static double INCREASE_STEP = 1.15;
static double DECREASE_STEP = 0.9;
static int HIGH_STEP_BOUND = 5;
static int LOW_STEP_BOUND = 3;
const int UPDATE_TIME = 5;
const unsigned int MAXPREVIOUSPOINTS = 5000;
const unsigned int RMIN_POINTS = 120;
const double RBREAK = 1.2;
const unsigned int MIN_SIZE_FOR_FREEZE_IN = 500;

extern void test_likelihood_(int *inputflag, double *cl_in_tt, double *cl_in_te, double *cl_in_ee, double *cl_in_bb);
double drfactor;
int clik_process_initialized = 0;

typedef struct Task { double f[MAX_TASKARRAY_SIZE]; int Multiplicity; int ReallyInvestigated; } Task;

Task *free_Task() {
    Task *tsk = (Task *)malloc(sizeof(Task));
    for (int i = 0; i < MAX_TASKARRAY_SIZE; i++) tsk->f[i] = 0.0;
    tsk->Multiplicity = 0;
    tsk->ReallyInvestigated = 0;
    return tsk;
}

void Task_copy(Task *tska, Task *tskb) {
    for (int i = 0; i < MAX_TASKARRAY_SIZE; i++) tska->f[i] = tskb->f[i];
    tska->Multiplicity = tskb->Multiplicity;
    tska->ReallyInvestigated = tskb->ReallyInvestigated;
}

typedef struct RollingAverage { unsigned int Size, Number; double Sum; double *y; unsigned int idx; int PerformFullCounter; } RollingAverage;

void Rolling_Average_push(RollingAverage *RA, double x) {
    RA->Sum -= RA->y[RA->idx];
    RA->Sum += x;
    RA->y[RA->idx++] = x;
    if (RA->idx == RA->Size) RA->idx = 0;
    RA->Number++;
    if (RA->Number >= RA->Size) RA->Number = RA->Size;
}

double RollingAverage_average(RollingAverage *RA) {
    if (RA->Number == 0) return 0.0;
    if (RA->PerformFullCounter++ < 10) return RA->Sum / RA->Number;
    RA->PerformFullCounter = 0;
    double Sum = 0.0;
    for (unsigned int i = 0; i < RA->Number; i++) Sum += RA->y[i];
    return Sum / RA->Number;
}

RollingAverage *new_RollingAverage(int Size) {
    RollingAverage *RA = (RollingAverage*)malloc(sizeof(RollingAverage));
    RA->y = (double*)malloc(Size * sizeof(double));
    RA->Size = Size;
    RA->idx = 0;
    RA->Number = 0;
    RA->Sum = 0.0;
    RA->PerformFullCounter = 0;
    for (int i = 0; i < Size; i++) RA->y[i] = 0.0;
    return RA;
}

typedef struct MultiGaussian {
  double Scale; unsigned int SIZE; double **MasterMatrix;
  double *eigenvalues; double *generatedValues;
  double *lbounds; double *hbounds; double *randomq; double *center;
} MultiGaussian;

MultiGaussian *new_MultiGaussian(int Size) {
  MultiGaussian *MG = (MultiGaussian *)malloc(sizeof(MultiGaussian));
  MG->SIZE = Size;
  MG->MasterMatrix = (double **)malloc(Size * sizeof(double *));
  for (unsigned int i = 0; i < Size; i++) MG->MasterMatrix[i] = (double *)malloc(Size * sizeof(double));
  MG->eigenvalues = (double *)malloc(Size * sizeof(double));
  MG->generatedValues = (double *)malloc(Size * sizeof(double));
  MG->lbounds = (double *)malloc(Size * sizeof(double));
  MG->hbounds = (double *)malloc(Size * sizeof(double));
  MG->center = (double *)malloc(Size * sizeof(double));
  MG->randomq = (double *)malloc(Size * sizeof(double));
  return MG;
}

void MultiGaussian_setBounds(MultiGaussian *MG, double lowBound[], double highBound[]) {
  for (unsigned int i = 0; i < MG->SIZE; i++) { MG->lbounds[i] = lowBound[i]; MG->hbounds[i] = highBound[i]; }
}

void printInfo(MultiGaussian *MG) {
  DEBUG_PRINT("************************************\n");
  DEBUG_PRINT("MultiGaussian::printInfo:\n");
  DEBUG_PRINT("Eigenvalues (we used scale = %f):\n", MG->Scale);
  for (unsigned int i = 0; i < MG->SIZE; i++) {
    DEBUG_PRINT("%e  --> sigma:  %e  ", MG->eigenvalues[i], sqrt(MG->eigenvalues[i]));
    DEBUG_PRINT("eigenvector w/o scale: %e  ", MG->eigenvalues[i] / MG->Scale);
    DEBUG_PRINT("  --> sigma: %e\n", sqrt(MG->eigenvalues[i] / MG->Scale));
  }
  DEBUG_PRINT("\nEigenvectors:\n");
  for (unsigned int i = 0; i < MG->SIZE; i++) {
    for (unsigned int j = 0; j < MG->SIZE; j++) DEBUG_PRINT("%e  ", MG->MasterMatrix[i][j]);
    DEBUG_PRINT("\n");
  }
  DEBUG_PRINT("***********************************\n");
}

void nrerror(char s[]) { fprintf(stderr, "%s", s); exit(1); }

double **convert_matrix(double *a, long nrl, long nrh, long ncl, long nch) {
  long i, j, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
  double **m = (double **)malloc((unsigned int)((nrow + NR_END) * sizeof(double *)));
  if (!m) nrerror("allocation failure in convert_matrix()");
  m += NR_END; m -= nrl;
  m[nrl] = a - ncl;
  for (i = 1, j = nrl + 1; i < nrow; i++, j++) m[j] = m[j - 1] + ncol;
  return m;
}
void free_convert_matrix(double **b, long nrl, long nrh, long ncl, long nch) { free((FREE_ARG)(b + nrl - NR_END)); }

double *vector(long nl, long nh) {
  double *v = (double *)malloc((unsigned int)((nh - nl + 1 + NR_END) * sizeof(double)));
  if (!v) nrerror("allocation failure in vector()");
  return v - nl + NR_END;
}
void free_vector(double *v, long nl, long nh) { free((FREE_ARG)(v + nl - NR_END)); }

double *new_double(unsigned short int n) { return (double *)malloc(n * sizeof(double)); }
double fabs(double take) { return (take < 0) ? -take : take; }
double posRnd(double max) { return rand() * (1.0 / RAND_MAX) * max; }
double ran1(float x) { return 2 * (rand() * (1.0 / RAND_MAX) - 0.5) * x; }

double gasdev(double mean, double std) {
  double rsq, v1, v2;
  do { v1 = ran1(1.0); v2 = ran1(1.0); rsq = v1*v1 + v2*v2; } while (rsq >= 1.0 || rsq == 0.0);
  return v1 * sqrt(-2.0 * log(rsq) / rsq) * std + mean;
}

void generateRandom(MultiGaussian *MG) {
  double *y = (double *)malloc(MG->SIZE * sizeof(double));
  for (unsigned int i = 0; i < MG->SIZE; i++) y[i] = gasdev(0.0, sqrt(MG->eigenvalues[i]));
  for (unsigned int i = 0; i < MG->SIZE; i++) {
    MG->randomq[i] = y[i];
    MG->generatedValues[i] = 0;
    for (unsigned int j = 0; j < MG->SIZE; j++) MG->generatedValues[i] += MG->MasterMatrix[i][j] * y[j];
  }
  free(y);
}

int throwDice(Task chain, Task *next, MultiGaussian *MG, int scale) {
  for (unsigned int i = 0; i < MG->SIZE; i++) MG->center[i] = chain.f[i];
  generateRandom(MG);
  for (unsigned int i = 0; i < PARAMETERS; i++) {
    next->f[i] = MG->generatedValues[i] * pow(drfactor, scale) + MG->center[i];
    next->f[RANDPOS + i] = MG->randomq[i] * pow(drfactor, scale);
  }
  for (unsigned int i = 0; i < MG->SIZE; i++) {
    double v = MG->generatedValues[i] * pow(drfactor, scale) + MG->center[i];
    if (v < MG->lbounds[i] || v > MG->hbounds[i]) return 0;
  }
  return 1;
}

void generateEigenvectors(MultiGaussian *MG, double **covarianceMatrix, double scale) {
  MG->Scale = scale;
  double *data = (double *)malloc((MG->SIZE) * (MG->SIZE) * sizeof(double));
  int k = 0;
  for (unsigned int j = 0; j < MG->SIZE; j++)
    for (unsigned int i = 0; i < MG->SIZE; i++) { data[k] = covarianceMatrix[i][j]; k++; }
  gsl_matrix_view m = gsl_matrix_view_array(data, MG->SIZE, MG->SIZE);
  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(MG->SIZE);
  gsl_vector *eval = gsl_vector_alloc(MG->SIZE);
  gsl_matrix *evec = gsl_matrix_alloc(MG->SIZE, MG->SIZE);
  gsl_eigen_symmv(&m.matrix, eval, evec, w);
  gsl_eigen_symmv_free(w);
  gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_ASC);
  for (unsigned short int i = 0; i < MG->SIZE; i++) {
    MG->eigenvalues[i] = gsl_vector_get(eval, i);
    for (unsigned short int j = 0; j < MG->SIZE; j++) MG->MasterMatrix[i][j] = gsl_matrix_get(evec, i, j);
  }
  for (unsigned short int i = 0; i < MG->SIZE; i++) MG->eigenvalues[i] *= scale;
  gsl_vector_free(eval); gsl_matrix_free(evec); free(data);
}

unsigned short int ADAPTIVE = 1;
unsigned short int FREEZE_IN = 1;

void printExtInfo(MultiGaussian *MG) {
  DEBUG_PRINT("\n********************************************\n");
  DEBUG_PRINT("EXTENDED information :\nSize : %d\nBounds:\n", MG->SIZE);
  for (unsigned int i = 0; i < MG->SIZE; i++) DEBUG_PRINT("%d) %e  %e  %e\n", i, MG->lbounds[i], MG->hbounds[i], MG->generatedValues[i]);
  DEBUG_PRINT("\n********************************************\n\n\n");
  for (unsigned int i = 0; i < MG->SIZE; i++) {
    DEBUG_PRINT("%e  :  ", MG->eigenvalues[i]);
    for (unsigned int j = 0; j < MG->SIZE; j++) DEBUG_PRINT("  %.2e", MG->MasterMatrix[i][j]);
    DEBUG_PRINT("\n");
  }
  DEBUG_PRINT("\n********************************************\n");
}

double normd(double a[], int size) { double s = 0.0; for (int i = 0; i < size; i++) s += a[i]*a[i]; return s; }

void get_parameter_bounds_from_config(double *lowbound, double *highbound, double *initial_sigma) {
    int param_index = 0;
    for (int i = 0; i < global_config.param_count && param_index < PARAMETERS; i++) {
        if (global_config.params[i].state == PARAM_SAMPLED) {
            lowbound[param_index]      = global_config.params[i].lower_bound;
            highbound[param_index]     = global_config.params[i].upper_bound;
            initial_sigma[param_index] = global_config.params[i].proposal_sigma;
            DEBUG_PRINT("Chain Param %d: %s [%.3f, %.3f]\n",
                        param_index, global_config.params[i].name,
                        lowbound[param_index], highbound[param_index]);
            param_index++;
        }
    }
    for (; param_index < PARAMETERS; param_index++) {
        lowbound[param_index] = 0.0;
        highbound[param_index] = 1.0;
        initial_sigma[param_index] = 0.1;
    }
}

int myslave_rank(int ii)   { return ii + 1; }
int middleman(int rank)    { (void)rank; return 1; }
int mymiddlemann(int rank) { return rank; }

int throwSlowDice(Task chain_pt, Task *next, MultiGaussian *MGs, int scale) {
    for (unsigned int k = 0; k < MGs->SIZE; k++) MGs->center[k] = chain_pt.f[slow_idx[k]];
    generateRandom(MGs);
    for (unsigned int i = 0; i < PARAMETERS; i++) next->f[i] = chain_pt.f[i];
    for (unsigned int k = 0; k < MGs->SIZE; k++) {
        int idx = slow_idx[k];
        next->f[idx] = MGs->generatedValues[k] * pow(drfactor, scale) + MGs->center[k];
        if (next->f[idx] < MGs->lbounds[k] || next->f[idx] > MGs->hbounds[k]) return 0;
    }
    return 1;
}

void broadcast_fast_covariance(double **cov_full, int worldsize) {
    if (N_FAST == 0) return;
    static MultiGaussian *MGfast = NULL;
    if (!MGfast) MGfast = new_MultiGaussian(N_FAST);

    double **covFast = (double**)malloc(N_FAST * sizeof(double*));
    for (int k = 0; k < N_FAST; k++) {
        covFast[k] = (double*)malloc(N_FAST * sizeof(double));
        for (int j = 0; j < N_FAST; j++) covFast[k][j] = cov_full[fast_idx[k]][fast_idx[j]];
    }
    generateEigenvectors(MGfast, covFast, 1.0);

    static double packed[MAX_FAST_PACKED];
    packed[0] = (double)N_FAST;
    for (int k = 0; k < N_FAST; k++) packed[1+k] = MGfast->eigenvalues[k];
    int p = 1 + N_FAST;
    for (int i = 0; i < N_FAST; i++)
        for (int j = 0; j < N_FAST; j++) packed[p++] = MGfast->MasterMatrix[i][j];
    for (int r = 1; r < worldsize; r++)
        MPI_Send(packed, p, MPI_DOUBLE, r, TAG_UPDATE_COV, MPI_COMM_WORLD);
    for (int k = 0; k < N_FAST; k++) free(covFast[k]);
    free(covFast);
}

void update_local_fast_covariance(double *buf) {
    int n = (int)buf[0];
    if (n != N_FAST) return;
    for (int k = 0; k < N_FAST; k++) fast_eval[k] = buf[1+k];
    int p = 1 + N_FAST;
    for (int i = 0; i < N_FAST; i++)
        for (int j = 0; j < N_FAST; j++) fast_evec[i][j] = buf[p++];
    fast_cov_ready = 1;
}

static int load_covariance_from_file(const char *path, int n, double **cov_out, double *entire_factor_out) {
    if (!path || n <= 0 || !cov_out) return 0;
    FILE *fp = fopen(path, "r");
    if (!fp) { fprintf(stderr, "load_covariance: cannot open '%s'\n", path); return 0; }

    double **tmp = (double**)malloc((size_t)n * sizeof(double*));
    if (!tmp) { fclose(fp); return 0; }
    for (int i = 0; i < n; i++) {
        tmp[i] = (double*)malloc((size_t)n * sizeof(double));
        if (!tmp[i]) { for (int k = 0; k < i; k++) free(tmp[k]); free(tmp); fclose(fp); return 0; }
    }

    char line[8192];
    int row = 0, in_block = 0, have_mat = 0;
    double ef = 1.0;
    while (fgets(line, sizeof(line), fp)) {
        if (strncmp(line, "Chain:", 6) == 0) { in_block = 1; row = 0; ef = 1.0; continue; }
        if (strncmp(line, "Entire factor:", 14) == 0) {
            if (in_block && row == n) {
                double v = 1.0;
                if (sscanf(line, "Entire factor: %lf", &v) == 1) ef = v;
                for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) cov_out[i][j] = tmp[i][j];
                if (entire_factor_out) *entire_factor_out = ef;
                have_mat = 1;
            }
            row = 0; continue;
        }
        if (line[0] == '*' || line[0] == '\n' || line[0] == '\r') continue;
        if (in_block && row < n) {
            double vals[512];
            if (n > (int)(sizeof(vals)/sizeof(vals[0]))) break;
            int cnt = 0;
            const char *p = line;
            while (*p && cnt < n) {
                char *end;
                double v = strtod(p, &end);
                if (end == p) break;
                vals[cnt++] = v;
                p = end;
            }
            if (cnt == n) { for (int j = 0; j < n; j++) tmp[row][j] = vals[j]; row++; }
            else { row = 0; in_block = 0; }
        }
    }
    fclose(fp);
    for (int i = 0; i < n; i++) free(tmp[i]);
    free(tmp);
    return have_mat;
}

// =====================================================================
// MASTER
// =====================================================================
int master(double **startnum, unsigned short int Restart) {
    DEBUG_PRINT("\nStarting master with %d chains and max chain length %d", CHAINS, MAXCHAINLENGTH);
    int total_proposals = 0;

    unsigned int chainSize[CHAINS] = {0};
    unsigned int chainTotalPerformed[CHAINS] = {0};
    unsigned int chainBack[CHAINS] = {0};
    unsigned int checkSize = 0;

    Task **chain = (Task**)malloc(CHAINS * sizeof(Task*));
    for (int c = 0; c < CHAINS; c++) chain[c] = (Task*)malloc(MAXCHAINLENGTH * sizeof(Task));

    int steps_since_update[CHAINS] = {0};
    double EntireFactor[CHAINS];
    RollingAverage *roll[CHAINS];
    double *sigma[CHAINS];
    double **covMatrix[CHAINS];
    for (int i = 0; i < CHAINS; i++) {
        sigma[i] = new_double(TASKARRAY_SIZE);
        covMatrix[i] = (double**)malloc(PARAMETERS * sizeof(double*));
        for (int j = 0; j < PARAMETERS; j++) covMatrix[i][j] = (double*)malloc(PARAMETERS * sizeof(double));
    }

    Task *next[CHAINS];
    MultiGaussian *mGauss[CHAINS];
    MultiGaussian *mGaussSlow[CHAINS];
    double lowboundSlow[NOPARAM], highboundSlow[NOPARAM];

    double *lowbound, *highbound, *initial_sigma;
    unsigned int break_at, N;
    double *y[CHAINS], *dist_y, *B, *W, *R, *average;
    double **cov;

    lowbound = new_double(PARAMETERS);
    highbound = new_double(PARAMETERS);
    initial_sigma = new_double(PARAMETERS);
    dist_y = new_double(PARAMETERS);
    B = new_double(PARAMETERS);
    W = new_double(PARAMETERS);
    R = new_double(PARAMETERS);
    average = new_double(PARAMETERS);
    for (int i = 0; i < CHAINS; i++) y[i] = new_double(PARAMETERS);
    cov = (double**)malloc(PARAMETERS * sizeof(double*));
    for (int i = 0; i < PARAMETERS; i++) cov[i] = new_double(PARAMETERS);

    get_parameter_bounds_from_config(lowbound, highbound, initial_sigma);
    srand((unsigned)time(NULL));

    int have_loaded_cov = 0;
    double **loaded_cov = NULL;
    double loaded_ef = 1.0;
    if (LOAD_COV_PATH) {
        loaded_cov = (double**)malloc(PARAMETERS * sizeof(double*));
        for (int i = 0; i < PARAMETERS; i++) loaded_cov[i] = (double*)malloc(PARAMETERS * sizeof(double));
        if (load_covariance_from_file(LOAD_COV_PATH, PARAMETERS, loaded_cov, &loaded_ef)) {
            have_loaded_cov = 1;
            printf("Loaded covariance from %s (EntireFactor = %g)\n", LOAD_COV_PATH, loaded_ef);
        } else {
            fprintf(stderr, "WARNING: could not load covariance from %s\n", LOAD_COV_PATH);
            for (int i = 0; i < PARAMETERS; i++) free(loaded_cov[i]);
            free(loaded_cov); loaded_cov = NULL;
        }
    }

    int intelligentstartnum = 1;
    for (int i = 0; i < CHAINS; i++) {
        roll[i] = new_RollingAverage(500);
        Task t;
        for (int k = 0; k < TASKARRAY_SIZE; k++) t.f[k] = 0;
        mGauss[i] = new_MultiGaussian(PARAMETERS);
        MultiGaussian_setBounds(mGauss[i], lowbound, highbound);

        if (!intelligentstartnum)
            for (int j = 0; j < PARAMETERS; j++) t.f[j] = lowbound[j] + posRnd(highbound[j] - lowbound[j]);
        else
            for (int j = 0; j < PARAMETERS; j++) t.f[j] = startnum[i][j];
        for (int j = 0; j < PARAMETERS; j++) t.f[CURRENTSTATEPOS + j] = t.f[j];

        for (int j = 0; j < PARAMETERS; j++) sigma[i][j] = initial_sigma[j];

        if (have_loaded_cov) {
            for (int k = 0; k < PARAMETERS; k++)
                for (int j = 0; j < PARAMETERS; j++) covMatrix[i][k][j] = loaded_cov[k][j];
            EntireFactor[i] = loaded_ef;
        } else {
            for (int k = 0; k < PARAMETERS; k++)
                for (int j = 0; j < PARAMETERS; j++)
                    covMatrix[i][k][j] = (j == k) ? sigma[i][k]*sigma[i][k] : 0.0;
            EntireFactor[i] = 1.0;
        }

        generateEigenvectors(mGauss[i], covMatrix[i], EntireFactor[i]*EntireFactor[i]);
        steps_since_update[i] = 0;

        t.f[LOGLIKEPOS] = -0.5 * startnum[i][PARAMETERS];
        t.Multiplicity = 0;
        t.ReallyInvestigated = 0;
        chainBack[i]++;

        Task_copy(&chain[i][chainBack[i]-1], &t);
        chainSize[i] = 0;
        chainTotalPerformed[i] = 0;

        MPI_Send(t.f, TASKARRAY_SIZE, MPI_DOUBLE, myslave_rank(i), TAKETASK, MPI_COMM_WORLD);

        for (int k = 0; k < N_SLOW; k++) {
            lowboundSlow[k]  = lowbound[slow_idx[k]];
            highboundSlow[k] = highbound[slow_idx[k]];
        }
        mGaussSlow[i] = new_MultiGaussian(N_SLOW);
        MultiGaussian_setBounds(mGaussSlow[i], lowboundSlow, highboundSlow);

        double **covSlow0 = (double**)malloc(N_SLOW * sizeof(double*));
        for (int k = 0; k < N_SLOW; k++) {
            covSlow0[k] = (double*)calloc(N_SLOW, sizeof(double));
            if (have_loaded_cov)
                for (int j = 0; j < N_SLOW; j++) covSlow0[k][j] = loaded_cov[slow_idx[k]][slow_idx[j]];
            else
                covSlow0[k][k] = sigma[i][slow_idx[k]] * sigma[i][slow_idx[k]];
        }
        generateEigenvectors(mGaussSlow[i], covSlow0, EntireFactor[i]*EntireFactor[i]);
        for (int k = 0; k < N_SLOW; k++) free(covSlow0[k]);
        free(covSlow0);
    }

    if (have_loaded_cov && USE_DRAGGING && N_FAST > 0) {
        int worldsz; MPI_Comm_size(MPI_COMM_WORLD, &worldsz);
        broadcast_fast_covariance(loaded_cov, worldsz);
    }

    FILE *data[CHAINS], *head[CHAINS], *investigated[CHAINS];
    char montecarlo_name[] = "montecarlo_chain", head_name[] = "head", investigated_name[] = "investigated";
    char path[512];
    for (int k = 1; k <= CHAINS; k++) {
        snprintf(path, sizeof(path), "%s/%s_%d.d", outdir_buf, montecarlo_name, k);
        data[k-1] = fopen(path, "a+");
        snprintf(path, sizeof(path), "%s/%s_%d.d", outdir_buf, head_name, k);
        head[k-1] = fopen(path, "a+");
        snprintf(path, sizeof(path), "%s/%s_%d.d", outdir_buf, investigated_name, k);
        investigated[k-1] = fopen(path, "a+");

        if (Restart == 0) {
            fprintf(data[k-1], "%-5s %-16s", "Mult", "Chi2");
            int printed = 0;
            for (int p = 0; p < global_config.param_count && printed < PARAMETERS; p++) {
                if (global_config.params[p].state == PARAM_SAMPLED) {
                    fprintf(data[k-1], " %-18s", global_config.params[p].name);
                    printed++;
                }
            }
            fprintf(data[k-1], "\n");
            fflush(data[k-1]);
        }
    }

    snprintf(path, sizeof(path), "%s/progress.txt", outdir_buf);
    FILE *progress = fopen(path, "a+");
    snprintf(path, sizeof(path), "%s/gelmanRubin.txt", outdir_buf);
    FILE *gelmanRubin = fopen(path, "a+");
    snprintf(path, sizeof(path), "%s/covMatrix.txt", outdir_buf);
    FILE *covarianceMatrices = fopen(path, "a+");

    if (Restart == 1) fprintf(progress, "\n\n::::::::::::::::::: RESTARTED FROM HERE ::::::::::::::::::::::::\n\n");

    short int work = 1;
    MPI_Status status;
    double result[MAX_TASKARRAY_SIZE];
    int mult[CHAINS];
    int take = 0;
    unsigned int min_size = 100*100;
    int loopstop = 0;
    int global_step = 0;

    do {
        global_step++;
        MPI_Recv(result, TASKARRAY_SIZE, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == GIVEMETASK) {
            int i = status.MPI_SOURCE - 1;
            /* FIX #3: use BEGINDRAGGING */
            int use_dragging = (USE_DRAGGING && chainSize[i] >= BEGINDRAGGING) ? 1 : 0;
            total_proposals += 1;

            mult[i] = 0;
            next[i] = free_Task();
            for (;;) {
                if (use_dragging) { if (throwSlowDice(chain[i][chainBack[i]-1], next[i], mGaussSlow[i], 0)) break; }
                else              { if (throwDice(chain[i][chainBack[i]-1], next[i], mGauss[i], 0)) break; }
                mult[i]++;
                if (mult[i] > 100) {
                    EntireFactor[i] *= 0.9;
                    generateEigenvectors(mGauss[i], covMatrix[i], EntireFactor[i]*EntireFactor[i]);
                    if (use_dragging) {
                        double **covSlow = (double**)malloc(N_SLOW * sizeof(double*));
                        for (int k = 0; k < N_SLOW; k++) {
                            covSlow[k] = (double*)malloc(N_SLOW * sizeof(double));
                            for (int j = 0; j < N_SLOW; j++) covSlow[k][j] = covMatrix[i][slow_idx[k]][slow_idx[j]];
                        }
                        generateEigenvectors(mGaussSlow[i], covSlow, EntireFactor[i]*EntireFactor[i]);
                        for (int k = 0; k < N_SLOW; k++) free(covSlow[k]);
                        free(covSlow);
                    }
                    mult[i] = 0;
                }
                if (mult[i] > 100000) MPI_Abort(MPI_COMM_WORLD, 1);
            }
            for (int k = 0; k < PARAMETERS; k++) next[i]->f[CURRENTSTATEPOS + k] = chain[i][chainBack[i]-1].f[k];

            next[i]->f[STEP_POS]        = (double)global_step;
            next[i]->f[PROBPOS]         = posRnd(1.0);
            next[i]->f[ADAPTIVEPOS]     = (double)ADAPTIVE;
            next[i]->f[DRAGPOS]         = (double)use_dragging;
            next[i]->f[FASTFACTORPOS]   = fast_EntireFactor;
            next[i]->f[TAKEPOS]         = (double)ADAPTIVE;
            next[i]->f[LOGLIKEPOS]      = chain[i][chainBack[i]-1].f[LOGLIKEPOS];
            next[i]->f[MULTIPURPOSEPOS] = (double)chain[i][chainBack[i]-1].ReallyInvestigated;

            for (int k = 0; k < PARAMETERS; k++) fprintf(head[i], "%e\t", next[i]->f[k]);
            fprintf(head[i], "\n");

            MPI_Send(next[i]->f, TASKARRAY_SIZE, MPI_DOUBLE, myslave_rank(i), TAKETASK, MPI_COMM_WORLD);
        }

        if (status.MPI_TAG == TAKERESULT) {
            int i = status.MPI_SOURCE - 1;

            for (int k = 0; k <= PARAMETERS; k++) fprintf(investigated[i], "%e  ", result[k]);
            fprintf(investigated[i], " %d\n", i);

            take = (int)result[TAKEPOS];
            chainTotalPerformed[i] += 1;
            chain[i][chainBack[i]-1].Multiplicity += mult[i];

            if (take == 1) {
                for (int k = 0; k < PARAMETERS; k++)
                    fprintf(data[i], "%e  ", chain[i][chainBack[i]-1].f[k]);
                fprintf(data[i], "%e  %e  ", chain[i][chainBack[i]-1].f[LOGLIKEPOS],
                        chain[i][chainBack[i]-1].f[LOGLIKEPOS]);
                fprintf(data[i], "%d \n", chain[i][chainBack[i]-1].Multiplicity);

                if (ADAPTIVE == 1)
                    if (chain[i][chainBack[i]-1].ReallyInvestigated < LOW_STEP_BOUND)
                        if (EntireFactor[i] < 3) {
                            EntireFactor[i] *= INCREASE_STEP;
                            generateEigenvectors(mGauss[i], covMatrix[i], EntireFactor[i]*EntireFactor[i]);
                        }

                if (ADAPTIVE == 1)
                    Rolling_Average_push(roll[i], pow(drfactor, (int)result[PROBPOS] - 1));

                for (int k = 0; k < TASKARRAY_SIZE; k++) chain[i][chainBack[i]].f[k] = result[k];
                chainBack[i]++;
                steps_since_update[i]++;
                chainSize[i]++;
                chain[i][chainBack[i]-1].Multiplicity = 1;
                chain[i][chainBack[i]-1].ReallyInvestigated = 1;
            } else {
                chain[i][chainBack[i]-1].Multiplicity += 1;
                chain[i][chainBack[i]-1].ReallyInvestigated += 1;
            }

            if (ADAPTIVE == 1 && steps_since_update[i] >= UPDATE_TIME && chainSize[i] >= BEGINCOVUPDATE) {
                int TotalSize = 0;
                unsigned int covSize[CHAINS];
                for (int ii = 0; ii < CHAINS; ii++) {
                    if (chainSize[ii] < 2 * BEGINCOVUPDATE) covSize[ii] = chainSize[ii] / 2;
                    else                                     covSize[ii] = chainSize[ii] - BEGINCOVUPDATE;
                    if (covSize[ii] > MAXPREVIOUSPOINTS) covSize[ii] = MAXPREVIOUSPOINTS;
                }
                for (int k = 0; k < PARAMETERS; k++) {
                    average[k] = 0.0;
                    for (int j = 0; j < PARAMETERS; j++) cov[k][j] = 0.0;
                }
                for (int ii = 0; ii < CHAINS; ii++)
                    for (unsigned int n = chainSize[ii] - covSize[ii]; n < chainSize[ii]; n++) {
                        for (int k = 0; k < PARAMETERS; k++)
                            average[k] += chain[ii][n].f[k] * chain[ii][n].Multiplicity;
                        TotalSize += chain[ii][n].Multiplicity;
                    }
                for (int k = 0; k < PARAMETERS; k++) average[k] /= (double)TotalSize;
                for (int ii = 0; ii < CHAINS; ii++)
                    for (unsigned int n = chainSize[ii] - covSize[ii]; n < chainSize[ii]; n++)
                        for (int k = 0; k < PARAMETERS; k++)
                            for (int j = 0; j < PARAMETERS; j++)
                                cov[k][j] += (chain[ii][n].f[k] - average[k]) *
                                             (chain[ii][n].f[j] - average[j]) *
                                             chain[ii][n].Multiplicity;
                for (int k = 0; k < PARAMETERS; k++)
                    for (int j = 0; j < PARAMETERS; j++) cov[k][j] /= (TotalSize - 1.0);

                for (int ii = 0; ii < CHAINS; ii++)
                    for (int k = 0; k < PARAMETERS; k++)
                        for (int j = 0; j < PARAMETERS; j++) covMatrix[ii][k][j] = cov[k][j];

                for (int ii = 0; ii < CHAINS; ii++) {
                    generateEigenvectors(mGauss[ii], cov, EntireFactor[ii]*EntireFactor[ii]);
                    double **covSlow = (double**)malloc(N_SLOW * sizeof(double*));
                    for (int k = 0; k < N_SLOW; k++) {
                        covSlow[k] = (double*)malloc(N_SLOW * sizeof(double));
                        for (int j = 0; j < N_SLOW; j++) covSlow[k][j] = cov[slow_idx[k]][slow_idx[j]];
                    }
                    generateEigenvectors(mGaussSlow[ii], covSlow, EntireFactor[ii]*EntireFactor[ii]);
                    for (int k = 0; k < N_SLOW; k++) free(covSlow[k]);
                    free(covSlow);
                }

                if (USE_DRAGGING) {
                    int worldsz; MPI_Comm_size(MPI_COMM_WORLD, &worldsz);
                    broadcast_fast_covariance(cov, worldsz);
                }

                fprintf(covarianceMatrices, "Chain: %d Step: %d Points used: %d\n", i+1, chainSize[i], covSize[i]);
                for (int j = 0; j < PARAMETERS; j++) {
                    for (int k = 0; k < PARAMETERS; k++) fprintf(covarianceMatrices, "%e  ", cov[j][k]);
                    fprintf(covarianceMatrices, "\n");
                }
                fprintf(covarianceMatrices, "\nEntire factor: %e", EntireFactor[i]);
                fprintf(covarianceMatrices, "\n***********************************************************\n");

                for (int ii = 0; ii < CHAINS; ii++) steps_since_update[ii] = 0;
            }

            if (ADAPTIVE == 1) {
                if (chain[i][chainBack[i]-1].f[LOGLIKEPOS] < -1e9) {
                    EntireFactor[i] = 1.0;
                    generateEigenvectors(mGauss[i], covMatrix[i], EntireFactor[i]*EntireFactor[i]);
                } else if (chain[i][chainBack[i]-1].ReallyInvestigated > HIGH_STEP_BOUND) {
                    if (EntireFactor[i] > 1e-5) {
                        EntireFactor[i] *= DECREASE_STEP;
                        generateEigenvectors(mGauss[i], covMatrix[i], EntireFactor[i]*EntireFactor[i]);
                        double **covSlow = (double**)malloc(N_SLOW * sizeof(double*));
                        for (int k = 0; k < N_SLOW; k++) {
                            covSlow[k] = (double*)malloc(N_SLOW * sizeof(double));
                            for (int j = 0; j < N_SLOW; j++) covSlow[k][j] = covMatrix[i][slow_idx[k]][slow_idx[j]];
                        }
                        generateEigenvectors(mGaussSlow[i], covSlow, EntireFactor[i]*EntireFactor[i]);
                        for (int k = 0; k < N_SLOW; k++) free(covSlow[k]);
                        free(covSlow);
                    }
                }
            }

            min_size = 1000*1000;
            for (unsigned int ii = 0; ii < CHAINS; ii++)
                if (chainSize[ii] < min_size) min_size = chainSize[ii];

            fprintf(progress, "performed/Chainsize: ");
            for (unsigned int ii = 0; ii < CHAINS; ii++)
                fprintf(progress, "%d/%d  ", chainTotalPerformed[ii], chainSize[ii]);
            fprintf(progress, "min_size: %d    Multiplicity [Really]: ", min_size);
            for (unsigned int ii = 0; ii < CHAINS; ii++)
                fprintf(progress, "%d[%d]  ", chain[ii][chainBack[ii]-1].Multiplicity,
                        chain[ii][chainBack[ii]-1].ReallyInvestigated);
            fprintf(progress, "\n");

            if ((min_size > RMIN_POINTS) && (posRnd(1.0) > 0.95)) {
                break_at = min_size / 2;
                N = min_size - break_at;
                for (int k = 0; k < PARAMETERS; k++) {
                    dist_y[k] = 0; B[k] = 0; W[k] = 0;
                    for (int ii = 0; ii < CHAINS; ii++) y[ii][k] = 0;
                }
                double M = (double)CHAINS;
                for (int ii = 0; ii < CHAINS; ii++) {
                    int TotalMultiplicity = 0;
                    for (unsigned int n = break_at; n < min_size; n++) {
                        for (int k = 0; k < PARAMETERS; k++)
                            y[ii][k] += chain[ii][n].f[k] * chain[ii][n].Multiplicity;
                        TotalMultiplicity += chain[ii][n].Multiplicity;
                    }
                    N = TotalMultiplicity;
                    for (int k = 0; k < PARAMETERS; k++) {
                        y[ii][k] /= (double)N;
                        dist_y[k] += y[ii][k] / M;
                    }
                }
                for (int k = 0; k < PARAMETERS; k++)
                    for (int ii = 0; ii < CHAINS; ii++) {
                        double t1 = y[ii][k] - dist_y[k];
                        B[k] += t1 * t1 / (M - 1.0);
                    }
                for (int ii = 0; ii < CHAINS; ii++)
                    for (unsigned int n = break_at; n < min_size; n++)
                        for (int k = 0; k < PARAMETERS; k++) {
                            double t2 = chain[ii][n].f[k] - y[ii][k];
                            W[k] += t2 * t2 * chain[ii][n].Multiplicity;
                        }
                for (int k = 0; k < PARAMETERS; k++) {
                    W[k] /= (M * (N - 1.0));
                    R[k] = (N - 1.0) / N * W[k] + B[k] * (1.0 + 1.0 / N);
                    R[k] /= W[k];
                }

                if (FREEZE_IN == 1 && min_size > MIN_SIZE_FOR_FREEZE_IN) {
                    unsigned short int ConvergenceReached = 1;
                    for (int m = 0; m < PARAMETERS; m++) if (R[m] > RBREAK) ConvergenceReached = 0;
                    if (ConvergenceReached == 1) {
                        ADAPTIVE = 0; FREEZE_IN = 0; drfactor = 1.0;
                        FILE *final;
                        snprintf(path, sizeof(path), "%s/freezeInEigenvectors.txt", outdir_buf);
                        final = fopen(path, "a+");
                        for (int k = 1; k <= CHAINS; k++) {
                            fclose(data[k-1]);
                            snprintf(path, sizeof(path), "%s/%s_freeze_%d.d", outdir_buf, montecarlo_name, k);
                            data[k-1] = fopen(path, "a+");
                            fprintf(data[k-1], "%-5s %-16s", "Mult", "Chi2");
                            int printed = 0;
                            for (int p = 0; p < global_config.param_count && printed < PARAMETERS; p++) {
                                if (global_config.params[p].state == PARAM_SAMPLED) {
                                    fprintf(data[k-1], " %-18s", global_config.params[p].name);
                                    printed++;
                                }
                            }
                            fprintf(data[k-1], "\n");
                            fflush(data[k-1]);
                        }
                        for (int j = 0; j < CHAINS; j++) {
                            Task *keep = chain[j] + chainBack[j] - 1;
                            Task_copy(chain[j], keep);
                            double OptimalFactor = 2.4 / sqrt((double)PARAMETERS);
                            OptimalFactor = RollingAverage_average(roll[j]);
                            EntireFactor[j] = OptimalFactor;
                            fprintf(final, "CHAIN[%d] Eigenvector and value info: ", j);
                            fprintf(final, "************************************\n");
                            fprintf(final, "MultiGaussian::printInfo:\n");
                            fprintf(final, "Eigenvalues (scale = %f):\n", mGauss[j]->Scale);
                            for (unsigned int ii = 0; ii < mGauss[j]->SIZE; ii++) {
                                fprintf(final, "%e  --> sigma: %e  ", mGauss[j]->eigenvalues[ii], sqrt(mGauss[j]->eigenvalues[ii]));
                                fprintf(final, "eigenvector w/o scale: %e  ", mGauss[j]->eigenvalues[ii]/mGauss[j]->Scale);
                                fprintf(final, "  --> sigma: %e\n", sqrt(mGauss[j]->eigenvalues[ii]/mGauss[j]->Scale));
                            }
                            fprintf(final, "\nEigenvectors:\n");
                            for (unsigned int ii = 0; ii < mGauss[j]->SIZE; ii++) {
                                for (unsigned int jj = 0; jj < mGauss[j]->SIZE; jj++)
                                    fprintf(final, "%e  ", mGauss[j]->MasterMatrix[ii][jj]);
                                fprintf(final, "\n");
                            }
                            fprintf(final, "***********************************\n");
                            printInfo(mGauss[j]);
                            fprintf(final, "\n\n");
                        }
                        fclose(final);
                        fprintf(covarianceMatrices, "Stopped adaptive stepsize at %d for all chains (FREEZE_IN=true)", min_size);
                    }
                }
                fprintf(progress, "Statistics: \n");
                for (int k = 0; k < PARAMETERS; k++)
                    fprintf(progress, "%d  dist_y: %e   B: %e   W: %e     R[k]:  %e \n",
                            k, dist_y[k], B[k], W[k], R[k]);
                if (FREEZE_IN) fprintf(progress, "Multiplicity (and EntireFactor): ");
                else           fprintf(progress, "Multiplicity (and frozen EntireFactor): ");
                for (int ii = 0; ii < CHAINS; ii++)
                    fprintf(progress, "%d (%e)  ", chain[ii][chainBack[ii]-1].Multiplicity, EntireFactor[ii]);
                fprintf(progress, "\n");
                if (min_size != checkSize) {
                    fprintf(gelmanRubin, "%d   ", break_at);
                    for (int n = 0; n < PARAMETERS; n++) fprintf(gelmanRubin, "%e   ", R[n]);
                    fprintf(gelmanRubin, "%d \n", min_size);
                    checkSize = min_size;
                }
            }

            if (FREEZE_IN == 0) loopstop++;
            unsigned int max_size = 0;
            for (int ii = 0; ii < CHAINS; ii++) if (chainBack[ii] > max_size) max_size = chainBack[ii];
            if (max_size >= MAXCHAINLENGTH || total_proposals > 2500000) work = 0;

            fflush(data[i]); fflush(head[i]); fflush(investigated[i]);
            fflush(progress); fflush(gelmanRubin); fflush(covarianceMatrices);
        }
    } while (work);

    fclose(progress); fclose(gelmanRubin); fclose(covarianceMatrices);
    for (int k = 1; k <= CHAINS; k++) {
        fclose(data[k-1]); fclose(head[k-1]); fclose(investigated[k-1]);
    }

    if (loaded_cov) {
        for (int i = 0; i < PARAMETERS; i++) free(loaded_cov[i]);
        free(loaded_cov); loaded_cov = NULL;
    }

    DEBUG_PRINT("\nMCMC finished. Sending stop signal to all slaves...\n");
    int worldsize; MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
    for (int r = 1; r < worldsize; r++)
        MPI_Send(NULL, 0, MPI_INT, r, TAG_STOP, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}

int fact(int a) { return (a <= 1) ? 1 : a * fact(a - 1); }
double mind(double x, double y) { return (x < y) ? x : y; }

double eval_full_chi2(int rank, double *theta, double *Cl_TT, double *Cl_TE, double *Cl_EE, double *Cl_BB) {
    int ok = param_iface(rank, theta, Cl_TT, Cl_TE, Cl_EE, Cl_BB);
    if (!ok) return 1e30;
    double dummy = 0.0;
    double chi2 = run_plc(rank, theta, Cl_TT, Cl_TE, Cl_EE, Cl_BB, 0, &dummy);
    return chi2 + compute_prior_penalty(theta);
}

// =====================================================================
// evaluate_proposal
//
// Note on the MH ratio in the hybrid (emu/CAMB) case:
//   `loglike_cur` comes from the previous accept and may have been
//   produced by a different backend (emu vs CAMB) than the current
//   proposal. This introduces a small mode-mismatch bias bounded by
//   the emulator's accuracy (~ FALLBACK_THRESHOLD). Accepted as a
//   trade-off for speed; the stuck-chain rescue at really_inv > 10
//   forces a consistent CAMB recompute when it matters.
// =====================================================================
static void evaluate_proposal(
    int rank,
    double task[MAX_TASKARRAY_SIZE],
    double *Cl_TT_A, double *Cl_TE_A, double *Cl_EE_A, double *Cl_BB_A,
    double *Cl_TT_B, double *Cl_TE_B, double *Cl_EE_B, double *Cl_BB_B,
    double g_lowbound[], double g_highbound[], double g_sigma[],
    int *use_emu_out, double *uncertainty_out)
{
    double theta_A[MAX_TASKARRAY_SIZE] = {0}, theta_B[MAX_TASKARRAY_SIZE] = {0};
    for (int k = 0; k < PARAMETERS; k++) {
        theta_A[k] = task[CURRENTSTATEPOS + k];
        theta_B[k] = task[k];
    }

    double final_chi2 = 1e30;
    double proposal_chi2 = 1e30;
    int use_dragging = (int)task[DRAGPOS];
    double fast_EntireFactor_local = task[FASTFACTORPOS];

    double uncertainty_A = 0.0, uncertainty_B = 0.0;
    double emu_cl_tt_A[2601], emu_cl_te_A[2601], emu_cl_ee_A[2601], emu_cl_bb_A[2601];
    double emu_cl_tt_B[2601], emu_cl_te_B[2601], emu_cl_ee_B[2601], emu_cl_bb_B[2601];

    int emu_use_A = 0, emu_use_B = 0;

    double cosmo_A[N_SLOW], cosmo_B[N_SLOW];
    for (int k = 0; k < N_SLOW; k++) {
        cosmo_A[k] = theta_A[slow_idx[k]];
        cosmo_B[k] = theta_B[slow_idx[k]];
    }

    if (USE_EMULATOR && g_emulator && emulator_is_ready(g_emulator)) {
        int okA = emulator_predict(g_emulator, cosmo_A, emu_cl_tt_A, emu_cl_te_A, emu_cl_ee_A, emu_cl_bb_A, &uncertainty_A);
        int okB = emulator_predict(g_emulator, cosmo_B, emu_cl_tt_B, emu_cl_te_B, emu_cl_ee_B, emu_cl_bb_B, &uncertainty_B);
        emu_use_A = okA && (uncertainty_A <= FALLBACK_THRESHOLD);
        emu_use_B = okB && (uncertainty_B <= FALLBACK_THRESHOLD);
    }
    int use_emu = (emu_use_A && emu_use_B) ? 1 : 0;

    if (!use_dragging) {
        double cur_chi2 = -2.0 * task[LOGLIKEPOS];
        int really_inv = (int)task[MULTIPURPOSEPOS];

        if (emu_use_B) {
            proposal_chi2 = run_plc(rank, theta_B, emu_cl_tt_B, emu_cl_te_B, emu_cl_ee_B, emu_cl_bb_B, 0, NULL)
                            + compute_prior_penalty(theta_B);
            final_chi2 = proposal_chi2;
        } else {
            int ok = param_iface(rank, theta_B, Cl_TT_B, Cl_TE_B, Cl_EE_B, Cl_BB_B);
            if (ok) {
                double dummy = 0.0;
                proposal_chi2 = run_plc(rank, theta_B, Cl_TT_B, Cl_TE_B, Cl_EE_B, Cl_BB_B, 0, &dummy)
                                + compute_prior_penalty(theta_B);
                final_chi2 = proposal_chi2;
            } else { proposal_chi2 = 1e30; final_chi2 = 1e30; }
        }

        if (really_inv > 10) {
            double cur_re_eval = 1e30;
            if (emu_use_A) {
                cur_re_eval = run_plc(rank, theta_A, emu_cl_tt_A, emu_cl_te_A, emu_cl_ee_A, emu_cl_bb_A, 0, NULL)
                              + compute_prior_penalty(theta_A);
            } else {
                int okA = param_iface(rank, theta_A, Cl_TT_A, Cl_TE_A, Cl_EE_A, Cl_BB_A);
                if (okA) {
                    double dummy = 0.0;
                    cur_re_eval = run_plc(rank, theta_A, Cl_TT_A, Cl_TE_A, Cl_EE_A, Cl_BB_A, 0, &dummy)
                                  + compute_prior_penalty(theta_A);
                }
            }
            if (cur_re_eval < 1e29) cur_chi2 = cur_re_eval;
        }

        task[PARAMETERS] = final_chi2;
        task[LOGLIKEPOS] = (final_chi2 < 1e29) ? -0.5 * final_chi2 : -1e30;
        *use_emu_out = emu_use_B;
        *uncertainty_out = uncertainty_A;
        return;
    }

    double chi2_A_state, chi2_B_state;
    if (emu_use_A) {
        chi2_A_state = run_plc(rank, theta_A, emu_cl_tt_A, emu_cl_te_A, emu_cl_ee_A, emu_cl_bb_A, 0, NULL)
                       + compute_prior_penalty(theta_A);
    } else {
        param_iface(rank, theta_A, Cl_TT_A, Cl_TE_A, Cl_EE_A, Cl_BB_A);
        chi2_A_state = run_plc(rank, theta_A, Cl_TT_A, Cl_TE_A, Cl_EE_A, Cl_BB_A, 0, NULL)
                       + compute_prior_penalty(theta_A);
    }
    if (emu_use_B) {
        chi2_B_state = run_plc(rank, theta_B, emu_cl_tt_B, emu_cl_te_B, emu_cl_ee_B, emu_cl_bb_B, 0, NULL)
                       + compute_prior_penalty(theta_B);
    } else {
        param_iface(rank, theta_B, Cl_TT_B, Cl_TE_B, Cl_EE_B, Cl_BB_B);
        chi2_B_state = run_plc(rank, theta_B, Cl_TT_B, Cl_TE_B, Cl_EE_B, Cl_BB_B, 0, NULL)
                       + compute_prior_penalty(theta_B);
    }
    proposal_chi2 = chi2_B_state;

    double *cl_A_tt = emu_use_A ? emu_cl_tt_A : Cl_TT_A;
    double *cl_A_te = emu_use_A ? emu_cl_te_A : Cl_TE_A;
    double *cl_A_ee = emu_use_A ? emu_cl_ee_A : Cl_EE_A;
    double *cl_A_bb = emu_use_A ? emu_cl_bb_A : Cl_BB_A;
    double *cl_B_tt = emu_use_B ? emu_cl_tt_B : Cl_TT_B;
    double *cl_B_te = emu_use_B ? emu_cl_te_B : Cl_TE_B;
    double *cl_B_ee = emu_use_B ? emu_cl_ee_B : Cl_EE_B;
    double *cl_B_bb = emu_use_B ? emu_cl_bb_B : Cl_BB_B;

    if (chi2_A_state < 1e29 && chi2_B_state < 1e29) {
        double chi2_A_original = chi2_A_state;
        double work_sum = chi2_B_state - chi2_A_state;
        double theta_f_path[MAX_TASKARRAY_SIZE] = {0};
        for (int k = 0; k < PARAMETERS; k++) theta_f_path[k] = theta_A[k];

        for (int d = 1; d <= N_DRAG; d++) {
            double w = (double)d / (double)N_DRAG;
            double trial[MAX_TASKARRAY_SIZE] = {0};
            for (int k = 0; k < N_SLOW; k++) {
                int idx = slow_idx[k];
                trial[idx] = (1.0 - w) * theta_A[idx] + w * theta_B[idx];
            }
            for (int k = 0; k < N_FAST; k++) {
                int idx = fast_idx[k];
                trial[idx] = theta_f_path[idx];
            }

            int fast_oob = 0;
            if (fast_cov_ready && N_FAST > 0) {
                double jumps_f[MAX_FAST];
                for (int k = 0; k < N_FAST; k++) {
                    double stddev = (fast_eval[k] > 0) ? sqrt(fast_eval[k]) : 0.0;
                    jumps_f[k] = gasdev(0.0, stddev * fast_EntireFactor_local);
                }
                for (int k = 0; k < N_FAST; k++) {
                    int idx = fast_idx[k];
                    double delta = 0.0;
                    for (int j = 0; j < N_FAST; j++) delta += fast_evec[k][j] * jumps_f[j];
                    trial[idx] = theta_f_path[idx] + delta;
                    if (trial[idx] < g_lowbound[idx] || trial[idx] > g_highbound[idx]) fast_oob = 1;
                }
            }

            double chi2_A_trial = 1e30, chi2_B_trial = 1e30;
            if (!fast_oob) {
                double thA[MAX_TASKARRAY_SIZE] = {0}, thB[MAX_TASKARRAY_SIZE] = {0};
                for (int k = 0; k < PARAMETERS; k++) { thA[k] = trial[k]; thB[k] = trial[k]; }
                for (int k = 0; k < N_SLOW; k++) {
                    thA[slow_idx[k]] = theta_A[slow_idx[k]];
                    thB[slow_idx[k]] = theta_B[slow_idx[k]];
                }

                chi2_A_trial = run_plc(rank, thA, cl_A_tt, cl_A_te, cl_A_ee, cl_A_bb, 0, NULL)
                               + compute_prior_penalty(thA);
                chi2_B_trial = run_plc(rank, thB, cl_B_tt, cl_B_te, cl_B_ee, cl_B_bb, 0, NULL)
                               + compute_prior_penalty(thB);
            }

            double L_cur   = (1.0 - w) * chi2_A_state + w * chi2_B_state;
            double L_trial = (1.0 - w) * chi2_A_trial + w * chi2_B_trial;

            if (!fast_oob && chi2_A_trial < 1e29 && chi2_B_trial < 1e29 &&
                posRnd(1.0) < exp(-0.5 * (L_trial - L_cur))) {
                for (int k = 0; k < PARAMETERS; k++) theta_f_path[k] = trial[k];
                chi2_A_state = chi2_A_trial;
                chi2_B_state = chi2_B_trial;
            }
            if (d < N_DRAG) work_sum += (chi2_B_state - chi2_A_state);
        }

        double ln_alpha = -0.5 * work_sum / (double)N_DRAG;
        double alpha = (ln_alpha >= 0.0) ? 1.0 : exp(ln_alpha);

        if (posRnd(1.0) < alpha) {
            for (int k = 0; k < PARAMETERS; k++) task[k] = theta_f_path[k];
            for (int k = 0; k < N_SLOW; k++) task[slow_idx[k]] = theta_B[slow_idx[k]];
            final_chi2 = chi2_B_state;
            proposal_chi2 = chi2_B_state;
            task[TAKEPOS] = 1.0;
        } else {
            for (int k = 0; k < PARAMETERS; k++) task[k] = theta_A[k];
            final_chi2 = chi2_A_original;
            proposal_chi2 = chi2_A_original;
            task[TAKEPOS] = 0.0;
        }
    } else {
        for (int k = 0; k < PARAMETERS; k++) task[k] = theta_A[k];
        final_chi2 = 1e30; proposal_chi2 = 1e30;
        task[TAKEPOS] = 0.0;
    }

    task[PARAMETERS] = proposal_chi2;
    task[LOGLIKEPOS] = (final_chi2 < 1e29) ? -0.5 * final_chi2 : -1e30;
    *use_emu_out = use_emu;
    *uncertainty_out = uncertainty_A;
}

// =====================================================================
// SLAVE
// =====================================================================
int slave(int rank) {
    DEBUG_PRINT("\nI am from slave %d", rank);
    MPI_Status status;
    double task[MAX_TASKARRAY_SIZE];

    int l_max_alloc = 3000;
    double *Cl_TT_A = (double*)malloc((l_max_alloc+1)*sizeof(double));
    double *Cl_TE_A = (double*)malloc((l_max_alloc+1)*sizeof(double));
    double *Cl_EE_A = (double*)malloc((l_max_alloc+1)*sizeof(double));
    double *Cl_BB_A = (double*)malloc((l_max_alloc+1)*sizeof(double));
    double *Cl_TT_B = (double*)malloc((l_max_alloc+1)*sizeof(double));
    double *Cl_TE_B = (double*)malloc((l_max_alloc+1)*sizeof(double));
    double *Cl_EE_B = (double*)malloc((l_max_alloc+1)*sizeof(double));
    double *Cl_BB_B = (double*)malloc((l_max_alloc+1)*sizeof(double));

    double g_lowbound[NOPARAM], g_highbound[NOPARAM], g_sigma[NOPARAM];
    get_parameter_bounds_from_config(g_lowbound, g_highbound, g_sigma);

    if (!mapping_done) {
        g_n_cosmo = N_SLOW;
        if (g_n_cosmo == 0) g_n_cosmo = 6;
        mapping_done = 1;
    }
    if (USE_EMULATOR && g_emulator == NULL) {
        EmulatorConfig emu_cfg = {
            .n_params = g_n_cosmo,
            .buffer_capacity = 500,
            .max_pca_modes = 40,
            .pca_interval = 500,
            .gp_interval = 50,
            .min_train_points = 200,
            .use_emulator_after = 250,
            .model_dir = "models",
            .emulator_type = EMULATOR_TYPE,
            .poly_degree = POLY_DEGREE
        };
        g_emulator = emulator_init(&emu_cfg);
        if (!g_emulator)
            fprintf(stderr, "Rank %d: Emulator init failed. Running without emulator.\n", rank);
        else
            DEBUG_PRINT("Rank %d: Emulator initialised (type %s) with %d cosmological parameters.\n",
                        rank, (EMULATOR_TYPE == 0) ? "GP" : "Polynomial", g_n_cosmo);
    }

    if (VERBOSE) {
        printf("\n"
               "------------------------------------------------------------------------------------------------------------------------------\n"
               " Rank  MCMC_step  Emu  Fallback  Uncertainty    chi2      logL       Omega_m    Omega_b       h        tau      n_s       A_s\n"
               "------------------------------------------------------------------------------------------------------------------------------\n");
        fflush(stdout);
    }

    for (;;) {
        MPI_Status probe_status;
        int flag = 0;
        MPI_Iprobe(0, TAG_STOP, MPI_COMM_WORLD, &flag, &probe_status);
        if (flag) { MPI_Recv(NULL, 0, MPI_INT, 0, TAG_STOP, MPI_COMM_WORLD, &probe_status); break; }

        MPI_Send(task, TASKARRAY_SIZE, MPI_DOUBLE, 0, GIVEMETASK, MPI_COMM_WORLD);

        for (;;) {
            MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            if (status.MPI_TAG == TAG_UPDATE_COV) {
                MPI_Recv(fast_cov_buffer, MAX_FAST_PACKED, MPI_DOUBLE, 0, TAG_UPDATE_COV, MPI_COMM_WORLD, &status);
                update_local_fast_covariance(fast_cov_buffer);
                continue;
            }
            break;
        }
        MPI_Recv(task, TASKARRAY_SIZE, MPI_DOUBLE, 0, TAKETASK, MPI_COMM_WORLD, &status);

        double loglike_cur = task[LOGLIKEPOS];
        int use_dragging = (int)task[DRAGPOS];

        int use_emu = 0;
        double uncertainty = 0.0;
        evaluate_proposal(rank, task,
                          Cl_TT_A, Cl_TE_A, Cl_EE_A, Cl_BB_A,
                          Cl_TT_B, Cl_TE_B, Cl_EE_B, Cl_BB_B,
                          g_lowbound, g_highbound, g_sigma,
                          &use_emu, &uncertainty);

        int take;
        if (use_dragging) {
            take = (int)task[TAKEPOS];
        } else {
            double loglike_prop = task[LOGLIKEPOS];
            double alpha = mind(1.0, exp(loglike_prop - loglike_cur));
            take = (posRnd(1.0) < alpha) ? 1 : 0;
            if (take == 0) {
                for (int k = 0; k < PARAMETERS; k++) task[k] = task[CURRENTSTATEPOS + k];
                task[LOGLIKEPOS] = loglike_cur;
                task[PARAMETERS] = -2.0 * loglike_cur;
            }
        }

        task[TAKEPOS] = (double)take;
        task[PROBPOS] = (take == 1) ? 1.0 : 0.0;

        /* Emulator training: use TRUE CAMB Cls AND the logL recomputed from them */
        if (take && g_emulator) {
            double *Cl_TT_true = (double*)malloc(2601 * sizeof(double));
            double *Cl_TE_true = (double*)malloc(2601 * sizeof(double));
            double *Cl_EE_true = (double*)malloc(2601 * sizeof(double));
            double *Cl_BB_true = (double*)malloc(2601 * sizeof(double));

            if (Cl_TT_true && Cl_TE_true && Cl_EE_true && Cl_BB_true) {
                if (param_iface(rank, task, Cl_TT_true, Cl_TE_true, Cl_EE_true, Cl_BB_true)) {
                    /* FIX #2: recompute true logL from CAMB Cls */
                    double dummy_true = 0.0;
                    double true_chi2 = run_plc(rank, task,
                                               Cl_TT_true, Cl_TE_true,
                                               Cl_EE_true, Cl_BB_true,
                                               0, &dummy_true)
                                       + compute_prior_penalty(task);
                    double true_logL = -0.5 * true_chi2;

                    double cosmo_final[N_SLOW];
                    for (int k = 0; k < N_SLOW; k++) cosmo_final[k] = task[slow_idx[k]];

                    int taken = emulator_update(g_emulator, local_step++, cosmo_final,
                                                Cl_TT_true, Cl_TE_true,
                                                Cl_EE_true, Cl_BB_true, true_logL);
                    if (!taken) {
                        free(Cl_TT_true); free(Cl_TE_true);
                        free(Cl_EE_true); free(Cl_BB_true);
                    }
                } else {
                    free(Cl_TT_true); free(Cl_TE_true);
                    free(Cl_EE_true); free(Cl_BB_true);
                }
            } else {
                free(Cl_TT_true); free(Cl_TE_true);
                free(Cl_EE_true); free(Cl_BB_true);
            }
        }

        if (g_emulator && !emulator_is_ready(g_emulator) && emulator_buffer_size(g_emulator) >= 200) {
            DEBUG_PRINT("Rank %d: Forcing emulator training.\n", rank);
            emulator_train(g_emulator);
        }

        if (VERBOSE) {
            int mcmc_step = (int)task[STEP_POS];
            double chi2_to_print = -2.0 * task[LOGLIKEPOS];
            double logL_to_print = task[LOGLIKEPOS];
            int max_print = (g_n_cosmo < 6) ? g_n_cosmo : 6;
            printf(" %2d     %6d    %1d      %1d     %8.4e   %8.2f %8.2f   ",
                   rank, mcmc_step, use_emu, (!use_emu), uncertainty,
                   chi2_to_print, logL_to_print);
            for (int i = 0; i < max_print; i++) printf("%8.4f ", task[i]);
            for (int i = max_print; i < 6; i++) printf("%8s ", " ");
            printf("\n"); fflush(stdout);
        }

        MPI_Send(task, TASKARRAY_SIZE, MPI_DOUBLE, 0, TAKERESULT, MPI_COMM_WORLD);
    }

    free(Cl_TT_A); free(Cl_TE_A); free(Cl_EE_A); free(Cl_BB_A);
    free(Cl_TT_B); free(Cl_TE_B); free(Cl_EE_B); free(Cl_BB_B);
    return 0;
}

int slave0(int rank,int runperchain) {
  MPI_Status  status;
  double *task;
  double Chi2X;
  int magicnum1;
  int l_max_alloc = 3000;
  double *Cl_TT = (double*)malloc((l_max_alloc + 1) * sizeof(double));
  double *Cl_TE = (double*)malloc((l_max_alloc + 1) * sizeof(double));
  double *Cl_EE = (double*)malloc((l_max_alloc + 1) * sizeof(double));
  double *Cl_BB = (double*)malloc((l_max_alloc + 1) * sizeof(double));

  if (!Cl_TT || !Cl_TE || !Cl_EE || !Cl_BB) {
      fprintf(stderr, "Rank %d: Failed to allocate memory for Cl arrays.\n", rank);
      MPI_Abort(MPI_COMM_WORLD, 1);
  }

  task = new_double(PARAMETERS+1);
  DEBUG_PRINT("My rank is here %d",rank);
  for (int i=0;i<runperchain;i++) {
      MPI_Recv(task,PARAMETERS+1,MPI_DOUBLE,0,200,MPI_COMM_WORLD,&status);
      magicnum1 = param_iface(rank, task, Cl_TT, Cl_TE, Cl_EE, Cl_BB);
      if (magicnum1 != 0) {
          double dummy = 0.0;
          Chi2X = run_plc(rank, task, Cl_TT, Cl_TE, Cl_EE, Cl_BB, 0, &dummy) + compute_prior_penalty(task);
      } else {
          Chi2X = 10.0e10;
      }
      if (WMAP7 == 1) task[PARAMETERS] = Chi2X;
      for (unsigned int k = 0;k < PARAMETERS+1; k++) {
          if (isnan(task[k])) {
              fprintf(stderr, "isnan element\n");
              free(Cl_TT); free(Cl_TE); free(Cl_EE); free(Cl_BB);
              MPI_Abort(MPI_COMM_WORLD, 1);
          }
      }
      MPI_Send(task,PARAMETERS+1, MPI_DOUBLE,0,201,MPI_COMM_WORLD);
  }
  free(Cl_TT); free(Cl_TE); free(Cl_EE); free(Cl_BB);
  DEBUG_PRINT("\nI am now working fine..");
  return 0;
}

/* NOTE: `worldsize` here carries the *real* MPI world size (passed by main). */
double **master0(int worldsize, int runperchain)
{
  MPI_Status status;
  double *startval[100];
  for (int i = 0; i < 100; i++) startval[i] = new_double(PARAMETERS + 1);

  double *lowbound = new_double(PARAMETERS);
  double *highbound = new_double(PARAMETERS);
  double *initial_sigma_dummy = new_double(PARAMETERS);
  get_parameter_bounds_from_config(lowbound, highbound, initial_sigma_dummy);

  /* FIX #4: single loop; random fill from config only */
  for (int i = 0; i < 100; i++) {
    int param_idx = 0;
    for (int p = 0; p < global_config.param_count && param_idx < PARAMETERS; p++) {
      if (global_config.params[p].state == PARAM_SAMPLED) {
        double val;
        if (global_config.params[p].prior_type == PRIOR_GAUSSIAN) {
          val = gasdev(global_config.params[p].gaussian_mean, global_config.params[p].gaussian_sigma);
        } else {
          double center = (highbound[param_idx] + lowbound[param_idx]) / 2.0;
          val = center + gasdev(0.0, initial_sigma_dummy[param_idx] * 2.0);
        }
        if (val < lowbound[param_idx]) val = lowbound[param_idx];
        if (val > highbound[param_idx]) val = highbound[param_idx];
        startval[i][param_idx] = val;
        param_idx++;
      }
    }
    for (; param_idx < PARAMETERS; param_idx++)
      startval[i][param_idx] = lowbound[param_idx] + posRnd(highbound[param_idx] - lowbound[param_idx]);
    startval[i][PARAMETERS] = 0.0;
  }
  free(initial_sigma_dummy);

  double *startvaltemp = new_double(PARAMETERS + 1);

  /* n_slaves is now the real slave count because main passes real `worldsize`. */
  int n_slaves = worldsize - 1;
  for (int i = 0; i < runperchain; i++) {
    for (int j = 0; j < n_slaves; j++)
      MPI_Send(startval[i + j * runperchain], PARAMETERS + 1, MPI_DOUBLE, j + 1, 200, MPI_COMM_WORLD);
    for (int j = 0; j < n_slaves; j++) {
      MPI_Recv(startvaltemp, PARAMETERS + 1, MPI_DOUBLE, MPI_ANY_SOURCE, 201, MPI_COMM_WORLD, &status);
      for (int k = 0; k <= PARAMETERS; k++)
        startval[i + (status.MPI_SOURCE - 1) * runperchain][k] = startvaltemp[k];
    }
  }

  int temparrange[100];
  double temploglike[100];
  for (int i = 0; i < 100; i++) { temparrange[i] = i; temploglike[i] = startval[i][PARAMETERS]; }

  for (int i = 0; i < CHAINS; i++)
    for (int j = i; j < 100; j++) {
      if (temploglike[i] > temploglike[j]) {
        int    tf = temparrange[i];    temparrange[i] = temparrange[j];    temparrange[j] = tf;
        double lf = temploglike[i];    temploglike[i] = temploglike[j];    temploglike[j] = lf;
      }
    }

  double **startvalfinal = (double **)malloc(CHAINS * sizeof(double *));
  for (int i = 0; i < CHAINS; i++) startvalfinal[i] = new_double(PARAMETERS + 1);
  for (int i = 0; i < CHAINS; i++)
    for (int j = 0; j <= PARAMETERS; j++)
      startvalfinal[i][j] = startval[temparrange[i]][j];
  return startvalfinal;
}

void classify_parameters(void) {
    N_SLOW = 0; N_FAST = 0;
    int param_idx = 0;
    for (int i = 0; i < global_config.param_count && param_idx < PARAMETERS; i++) {
        if (global_config.params[i].state == PARAM_SAMPLED) {
            if (global_config.params[i].speed == PARAM_SPEED_SLOW) slow_idx[N_SLOW++] = param_idx;
            else                                                 fast_idx[N_FAST++] = param_idx;
            param_idx++;
        }
    }
}

void setVariables() {
    if (!load_config("param_reader.ini", &global_config)) {
        printf("CRITICAL ERROR: Could not load param_reader.ini\n");
        exit(1);
    }
    DEBUG_PRINT("Loaded configuration from param_reader.ini.\n");

    int total_estimated = 0;
    for (int i = 0; i < global_config.param_count; i++)
        if (global_config.params[i].state == PARAM_SAMPLED) total_estimated++;

    PARAMETERS = (global_config.num_sampled_total > 0) ? global_config.num_sampled_total : NOPARAM;

    LOGLIKEPOS = PARAMETERS + 1;
    LOGLIKEPOS = (LOGLIKEPOS > 15) ? LOGLIKEPOS : 15;

    MULTIPURPOSEPOS = LOGLIKEPOS + 22;
    ADAPTIVEPOS = MULTIPURPOSEPOS + 1;
    DRAGPOS = ADAPTIVEPOS + 1;
    FASTFACTORPOS = DRAGPOS + 1;
    TAKEPOS = FASTFACTORPOS + 1;
    PROBPOS = TAKEPOS + 1;
    RANDPOS = PROBPOS + 1;

    STEP_POS = RANDPOS + PARAMETERS + MULTIPURPOSE_REQUIRED + 1;
    CURRENTSTATEPOS = STEP_POS + 1;
    TASKARRAY_SIZE  = CURRENTSTATEPOS + PARAMETERS;

    if (TASKARRAY_SIZE > MAX_TASKARRAY_SIZE) {
        printf("ERROR: TASKARRAY_SIZE (%d) exceeds MAX_TASKARRAY_SIZE (%d).\n",
               TASKARRAY_SIZE, MAX_TASKARRAY_SIZE);
        exit(1);
    }
    classify_parameters();
    DEBUG_PRINT("Task Array Size: %d | Slow: %d | Fast: %d\n", TASKARRAY_SIZE, N_SLOW, N_FAST);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int myrank, worldsize;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

    srand((unsigned)(time(NULL) ^ (myrank * 2654435761u)));

    unsigned short int Restart = 0;
    const char *outdir_default = ".";
    const char *outdir = outdir_default;
    const char *load_cov_path = NULL;

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "-restart") == 0)                        Restart = 1;
        else if (strcmp(argv[i], "-o") == 0 && i+1 < argc)                outdir = argv[++i];
        else if (strcmp(argv[i], "-load-cov") == 0 && i+1 < argc)         load_cov_path = argv[++i];
        else if (strcmp(argv[i], "-emu") == 0)                            USE_EMULATOR = 1;
        else if (strcmp(argv[i], "-no-emu") == 0)                         USE_EMULATOR = 0;
        else if (strcmp(argv[i], "-drag") == 0)                           USE_DRAGGING = 1;
        else if (strcmp(argv[i], "-no-drag") == 0)                        USE_DRAGGING = 0;
        else if (strcmp(argv[i], "-print") == 0)                          VERBOSE = 1;
        else if (strcmp(argv[i], "-no-print") == 0)                       VERBOSE = 0;
        else {
            if (myrank == 0)
                printf("Usage: %s [-restart] [-o outdir] [-load-cov file] "
                       "[-emu|-no-emu] [-drag|-no-drag] [-print|-no-print]\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    LOAD_COV_PATH = load_cov_path;

    int expected_ranks = 1 + CHAINS * SLAVEPARCHAIN;
    if (worldsize != expected_ranks) {
        if (myrank == 0)
            fprintf(stderr, "ERROR: need exactly %d MPI ranks (1 master + %d chains * %d slave per chain).\n",
                    expected_ranks, CHAINS, SLAVEPARCHAIN);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (myrank == 0) {
        char mkdir_cmd[512];
        snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", outdir);
        system(mkdir_cmd);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    strncpy(outdir_buf, outdir, sizeof(outdir_buf)-1);
    outdir_buf[sizeof(outdir_buf)-1] = '\0';
    MPI_Bcast(outdir_buf, sizeof(outdir_buf), MPI_CHAR, 0, MPI_COMM_WORLD);

    setVariables();
    DEBUG_PRINT("Myrank : %d", myrank);

    print_and_validate_parameters(myrank);

    int totalrun, runperchain;
    drfactor = 0.7;
    if (worldsize > 101) { totalrun = 100; runperchain = 1; }
    else                 { runperchain = 100 / (worldsize - 1); totalrun = runperchain * (worldsize - 1); }

    double **startnum;
    double start_time = 0.0;
    if (myrank == 0) start_time = get_time();

    if (myrank == 0) {
        DEBUG_PRINT("\nRestart = %d", Restart);
        /* FIX #1: pass the REAL world size, not totalrun */
        startnum = master0(worldsize, runperchain);
    } else {
        slave0(myrank, runperchain);
    }

    if (myrank == 0) {
        DEBUG_PRINT("\nRestart = %d", Restart);
        master(startnum, Restart);
    } else {
        DEBUG_PRINT("Slave is working");
        slave(myrank);
    }

    if (myrank == 0) {
        double end_time = get_time();
        double elapsed = end_time - start_time;
        int h = (int)(elapsed / 3600);
        int m = (int)((elapsed - h * 3600) / 60);
        int s = (int)(elapsed - h * 3600 - m * 60);
        printf("\n========================================\n");
        printf("Total elapsed time: %02d:%02d:%02d (hh:mm:ss)\n", h, m, s);
        printf("========================================\n");
    }

    if (g_emulator) emulator_free(g_emulator);

    MPI_Finalize();
    return 0;
}
