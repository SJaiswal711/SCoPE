/*
 * serial_scope.c - Serial MCMC with online PCA+GP emulator
 * Features:
 *   - Online training: always adds true evaluations to buffer (active learning)
 *   - Uses GP predictive uncertainty to decide when to call true model
 *   - Periodically retrains emulator with new data
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>

#include "param_config_single.h"
#include "newpar_new.h"
#include "nrunplc_new.h"

/* ======================= Emulator headers ======================= */
#include "emulator/include/emulator.h"

/* ======================= Emulator selection ======================= */
#define EMULATOR_TYPE 1   /* 0 = GP, 1 = Polynomial */
#define POLY_DEGREE 1     /* only used if EMULATOR_TYPE == 1 */

/* ---------- constants ---------- */
#define NOPARAM 20
#define MAX_TASKARRAY_SIZE 150
#define NR_END 1
#define FREE_ARG char *

#define CHAINS 1
#define MAXCHAINLENGTH 50000
#define BEGINCOVUPDATE 200
#define UPDATE_TIME 5
#define MAXPREVIOUSPOINTS 5000
#define RMIN_POINTS 200

#define N_SPECTRA 2601   /* number of multipoles (l=0..2600) */

/* Step size adaptation - more stable */
static double INCREASE_STEP = 1.02;
static double DECREASE_STEP = 0.98;
static double ACCEPT_TARGET_LOW = 0.23;
static double ACCEPT_TARGET_HIGH = 0.40;
static int ADAPTATION_INTERVAL = 500;

/* Step factor limits - increased minimum to allow movement */
#define MAX_STEP_FACTOR 1.0
#define MIN_STEP_FACTOR 0.3

unsigned short int ADAPTIVE = 1;
unsigned short int FREEZE_IN = 0;

Config global_config;
short int PARAMETERS = NOPARAM;

unsigned short int TASKARRAY_SIZE;
unsigned short int MULTIPURPOSEPOS;
unsigned short int TAKEPOS, PROBPOS;
unsigned short int ADAPTIVEPOS;
unsigned int LOGLIKEPOS;
int RANDPOS;

static unsigned int MULTIPURPOSE_REQUIRED = 1;
static short int WMAP7 = 1;

/* ---------- structures ---------- */
typedef struct Task {
    double f[MAX_TASKARRAY_SIZE];
    int Multiplicity;
    int ReallyInvestigated;
} Task;

typedef struct RollingAverage {
    unsigned int Size;
    unsigned int Number;
    double Sum;
    double *y;
    unsigned int idx;
    int PerformFullCounter;
} RollingAverage;

typedef struct MultiGaussian {
    double Scale;
    unsigned int SIZE;
    double **MasterMatrix;
    double *eigenvalues;
    double *generatedValues;
    double *lbounds;
    double *hbounds;
    double *randomq;
    double *center;
} MultiGaussian;

/* ---------- forward declarations ---------- */
static double compute_loglike_true(double *params);
static double compute_loglike_true_and_update_buffer(double *params, int step);
static double compute_loglike_emu(double *params);
static double compute_loglike(double *params);
static void add_point_to_buffer(double *params, int step, double logL);

/* ---------- Logging: CAMB calls and spectra ---------- */
static FILE *g_log_file = NULL;          /* text log of CAMB calls */
static FILE *g_spec_file = NULL;         /* text log of buffer additions */

static int is_spectrum_valid(const double *cl_tt, const double *cl_te,
                             const double *cl_ee, const double *cl_bb, int step) {
    const double MIN_VAL = 1e-30;
    const double EPS_CHECK = 1e-60;   // for denom in rho

    // Check finite and non‑negative for TT, EE, BB (TE can be negative)
    for (int i = 0; i < N_SPECTRA; i++) {
        if (!isfinite(cl_tt[i]) || !isfinite(cl_te[i]) ||
            !isfinite(cl_ee[i]) || !isfinite(cl_bb[i])) {
            fprintf(stderr, "Invalid at step %d: non-finite value\n", step);
            return 0;
        }
        if (cl_tt[i] < 0.0 || cl_ee[i] < 0.0 || cl_bb[i] < 0.0) {
            fprintf(stderr, "Invalid at step %d: negative Cl\n", step);
            return 0;
        }
        // TE can be negative, so no check
    }

    // Reject only if the entire spectrum is too small (all TT, EE, BB near zero)
    double max_tt = 0.0, max_ee = 0.0, max_bb = 0.0;
    for (int i = 0; i < N_SPECTRA; i++) {
        if (cl_tt[i] > max_tt) max_tt = cl_tt[i];
        if (cl_ee[i] > max_ee) max_ee = cl_ee[i];
        if (cl_bb[i] > max_bb) max_bb = cl_bb[i];
    }
    if (max_tt < MIN_VAL && max_ee < MIN_VAL && max_bb < MIN_VAL) {
        fprintf(stderr, "Invalid at step %d: all Cl too small (max_tt=%g, max_ee=%g, max_bb=%g)\n",
                step, max_tt, max_ee, max_bb);
        return 0;
    }

    // Check rho = TE / sqrt(TT*EE) is within [-1,1] where denom is not too small
    for (int i = 0; i < N_SPECTRA; i++) {
        double tt = cl_tt[i];
        double ee = cl_ee[i];
        double te = cl_te[i];
        double denom = sqrt(tt * ee);
        // If denom is extremely small, rho is ill‑defined; skip check (it's fine)
        if (denom < 1e-60) continue;
        double rho = te / denom;
        if (rho < -1.0 || rho > 1.0) {
            fprintf(stderr, "Invalid rho at step %d (l=%d): rho=%g\n", step, i, rho);
            return 0;
        }
    }

    return 1;
}

static void log_camb_call(int step, const double *params, double logL, const char *tag) {
    if (!g_log_file) return;
    fprintf(g_log_file, "%d %s", step, tag);
    for (int i = 0; i < PARAMETERS; i++) {
        fprintf(g_log_file, " %.12e", params[i]);
    }
    fprintf(g_log_file, " %.12e\n", logL);
    fflush(g_log_file);
}

static void log_buffer_add(int step, const double *params, double logL,
                           const double *cl_tt, const double *cl_te,
                           const double *cl_ee, const double *cl_bb) {
    if (!g_spec_file) return;
    if (!is_spectrum_valid(cl_tt, cl_te, cl_ee, cl_bb, step)) {
        fprintf(stderr, "WARNING: Skipping buffer log of invalid spectra at step %d\n", step);
        return;
    }
    fprintf(g_spec_file, "%d", step);
    for (int i = 0; i < PARAMETERS; i++) {
        fprintf(g_spec_file, " %.12e", params[i]);
    }
    fprintf(g_spec_file, " %.12e", logL);
    // Compute min, max, mean for each spectrum
    double tt_min=1e100, tt_max=-1e100, tt_sum=0.0;
    double te_min=1e100, te_max=-1e100, te_sum=0.0;
    double ee_min=1e100, ee_max=-1e100, ee_sum=0.0;
    double bb_min=1e100, bb_max=-1e100, bb_sum=0.0;
    for (int i=0; i<N_SPECTRA; i++) {
        if (cl_tt[i] < tt_min) tt_min = cl_tt[i];
        if (cl_tt[i] > tt_max) tt_max = cl_tt[i];
        tt_sum += cl_tt[i];
        if (cl_te[i] < te_min) te_min = cl_te[i];
        if (cl_te[i] > te_max) te_max = cl_te[i];
        te_sum += cl_te[i];
        if (cl_ee[i] < ee_min) ee_min = cl_ee[i];
        if (cl_ee[i] > ee_max) ee_max = cl_ee[i];
        ee_sum += cl_ee[i];
        if (cl_bb[i] < bb_min) bb_min = cl_bb[i];
        if (cl_bb[i] > bb_max) bb_max = cl_bb[i];
        bb_sum += cl_bb[i];
    }
    fprintf(g_spec_file, " TT:min=%.6e max=%.6e mean=%.6e", tt_min, tt_max, tt_sum/N_SPECTRA);
    fprintf(g_spec_file, " TE:min=%.6e max=%.6e mean=%.6e", te_min, te_max, te_sum/N_SPECTRA);
    fprintf(g_spec_file, " EE:min=%.6e max=%.6e mean=%.6e", ee_min, ee_max, ee_sum/N_SPECTRA);
    fprintf(g_spec_file, " BB:min=%.6e max=%.6e mean=%.6e\n", bb_min, bb_max, bb_sum/N_SPECTRA);
    fflush(g_spec_file);
}

/* ---------- rolling average ---------- */
void Rolling_Average_push(RollingAverage *RA, double x)
{
    RA->Sum -= RA->y[RA->idx];
    RA->Sum += x;
    RA->y[RA->idx++] = x;
    if (RA->idx == RA->Size) RA->idx = 0;
    RA->Number++;
    if (RA->Number >= RA->Size) RA->Number = RA->Size;
}

double RollingAverage_average(RollingAverage *RA)
{
    if (RA->Number == 0) return 0;
    if (RA->PerformFullCounter++ < 10) return RA->Sum / RA->Number;
    RA->PerformFullCounter = 0;
    double Sum = 0.0;
    for (unsigned int i = 0; i < RA->Number; i++) Sum += RA->y[i];
    return Sum / RA->Number;
}

RollingAverage *new_RollingAverage(int Size)
{
    RollingAverage *RA = malloc(sizeof(RollingAverage));
    RA->y = malloc(Size * sizeof(double));
    RA->Size = Size;
    RA->idx = 0;
    RA->Number = 0;
    RA->Sum = 0.0;
    RA->PerformFullCounter = 0;
    return RA;
}

/* ---------- MultiGaussian functions ---------- */
MultiGaussian *new_MultiGaussian(int Size)
{
    MultiGaussian *MG = malloc(sizeof(MultiGaussian));
    MG->SIZE = Size;
    MG->MasterMatrix = malloc(Size * sizeof(double*));
    for (int i = 0; i < Size; i++)
        MG->MasterMatrix[i] = malloc(Size * sizeof(double));
    MG->eigenvalues = malloc(Size * sizeof(double));
    MG->generatedValues = malloc(Size * sizeof(double));
    MG->lbounds = malloc(Size * sizeof(double));
    MG->hbounds = malloc(Size * sizeof(double));
    MG->center = malloc(Size * sizeof(double));
    MG->randomq = malloc(Size * sizeof(double));
    return MG;
}

void MultiGaussian_setBounds(MultiGaussian *MG, double lowBound[], double highBound[])
{
    for (unsigned int i = 0; i < MG->SIZE; i++) {
        MG->lbounds[i] = lowBound[i];
        MG->hbounds[i] = highBound[i];
    }
}

double posRnd(double max) { return (rand() / (double)RAND_MAX) * max; }
double ran1(float x) { return 2.0 * (rand() / (double)RAND_MAX - 0.5) * x; }

double gasdev(double mean, double std)
{
    double rsq, v1, v2;
    do {
        v1 = ran1(1.0);
        v2 = ran1(1.0);
        rsq = v1*v1 + v2*v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    double fac = v1 * sqrt(-2.0 * log(rsq) / rsq);
    return fac * std + mean;
}

void generateRandom(MultiGaussian *MG)
{
    double *y = malloc(MG->SIZE * sizeof(double));
    for (unsigned int i = 0; i < MG->SIZE; i++)
        y[i] = gasdev(0.0, sqrt(MG->eigenvalues[i]));
    for (unsigned int i = 0; i < MG->SIZE; i++) {
        MG->randomq[i] = y[i];
        MG->generatedValues[i] = 0.0;
        for (unsigned int j = 0; j < MG->SIZE; j++)
            MG->generatedValues[i] += MG->MasterMatrix[i][j] * y[j];
    }
    free(y);
}

void generateEigenvectors(MultiGaussian *MG, double **covarianceMatrix, double scale)
{
    MG->Scale = scale;
    double *data = malloc(MG->SIZE * MG->SIZE * sizeof(double));
    int k = 0;
    for (unsigned int j = 0; j < MG->SIZE; j++)
        for (unsigned int i = 0; i < MG->SIZE; i++)
            data[k++] = covarianceMatrix[i][j];
    gsl_matrix_view m = gsl_matrix_view_array(data, MG->SIZE, MG->SIZE);
    gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(MG->SIZE);
    gsl_vector *eval = gsl_vector_alloc(MG->SIZE);
    gsl_matrix *evec = gsl_matrix_alloc(MG->SIZE, MG->SIZE);
    gsl_eigen_symmv(&m.matrix, eval, evec, w);
    gsl_eigen_symmv_free(w);
    gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_ASC);
    for (unsigned int i = 0; i < MG->SIZE; i++) {
        MG->eigenvalues[i] = gsl_vector_get(eval, i);
        for (unsigned int j = 0; j < MG->SIZE; j++)
            MG->MasterMatrix[i][j] = gsl_matrix_get(evec, i, j);
    }
    gsl_vector_free(eval);
    gsl_matrix_free(evec);
    free(data);
    for (unsigned int i = 0; i < MG->SIZE; i++)
        MG->eigenvalues[i] *= scale;
}

int throwDice(Task chain, Task *next, MultiGaussian *MG, double step_factor)
{
    for (unsigned int i = 0; i < MG->SIZE; i++)
        MG->center[i] = chain.f[i];
    generateRandom(MG);
    for (unsigned int i = 0; i < PARAMETERS; i++) {
        next->f[i] = MG->generatedValues[i] * step_factor + MG->center[i];
        if (next->f[i] < MG->lbounds[i] || next->f[i] > MG->hbounds[i])
            return 0;
    }
    return 1;
}

/* ======================= Emulator integration ======================= */

static Emulator *g_emulator = NULL;
static int g_step = 0;
static int g_use_emulator = 0;

/* Helper: add a point (already evaluated) to the training buffer */
static void add_point_to_buffer(double *params, int step, double logL) {
    if (!g_emulator) return;
    double *cl_tt = malloc(N_SPECTRA * sizeof(double));
    double *cl_te = malloc(N_SPECTRA * sizeof(double));
    double *cl_ee = malloc(N_SPECTRA * sizeof(double));
    double *cl_bb = malloc(N_SPECTRA * sizeof(double));
    if (cl_tt && cl_te && cl_ee && cl_bb) {
        if (param_iface(0, params, cl_tt, cl_te, cl_ee, cl_bb)) {
            if (is_spectrum_valid(cl_tt, cl_te, cl_ee, cl_bb, step)) {
                emulator_update(g_emulator, step, params, cl_tt, cl_te, cl_ee, cl_bb, logL);
                // buffer now owns the pointers – do NOT free
            } else {
                free(cl_tt); free(cl_te); free(cl_ee); free(cl_bb);
            }
        } else {
            free(cl_tt); free(cl_te); free(cl_ee); free(cl_bb);
        }
    } else {
        free(cl_tt); free(cl_te); free(cl_ee); free(cl_bb);
    }
}

/* ----------------------------------------------------------------------
   True model evaluation without buffer update (used for proposal when emulator is off)
   ---------------------------------------------------------------------- */
static double compute_loglike_true(double *params) {
    double *cl_tt = malloc(N_SPECTRA * sizeof(double));
    double *cl_te = malloc(N_SPECTRA * sizeof(double));
    double *cl_ee = malloc(N_SPECTRA * sizeof(double));
    double *cl_bb = malloc(N_SPECTRA * sizeof(double));
    if (!cl_tt || !cl_te || !cl_ee || !cl_bb) {
        free(cl_tt); free(cl_te); free(cl_ee); free(cl_bb);
        return -1e100;
    }

    if (!param_iface(0, params, cl_tt, cl_te, cl_ee, cl_bb)) {
        free(cl_tt); free(cl_te); free(cl_ee); free(cl_bb);
        return -1e100;
    }

    double chi2 = run_plc(0, params, cl_tt, cl_te, cl_ee, cl_bb);
    double logL = -0.5 * chi2;

    log_camb_call(g_step, params, logL, "TRUE");

    free(cl_tt); free(cl_te); free(cl_ee); free(cl_bb);
    return logL;
}

/* ----------------------------------------------------------------------
   True model evaluation that also updates the training buffer (active learning)
   Used for fallback points and accepted points (the latter already handled separately)
   ---------------------------------------------------------------------- */
static double compute_loglike_true_and_update_buffer(double *params, int step) {
    double *cl_tt = malloc(N_SPECTRA * sizeof(double));
    double *cl_te = malloc(N_SPECTRA * sizeof(double));
    double *cl_ee = malloc(N_SPECTRA * sizeof(double));
    double *cl_bb = malloc(N_SPECTRA * sizeof(double));
    double logL = -1e100;

    if (cl_tt && cl_te && cl_ee && cl_bb) {
        if (param_iface(0, params, cl_tt, cl_te, cl_ee, cl_bb)) {
            double chi2 = run_plc(0, params, cl_tt, cl_te, cl_ee, cl_bb);
            logL = -0.5 * chi2;
            log_camb_call(step, params, logL, "TRUE_BUFFER");
            if (g_emulator) {
                if (is_spectrum_valid(cl_tt, cl_te, cl_ee, cl_bb, step)) {
                    log_buffer_add(step, params, logL, cl_tt, cl_te, cl_ee, cl_bb);
                    emulator_update(g_emulator, step, params, cl_tt, cl_te, cl_ee, cl_bb, logL);
                    // buffer owns pointers – do NOT free
                    return logL;
                } else {
                    free(cl_tt); free(cl_te); free(cl_ee); free(cl_bb);
                    return logL;
                }
            }
        }
    }
    // If we reach here, param_iface failed or g_emulator is NULL
    free(cl_tt); free(cl_te); free(cl_ee); free(cl_bb);
    return logL;
}

/* ----------------------------------------------------------------------
   Emulator prediction with uncertainty-based fallback
   ---------------------------------------------------------------------- */
static double compute_loglike_emu(double *params) {
    double cl_tt[N_SPECTRA], cl_te[N_SPECTRA], cl_ee[N_SPECTRA], cl_bb[N_SPECTRA];
    double uncertainty;
    if (!emulator_predict(g_emulator, params, cl_tt, cl_te, cl_ee, cl_bb, &uncertainty)) {
        // Prediction failed: fall back to true model and update buffer
        return compute_loglike_true_and_update_buffer(params, g_step);
    }
    if (uncertainty > 0.05) {
        printf("[STEP %d] FALLBACK: uncertainty = %.4f > 0.05\n", g_step, uncertainty);
        fflush(stdout);
        return compute_loglike_true_and_update_buffer(params, g_step);
    }
    double chi2 = run_plc(0, params, cl_tt, cl_te, cl_ee, cl_bb);
    return -0.5 * chi2;
}

static double compute_loglike(double *params) {
    if (g_use_emulator && emulator_is_ready(g_emulator))
        return compute_loglike_emu(params);
    else
        return compute_loglike_true(params);
}

/* ---------- configuration functions ---------- */
void get_parameter_bounds_from_config(double *lowbound, double *highbound, double *initial_sigma) {
    int param_index = 0;
    for (int i = 0; i < global_config.param_count && param_index < PARAMETERS; i++) {
        if (global_config.params[i].is_estimated) {
            lowbound[param_index] = global_config.params[i].lower_bound;
            highbound[param_index] = global_config.params[i].upper_bound;
            initial_sigma[param_index] = global_config.params[i].sigma;
            param_index++;
        }
    }
    for (; param_index < PARAMETERS; param_index++) {
        lowbound[param_index] = 0.0;
        highbound[param_index] = 1.0;
        initial_sigma[param_index] = 0.1;
    }
}

void setVariables() {
    if (!load_config("param.ini", &global_config)) {
        fprintf(stderr, "Failed to load param.ini\n");
        exit(1);
    }
    int total_estimated = 0;
    for (int i = 0; i < global_config.param_count; i++)
        if (global_config.params[i].is_estimated) total_estimated++;
    PARAMETERS = (total_estimated > 0) ? total_estimated : NOPARAM;
    LOGLIKEPOS = PARAMETERS + 1;
    if (LOGLIKEPOS < 15) LOGLIKEPOS = 15;
    MULTIPURPOSEPOS = LOGLIKEPOS + 22;
    ADAPTIVEPOS = MULTIPURPOSEPOS + 1;
    TAKEPOS = ADAPTIVEPOS + 1;
    PROBPOS = TAKEPOS + 1;
    RANDPOS = PROBPOS + 1;
    TASKARRAY_SIZE = PROBPOS + PARAMETERS + MULTIPURPOSE_REQUIRED + 1;
    printf("Serial SCoPE: %d estimated parameters.\n", PARAMETERS);
    fflush(stdout);
}

/* ---------- main ---------- */
int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setVariables();
    srand(time(NULL));

    /* Open log files */
    g_log_file = fopen("camb_calls_1params6.log", "w");
    if (!g_log_file) fprintf(stderr, "Warning: could not open camb_calls_1params6.log\n");
    g_spec_file = fopen("buffer_log_1params6poly.txt", "w");
    if (!g_spec_file) fprintf(stderr, "Warning: could not open buffer_log_1params6poly.txt\n");

    double *lowbound = malloc(PARAMETERS * sizeof(double));
    double *highbound = malloc(PARAMETERS * sizeof(double));
    double *initial_sigma = malloc(PARAMETERS * sizeof(double));
    get_parameter_bounds_from_config(lowbound, highbound, initial_sigma);

    double *current = malloc(PARAMETERS * sizeof(double));
    for (int i = 0; i < PARAMETERS; i++)
        current[i] = (lowbound[i] + highbound[i]) * 0.5;

    MultiGaussian *mg = new_MultiGaussian(PARAMETERS);
    MultiGaussian_setBounds(mg, lowbound, highbound);
    double **cov = malloc(PARAMETERS * sizeof(double*));
    for (int i = 0; i < PARAMETERS; i++) {
        cov[i] = malloc(PARAMETERS * sizeof(double));
        for (int j = 0; j < PARAMETERS; j++)
            cov[i][j] = (i == j) ? initial_sigma[i]*initial_sigma[i] : 0.0;
    }
    double step_factor = 0.2;
    generateEigenvectors(mg, cov, step_factor * step_factor);

    Task *chain = malloc(MAXCHAINLENGTH * sizeof(Task));
    int chainSize = 0, chainBack = 0;

    /* ----------------------- Emulator initialisation ----------------------- */
    EmulatorConfig emu_cfg = {
        .n_params = 6,
        .buffer_capacity = 200,
        .max_pca_modes = 40,
        .pca_interval = 50,
        .gp_interval = 10,
        .min_train_points = 50,
        .use_emulator_after = 100,
        .model_dir = "models",
        .emulator_type = EMULATOR_TYPE,
        .poly_degree = POLY_DEGREE
    };
    g_emulator = emulator_init(&emu_cfg);
    if (!g_emulator) {
        printf("Warning: Emulator initialisation failed. Running without emulator.\n");
    } else {
        printf("Emulator initialised. Type: %s, buffer capacity = %d, max PCA modes = %d\n",
               (EMULATOR_TYPE == 0) ? "GP" : "Polynomial",
               emu_cfg.buffer_capacity, emu_cfg.max_pca_modes);
    }
    /* ----------------------------------------------------------------------- */

    double current_loglike = compute_loglike_true(current);
    // Add initial point to buffer for training (also logs spectra)
    compute_loglike_true_and_update_buffer(current, 0);
    printf("Initial log-likelihood: %f\n", current_loglike);
    fflush(stdout);

    Task t;
    for (int i = 0; i < PARAMETERS; i++) t.f[i] = current[i];
    t.f[LOGLIKEPOS] = current_loglike;
    t.Multiplicity = 1;
    t.ReallyInvestigated = 0;
    chain[chainBack++] = t;
    chainSize++;

    FILE *chain_file = fopen("chain_serial_1params6poly.txt", "w");
    fprintf(chain_file, "# step");
    for (int i = 0; i < PARAMETERS; i++) fprintf(chain_file, " param_%d", i);
    fprintf(chain_file, " loglike\n");

    RollingAverage *roll = new_RollingAverage(500);
    int steps_since_update = 0;
    double EntireFactor = step_factor;

    int burnin = 200;
    int max_steps = 50000;
    int consecutive_rejections = 0;
    int stuck_counter = 0;
    int adapt_counter = 0;
    int last_100_accepted[100] = {0};
    int circ_idx = 0;

    int initially_trained = 0;
    // int last_retrain_step = 0;
    // int retrain_interval = 200;
    // int retrain_threshold = 50;
    // int points_since_last_retrain = 0;

    int emu_reject_count = 0;

    printf("Starting MCMC loop with max_steps = %d\n", max_steps);
    printf("Target acceptance rate: %.0f-%.0f%%\n", ACCEPT_TARGET_LOW*100, ACCEPT_TARGET_HIGH*100);
    fflush(stdout);

    for (int step = 0; step < max_steps; step++) {
        g_step = step;
        Task next;
        int accepted = 0;
        int n_tries = 0;

        do {
            if (!throwDice(chain[chainBack-1], &next, mg, EntireFactor)) {
                n_tries++;
                if (n_tries > 20 && step % 100 == 0)
                    printf("Step %d: proposal generation failed after 200 tries\n", step);
                break;
            }
            double prop_loglike = compute_loglike(next.f);
            if (!isfinite(prop_loglike)) {
                n_tries++;
                continue;
            }
            double log_ratio = prop_loglike - current_loglike;
            if (log_ratio > 0 || exp(log_ratio) > (rand()/(double)RAND_MAX)) {
                accepted = 1;
                current_loglike = prop_loglike;
                for (int i = 0; i < PARAMETERS; i++) current[i] = next.f[i];
                for (int i = 0; i < PARAMETERS; i++) t.f[i] = next.f[i];
                t.f[LOGLIKEPOS] = current_loglike;
                t.Multiplicity = 1;
                t.ReallyInvestigated = 0;
                chain[chainBack++] = t;
                chainSize++;
                if (chainBack >= MAXCHAINLENGTH) goto done;

                /* --- BUFFER UPDATE: ALWAYS add true evaluation for accepted points --- */
                double *cl_tt = malloc(N_SPECTRA * sizeof(double));
                double *cl_te = malloc(N_SPECTRA * sizeof(double));
                double *cl_ee = malloc(N_SPECTRA * sizeof(double));
                double *cl_bb = malloc(N_SPECTRA * sizeof(double));
                if (param_iface(0, current, cl_tt, cl_te, cl_ee, cl_bb)) {
                    double true_logL = -0.5 * run_plc(0, current, cl_tt, cl_te, cl_ee, cl_bb);
                    log_camb_call(step, current, true_logL, "ACCEPTED");
                    if (is_spectrum_valid(cl_tt, cl_te, cl_ee, cl_bb, step)) {
                        log_buffer_add(step, current, true_logL, cl_tt, cl_te, cl_ee, cl_bb);
                        emulator_update(g_emulator, step, current, cl_tt, cl_te, cl_ee, cl_bb, true_logL);
                        // points_since_last_retrain++;
                    } else {
                        free(cl_tt); free(cl_te); free(cl_ee); free(cl_bb);
                    }
                } else {
                    free(cl_tt); free(cl_te); free(cl_ee); free(cl_bb);
                }
            }
            break;
        } while (1);

        chain[chainBack-1].ReallyInvestigated++;

        if (!accepted) consecutive_rejections++;
        else consecutive_rejections = 0;

        if (g_use_emulator && !accepted) {
            emu_reject_count++;
        } else {
            emu_reject_count = 0;
        }
        if (emu_reject_count > 200) {
            printf("Step %d: Disabling emulator due to %d consecutive rejections\n", step, emu_reject_count);
            g_use_emulator = 0;
            EntireFactor = 0.5;
            generateEigenvectors(mg, cov, EntireFactor * EntireFactor);
            emu_reject_count = 0;
        }

        last_100_accepted[circ_idx++] = accepted;
        if (circ_idx >= 100) circ_idx = 0;
        double acc_rate = 0.0;
        for (int i = 0; i < 100; i++) acc_rate += last_100_accepted[i];
        acc_rate /= 100.0;

        static double last_logL = 0;
        if (step > burnin && fabs(current_loglike - last_logL) < 0.1) stuck_counter++;
        else { stuck_counter = 0; last_logL = current_loglike; }
        if (stuck_counter > 500 && EntireFactor > MIN_STEP_FACTOR) {
            printf("Step %d: WARNING - Chain stuck at logL = %f. Resetting step factor.\n", step, current_loglike);
            EntireFactor = 0.5;
            generateEigenvectors(mg, cov, EntireFactor * EntireFactor);
            stuck_counter = 0;
        }

        /* Step size adaptation */
        if (step >= burnin && ADAPTIVE) {
            adapt_counter++;
            if (adapt_counter >= ADAPTATION_INTERVAL) {
                adapt_counter = 0;
                Rolling_Average_push(roll, EntireFactor);
                if (acc_rate > ACCEPT_TARGET_HIGH && EntireFactor < MAX_STEP_FACTOR) {
                    EntireFactor *= INCREASE_STEP;
                    if (EntireFactor > MAX_STEP_FACTOR) EntireFactor = MAX_STEP_FACTOR;
                    generateEigenvectors(mg, cov, EntireFactor * EntireFactor);
                    printf("Step %d: Acc rate %.2f -> increasing factor to %.4f\n", step, acc_rate, EntireFactor);
                    fflush(stdout);
                } else if (acc_rate < ACCEPT_TARGET_LOW && EntireFactor > MIN_STEP_FACTOR) {
                    EntireFactor *= DECREASE_STEP;
                    if (EntireFactor < MIN_STEP_FACTOR) EntireFactor = MIN_STEP_FACTOR;
                    generateEigenvectors(mg, cov, EntireFactor * EntireFactor);
                    printf("Step %d: Acc rate %.2f -> decreasing factor to %.4f\n", step, acc_rate, EntireFactor);
                    fflush(stdout);
                }
                steps_since_update++;
            }
        }

        /* Covariance adaptation */
        if (ADAPTIVE && steps_since_update >= UPDATE_TIME && chainSize >= BEGINCOVUPDATE) {
            int nsamples = (chainSize < 2*BEGINCOVUPDATE) ? chainSize/2 : chainSize - BEGINCOVUPDATE;
            if (nsamples > MAXPREVIOUSPOINTS) nsamples = MAXPREVIOUSPOINTS;
            double *mean = calloc(PARAMETERS, sizeof(double));
            int total_weight = 0;
            for (int n = chainSize - nsamples; n < chainSize; n++) {
                for (int i = 0; i < PARAMETERS; i++) mean[i] += chain[n].f[i];
                total_weight++;
            }
            for (int i = 0; i < PARAMETERS; i++) mean[i] /= total_weight;
            for (int i = 0; i < PARAMETERS; i++)
                for (int j = 0; j < PARAMETERS; j++)
                    cov[i][j] = 0.0;
            for (int n = chainSize - nsamples; n < chainSize; n++) {
                for (int i = 0; i < PARAMETERS; i++)
                    for (int j = 0; j < PARAMETERS; j++)
                        cov[i][j] += (chain[n].f[i] - mean[i]) * (chain[n].f[j] - mean[j]);
            }
            for (int i = 0; i < PARAMETERS; i++)
                for (int j = 0; j < PARAMETERS; j++)
                    cov[i][j] /= (total_weight - 1);
            generateEigenvectors(mg, cov, EntireFactor * EntireFactor);
            free(mean);
            steps_since_update = 0;
            printf("Step %d: Covariance updated. Factor = %.4f, Acc rate = %.2f\n", step, EntireFactor, acc_rate);
            fflush(stdout);
        }

        /* Emulator training & retraining */
        if (!initially_trained && step >= burnin && emulator_buffer_size(g_emulator) >= emu_cfg.min_train_points && g_emulator) {
            printf("Initial training at step %d with %d points...\n", step, emulator_buffer_size(g_emulator));
            if (g_log_file) {
                fprintf(g_log_file, "TRAIN_INIT %d %d\n", step, emulator_buffer_size(g_emulator));
                fflush(g_log_file);
            }
            emulator_train(g_emulator);
            initially_trained = 1;
        }

        if (initially_trained && step >= emu_cfg.use_emulator_after && emulator_is_ready(g_emulator)) {
            g_use_emulator = 1;
        }

        // Retraining is now handled internally by emulator_update() via pca_interval/gp_interval

        /* Write to output file */
        fprintf(chain_file, "%d", step);
        for (int i = 0; i < PARAMETERS; i++) fprintf(chain_file, " %e", current[i]);
        fprintf(chain_file, " %e\n", current_loglike);
        fflush(chain_file);

        if (step % 10 == 0) {
            printf("Step %d, logL = %.2f, factor = %.4f, acc = %d, acc_rate = %.2f, rej = %d, emu=%d\n",
                   step, current_loglike, EntireFactor, accepted, acc_rate, consecutive_rejections, g_use_emulator);
            fflush(stdout);
        }
    }

done:
    fclose(chain_file);
    if (g_log_file) fclose(g_log_file);
    if (g_spec_file) fclose(g_spec_file);
    printf("Chain written to chain_serial_1params6poly.txt\n");
    printf("Final log-likelihood: %.2f\n", current_loglike);
    printf("Final step factor: %.4f\n", EntireFactor);
    fflush(stdout);

    for (int i = 0; i < PARAMETERS; i++) free(cov[i]);
    free(cov);
    free(lowbound); free(highbound); free(initial_sigma);
    free(current);
    free(chain);
    if (g_emulator) emulator_free(g_emulator);
    return 0;
}
