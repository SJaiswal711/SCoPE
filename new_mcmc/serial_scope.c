/*
 * serial_scope.c - Serial MCMC version of SCoPE (single chain, no MPI)
 * step size adaptation, bounds checking, initial scale.
 * ReallyInvestigated counter, stdout flushing, escape mechanism,
 *        acceptance-rate-based step size control.
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

/* Step size adaptation  */
static double INCREASE_STEP = 1.02;     /* Gentle increase */
static double DECREASE_STEP = 0.98;     /* Gentle decrease */
static double ACCEPT_TARGET_LOW = 0.23;  /* Target acceptance rate lower bound */
static double ACCEPT_TARGET_HIGH = 0.40; /* Target acceptance rate upper bound */
static int ADAPTATION_INTERVAL = 10;     /* Adapt every N steps */

/* Maximum step factor */
#define MAX_STEP_FACTOR 1.0
#define MIN_STEP_FACTOR 0.1

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

/* ---------- rolling average functions ---------- */
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

/* ---------- likelihood functions ---------- */
double compute_loglike(double *params)
{
    double cl_tt[2601] = {0}, cl_te[2601] = {0}, cl_ee[2601] = {0}, cl_bb[2601] = {0};
    if (!param_iface(0, params, cl_tt, cl_te, cl_ee, cl_bb))
        return -1e100;
    double chi2 = run_plc(0, params, cl_tt, cl_te, cl_ee, cl_bb);
    return -0.5 * chi2;
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

void setVariables()
{
    if (!load_config("param.ini", &global_config)) {
        fprintf(stderr, "Failed to load param.ini\n");
        exit(1);
    }
    int total_estimated = 0;
    for (int i = 0; i < global_config.param_count; i++)
        if (global_config.params[i].is_estimated)
            total_estimated++;
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

/* ---------- main program ---------- */
int main(int argc, char **argv)
{
    /* Disable stdout buffering for immediate output */
    setvbuf(stdout, NULL, _IONBF, 0);
    
    setVariables();
    srand(time(NULL));

    /* Allocate and initialize bounds */
    double *lowbound = malloc(PARAMETERS * sizeof(double));
    double *highbound = malloc(PARAMETERS * sizeof(double));
    double *initial_sigma = malloc(PARAMETERS * sizeof(double));
    get_parameter_bounds_from_config(lowbound, highbound, initial_sigma);

    double *current = malloc(PARAMETERS * sizeof(double));
    for (int i = 0; i < PARAMETERS; i++)
        current[i] = (lowbound[i] + highbound[i]) * 0.5;

    /* Initialize proposal distribution */
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

    /* Initialize chain */
    Task *chain = malloc(MAXCHAINLENGTH * sizeof(Task));
    int chainSize = 0, chainBack = 0;

    double current_loglike = compute_loglike(current);
    printf("Initial log-likelihood: %f\n", current_loglike);
    fflush(stdout);

    Task t;
    for (int i = 0; i < PARAMETERS; i++) t.f[i] = current[i];
    t.f[LOGLIKEPOS] = current_loglike;
    t.Multiplicity = 1;
    t.ReallyInvestigated = 0;
    chain[chainBack++] = t;
    chainSize++;

    /* Open output file */
    FILE *chain_file = fopen("chain_serial.txt", "w");
    fprintf(chain_file, "# step");
    for (int i = 0; i < PARAMETERS; i++)
        fprintf(chain_file, " param_%d", i);
    fprintf(chain_file, " loglike\n");

    RollingAverage *roll = new_RollingAverage(500);
    int steps_since_update = 0;
    double EntireFactor = step_factor;

    int burnin = 200;
    int max_steps = 100000;
    int consecutive_rejections = 0;
    int stuck_counter = 0;
    int adapt_counter = 0;
    
    /* Rolling acceptance rate tracking (last 100 steps) */
    int last_100_accepted[100] = {0};
    int circ_idx = 0;

    printf("Starting MCMC loop with max_steps = %d\n", max_steps);
    printf("Target acceptance rate: %.0f-%.0f%%\n", ACCEPT_TARGET_LOW*100, ACCEPT_TARGET_HIGH*100);
    fflush(stdout);

    /* Main MCMC loop */
    for (int step = 0; step < max_steps; step++) {
        Task next;
        int accepted = 0;
        int n_tries = 0;
        
        /* Propose and evaluate */
        do {
            if (!throwDice(chain[chainBack-1], &next, mg, EntireFactor)) {
                n_tries++;
                if (n_tries > 200) {
                    if (step % 100 == 0)
                        printf("Step %d: proposal generation failed after 200 tries\n", step);
                    break;
                }
                continue;
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
                if (chainBack >= MAXCHAINLENGTH) {
                    printf("Chain length limit reached.\n");
                    goto done;
                }
            }
            break;
        } while (1);
        
        /* Update investigation counter */
        chain[chainBack-1].ReallyInvestigated++;
        
        /* Track consecutive rejections */
        if (!accepted) {
            consecutive_rejections++;
        } else {
            consecutive_rejections = 0;
        }
        
        /* Update rolling acceptance rate */
        last_100_accepted[circ_idx++] = accepted;
        if (circ_idx >= 100) circ_idx = 0;
        double acc_rate = 0.0;
        for (int i = 0; i < 100; i++) acc_rate += last_100_accepted[i];
        acc_rate /= 100.0;
        
        /* Stuck detection */
        static double last_logL = 0;
        if (step > burnin && fabs(current_loglike - last_logL) < 0.1) {
            stuck_counter++;
        } else {
            stuck_counter = 0;
            last_logL = current_loglike;
        }
        
        if (stuck_counter > 500 && EntireFactor > MIN_STEP_FACTOR) {
            printf("Step %d: WARNING - Chain stuck at logL = %f. Resetting step factor.\n", 
                   step, current_loglike);
            EntireFactor = 0.3;
            generateEigenvectors(mg, cov, EntireFactor * EntireFactor);
            stuck_counter = 0;
        }
        
        /* Step size adaptation using acceptance rate (every ADAPTATION_INTERVAL steps) */
        if (step >= burnin && ADAPTIVE) {
            adapt_counter++;
            if (adapt_counter >= ADAPTATION_INTERVAL) {
                adapt_counter = 0;
                
                Rolling_Average_push(roll, EntireFactor);
                
                if (acc_rate > ACCEPT_TARGET_HIGH && EntireFactor < MAX_STEP_FACTOR) {
                    EntireFactor *= INCREASE_STEP;
                    if (EntireFactor > MAX_STEP_FACTOR) EntireFactor = MAX_STEP_FACTOR;
                    generateEigenvectors(mg, cov, EntireFactor * EntireFactor);
                    printf("Step %d: Acc rate %.2f -> increasing factor to %.4f\n", 
                           step, acc_rate, EntireFactor);
                    fflush(stdout);
                } 
                else if (acc_rate < ACCEPT_TARGET_LOW && EntireFactor > MIN_STEP_FACTOR) {
                    EntireFactor *= DECREASE_STEP;
                    if (EntireFactor < MIN_STEP_FACTOR) EntireFactor = MIN_STEP_FACTOR;
                    generateEigenvectors(mg, cov, EntireFactor * EntireFactor);
                    printf("Step %d: Acc rate %.2f -> decreasing factor to %.4f\n", 
                           step, acc_rate, EntireFactor);
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
            
            printf("Step %d: Covariance updated. Factor = %.4f, Acc rate = %.2f\n", 
                   step, EntireFactor, acc_rate);
            fflush(stdout);
        }
        
        /* Write to output file */
        fprintf(chain_file, "%d", step);
        for (int i = 0; i < PARAMETERS; i++)
            fprintf(chain_file, " %e", current[i]);
        fprintf(chain_file, " %e\n", current_loglike);
        fflush(chain_file);
        
        /* Progress output */
        if (step % 10 == 0) {
            printf("Step %d, logL = %.2f, factor = %.4f, acc = %d, acc_rate = %.2f, rej = %d\n", 
                   step, current_loglike, EntireFactor, accepted, acc_rate, consecutive_rejections);
            fflush(stdout);
        }
    }

done:
    fclose(chain_file);
    printf("Chain written to chain_serial.txt\n");
    printf("Final log-likelihood: %.2f\n", current_loglike);
    printf("Final step factor: %.4f\n", EntireFactor);
    fflush(stdout);

    /* Cleanup */
    for (int i = 0; i < PARAMETERS; i++) free(cov[i]);
    free(cov);
    free(lowbound); free(highbound); free(initial_sigma);
    free(current);
    free(chain);
    
    return 0;
}
