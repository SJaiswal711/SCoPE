// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// //                                                                     Version includes timing, fast_mode, dragging method and emulator
// //                                                                     with multiple slaves per chain (SLAVEPARCHAIN)

// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <mpi.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>
#include "param_config_single.h"
#include "newpar_new.h"
double get_time(void);
#include "nrunplc_new0.h"

/* ======================= Emulator includes ======================= */
#include "emulator/include/emulator.h"

/* ======================= Emulator selection ======================= */
#define EMULATOR_TYPE 0   /* 0 = GP, 1 = Polynomial */
#define POLY_DEGREE 2     /* only used if EMULATOR_TYPE == 1 */
#define FALLBACK_THRESHOLD 0.05

// ============================ Runtime flags ============================
int USE_DR = 1;          /* 1 = enable delayed rejection, 0 = disable */
int USE_EMULATOR = 1;    /* 1 = use emulator when available, 0 = always fallback to CAMB */

/* Emulator instance and helper variables (per slave) */
static Emulator *g_emulator = NULL;
static int local_step = 0;          /* Counter for true evaluations (buffer updates) */
static int g_n_cosmo = 0;           /* Number of cosmological (slow) parameters */
static int mapping_done = 0;        /* Ensure we initialise emulator only once */

/* Output directory (broadcast to all ranks) */
static char outdir_buf[512];

// CMBAns header files
// ---------------------------------------------------------------------

#define NOPARAM 20             // The number of parameters in our model
#define MAX_TASKARRAY_SIZE 150 // this is the maximum task array size
#define NR_END 1              // Numerical Recipr function
#define FREE_ARG char *

// The number of chains; 1-9 chains are supported, if you want more,
// you'll have to slightly modify the code. No big problem, though.
//----------------------------------------------------------------
#define CHAINS 2
static short int Astier = 0;
short int PARAMETERS = NOPARAM; // The number of parameters in our model

double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + 1e-9 * ts.tv_nsec;
}

/* =========================================================
   DRAGGING METHOD ADDITIONS
   ========================================================= */
#define TAG_UPDATE_COV 400     // master -> all slaves: fast-covariance broadcast
#define N_DRAG 3               // fast intermediate drag steps
#define MAX_FAST 20            // safety cap for fast-parameter covariance buffer
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

// Global flag for fast dragging (now user‑controllable)
int USE_DRAGGING = 1;   // default: ON

// Global configuration
Config global_config;

// The size of the array, i.e. all possible values
// (parameters, likelihoods, derived parameters) that we wish
// to write to a file. Automatically set by the program
//-------------------------------------------------------------------------------------
unsigned short int TASKARRAY_SIZE;
unsigned short int MULTIPURPOSEPOS; // will be set automatically
unsigned short int TAKEPOS, PROBPOS;
unsigned short int ADAPTIVEPOS;
unsigned int LOGLIKEPOS;
int RANDPOS;
unsigned short int DRAGPOS;
unsigned short int FASTFACTORPOS;
int STEP_POS;                     // Position for MCMC step number

// Other information (derived parameters such as sigma 8 etc) can be stored from this
// position on.  MULTIPURPOSE_REQUIRED is the number of variables that you can
// store in the chain on top of the parameters and likelihoods.
//-------------------------------------------------------------------------------------
static unsigned int MULTIPURPOSE_REQUIRED = 1;

// SDSS Flag
//-------------------------------------------------------------------------------------
static short int SDSS = 0;
static short int TwodF = 0;
static short int SDSS_BAO = 0;     // Baryon acoustic peak
static short int SDSSLRG = 0;      // Luminous reg galaxies
static short int LYA_MCDONALD = 0; // Lyman alpha code of Pat McDonald

// Other Flags
//-------------------------------------------------------------------------------------
static short int WMAP7 = 1;       // WMAP 3-year yes or no ?
static short int BOOMERANG03 = 0; // Boomerang from 2003 flight
static short int VSA = 0;
static short int ACBAR = 0;
static short int CBI = 0;
static short int Riess06 = 0;

// Needed to keep track of parameters needed for experiments
//-------------------------------------------------------------------------------------
short int AstierParameterPos = 0;
short int WMAP5ParameterPos = 0;
short int SDSSLRGParameterPos = 0;

// Start using the estimated covariance matrix for steps after BEGINCOVUPDATE points have
// been computed in a chain
//-------------------------------------------------------------------------------------
static unsigned int BEGINCOVUPDATE = 120;
static unsigned int MAXCHAINLENGTH = 50000;

// Some constants needed for multiprocessor application
//-------------------------------------------------------------------------------------
static int GIVEMETASK = 1; // < MPI flag indicating slave()'s wish to receive a task from master()
static int TAKERESULT = 2; // < MPI flag indicating slave()'s request to master() to take result
static int TAKETASK = 3;   // < MPI flag indicating master()'s request to slave to take task.

// ============================ ADDED: TAG_STOP ============================
#define TAG_STOP 4   // master -> slave: stop signal
// =========================================================================

// SLAVEPARCHAIN: number of slaves per chain
int SLAVEPARCHAIN = 3;


// ===============================================================================

// For the adaptive stepsize algoritm, variable step length; these numbers are
// rather arbitrary, but seem to work well
// Please note, that after FREEZE_IN, the covariance matrix without the
// step factor will be used. Hence, the average stay at a certain parameter point
// is approx 3
//-------------------------------------------------------------------------------------
static double INCREASE_STEP = 1.15; // 1.15
static double DECREASE_STEP = 0.9;  // 0.9
static int HIGH_STEP_BOUND = 5;
static int LOW_STEP_BOUND = 3;

// For the covariance updater; recompute covariance matrix after UPDATE_TIME steps
const int UPDATE_TIME = 5;

// For the adaptive stepsize: Do not compute covariance Matrix with more than MAXPREVIOUSPOINTS
const unsigned int MAXPREVIOUSPOINTS = 5000;

// Minimum number of points for R-statistic calculation
const unsigned int RMIN_POINTS = 120; // minimum number of points for R-statistic calculation

// only if FREEZE_IN=true: If R-statistic for all parameters less than RBREAK and
// number of points more than MIN_SIZE_FOR_FREEZE_IN then freeze-in.
// If you want to be really conservative, you can set RBREAK=1.1
//---------------------------------------------------------------------------------------
const double RBREAK = 1.2;
const unsigned int MIN_SIZE_FOR_FREEZE_IN = 500;

extern void test_likelihood_(int *inputflag, double *cl_in_tt, double *cl_in_te, double *cl_in_ee, double *cl_in_bb);

double drfactor;
int clik_process_initialized = 0;

// A task is just a double array. The benefit of having
// this structure anyhow is that we can very easily store tasks
// in lists and all that.
// Each double variable corresponds to either a parameter,
// a likelihood, some multipurpose stuff or just no information
// at all.
// I find it convenient to leave [0..7] for parameters
// [8..14] is for likelihoods and after that come multipurpose
// stuff. If you only need 4 parameters, there will be garbage
// in [5..7], but as you don't have to look at it, who cares, if
// you don't ?
//---------------------------------------------------------------------------------------

typedef struct Task
{                               // The weigh of this point in the chain,
  double f[MAX_TASKARRAY_SIZE]; // i.e. how long does it rest at the point
  int Multiplicity;             //
                                // The number of times a different model has been
                                // simulated until the step has been taken.
                                // This is Multiplicity minus the number of times the
                                // step proposal crossed the boundaries of parameter Space.
                                // We take this number as a meassure to increase or
  int ReallyInvestigated;       // decrease step sizes
} Task;                         //-----------------------------------------------------------

Task *free_Task()
{
  Task *tsk;
  tsk = (Task *)malloc(sizeof(Task));
  for (unsigned int i = 0; i < MAX_TASKARRAY_SIZE; i++)
    tsk->f[i] = 0.0;
  tsk->Multiplicity = 0;
  tsk->ReallyInvestigated = 0;
  return tsk;
}

void Task_copy(Task *tska, Task *tskb)
{
  for (unsigned int i = 0; i < MAX_TASKARRAY_SIZE; i++)
    tska->f[i] = tskb->f[i];
  tska->Multiplicity = tskb->Multiplicity;
  tska->ReallyInvestigated = tskb->ReallyInvestigated;
}

// Small class to compute a "rolling" average over values.
// The constructer takes as an argument the maximum number
// of values. By calling push(x), the value x will be stored and
// if the number of values exceeds the limit size, the one stored
// earliest will be erased. Mathematically speaking:
//
// average = (\sum_0^size x_i ) / min(size,#values stored)
// and at each push(), x_i = x_i+1
//
// The average is updated at each call to average() by updating
// the sum of all values. Each 100 calls to average(), it is computed
// from scratch to limit elimination of significant figures
// -------------------------------------------------------------------------------------------

typedef struct RollingAverage
{
  unsigned int Size;      // the maximum number of values
  unsigned int Number;    // the number of values
  double Sum;             // the sum of all values
  double *y;              // the values
  unsigned int idx;       // current position for push()
  int PerformFullCounter; // Every 100 calls to average(),
                          // we re-compute the average from scratch
} RollingAverage;

void Rolling_Average_push(RollingAverage *RA, double x) // push x into the average-array
{
  RA->Sum -= RA->y[RA->idx]; // subtract the value we will overwrite
  RA->Sum += x;
  RA->y[RA->idx++] = x; // store the new value
  if (RA->idx == RA->Size)
    RA->idx = 0; // roll-over
  RA->Number++;
  if (RA->Number >= RA->Size)
    RA->Number = RA->Size; // at most we have Size numbers in the array
}

void RollingAverage_clear(); // clear the array

double RollingAverage_average(RollingAverage *RA) // compute the average
{
  if (RA->Number == 0)
  {
    return 0;
  }
  if (RA->PerformFullCounter++ < 10)
    return RA->Sum / RA->Number;

  RA->PerformFullCounter = 0; // reset
  double Sum = 0.0;
  for (unsigned int i = 0; i < RA->Number; i++)
  { // at most to number, which might later be size
    Sum += RA->y[i];
  }
  return RA->Sum / RA->Number;
}

RollingAverage *new_RollingAverage(int Size)
{
  RollingAverage *RA;
  RA = (RollingAverage *)malloc(sizeof(RollingAverage));
  RA->y = (double *)malloc(Size * sizeof(double));
  RA->Size = Size;
  RA->idx = 0;
  if (!RA)
  {
    printf("Error in allocating Rolling average.");
    exit(1);
  }
  return RA;
}

typedef struct MultiGaussian
{
  double Scale; // bool Lock;
                // void generateAndTransform();
                // The row number of the covariance matrix and number of random variables sought.
  unsigned int SIZE;
  double **MasterMatrix;   // Encodes the eigenvectors/transformation matrix
  double *eigenvalues;     // The eigenvalues
  double *generatedValues; // The random values, generated anew if throwDice() is called

  double *lbounds; // Upper and lower bounds are encoded here
  double *hbounds;

  double *randomq;

  double *center; // The mean of the gaussian to sample from
} MultiGaussian;

MultiGaussian *new_MultiGaussian(int Size)
{
  MultiGaussian *MG;
  MG = (MultiGaussian *)malloc(sizeof(MultiGaussian));
  MG->SIZE = Size;
  MG->MasterMatrix = (double **)malloc(Size * sizeof(double *));
  for (unsigned int i = 0; i < Size; i++)
    MG->MasterMatrix[i] = (double *)malloc(Size * sizeof(double));

  MG->eigenvalues = (double *)malloc(Size * sizeof(double));
  MG->generatedValues = (double *)malloc(Size * sizeof(double));
  MG->lbounds = (double *)malloc(Size * sizeof(double));
  MG->hbounds = (double *)malloc(Size * sizeof(double));
  MG->center = (double *)malloc(Size * sizeof(double));
  MG->randomq = (double *)malloc(Size * sizeof(double));
  return MG;
}

// Set bounds for the random variables. This needs to be called before any other
// function, or unpredictive behaviour will result. In most applications, you will
// probably set this only once.
// \param  lowBound a vector with lower Bounds for the random variables
// \param highBound a vector with upper Bounds for the random variables
//-----------------------------------------------------------------------------------------------------
void MultiGaussian_setBounds(MultiGaussian *MG, double lowBound[], double highBound[])
{
  for (unsigned int i = 0; i < MG->SIZE; i++)
  {
    MG->lbounds[i] = lowBound[i];
    MG->hbounds[i] = highBound[i];
  }
}

void printInfo(MultiGaussian *MG)
{
  printf("************************************\n");
  printf("MultiGaussian::printInfo:\n");
  printf("Eigenvalues (we used scale = %f):\n", MG->Scale);
  for (unsigned int i = 0; i < MG->SIZE; i++)
  {
    printf("%e  --> sigma:  %e  ", MG->eigenvalues[i], sqrt(MG->eigenvalues[i]));
    printf("eigenvector w/o scale: %e  ", MG->eigenvalues[i] / MG->Scale);
    printf("  --> sigma: %e\n", sqrt(MG->eigenvalues[i] / MG->Scale));
  }

  printf("\nEigenvectors:\n");
  for (unsigned int i = 0; i < MG->SIZE; i++)
  {
    for (unsigned int j = 0; j < MG->SIZE; j++)
    {
      printf("%e  ", MG->MasterMatrix[i][j]);
    }
    printf("\n");
  }
  printf("***********************************\n");
}

void nrerror(char s[])
{
  printf("%s", s);
  exit(1);
}

double **convert_matrix(double *a, long nrl, long nrh, long ncl, long nch)
// allocate a float matrix m[nrl..nrh][ncl..nch] that points to the matrix
// declared in the standard C manner as a[nrow][ncol], where nrow=nrh-nrl+1
// and ncol=nch-ncl+1. The routine should be called with the address
// &a[0][0] as the first argument.
{
  long i, j, nrow = nrh - nrl + 1, ncol = nch - ncl + 1;
  double **m;

  // allocate pointers to rows
  m = (double **)malloc((unsigned int)((nrow + NR_END) * sizeof(double *)));
  if (!m)
    nrerror("allocation failure in convert_matrix()");
  m += NR_END;
  m -= nrl;

  // set pointers to rows
  m[nrl] = a - ncl;
  for (i = 1, j = nrl + 1; i < nrow; i++, j++)
    m[j] = m[j - 1] + ncol;
  // return pointer to array of pointers to rows
  return m;
}

void free_convert_matrix(double **b, long nrl, long nrh, long ncl, long nch)
{
  free((FREE_ARG)(b + nrl - NR_END)); // free a matrix allocated by convert_matrix()
}

double *vector(long nl, long nh) // allocate a float vector with subscript range v[nl..nh]
{
  double *v;

  v = (double *)malloc((unsigned int)((nh - nl + 1 + NR_END) * sizeof(double)));
  if (!v)
    nrerror("allocation failure in vector()");
  return v - nl + NR_END;
}

void free_vector(double *v, long nl, long nh)
{ // free a float vector allocated with vector()
  free((FREE_ARG)(v + nl - NR_END));
}

double *new_double(unsigned short int n)
{
  double *v;
  v = (double *)malloc(n * sizeof(double));
  return v;
}

double fabs(double take)
{
  if (take < 0)
    return -take;
  return take;
}

double posRnd(double max)
{
  double rand_max = 1.0 / RAND_MAX;

  return rand() * rand_max * max;
}

double ran1(float x)
{
  double rand_max = 1.0 / RAND_MAX;
  return 2 * (rand() * rand_max - 0.5) * x;
}

double gasdev(double mean, double std)
{
  double rsq, v1, v2;
  do
  {
    v1 = ran1(1.0);
    v2 = ran1(1.0);
    rsq = v1 * v1 + v2 * v2;
  } while (rsq >= 1.0 || rsq == 0.0);

  double fac = v1 * sqrt(-2.0 * log(rsq) / rsq);
  return fac * std + mean;
}

void generateRandom(MultiGaussian *MG)
{
  double *y;
  y = (double *)malloc(MG->SIZE * sizeof(double));

  for (unsigned int i = 0; i < MG->SIZE; i++)
    y[i] = gasdev(0.0, sqrt(MG->eigenvalues[i]));

  for (unsigned int i = 0; i < MG->SIZE; i++)
  {
    MG->randomq[i] = y[i];
    MG->generatedValues[i] = 0;
    for (unsigned int j = 0; j < MG->SIZE; j++)
    {
      MG->generatedValues[i] += MG->MasterMatrix[i][j] * y[j];
    }
  }
}

int throwDice(Task chain, Task *next, MultiGaussian *MG, int scale)
{
  for (unsigned int i = 0; i < MG->SIZE; i++)
  {
    MG->center[i] = chain.f[i];
  }

  generateRandom(MG);

  for (unsigned int i = 0; i < PARAMETERS; i++)
  {
    next->f[i] = MG->generatedValues[i] * pow(drfactor, scale) + MG->center[i];
    next->f[RANDPOS + i] = MG->randomq[i] * pow(drfactor, scale);
  }

  for (unsigned int i = 0; i < MG->SIZE; i++)
  {
    if (MG->generatedValues[i] * pow(drfactor, scale) + MG->center[i] < MG->lbounds[i])
      return 0;
    if (MG->generatedValues[i] * pow(drfactor, scale) + MG->center[i] > MG->hbounds[i])
      return 0;
  }

  return 1;
}

// Generate a set of eigenvectors and eigenvalues from the covariance matrix.
// Thus, if you whish to sample from a new distribution with covariance matrix S,
// you need to call this function again. This function uses the eigenvalue and
// eigenvector routines of the GNU Scientific Library, so if you need this for
// large matrices it is probably best to change this code and use LAPACK or
// something similar.
// param covarianceMatrix the covariance matrix of the distribution you want to draw from
// -------------------------------------------------------------------------------------------------------------

void generateEigenvectors(MultiGaussian *MG, double **covarianceMatrix, double scale)
{
  MG->Scale = scale;
  double *data;
  data = (double *)malloc((MG->SIZE) * (MG->SIZE) * sizeof(double)); // reformat [][] into [] for creating matrix
  int k = 0;
  for (unsigned int j = 0; j < MG->SIZE; j++) // copy into tempArray
    for (unsigned int i = 0; i < MG->SIZE; i++)
    {
      data[k] = covarianceMatrix[i][j];
      k++;
    }

  gsl_matrix_view m = gsl_matrix_view_array(data, MG->SIZE, MG->SIZE);
  gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(MG->SIZE);

  gsl_vector *eval = gsl_vector_alloc(MG->SIZE);
  gsl_matrix *evec = gsl_matrix_alloc(MG->SIZE, MG->SIZE);

  gsl_eigen_symmv(&m.matrix, eval, evec, w);

  gsl_eigen_symmv_free(w);
  gsl_eigen_symmv_sort(eval, evec, GSL_EIGEN_SORT_ABS_ASC);

  for (unsigned short int i = 0; i < MG->SIZE; i++)
  {
    MG->eigenvalues[i] = gsl_vector_get(eval, i);
    for (unsigned short int j = 0; j < MG->SIZE; j++)
      MG->MasterMatrix[i][j] = gsl_matrix_get(evec, i, j);
  }

  for (unsigned short int i = 0; i < MG->SIZE; i++)
    MG->eigenvalues[i] *= scale;

  gsl_vector_free(eval);
  gsl_matrix_free(evec);
  free(data);
}

unsigned short int ADAPTIVE = 1;
unsigned short int FREEZE_IN = 1;

void printExtInfo(MultiGaussian *MG)
{
  printf("\n******************************************************************************\n");
  printf("EXTENDED information :\n");
  printf("Size : %d\n", MG->SIZE);
  printf("Bounds:\n");
  for (unsigned int i = 0; i < MG->SIZE; i++)
    printf("%d) %e  %e  %e\n", i, MG->lbounds[i], MG->hbounds[i], MG->generatedValues[i]);

  printf("\n******************************************************************************\n");
  printf("\n\n");
  for (unsigned int i = 0; i < MG->SIZE; i++)
  {
    printf("%e  :  ", MG->eigenvalues[i]);
    for (unsigned int j = 0; j < MG->SIZE; j++)
      printf("  %.2e", MG->MasterMatrix[i][j]);
    printf("\n");
  }
  printf("\n******************************************************************************\n");
}


double normd(double a[], int size)
{
  double norm = 0.0;
  for (int ii = 0; ii < size; ii++)
    norm += a[ii] * a[ii];
  return norm;
}
// Get parameter bounds from config - for all non-nuisance parameters
// Get parameter bounds from config - for ALL estimated parameters (Cosmo + Nuisance)
void get_parameter_bounds_from_config(double *lowbound, double *highbound, double *initial_sigma) {
    int param_index = 0;
    
    // We loop through the config and take EVERYTHING that is marked for estimation
    for(int i = 0; i < global_config.param_count && param_index < PARAMETERS; i++) {
        if(global_config.params[i].is_estimated) {
            
            // 1. Copy bounds from config
            lowbound[param_index] = global_config.params[i].lower_bound;
            highbound[param_index] = global_config.params[i].upper_bound;
            initial_sigma[param_index] = global_config.params[i].sigma;
            
            printf("Chain Param %d: %s [%.3f, %.3f]\n", 
                   param_index, 
                   global_config.params[i].name, 
                   lowbound[param_index], 
                   highbound[param_index]);
            
            param_index++;
        }
    }
    
    // Safety: Fill defaults if for some reason we have fewer config params than PARAMETERS
    for(; param_index < PARAMETERS; param_index++) {
        lowbound[param_index] = 0.0;
        highbound[param_index] = 1.0;
        initial_sigma[param_index] = 0.1;
        printf("WARNING: Using default bounds for missing param index %d\n", param_index);
    }
}
// The master is responsible for sending tasks to the slaves and collecting
// the information from the slaves after the calculation. In other words,
// this is the coordinator. The slaves don't know about each others existence.
// The master also monitors the output. There is only one master, but an
// arbitrary number of slaves.
//-------------------------------------------------------------------------------------

// Helper functions for multiple slaves per chain
int myslave_rank(int ii) { return ii * SLAVEPARCHAIN + 1; } // first slave of chain ii (middleman)
int middleman(int rank) {
    if (rank % SLAVEPARCHAIN == 1) return 1;
    return 0;
}
int mymiddlemann(int rank) {
    return ((rank - 1) / SLAVEPARCHAIN) * SLAVEPARCHAIN + 1;
}

//generating 
int throwSlowDice(Task chain_pt, Task *next, MultiGaussian *MGs, int scale) {
    for (unsigned int k = 0; k < MGs->SIZE; k++)
        MGs->center[k] = chain_pt.f[slow_idx[k]];

    generateRandom(MGs);

    for (unsigned int i = 0; i < PARAMETERS; i++)
        next->f[i] = chain_pt.f[i];

    for (unsigned int k = 0; k < MGs->SIZE; k++) {
        int idx = slow_idx[k];
        next->f[idx] = MGs->generatedValues[k] * pow(drfactor, scale) + MGs->center[k];
        if (next->f[idx] < MGs->lbounds[k] || next->f[idx] > MGs->hbounds[k])
            return 0;
    }
    return 1;
}

// sending the fast covariance matrix to all slaves
void broadcast_fast_covariance(double **cov_full, int worldsize) {
    if (N_FAST == 0) return;
    static MultiGaussian *MGfast = NULL;
    if (!MGfast) MGfast = new_MultiGaussian(N_FAST);

    double **covFast = (double**)malloc(N_FAST * sizeof(double*));
    for (int k = 0; k < N_FAST; k++) {
        covFast[k] = (double*)malloc(N_FAST * sizeof(double));
        for (int j = 0; j < N_FAST; j++)
            covFast[k][j] = cov_full[fast_idx[k]][fast_idx[j]];
    }
    generateEigenvectors(MGfast, covFast, 1.0);  
    
    static double packed[MAX_FAST_PACKED];
    packed[0] = (double)N_FAST;
    for (int k = 0; k < N_FAST; k++) packed[1+k] = MGfast->eigenvalues[k];
    int p = 1 + N_FAST;
    for (int i = 0; i < N_FAST; i++)
        for (int j = 0; j < N_FAST; j++)
            packed[p++] = MGfast->MasterMatrix[i][j];

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
        for (int j = 0; j < N_FAST; j++)
            fast_evec[i][j] = buf[p++];
    fast_cov_ready = 1;
}

/* =========================================================
   MODIFIED MASTER: added proposal limit, graceful exit, adaptive step reduction
   ========================================================= */
int master(double **startnum, unsigned short int Restart)
{
  printf("\nStarting master with %d chains and max chain length %d", CHAINS, MAXCHAINLENGTH);

  // <-- ADDED: total_proposals counter
  int total_proposals = 0;

  unsigned int chainSize[CHAINS];           // chain size
  unsigned int chainTotalPerformed[CHAINS]; // total number of performed models
  unsigned int chainBack[CHAINS] = {0};     // int count[CHAINS];
  unsigned int checkSize = 0;               // Last time when GR statistics was calculated
  
  Task **chain = (Task**)malloc(CHAINS * sizeof(Task*));
  for (int c = 0; c < CHAINS; c++) {
      chain[c] = (Task*)malloc(MAXCHAINLENGTH * sizeof(Task));
  }

  // For dynamical stepsize
  int steps_since_update[CHAINS] = {0};
  double EntireFactor[CHAINS];
  RollingAverage *roll[CHAINS]; // rolling average of EntireFactor for after burn-in

  double *sigma[CHAINS];
  double **covMatrix[CHAINS];
  for (unsigned int ii = 0; ii < CHAINS; ii++)
  {
    sigma[ii] = new_double(TASKARRAY_SIZE);
    covMatrix[ii] = (double **)malloc(PARAMETERS * sizeof(double *));
    for (unsigned int jj = 0; jj < PARAMETERS; jj++)
      covMatrix[ii][jj] = (double *)malloc(PARAMETERS * sizeof(double));
  }

  Task *next[CHAINS];
  double AverageMultiplicity[CHAINS]; // just for our information

  for (unsigned int i = 0; i < CHAINS; i++) // initialize to 0
    AverageMultiplicity[i] = 0;

  MultiGaussian *mGauss[CHAINS]; // multivariate gaussian random number generator
  MultiGaussian *mGaussSlow[CHAINS];
  double lowboundSlow[NOPARAM], highboundSlow[NOPARAM];

  double *lowbound;
  double *highbound;
  double *initial_sigma;

  unsigned int break_at;
  unsigned int N;

  // now let us calculate the mean of each parameter
  double *y[CHAINS];
  double *dist_y; // distribution mean
  double *B;      // variance between chains
  double *W;      // variance within chains
  double *R;      // monitoring parameter

  double *average; //[PARAMETERS];
  double **cov;    //[PARAMETERS][PARAMETERS];

  lowbound = new_double(PARAMETERS);
  highbound = new_double(PARAMETERS);
  initial_sigma = new_double(PARAMETERS);
  dist_y = new_double(PARAMETERS);
  B = new_double(PARAMETERS);
  W = new_double(PARAMETERS);
  R = new_double(PARAMETERS);
  average = new_double(PARAMETERS);

  for (unsigned int ii = 0; ii < CHAINS; ii++)
    y[ii] = new_double(PARAMETERS);

  cov = (double **)malloc(PARAMETERS * sizeof(double *));
  for (unsigned int ii = 0; ii < PARAMETERS; ii++)
    cov[ii] = new_double(PARAMETERS);

  // set the (flat) priors, i.e. parameter boundaries FROM CONFIG
  //------------------------------------------------------------------
  get_parameter_bounds_from_config(lowbound, highbound, initial_sigma);

  // Must check what is the RandMax defined in your stdlib.h. It must be > 5*10^4
  srand((unsigned)time(NULL)); // Generate a random seed
  
  int intelligentstartnum = 1;
  for (unsigned int i = 0; i < CHAINS; i++)
  {
    roll[i] = new_RollingAverage(500);

    Task t;
    for (unsigned int k = 0; k < TASKARRAY_SIZE; k++)
      t.f[k] = 0; // for safety reasons, initialize (otherwise garbage in first line)

    mGauss[i] = new_MultiGaussian(PARAMETERS);
    MultiGaussian_setBounds(mGauss[i], lowbound, highbound);

    // Generate Starting points (inside the prior)
    //---------------------------------------------------------------------------------------------
    if (!intelligentstartnum)
    {
      for (unsigned int j = 0; j < PARAMETERS; j++)
        t.f[j] = lowbound[j] + posRnd(highbound[j] - lowbound[j]);
    }
    else
    {
      for (unsigned int j = 0; j < PARAMETERS; j++)
        t.f[j] = startnum[i][j];
    }
    for (unsigned int j = 0; j < PARAMETERS; j++) {
        t.f[CURRENTSTATEPOS + j] = t.f[j];
    }
    
    // the sigmas from config
    //----------------------------------------------------------------------------------------------
    for (unsigned int j = 0; j < PARAMETERS; j++) {
        sigma[i][j] = initial_sigma[j];
    }

    for (unsigned int k = 0; k < PARAMETERS; k++) // fill with sigmas
      for (unsigned int j = 0; j < PARAMETERS; j++)
      {
        if (j == k)
          covMatrix[i][j][j] = sigma[i][j] * sigma[i][j];
        else
          covMatrix[i][j][k] = 0.0;
      }

    generateEigenvectors(mGauss[i], covMatrix[i], 1.0);

    steps_since_update[i] = 0;
    EntireFactor[i] = 1.0;

    // Set the true initial loglike calculated by slave0
    //---------------------------------------------------------------------------------------------------------------
    // t.f[LOGLIKEPOS] = -0.5 * startnum[i][PARAMETERS];
    t.f[LOGLIKEPOS] = -1e10;
    t.Multiplicity = 0;
    t.ReallyInvestigated = 0;
    if (Restart == 0)
      chainBack[i]++;

    Task_copy(&chain[i][chainBack[i] - 1], &t);

    chainSize[i] = 0;
    chainTotalPerformed[i] = 0;

    // Send initial point to the middleman slave of chain i
    MPI_Send(t.f, TASKARRAY_SIZE, MPI_DOUBLE, myslave_rank(i), TAKETASK, MPI_COMM_WORLD);

    // Init slow bounds
    for (int k = 0; k < N_SLOW; k++) {
        lowboundSlow[k]  = lowbound[slow_idx[k]];
        highboundSlow[k] = highbound[slow_idx[k]];
    }
    mGaussSlow[i] = new_MultiGaussian(N_SLOW);
    MultiGaussian_setBounds(mGaussSlow[i], lowboundSlow, highboundSlow);

    double **covSlow0 = (double**)malloc(N_SLOW * sizeof(double*));
    for (int k = 0; k < N_SLOW; k++) {
        covSlow0[k] = (double*)calloc(N_SLOW, sizeof(double));
        covSlow0[k][k] = sigma[i][slow_idx[k]] * sigma[i][slow_idx[k]];
    }
    generateEigenvectors(mGaussSlow[i], covSlow0, 1.0);
    for (int k = 0; k < N_SLOW; k++) free(covSlow0[k]);
    free(covSlow0);
  }

  // The chain output will be written to these files
  //------------------------------------------------------------------------------------------------------
  FILE *data[CHAINS];
  FILE *head[CHAINS];         // for information: this is the model currently at the top of the chain
  FILE *investigated[CHAINS]; // all models, whether taken or not taken and their chi^2's

  char montecarlo_name[] = "montecarlo_chain";
  char head_name[] = "head";
  char investigated_name[] = "investigated";
  char path[512];

  // Use outdir_buf for all output files
  for (int k = 1; k <= CHAINS; k++) {
    snprintf(path, sizeof(path), "%s/%s_%d.d", outdir_buf, montecarlo_name, k);
    data[k-1] = fopen(path, "a+");
    snprintf(path, sizeof(path), "%s/%s_%d.d", outdir_buf, head_name, k);
    head[k-1] = fopen(path, "a+");
    snprintf(path, sizeof(path), "%s/%s_%d.d", outdir_buf, investigated_name, k);
    investigated[k-1] = fopen(path, "a+");
  }

  snprintf(path, sizeof(path), "%s/progress.txt", outdir_buf);
  FILE *progress = fopen(path, "a+");
  snprintf(path, sizeof(path), "%s/gelmanRubin.txt", outdir_buf);
  FILE *gelmanRubin = fopen(path, "a+");
  snprintf(path, sizeof(path), "%s/covMatrix.txt", outdir_buf);
  FILE *covarianceMatrices = fopen(path, "a+");

  printf("\n******************************************************\n");

  if (Restart == 1)
    fprintf(progress, "\n\n::::::::::::::::::: RESTARTED FROM HERE ::::::::::::::::::::::::\n\n");

  short int work = 1; // as long as its value is true, the chains will run.
  MPI_Status status;
  double result[MAX_TASKARRAY_SIZE]; // should be larger than TASKARRAY_SIZE

  // Now that we have initialized everything, we go into an
  // eternal loop: We wait for a message from the slaves (either
  // GIMMETASK or TAKERESULT
  // and send out work and receive the   results.
  // The master will also calculate the Gelman-Rubin statistic,
  // which takes up most of it's size
  //-------------------------------------------------------------------------------------------------
  int mult[CHAINS][SLAVEPARCHAIN];
  int tempi = 0;
  int take = 0;
  unsigned int min_size = 100 * 100;
  int loopstop = 0;
  double randq[MAX_TASKARRAY_SIZE], randq1[MAX_TASKARRAY_SIZE], randold;
  int localpeakadjust;
  int global_step = 0;  // MCMC iteration counter (sent to slaves)
  do
  {
    global_step++;  // increment each loop iteration

    MPI_Recv(result, TASKARRAY_SIZE, MPI_DOUBLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

    if (status.MPI_TAG == GIVEMETASK)
    {
      int i = (status.MPI_SOURCE - 1) / SLAVEPARCHAIN; // which chain's middleman sent this?
      // Only enable dragging if both the global flag is ON and the chain has enough points.
      int use_dragging = (USE_DRAGGING && chainSize[i] >= BEGINCOVUPDATE) ? 1 : 0;
      // <-- MODIFIED: if DR off, only one proposal
      int num_proposals = USE_DR ? SLAVEPARCHAIN : 1;
      total_proposals += num_proposals;   // <-- ADDED

      for (int ichains = 0; ichains < num_proposals; ichains++) {
          mult[i][ichains] = 0;
          next[i] = free_Task();
          for (;;)
          {
            if (use_dragging) {
                if (throwSlowDice(chain[i][chainBack[i] - 1], next[i], mGaussSlow[i], ichains)) break;
            } else {
                if (throwDice(chain[i][chainBack[i] - 1], next[i], mGauss[i], ichains)) break; 
            }
            mult[i][ichains]++;
            // <-- ADDED: adaptive step reduction on bounds failures (instead of fatal exit)
            if (mult[i][ichains] > 100) {
                printf("Chain %d: Reducing EntireFactor from %.4f to %.4f (bounds failures: %d)\n",
                       i, EntireFactor[i], EntireFactor[i] * 0.9, mult[i][ichains]);
                EntireFactor[i] *= 0.9;
                generateEigenvectors(mGauss[i], covMatrix[i], EntireFactor[i] * EntireFactor[i]);
                // Also update slow matrix if dragging
                if (use_dragging) {
                    double **covSlow = (double**)malloc(N_SLOW * sizeof(double*));
                    for (int k = 0; k < N_SLOW; k++) {
                        covSlow[k] = (double*)malloc(N_SLOW * sizeof(double));
                        for (int j = 0; j < N_SLOW; j++)
                            covSlow[k][j] = covMatrix[i][slow_idx[k]][slow_idx[j]];
                    }
                    generateEigenvectors(mGaussSlow[i], covSlow, EntireFactor[i] * EntireFactor[i]);
                    for (int k = 0; k < N_SLOW; k++) free(covSlow[k]);
                    free(covSlow);
                }
                mult[i][ichains] = 0;
            }
            if (mult[i][ichains] > 100000) {
                printf("\nFATAL: Too many bounds failures (%d). Aborting.\n", mult[i][ichains]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
          }
          // Attach the literal current state so the slave can compute dragging math
          for (unsigned int k = 0; k < PARAMETERS; k++)
              next[i]->f[CURRENTSTATEPOS + k] = chain[i][chainBack[i] - 1].f[k];

          // Store MCMC step number for slave
          next[i]->f[STEP_POS] = (double)global_step;

          // Set random numbers for the middleman logic (will be used in slave for MH)
          if (ichains == 0) {
              // For middleman, we need to compute the norm of random vector, etc.
              // In the earlier code, they set randq and randold; we can just use posRnd for PROBPOS
              next[i]->f[PROBPOS] = posRnd(1.0);
          } else {
              // For other slaves, we need to set some random numbers for the middleman's MH test
              // We'll set them as before
              next[i]->f[PROBPOS] = posRnd(1.0);
          }
          next[i]->f[ADAPTIVEPOS] = (double)ADAPTIVE;
          next[i]->f[DRAGPOS] = (double)use_dragging;
          next[i]->f[FASTFACTORPOS] = fast_EntireFactor;
          next[i]->f[TAKEPOS] = (double)ADAPTIVE;
          next[i]->f[LOGLIKEPOS] = chain[i][chainBack[i] - 1].f[LOGLIKEPOS];
          next[i]->f[MULTIPURPOSEPOS] = (double)chain[i][chainBack[i] - 1].ReallyInvestigated;

          for (unsigned int k = 0; k < PARAMETERS; k++) {
              fprintf(head[i], "%e\t", next[i]->f[k]);
          }
          fprintf(head[i], "\n");

          // Send to the specific slave
          int dest = myslave_rank(i) + ichains;
          MPI_Send(next[i]->f, TASKARRAY_SIZE, MPI_DOUBLE, dest, TAKETASK, MPI_COMM_WORLD);
      }
    }

    if (status.MPI_TAG == TAKERESULT) 
    {
      int i = (status.MPI_SOURCE - 1) / SLAVEPARCHAIN; // chain index

      // Write to investigated file: parameters + chi2 (result[0..PARAMETERS]) + chain index
      for (unsigned int k = 0; k <= PARAMETERS; k++)
        fprintf(investigated[i], "%e  ", result[k]);
      fprintf(investigated[i], " %d\n", i);

      take = (int)result[TAKEPOS];

      // Always increment by 1 since it's only one slave per chain
      chainTotalPerformed[i] += 1;
      
      // Count chain multiplicity based on rejection counter
      chain[i][chainBack[i] - 1].Multiplicity += (int)result[PROBPOS];

      double loglike1 = chain[i][chainBack[i] - 1].f[LOGLIKEPOS]; // get the Likelihood of the previous parameter set

      if (ADAPTIVE == 1)
      {
        Rolling_Average_push(roll[i], pow(drfactor, (int)result[PROBPOS] - 1)); 
      }

      //
      // we know now whether or not to take the step.
      // ----------------------------------------------------------------------------------------
      if (take == 1)
      {
        // Write chain file: parameters, then logL (twice to match code0), then multiplicity
        for (unsigned int k = 0; k < PARAMETERS; k++)
            fprintf(data[i], "%e  ", chain[i][chainBack[i] - 1].f[k]);
        fprintf(data[i], "%e  %e  ", 
                chain[i][chainBack[i] - 1].f[LOGLIKEPOS],
                chain[i][chainBack[i] - 1].f[LOGLIKEPOS]);
        fprintf(data[i], "%d \n", chain[i][chainBack[i] - 1].Multiplicity);

        // ADAPTIVE: increase stepsize, if we take steps  too often
        if (ADAPTIVE == 1)
          if (chain[i][chainBack[i] - 1].ReallyInvestigated < LOW_STEP_BOUND)
            if (EntireFactor[i] < 3)
            {
              EntireFactor[i] *= INCREASE_STEP;
              printf("\nEntire Factor : %e", EntireFactor[i]);
              generateEigenvectors(mGauss[i], covMatrix[i], EntireFactor[i] * EntireFactor[i]);
            }
        // end of ADAPTIVE increase

        Task t;
        for (unsigned int k = 0; k < TASKARRAY_SIZE; k++)
          chain[i][chainBack[i]].f[k] = result[k];
        chainBack[i]++;
        steps_since_update[i]++; // a taken step contributes towards steps_since_update counting
        chainSize[i]++;

        chain[i][chainBack[i] - 1].Multiplicity       = 1;      // Enhance multiplicity of the old one again
        chain[i][chainBack[i] - 1].ReallyInvestigated = 1; // we did a simulation and we didn't take the step
      }

      else
      {
        chain[i][chainBack[i] - 1].Multiplicity += 1;       // Enhance multiplicity of the old one again
        chain[i][chainBack[i] - 1].ReallyInvestigated += 1; // we did a simulation and we didn't take the step
      }

      //
      // Adaptivly setting stepsize according to covariance matrix
      // Here, we estimate the covariance
      // -----------------------------------------------------------------------------------

      if (ADAPTIVE == 1 && steps_since_update[i] >= UPDATE_TIME && chainSize[i] >= BEGINCOVUPDATE)
      {
        //  drfactor = 0.7;
        int TotalSize = 0;

        //
        // Here, we determine the number of distinct points in the chain
        // that will be taken into account to estimate the covariance matrix
        //------------------------------------------------------------------------------------------
        unsigned int covSize[CHAINS];
        if (chainSize[i] < 2 * BEGINCOVUPDATE) // early on
          covSize[i] = chainSize[i] / 2;
        else
          covSize[i] = chainSize[i] - BEGINCOVUPDATE; // all but the first few begincovupdate points

        covSize[i] = (covSize[i] < MAXPREVIOUSPOINTS) ? covSize[i] : MAXPREVIOUSPOINTS; // at most MAXPREVIOUSPOINTS

        //
        // Initialize
        // --------------------------------------------------------------------------
        for (unsigned int k = 0; k < PARAMETERS; k++)
        {
          average[k] = 0.0;
          for (unsigned int j = 0; j < PARAMETERS; j++)
            cov[k][j] = 0.0;
        }

        for (unsigned int ii = 0; ii < CHAINS; ii++)
        {
          for (unsigned int n = (chainSize[i] - covSize[ii]); n < chainSize[i]; n++)
          {
            for (unsigned int k = 0; k < PARAMETERS; k++)
              average[k] += chain[ii][n].f[k] * chain[ii][n].Multiplicity;
            TotalSize += chain[ii][n].Multiplicity; // total weight of all points
          }
        }

        for (unsigned int k = 0; k < PARAMETERS; k++) // normalize
        {
          average[k] = average[k] / ((double)TotalSize);
        }
        for (unsigned int ii = 0; ii < CHAINS; ii++)
        {
          for (unsigned int n = chainSize[ii] - covSize[ii]; n < chainSize[ii]; n++)
          { // run over all points
            for (unsigned int k = 0; k < PARAMETERS; k++)
              for (unsigned int j = 0; j < PARAMETERS; j++)
                cov[k][j] += (chain[ii][n].f[k] - average[k]) * (chain[ii][n].f[j] - average[j]) * chain[ii][n].Multiplicity;
          }
        }

        for (unsigned int k = 0; k < PARAMETERS; k++) // normalize
        {
          for (unsigned int j = 0; j < PARAMETERS; j++)
          {
            cov[k][j] /= (TotalSize - 1.0);
          }
        }

        for (unsigned int ii = 0; ii < CHAINS; ii++)
          for (unsigned int k = 0; k < PARAMETERS; k++)
            for (unsigned int j = 0; j < PARAMETERS; j++)
              covMatrix[ii][k][j] = cov[k][j];

        for (unsigned int ii = 0; ii < CHAINS; ii++)
        {
          generateEigenvectors(mGauss[ii], cov, EntireFactor[ii] * EntireFactor[ii]);
          
          // Fix: Must also update the Slow proposal matrix!
          double **covSlow = (double**)malloc(N_SLOW * sizeof(double*));
          for (int k = 0; k < N_SLOW; k++) {
              covSlow[k] = (double*)malloc(N_SLOW * sizeof(double));
              for (int j = 0; j < N_SLOW; j++)
                  covSlow[k][j] = cov[slow_idx[k]][slow_idx[j]];
          }
          generateEigenvectors(mGaussSlow[ii], covSlow, EntireFactor[ii] * EntireFactor[ii]);
          for (int k = 0; k < N_SLOW; k++) free(covSlow[k]);
          free(covSlow);
        }

        // Only broadcast fast covariance if dragging is enabled globally
        if (USE_DRAGGING) {
            int worldsz;
            MPI_Comm_size(MPI_COMM_WORLD, &worldsz);
            broadcast_fast_covariance(cov, worldsz);
        }

        // output information, in text file as well as binary format for re-starting
        fprintf(covarianceMatrices, "Chain: %d Step: %d Points used: %d\n", i + 1, chainSize[i], covSize[i]);

        if (chainSize[i] == BEGINCOVUPDATE) {
            printf("\n[INFO] Chain %d reached BEGINCOVUPDATE (%d). Building Cov Matrix and Starting Phase 2: DRAGGING!\n", i, BEGINCOVUPDATE);
        }

        for (unsigned int j = 0; j < PARAMETERS; j++)
        {
          for (unsigned int k = 0; k < PARAMETERS; k++)
            fprintf(covarianceMatrices, "%e  ", cov[j][k]);
          fprintf(covarianceMatrices, "\n");
        }

        fprintf(covarianceMatrices, "\nEntire factor: %e", EntireFactor[i]);
        fprintf(covarianceMatrices, "\n***********************************************************\n");

        for (unsigned int ii = 0; ii < CHAINS; ii++)
          steps_since_update[ii] = 0;
      }
      // END OF ADAPTIVE  COVARIANCE

      //
      // In addition and to help the adaptive covariance,
      // we use an adaptive step size multiplicator, called EntireFactor
      // Whenever we take a step, we consider increasing the step size,
      // as frequent approval corresponds to small stepsizes.
      // This is performed several lines above when storing the new step
      // in the chain. Here, we consider the opposite. If we hang around
      // for some time at the same spot, we decrease the step size..
      //-----------------------------------------------------------------------------------------------

      // ADAPTIVE: decrease if we didn't take for a long time
      if (ADAPTIVE == 1) {
        if (chain[i][chainBack[i] - 1].f[LOGLIKEPOS] < -1e9) {
            // FIX: Chain is stuck in a CAMB-failing region! Force a huge step size to escape.
            EntireFactor[i] = 1.0; 
            generateEigenvectors(mGauss[i], covMatrix[i], EntireFactor[i] * EntireFactor[i]);
        }
        else if (chain[i][chainBack[i] - 1].ReallyInvestigated > HIGH_STEP_BOUND) {
          if (EntireFactor[i] > 1e-5) {
            EntireFactor[i] *= DECREASE_STEP;
            generateEigenvectors(mGauss[i], covMatrix[i], EntireFactor[i] * EntireFactor[i]);

            double **covSlow = (double**)malloc(N_SLOW * sizeof(double*));
            for (int k = 0; k < N_SLOW; k++) {
                covSlow[k] = (double*)malloc(N_SLOW * sizeof(double));
                for (int j = 0; j < N_SLOW; j++)
                    covSlow[k][j] = covMatrix[i][slow_idx[k]][slow_idx[j]];
            }
            generateEigenvectors(mGaussSlow[i], covSlow, EntireFactor[i] * EntireFactor[i]);
            for (int k = 0; k < N_SLOW; k++) free(covSlow[k]);
            free(covSlow);
          }
        }
      }

      // END ADATPIVE decrease

      //
      // now, we re-evaluate
      // first, let us find out how long the shortest chain is
      //-----------------------------------------------------------------------------------------------
      min_size = 1000 * 1000;
      for (unsigned int ii = 0; ii < CHAINS; ii++) {
        if (chainSize[ii] < min_size)
          min_size = chainSize[ii];
      }

      // output of progress information (same format as code0)
      fprintf(progress, "performed/Chainsize: ");
      for (unsigned int ii = 0; ii < CHAINS; ii++)
        fprintf(progress, "%d/%d  ", chainTotalPerformed[ii], chainSize[ii]);

      fprintf(progress, "min_size: %d    Multiplicity [Really]: ", min_size);
      for (unsigned int ii = 0; ii < CHAINS; ii++)
        fprintf(progress, "%d[%d]  ", chain[ii][chainBack[ii] - 1].Multiplicity, chain[ii][chainBack[ii] - 1].ReallyInvestigated);

      fprintf(progress, "\n");

      //
      // Start with the Gelman and Rubin(1992) statistic;
      // compare variances within the chain with the variances between the chains
      // R[k] should be smaller than 1.1 for convergence
      // If you would like to run a synthetic distribution for checks, this statistics
      // will slow you down. You can speed things up if you replace the if
      // statement below by something like:
      //        if ( (min_size) > RMIN_POINTS && Miscmath::posRnd(1.0) > 0.95) {
      // hence, only about every 20 times, the statistics is re-calculated.
      // ---------------------------------------------------------------------------------------------

      if ((min_size > RMIN_POINTS) && (posRnd(1.0) > 0.95))
      {
        break_at = min_size / 2;
        N = min_size - break_at;

        //
        // Initialize
        // ------------------------------------------------------------
        for (unsigned int k = 0; k < PARAMETERS; k++)
        {
          dist_y[k] = 0;
          B[k] = 0;
          W[k] = 0;
          for (unsigned int ii = 0; ii < CHAINS; ii++)
            y[ii][k] = 0;
        }

        double M = (double)CHAINS;
        for (unsigned int ii = 0; ii < CHAINS; ii++)
        { // for all chains
          int TotalMultiplicity = 0;
          for (unsigned int n = break_at; n < min_size; n++)
          {                                                              // traverse through the chains
            for (unsigned int k = 0; k < PARAMETERS; k++)                // for all parameter
              y[ii][k] += chain[ii][n].f[k] * chain[ii][n].Multiplicity; // add up all parameters

            TotalMultiplicity += chain[ii][n].Multiplicity;
          }

          N = TotalMultiplicity;
          for (unsigned int k = 0; k < PARAMETERS; k++)
          {
            y[ii][k] /= (double)N;
            dist_y[k] += y[ii][k] / M; // M chains hence mean is sum / CHAINS
          }
        }

        // variance between chains
        for (unsigned int k = 0; k < PARAMETERS; k++)
          for (unsigned int ii = 0; ii < CHAINS; ii++)
          {
            double t1 = y[ii][k] - dist_y[k];
            B[k] += t1 * t1 / (M - 1.0); // as defined in (22) (why 1/(M-1) and not M ???)
          }
        // variance within a chain

        // once again, go back to break_at in the chains
        // get W, the variance within chains
        // run over all points
        for (unsigned int ii = 0; ii < CHAINS; ii++)         // run over all chains
          for (unsigned int n = break_at; n < min_size; n++) // run over all points
          {
            for (unsigned int k = 0; k < PARAMETERS; k++)
            {
              double t2 = chain[ii][n].f[k] - y[ii][k];
              W[k] += t2 * t2 * chain[ii][n].Multiplicity;
            }
          }

        // normalize W and get R
        for (unsigned int k = 0; k < PARAMETERS; k++)
        {
          W[k] /= (M * (N - 1.0));
          R[k] = (N - 1.0) / N * W[k] + B[k] * (1.0 + 1.0 / N);
          R[k] /= W[k];
        }

        // if we want FREEZE_IN, test if convergence has been reached:
        if (FREEZE_IN == 1 && min_size > MIN_SIZE_FOR_FREEZE_IN)
        {
          unsigned short int ConvergenceReached = 1;
          for (unsigned int m = 0; m < PARAMETERS; m++)
            if (R[m] > RBREAK)
              ConvergenceReached = 0;

          if (ConvergenceReached == 1)
          {
            ADAPTIVE = 0;  // stop with adaptive stepsize
            FREEZE_IN = 0; // never check again (otherwise no real freeze-in)
            drfactor = 1.0;
            FILE *final;
            snprintf(path, sizeof(path), "%s/freezeInEigenvectors.txt", outdir_buf);
            final = fopen(path, "a+");

            for (unsigned int k = 1; k <= CHAINS; k++)
            {
              fclose(data[k - 1]);
              snprintf(path, sizeof(path), "%s/%s_freeze_%d.d", outdir_buf, montecarlo_name, k);
              data[k - 1] = fopen(path, "a+");
              
              // Write clean header to the freeze-in files
              fprintf(data[k - 1], "%-5s %-16s", "Mult", "Chi2");
              int printed = 0;
              for (int p = 0; p < global_config.param_count && printed < PARAMETERS; p++) {
                  if (global_config.params[p].is_estimated) {
                      fprintf(data[k - 1], " %-18s", global_config.params[p].name);
                      printed++;
                  }
              }
              fprintf(data[k - 1], "\n");
              fflush(data[k - 1]);
            }

            for (unsigned int j = 0; j < CHAINS; j++)
            {
              Task *keep;
              keep = chain[j] + chainBack[j] - 1;
              // the next 2 lines write the flag to each montecarlo file
              // indicating that all models before this have to be discarded
              // the flag is a zero weight

              // Push back the last one...
              Task_copy(chain[j], keep);
              //	chainSize[j]=0;
              // USE Dunkley et. al. result for optimal sigma_t: 2.4 / sqrt(Parameters)
              double OptimalFactor = 2.4 / sqrt((double)PARAMETERS); // ........... Check it
              OptimalFactor = RollingAverage_average(roll[j]);
              EntireFactor[j] = OptimalFactor;
              if (EntireFactor[j] > 1.0)
                EntireFactor[j] = 0.2;
              fprintf(final, "CHAIN[%d] Eingenvector and value info: ", j);

              fprintf(final, "************************************\n");
              fprintf(final, "MultiGaussian::printInfo:\n");
              fprintf(final, "Eigenvalues (we used scale = %f):\n", mGauss[j]->Scale);
              for (unsigned int ii = 0; ii < mGauss[j]->SIZE; ii++)
              {
                fprintf(final, "%e  --> sigma:  %e  ", mGauss[j]->eigenvalues[ii], sqrt(mGauss[j]->eigenvalues[ii]));
                fprintf(final, "eigenvector w/o scale: %e  ", mGauss[j]->eigenvalues[ii] / mGauss[j]->Scale);
                fprintf(final, "  --> sigma: %e\n", sqrt(mGauss[j]->eigenvalues[ii] / mGauss[j]->Scale));
              }

              fprintf(final, "\nEigenvectors:\n");
              for (unsigned int ii = 0; ii < mGauss[j]->SIZE; ii++)
              {
                for (unsigned int jj = 0; jj < mGauss[j]->SIZE; jj++)
                {
                  fprintf(final, "%e  ", mGauss[j]->MasterMatrix[ii][jj]);
                }
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

        // Print statistics in the same format as code0
        fprintf(progress, "Statistics: \n");
        for (unsigned int k = 0; k < PARAMETERS; k++)
          fprintf(progress, "%d  dist_y: %e   B: %e   W: %e     R[k]:  %e \n", k, dist_y[k], B[k], W[k], R[k]);

        if (FREEZE_IN)
          fprintf(progress, "Multiplicity (and EntireFactor): ");
        else
          fprintf(progress, "Multiplicity (and frozen EntireFactor): ");

        for (unsigned int ii = 0; ii < CHAINS; ii++)
          fprintf(progress, "%d (%e)  ", chain[ii][chainBack[ii] - 1].Multiplicity, EntireFactor[ii]);
        fprintf(progress, "\n");

        // Write gelmanRubin.txt in code0 format
        if ((min_size) != checkSize)
        {
          fprintf(gelmanRubin, "%d   ", break_at);
          for (unsigned int n = 0; n < PARAMETERS; n++)
          {
            fprintf(gelmanRubin, "%e   ", R[n]);
          }
          fprintf(gelmanRubin, "%d \n", min_size);
          checkSize = min_size;
        }

      } // end if min_size > old_size

      if (FREEZE_IN == 0)
      {
        loopstop++;
        printf("\n**** %d", loopstop);
      }
      unsigned int max_size = 0;
      for (int ii = 0; ii < CHAINS; ii++)
        max_size = (max_size > chainBack[ii]) ? max_size : chainBack[ii];

      // <-- ADDED: total_proposals limit
      if ((FREEZE_IN == 0 && loopstop > 18000) || max_size > MAXCHAINLENGTH || total_proposals > 100000)
        work = 0;

      fflush(data[i]);
      fflush(head[i]);
      fflush(investigated[i]);
      fflush(progress);
      fflush(gelmanRubin);
      fflush(covarianceMatrices);

    } // if TAKE_RESULT

    ++tempi;
  } while (work);

  fclose(progress);
  fclose(gelmanRubin);
  fclose(covarianceMatrices);

  for (unsigned int k = 1; k <= CHAINS; k++)
  {
    fclose(data[k - 1]);
    fclose(head[k - 1]);
    fclose(investigated[k - 1]);
  }

  // <-- MODIFIED: graceful exit instead of exit(1)
  printf("\nMCMC finished. Sending stop signal to all slaves...\n");
  int worldsize;
  MPI_Comm_size(MPI_COMM_WORLD, &worldsize);
  for (int r = 1; r < worldsize; r++) {
      MPI_Send(NULL, 0, MPI_INT, r, TAG_STOP, MPI_COMM_WORLD);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  return 0;
}

int fact(int a)
{
  if (a <= 1)
    return 1;
  return a * fact(a - 1);
}

double mind(double x, double y)
{
  return (x < y) ? x : y;
}

//
// The slaves are responsible for computing  models and its associated likelihoods. They receive the
// parameters from the Master, report back to him the results of the computation and request new jobs
//--------------------------------------------------------------------------------------------------------
double eval_full_chi2(int rank, double *theta, double *Cl_TT, double *Cl_TE, double *Cl_EE, double *Cl_BB) {
    int ok = param_iface(rank, theta, Cl_TT, Cl_TE, Cl_EE, Cl_BB);
    if (!ok) return 1e30;
    double dummy = 0.0;
    return run_plc(rank, theta, Cl_TT, Cl_TE, Cl_EE, Cl_BB, 0, &dummy);
}

/* =========================================================
   MODIFIED SLAVE: added progress logging and stop signal check
   ========================================================= */
int slave(int rank) 
{
  printf("\nI am from slave %d", rank);
  MPI_Status status;
  double task[MAX_TASKARRAY_SIZE];
  
  int l_max_alloc = 3000;
  double *Cl_TT_A = (double*)malloc((l_max_alloc + 1) * sizeof(double));
  double *Cl_TE_A = (double*)malloc((l_max_alloc + 1) * sizeof(double));
  double *Cl_EE_A = (double*)malloc((l_max_alloc + 1) * sizeof(double));
  double *Cl_BB_A = (double*)malloc((l_max_alloc + 1) * sizeof(double));
  
  double *Cl_TT_B = (double*)malloc((l_max_alloc + 1) * sizeof(double));
  double *Cl_TE_B = (double*)malloc((l_max_alloc + 1) * sizeof(double));
  double *Cl_EE_B = (double*)malloc((l_max_alloc + 1) * sizeof(double));
  double *Cl_BB_B = (double*)malloc((l_max_alloc + 1) * sizeof(double));

  double g_lowbound[NOPARAM], g_highbound[NOPARAM], g_sigma[NOPARAM];
  get_parameter_bounds_from_config(g_lowbound, g_highbound, g_sigma);

  if (!mapping_done) {
      g_n_cosmo = N_SLOW;
      if (g_n_cosmo == 0) {
          fprintf(stderr, "Rank %d: WARNING: No slow parameters found! Using default 6.\n", rank);
          g_n_cosmo = 6;
      } else {
          printf("Rank %d: Found %d cosmological (slow) parameters for emulator.\n", rank, g_n_cosmo);
      }
      mapping_done = 1;
  }

  if (USE_EMULATOR && g_emulator == NULL) {
      EmulatorConfig emu_cfg = {
          .n_params = g_n_cosmo,
          .buffer_capacity = 200,
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
      if (!g_emulator) {
          fprintf(stderr, "Rank %d: Emulator initialisation failed. Running without emulator.\n", rank);
      } else {
          printf("Rank %d: Emulator initialised (type %s) with %d cosmological parameters.\n", rank,
                 (EMULATOR_TYPE == 0) ? "GP" : "Polynomial", g_n_cosmo);
      }
      if (g_emulator) {
          printf("Rank %d: emulator buffer size = %d, is_trained = %d\n",
                 rank, emulator_buffer_size(g_emulator), emulator_is_ready(g_emulator));
      }
  }

  char progress_path[512];
  snprintf(progress_path, sizeof(progress_path), "%s/progress_rank%d.log", outdir_buf, rank);
  FILE *progress_file = fopen(progress_path, "w");
  if (!progress_file) {
      fprintf(stderr, "Rank %d: Could not open progress file %s. Using stdout.\n", rank, progress_path);
      progress_file = stdout;
  }

  printf("\n"
         "------------------------------------------------------------------------------------------------------------------------------\n"
         " Rank  MCMC_step  Emu  Fallback  Uncertainty    chi2      logL       Omega_m    Omega_b       h        tau      n_s       A_s\n"
         "------------------------------------------------------------------------------------------------------------------------------\n");
  fflush(stdout);

  double start_wtime = MPI_Wtime();
  int proposal_count = 0;
  int emu_count = 0;

  int is_middleman = middleman(rank);
  int chain_index = (rank - 1) / SLAVEPARCHAIN;

  double loglike = -1.0e30;
  unsigned short int takenindex = 0, take = 0;

  for (;;) 
  {  
      MPI_Status probe_status;
      int flag = 0;
      MPI_Iprobe(0, TAG_STOP, MPI_COMM_WORLD, &flag, &probe_status);
      if (flag) {
          MPI_Recv(NULL, 0, MPI_INT, 0, TAG_STOP, MPI_COMM_WORLD, &probe_status);
          if (progress_file && progress_file != stdout) fclose(progress_file);
          break;
      }

      if (is_middleman) {
          MPI_Send(task, TASKARRAY_SIZE, MPI_DOUBLE, 0, GIVEMETASK, MPI_COMM_WORLD);
      }

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
      
      int mcmc_step = (int)task[STEP_POS];

      double theta_A[MAX_TASKARRAY_SIZE] = {0};
      double theta_B[MAX_TASKARRAY_SIZE] = {0};
      double final_theta[MAX_TASKARRAY_SIZE] = {0};
      for (int k = 0; k < PARAMETERS; k++) {
          theta_B[k] = task[k];
          theta_A[k] = task[CURRENTSTATEPOS + k];
      }

      double final_chi2 = 1e30;
      double proposal_chi2 = 1e30;
      int accepted = 0;
      int use_dragging = (int)task[DRAGPOS];
      double fast_EntireFactor_local = task[FASTFACTORPOS];

      int use_emu = 0;
      double uncertainty_A = 0.0, uncertainty_B = 0.0;
      double emu_cl_tt_A[2601], emu_cl_te_A[2601], emu_cl_ee_A[2601], emu_cl_bb_A[2601];
      double emu_cl_tt_B[2601], emu_cl_te_B[2601], emu_cl_ee_B[2601], emu_cl_bb_B[2601];
      int emu_ok_A = 0, emu_ok_B = 0;

      double cosmo_A[N_SLOW], cosmo_B[N_SLOW];
      for (int k = 0; k < N_SLOW; k++) {
          cosmo_A[k] = theta_A[slow_idx[k]];
          cosmo_B[k] = theta_B[slow_idx[k]];
      }

      if (USE_EMULATOR && g_emulator && emulator_is_ready(g_emulator)) {
          emu_ok_A = emulator_predict(g_emulator, cosmo_A,
                                       emu_cl_tt_A, emu_cl_te_A, emu_cl_ee_A, emu_cl_bb_A,
                                       &uncertainty_A);
          emu_ok_B = emulator_predict(g_emulator, cosmo_B,
                                       emu_cl_tt_B, emu_cl_te_B, emu_cl_ee_B, emu_cl_bb_B,
                                       &uncertainty_B);
          if (emu_ok_A && emu_ok_B && uncertainty_A <= FALLBACK_THRESHOLD && uncertainty_B <= FALLBACK_THRESHOLD) {
              use_emu = 1;
          } else {
              use_emu = 0;
          }
      }

      proposal_count++;
      if (use_emu) emu_count++;
      if (proposal_count % 500 == 0) {
          double elapsed = MPI_Wtime() - start_wtime;
          fprintf(progress_file, "[Rank %d] Proposals: %d, Emulator: %d, Time: %.2f s\n",
                  rank, proposal_count, emu_count, elapsed);
          fflush(progress_file);
      }

      // --- BURN IN / PRE-COVARIANCE MCMC ---
      if (!use_dragging) {
          double cur_chi2 = -2.0 * task[LOGLIKEPOS];
          int really_inv = (int)task[MULTIPURPOSEPOS];

          if (use_emu) {
              proposal_chi2 = run_plc(rank, theta_B, emu_cl_tt_B, emu_cl_te_B, emu_cl_ee_B, emu_cl_bb_B, 0, NULL);
              final_chi2 = proposal_chi2;
          } else {
              int ok = param_iface(rank, theta_B, Cl_TT_B, Cl_TE_B, Cl_EE_B, Cl_BB_B);
              if (ok) {
                  double dummy = 0.0;
                  proposal_chi2 = run_plc(rank, theta_B, Cl_TT_B, Cl_TE_B, Cl_EE_B, Cl_BB_B, 0, NULL);
                  final_chi2 = proposal_chi2;
              } else {
                  proposal_chi2 = 1e30;
                  final_chi2 = 1e30;
              }
          }

          if (really_inv > 10) {
              double cur_re_eval = 1e30;
              if (use_emu) {
                  if (g_emulator && emulator_is_ready(g_emulator)) {
                      int okA = emulator_predict(g_emulator, cosmo_A,
                                                  emu_cl_tt_A, emu_cl_te_A, emu_cl_ee_A, emu_cl_bb_A,
                                                  &uncertainty_A);
                      if (okA) cur_re_eval = run_plc(rank, theta_A, emu_cl_tt_A, emu_cl_te_A, emu_cl_ee_A, emu_cl_bb_A, 0, NULL);
                  } else {
                      int okA = param_iface(rank, theta_A, Cl_TT_A, Cl_TE_A, Cl_EE_A, Cl_BB_A);
                      if (okA) {
                          double dummy = 0.0;
                          cur_re_eval = run_plc(rank, theta_A, Cl_TT_A, Cl_TE_A, Cl_EE_A, Cl_BB_A, 0, NULL);
                      }
                  }
                  if (cur_re_eval < 1e29) cur_chi2 = cur_re_eval;
              }
          }

          if (final_chi2 < 1e29 && posRnd(1.0) < exp(-0.5 * (final_chi2 - cur_chi2))) {
              accepted = 1;
              for (int k = 0; k < PARAMETERS; k++) final_theta[k] = theta_B[k];
          }
      }
      // --- FAST DRAGGING MCMC with caching of "other" chi2 ---
      else {
          double dummy = 0.0;
          double chi2_A_state, chi2_B_state;
          double cached_other_A = 0.0, cached_other_B = 0.0;  // store -2*logL(Commander+Lowlike)

          // Compute full chi² for A and B, capturing the "other" part
          if (use_emu) {
              chi2_A_state = run_plc(rank, theta_A, emu_cl_tt_A, emu_cl_te_A, emu_cl_ee_A, emu_cl_bb_A, 0, &cached_other_A);
              chi2_B_state = run_plc(rank, theta_B, emu_cl_tt_B, emu_cl_te_B, emu_cl_ee_B, emu_cl_bb_B, 0, &cached_other_B);
              proposal_chi2 = chi2_B_state;
          } else {
              param_iface(rank, theta_A, Cl_TT_A, Cl_TE_A, Cl_EE_A, Cl_BB_A);
              param_iface(rank, theta_B, Cl_TT_B, Cl_TE_B, Cl_EE_B, Cl_BB_B);
              chi2_A_state = run_plc(rank, theta_A, Cl_TT_A, Cl_TE_A, Cl_EE_A, Cl_BB_A, 0, &cached_other_A);
              chi2_B_state = run_plc(rank, theta_B, Cl_TT_B, Cl_TE_B, Cl_EE_B, Cl_BB_B, 0, &cached_other_B);
              proposal_chi2 = chi2_B_state;
          }

          if (chi2_A_state < 1e29 && chi2_B_state < 1e29) {
              double work_sum = chi2_B_state - chi2_A_state;
              double theta_f_path[MAX_TASKARRAY_SIZE] = {0};
              for (int k = 0; k < PARAMETERS; k++) theta_f_path[k] = theta_B[k];

              for (int d = 1; d <= N_DRAG; d++) {
                  double w = (double)d / (double)N_DRAG;
                  double trial[MAX_TASKARRAY_SIZE] = {0};
                  for (int k = 0; k < PARAMETERS; k++) trial[k] = theta_f_path[k];

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

                      // *** USE FAST MODE WITH CACHED OTHER CHI2 ***
                      chi2_A_trial = run_plc(rank, thA,
                                             (use_emu ? emu_cl_tt_A : Cl_TT_A),
                                             (use_emu ? emu_cl_te_A : Cl_TE_A),
                                             (use_emu ? emu_cl_ee_A : Cl_EE_A),
                                             (use_emu ? emu_cl_bb_A : Cl_BB_A),
                                             1, &cached_other_A);   // fast_mode=1, reuse cached_other_A

                      chi2_B_trial = run_plc(rank, thB,
                                             (use_emu ? emu_cl_tt_B : Cl_TT_B),
                                             (use_emu ? emu_cl_te_B : Cl_TE_B),
                                             (use_emu ? emu_cl_ee_B : Cl_EE_B),
                                             (use_emu ? emu_cl_bb_B : Cl_BB_B),
                                             1, &cached_other_B);   // fast_mode=1, reuse cached_other_B
                  }

                  double L_cur   = (1.0-w)*chi2_A_state + w*chi2_B_state;
                  double L_trial = (1.0-w)*chi2_A_trial + w*chi2_B_trial;

                  if (!fast_oob && chi2_A_trial < 1e29 && chi2_B_trial < 1e29 &&
                      posRnd(1.0) < exp(-0.5*(L_trial - L_cur))) {
                      for (int k = 0; k < PARAMETERS; k++) theta_f_path[k] = trial[k];
                      chi2_A_state = chi2_A_trial;
                      chi2_B_state = chi2_B_trial;
                  }

                  if (d < N_DRAG) work_sum += (chi2_B_state - chi2_A_state);
              }

              double ln_alpha = -0.5 * work_sum / (double)N_DRAG; 
              double alpha = (ln_alpha >= 0.0) ? 1.0 : exp(ln_alpha);

              if (posRnd(1.0) < alpha) {
                  for (int k = 0; k < PARAMETERS; k++) final_theta[k] = theta_f_path[k];
                  for (int k = 0; k < N_SLOW; k++) final_theta[slow_idx[k]] = theta_B[slow_idx[k]];
                  final_chi2 = chi2_B_state;
                  proposal_chi2 = chi2_B_state;
                  accepted = 1;
              }
          }
      }

      task[PARAMETERS] = proposal_chi2;

      if (accepted && !use_emu && g_emulator) {
          double *Cl_TT_true = (double*)malloc(2601 * sizeof(double));
          double *Cl_TE_true = (double*)malloc(2601 * sizeof(double));
          double *Cl_EE_true = (double*)malloc(2601 * sizeof(double));
          double *Cl_BB_true = (double*)malloc(2601 * sizeof(double));
          if (Cl_TT_true && Cl_TE_true && Cl_EE_true && Cl_BB_true) {
              if (param_iface(rank, final_theta, Cl_TT_true, Cl_TE_true, Cl_EE_true, Cl_BB_true)) {
                  double true_logL = -0.5 * final_chi2;
                  double cosmo_final[N_SLOW];
                  for (int k = 0; k < N_SLOW; k++) cosmo_final[k] = final_theta[slow_idx[k]];
                  emulator_update(g_emulator, local_step++, cosmo_final,
                                  Cl_TT_true, Cl_TE_true, Cl_EE_true, Cl_BB_true, true_logL);
              } else {
                  free(Cl_TT_true); free(Cl_TE_true); free(Cl_EE_true); free(Cl_BB_true);
              }
          } else {
              free(Cl_TT_true); free(Cl_TE_true); free(Cl_EE_true); free(Cl_BB_true);
          }
      }

      if (g_emulator && !emulator_is_ready(g_emulator) && emulator_buffer_size(g_emulator) >= 200) {
          printf("Rank %d: Forcing emulator training. Buffer size = %d\n", rank, emulator_buffer_size(g_emulator));
          emulator_train(g_emulator);
      }

      if (accepted) {
          for (int k = 0; k < PARAMETERS; k++) task[k] = final_theta[k];
          task[LOGLIKEPOS] = -0.5 * final_chi2; 
          task[TAKEPOS] = 1.0;
      } else {
          task[TAKEPOS] = 0.0;
      }

      double chi2_to_print = (accepted) ? final_chi2 : -2.0 * task[LOGLIKEPOS];
      double logL_to_print = -0.5 * chi2_to_print;
      int max_print = (g_n_cosmo < 6) ? g_n_cosmo : 6;
      double *params_to_print = (accepted) ? final_theta : theta_A;
      printf(" %2d     %6d    %1d      %1d     %8.4e   %8.2f %8.2f   ",
             rank, mcmc_step, use_emu, (!use_emu), (use_emu ? uncertainty_A : 0.0),
             chi2_to_print, logL_to_print);
      for (int i = 0; i < max_print; i++) {
          printf("%8.4f ", params_to_print[i]);
      }
      for (int i = max_print; i < 6; i++) {
          printf("%8s ", " ");
      }
      printf("\n");
      fflush(stdout);

      MPI_Send(task, TASKARRAY_SIZE, MPI_DOUBLE, 0, TAKERESULT, MPI_COMM_WORLD);   
  } 
  
  if (progress_file && progress_file != stdout) fclose(progress_file);
  free(Cl_TT_A); free(Cl_TE_A); free(Cl_EE_A); free(Cl_BB_A);
  free(Cl_TT_B); free(Cl_TE_B); free(Cl_EE_B); free(Cl_BB_B);
  return 0;
}

// ====== slave0, master0, classify_parameters, setVariables ======
// These functions are unchanged from the original code.

int slave0(int rank,int runperchain) 
{
  MPI_Status  status;
  char inputfilename[30],command[500];
  double *task;
  double Chi2X;
  double ratio;
  FILE *fplens;
  int lfile;                    //We compute it if cmb required to get sigma_8 in multipurpose
  double loglike = -1.0e30;
  unsigned short int takenindex = 0,take =0;
  int magicnum1,magicnum2;
  char cfplens[100];
  double alpha12,alpha32,alpha13,l2,q1,newss;
// --- MODIFICATION START: Heap Allocation ---
  int l_max_alloc = 3000;
  double *Cl_TT = (double*)malloc((l_max_alloc + 1) * sizeof(double));
  double *Cl_TE = (double*)malloc((l_max_alloc + 1) * sizeof(double));
  double *Cl_EE = (double*)malloc((l_max_alloc + 1) * sizeof(double));
  double *Cl_BB = (double*)malloc((l_max_alloc + 1) * sizeof(double));

  if (!Cl_TT || !Cl_TE || !Cl_EE || !Cl_BB) {
      fprintf(stderr, "Rank %d: Failed to allocate memory for Cl arrays.\n", rank);
      MPI_Abort(MPI_COMM_WORLD, 1);
  }
  // --- MODIFICATION END --  

  task = new_double(PARAMETERS+1);
  //
  // start an infinite loop;
  // basically, a lot of this is just setting the parameters we have received from the master
  // and running cmbans, obtaining the likelihoods and derived parameters (e.g. sigma8 ) 
  // and sending the result to the master
  //-----------------------------------------------------------------------------------------------------
  printf("My rank is here %d",rank);
  int cnt = 0;
  for (int i=0;i<runperchain;i++) 
    {  

      printf("\nThis is a test\n");          
      MPI_Recv(task,PARAMETERS+1,MPI_DOUBLE,0,200,MPI_COMM_WORLD,&status); //master's answer   
         
      printf("\nReceived : %d",rank);  
      //
      // Specify the parameters in the CMBAns
      //----------------------------------------------------------------------------------
      magicnum1 = param_iface(rank, task, Cl_TT, Cl_TE, Cl_EE, Cl_BB);      
      printf("\nReceived (abcdefgh) : %d %d",rank,magicnum1);

      if(magicnum1 !=0)
      {
        //sprintf(cfplens,"test_%d_lensedCls.dat",rank);
  printf("CALLING RUN_PLC from slave0");
  double dummy = 0.0;
  Chi2X = run_plc(rank, task, Cl_TT, Cl_TE, Cl_EE, Cl_BB, 0, &dummy);
      }
      else
	{
	  Chi2X = 10.0e10; 
	}
  
      printf("\nChisquare : %e",Chi2X);    

      //
      // Having computed the model, we can compare it with the data
      // using the rich number of functions available in the analyzeit-class
      //-----------------------------------------------------------------------------------------
      
      if (WMAP7 == 1) 
	task[PARAMETERS] = Chi2X;
         
      printf("\nChisquare is (%d): %e",rank,Chi2X);
      //
      // Send the whole task - line back to master()+ last minute sanity check     
      // ------------------------------------------------------------------------------
      for (unsigned int k = 0;k < PARAMETERS+1; k++)  
	{
	  if (isnan(task[k])) 
	    {
	      printf("isnan element");
          // Cleanup
          free(Cl_TT); free(Cl_TE); free(Cl_EE); free(Cl_BB);
	      exit(1);
	    }
	}

      printf("\n\nThis is :\n\n");
      fprintf(stdout,"\nI will now send : %e",task[PARAMETERS]); 
      fflush(stdout);
      MPI_Send(task,PARAMETERS+1, MPI_DOUBLE,0,201,MPI_COMM_WORLD); 
    }
    // --- MODIFICATION: Cleanup ---
    free(Cl_TT); free(Cl_TE); free(Cl_EE); free(Cl_BB);
    
    printf("\nI am now working fine..");
    return 0;
}

double **master0(int worldsize, int runperchain)
{

  MPI_Status status;
  double *startval[100];
  for (int i = 0; i < 100; i++)
    startval[i] = new_double(PARAMETERS + 1);

  double *lowbound, *highbound;

  lowbound = new_double(PARAMETERS);
  highbound = new_double(PARAMETERS);

double *initial_sigma_dummy = new_double(PARAMETERS); // Temp array, not used here but needed by function
  
  // Use the function that reads from param.ini
  get_parameter_bounds_from_config(lowbound, highbound, initial_sigma_dummy);
  
  free(initial_sigma_dummy);

  printf("Hi I am here");

  for (int i = 0; i < 100; i++)
  {
    printf("\n %d) ", i);
    for (int j = 0; j <= PARAMETERS; j++)
    {
      startval[i][j] = lowbound[j] + posRnd(highbound[j] - lowbound[j]);
      printf("%e   ", startval[i][j]);
    }
  }
  double *startvaltemp;

  startvaltemp = new_double(PARAMETERS + 1);

  printf("Hi I am here");

  int slavenumber = worldsize / runperchain;

  for (int i = 0; i < runperchain; i++)
  {
    printf("\nSending Now : %d %d", runperchain, worldsize);
    for (int j = 0; j < worldsize / runperchain; j++)
      MPI_Send(startval[i + j * runperchain], PARAMETERS + 1, MPI_DOUBLE, j + 1, 200, MPI_COMM_WORLD);
    for (int j = 0; j < worldsize / runperchain; j++)
    {
      MPI_Recv(startvaltemp, PARAMETERS + 1, MPI_DOUBLE, MPI_ANY_SOURCE, 201, MPI_COMM_WORLD, &status);
      printf("Recv from %d %d", status.MPI_SOURCE, runperchain);
      for (int k = 0; k <= PARAMETERS; k++)
      {
        startval[i + (status.MPI_SOURCE - 1) * runperchain][k] = startvaltemp[k];
      }
    }
    printf("I am now here : %d ", runperchain);
  }

  int temparrange[100];
  double temploglike[100];

  for (int i = 0; i < 100; i++)
  {
    temparrange[i] = i;
    temploglike[i] = startval[i][PARAMETERS];
  }

  // Arrange

  int temparrangeflip;
  double temploglikeflip;

  for (int i = 0; i < CHAINS; i++)
    for (int j = i; j < 100; j++)
    {
      if (temploglike[i] > temploglike[j])
      {
        temploglikeflip = temploglike[i];
        temparrangeflip = temparrange[i];

        temploglike[i] = temploglike[j];
        temparrange[i] = temparrange[j];

        temploglike[j] = temploglikeflip;
        temparrange[j] = temparrangeflip;
      }
    }

  double **startvalfinal;
  startvalfinal = (double **)malloc(CHAINS * sizeof(double *));

  for (int i = 0; i < CHAINS; i++)
    startvalfinal[i] = new_double(PARAMETERS + 1);

  for (int i = 0; i < CHAINS; i++)
  {
    printf("\n");
    for (int j = 0; j <= PARAMETERS; j++)
    {
      startvalfinal[i][j] = startval[temparrange[i]][j];
      printf("%e  ", startvalfinal[i][j]);
    }
  }

  printf("This is final test");

  return startvalfinal;
}

void classify_parameters(void) {
    N_SLOW = 0; N_FAST = 0;
    int param_idx = 0;
    for (int i = 0; i < global_config.param_count && param_idx < PARAMETERS; i++) {
        if (global_config.params[i].is_estimated) {
            int is_slow = 0;
            for (int k = 0; k < N_SLOW_NAMES; k++) {
                if (strcmp(global_config.params[i].name, SLOW_PARAM_NAMES[k]) == 0) { is_slow = 1; break; }
            }
            if (is_slow) slow_idx[N_SLOW++] = param_idx;
            else         fast_idx[N_FAST++] = param_idx;
            param_idx++;
        }
    }
}


void setVariables() {
    if(!load_config("param.ini", &global_config)) {
        printf("CRITICAL ERROR: Could not load param.ini\n");
        exit(1);
    }

    printf("Loaded configuration from param.ini.\n");

    // Just count estimated parameters
    int total_estimated = 0;
    for(int i = 0; i < global_config.param_count; i++) {
        if(global_config.params[i].is_estimated)
            total_estimated++;
    }

    PARAMETERS = (total_estimated > 0) ? total_estimated : NOPARAM;

    LOGLIKEPOS = PARAMETERS + 1;
    LOGLIKEPOS = (LOGLIKEPOS > 15) ? LOGLIKEPOS : 15;

    MULTIPURPOSEPOS = LOGLIKEPOS + 22;
    ADAPTIVEPOS = MULTIPURPOSEPOS + 1;
    DRAGPOS = ADAPTIVEPOS + 1;
    FASTFACTORPOS = DRAGPOS + 1;
    TAKEPOS = FASTFACTORPOS + 1;
    PROBPOS = TAKEPOS + 1;
    RANDPOS = PROBPOS + 1;

    // Define STEP_POS and adjust CURRENTSTATEPOS and TASKARRAY_SIZE
    STEP_POS = RANDPOS + PARAMETERS + MULTIPURPOSE_REQUIRED + 1;
    CURRENTSTATEPOS = STEP_POS + 1;
    TASKARRAY_SIZE  = CURRENTSTATEPOS + PARAMETERS;

    if (TASKARRAY_SIZE > MAX_TASKARRAY_SIZE) {
        printf("ERROR: TASKARRAY_SIZE (%d) exceeds MAX_TASKARRAY_SIZE (%d).\n",
               TASKARRAY_SIZE, MAX_TASKARRAY_SIZE);
        exit(1);
    }
    
    classify_parameters();
    printf("Task Array Size: %d | Slow: %d | Fast: %d\n", TASKARRAY_SIZE, N_SLOW, N_FAST);
}

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  int myrank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  unsigned short int Restart = 0;
  const char *outdir_default = ".";
  const char *outdir = outdir_default;

  // <-- MODIFIED: parse all runtime flags
  for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-restart") == 0) {
          Restart = 1;
      } else if (strcmp(argv[i], "-o") == 0 && i+1 < argc) {
          outdir = argv[++i];
      } else if (strcmp(argv[i], "-dr") == 0) {
          USE_DR = 1;
      } else if (strcmp(argv[i], "-no-dr") == 0) {
          USE_DR = 0;
      } else if (strcmp(argv[i], "-emu") == 0) {
          USE_EMULATOR = 1;
      } else if (strcmp(argv[i], "-no-emu") == 0) {
          USE_EMULATOR = 0;
      } else if (strcmp(argv[i], "-drag") == 0) {
          USE_DRAGGING = 1;
      } else if (strcmp(argv[i], "-no-drag") == 0) {
          USE_DRAGGING = 0;
      } else {
          if (myrank == 0) {
              printf("Usage: %s [-restart] [-o output_dir] [-dr | -no-dr] [-emu | -no-emu] [-drag | -no-drag]\n", argv[0]);
          }
          MPI_Abort(MPI_COMM_WORLD, 1);
      }
  }

  // <-- ADDED: auto-set SLAVEPARCHAIN if DR disabled
  if (USE_DR == 0) {
      SLAVEPARCHAIN = 1;
      if (myrank == 0) {
          printf("Delayed rejection disabled: Setting SLAVEPARCHAIN = 1 to avoid idle slaves.\n");
      }
  }

  // Master creates the output directory
  if (myrank == 0) {
      char mkdir_cmd[512];
      snprintf(mkdir_cmd, sizeof(mkdir_cmd), "mkdir -p %s", outdir);
      system(mkdir_cmd);
  }
  MPI_Barrier(MPI_COMM_WORLD);  // ensure directory exists before opening files

  // Broadcast output directory to all ranks
  strncpy(outdir_buf, outdir, sizeof(outdir_buf)-1);
  outdir_buf[sizeof(outdir_buf)-1] = '\0';
  MPI_Bcast(outdir_buf, sizeof(outdir_buf), MPI_CHAR, 0, MPI_COMM_WORLD);

  setVariables();
  printf("Myrank : %d", myrank);

  int worldsize, totalrun, runperchain;
  drfactor = 0.7;
  MPI_Comm_size(MPI_COMM_WORLD, &worldsize);

  if (worldsize > 101)
  {
    totalrun = 100;
    runperchain = 1;
  }
  else
  {
    runperchain = 100 / (worldsize - 1);
    totalrun = runperchain * (worldsize - 1);
  }

  double **startnum;

  // <-- ADDED: total elapsed timer
  double start_time = 0.0;
  if (myrank == 0) {
      start_time = get_time();
  }

  if (myrank == 0)
  {
    printf("\nRestart = %d", Restart);
    startnum = master0(totalrun, runperchain);
    printf("\n\nWorking");
  }
  else
  {
    slave0(myrank, runperchain);
  }

  if (myrank == 0)
  {
    printf("\nRestart = %d", Restart);
    master(startnum, Restart);
  }
  else
  {
    printf("Slave is working");
    slave(myrank);
  }

  // <-- ADDED: print elapsed time
  if (myrank == 0) {
      double end_time = get_time();
      double elapsed = end_time - start_time;
      int hours   = (int)(elapsed / 3600);
      int minutes = (int)((elapsed - hours * 3600) / 60);
      int seconds = (int)(elapsed - hours * 3600 - minutes * 60);
      printf("\n========================================\n");
      printf("Total elapsed time: %02d:%02d:%02d (hh:mm:ss)\n", hours, minutes, seconds);
      printf("========================================\n");
  }

  MPI_Finalize();
  return 0;
}
