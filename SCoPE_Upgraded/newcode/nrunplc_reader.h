// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// //             Version with more general modification than above inculding build_fast_nuisance_map to handle the 
// //             new enums and gracefully support both sample + fixed parameters within the exact same likelihood struct

// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _NRUNPLC_H_
#define _NRUNPLC_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "clik.h"
#include "param_config_reader.h"

extern Config global_config;
extern double get_time(void);

#define MAX_CLVEC_SIZE 8000   
#define MAX_NUIS_BUF   50

// ============================================================================
//   Generalized Array-Based Likelihood Data Structures
// ============================================================================

typedef struct {
    int clik_index;         
    int task_index;         
    double default_val;     
    int is_estimated;       
} NuisanceMap;

typedef struct {
    clik_object *lik;
    char name[256];         
    int has_cl[6];          
    int lmax[6];            
    int num_nuis;
    parname *nuis_names;    
    NuisanceMap *map;       
    double *clvec_buffer;   
    double *nuis_buffer;    
} LikelihoodData;

typedef struct {
    LikelihoodData liks[MAX_LIKELIHOODS];
    int num_liks;
    int initialized;
    int max_lmax;            
    double norm_array[3000]; 
} ClikCache;

static ClikCache *get_clik_cache() {
    static ClikCache cache = {0};
    return &cache;
}

static inline void fill_clik_vector_fast(LikelihoodData *ld, double *TT, double *TE, double *EE, double *BB) 
{
    int clvec_idx = 0;
    double* spectra[6] = {TT, EE, BB, TE, NULL, NULL}; 
    
    for(int s = 0; s < 6; s++) {
        if (ld->has_cl[s]) {
            int len = ld->lmax[s] + 1;
            if (spectra[s] != NULL) {
                memcpy(&(ld->clvec_buffer[clvec_idx]), spectra[s], len * sizeof(double));
            } else {
                memset(&(ld->clvec_buffer[clvec_idx]), 0, len * sizeof(double));
            }
            clvec_idx += len;
        }
    }
    
    if (ld->num_nuis > 0) {
        memcpy(&(ld->clvec_buffer[clvec_idx]), ld->nuis_buffer, ld->num_nuis * sizeof(double));
    }
}

// ----------------------------------------------------------------------------
// Build O(1) Mapping
// ----------------------------------------------------------------------------
void build_fast_nuisance_map(LikelihoodData *ld, int lh_index) {
    if (ld->num_nuis <= 0) return;
    ld->map = malloc(ld->num_nuis * sizeof(NuisanceMap));

    for (int k = 0; k < ld->num_nuis; k++) {
        ld->map[k].clik_index = k;
        ld->map[k].task_index = -1; 
        ld->map[k].is_estimated = 0;
        ld->map[k].default_val = 0.0; 

        int global_task_idx = 0;
        for (int i = 0; i < global_config.param_count; i++) {
            ParameterConfig *p = &global_config.params[i];
            
            if (strcmp(ld->nuis_names[k], p->name) == 0) {
                if (p->state == PARAM_SAMPLED) {
                    ld->map[k].is_estimated = 1;
                    ld->map[k].task_index = global_task_idx;
                } else {
                    ld->map[k].default_val = p->fixed_value;
                    if (global_config.lh_nuisance_sample[lh_index] == 1) {
                        printf("  [WARNING] Likelihood %d permits nuisance sampling, but parameter '%s' is explicitly declared FIXED in param_reader.ini. Using fixed value: %g\n", lh_index + 1, p->name, p->fixed_value);
                    }
                }
                break;
            }
            if (p->state == PARAM_SAMPLED) global_task_idx++;
        }
    }
}

int initialize_clik_objects(error **err)
{
    ClikCache *cache = get_clik_cache();
    if (cache->initialized) return 1;

    for (int l = 2; l <= 2600; l++) {
        cache->norm_array[l] = 2.0 * M_PI / ((double)l * ((double)l + 1.0));
    }

    cache->num_liks = global_config.likelihood_count;
    cache->max_lmax = 0;

    for (int i = 0; i < cache->num_liks; i++) {
        char *path = global_config.likelihood_paths[i];
        
        char *last_slash = strrchr(path, '/');
        if (last_slash) strncpy(cache->liks[i].name, last_slash + 1, 255);
        else strncpy(cache->liks[i].name, path, 255);
        
        cache->liks[i].lik = clik_init(path, err);
        if (isError(*err)) return 0;

        clik_get_has_cl(cache->liks[i].lik, cache->liks[i].has_cl, err);
        clik_get_lmax(cache->liks[i].lik, cache->liks[i].lmax, err);
        
        for (int j = 0; j < 4; j++) { 
            if (cache->liks[i].has_cl[j] && cache->liks[i].lmax[j] > cache->max_lmax) {
                cache->max_lmax = cache->liks[i].lmax[j];
            }
        }
        
        int n_nuis = clik_get_extra_parameter_names(cache->liks[i].lik, &(cache->liks[i].nuis_names), err);
        cache->liks[i].num_nuis = n_nuis;

        int total_size = 0;
        for(int j=0; j<6; j++) {
            if(cache->liks[i].has_cl[j]) {
                total_size += (cache->liks[i].lmax[j] + 1);
            }
        }
        total_size += n_nuis;
        
        cache->liks[i].clvec_buffer = malloc(total_size * sizeof(double));
        cache->liks[i].nuis_buffer = malloc((n_nuis > 0 ? n_nuis : 1) * sizeof(double));

        build_fast_nuisance_map(&cache->liks[i], i);
    }
    
    if (cache->max_lmax > 2600) cache->max_lmax = 2600; 
    cache->initialized = 1;
    return 1;
}

static long plc_timing_calls = 0;
static double plc_lik_time[MAX_LIKELIHOODS][100];
static int plc_lik_valid[MAX_LIKELIHOODS][100];
static double plc_total_time[100];


static double run_plc(int rank, double *task, double *cl_tt_in, double *cl_te_in, double *cl_ee_in, double *cl_bb_in, int fast_mode, double *cached_slow_chi2)
{
    error *myerr = initError();
    if (!initialize_clik_objects(&myerr)) return 1e30;

    double plc_start_time = get_time();
    ClikCache *cache = get_clik_cache();
    double loglike = 0.0;

    static double TT[3000] = {0}, TE[3000] = {0}, EE[3000] = {0}, BB[3000] = {0};

    int lmax_needed = cache->max_lmax;
    #pragma omp simd
    for (int l = 2; l <= lmax_needed; l++) {
        double norm = cache->norm_array[l];
        TT[l] = cl_tt_in[l] * norm;
        TE[l] = cl_te_in[l] * norm;
        EE[l] = cl_ee_in[l] * norm;
        BB[l] = cl_bb_in[l] * norm;
    }

    for (int k = 0; k < cache->num_liks; k++) {
        LikelihoodData *ld = &cache->liks[k];

        for (int n = 0; n < ld->num_nuis; n++) {
            ld->nuis_buffer[n] = ld->map[n].is_estimated ? task[ld->map[n].task_index] : ld->map[n].default_val;
        }

        fill_clik_vector_fast(ld, TT, TE, EE, BB);

        double t0 = get_time();
        double res = clik_compute(ld->lik, ld->clvec_buffer, &myerr);
        double dt = get_time() - t0;

        plc_lik_time[k][plc_timing_calls % 100] = dt;
        plc_lik_valid[k][plc_timing_calls % 100] = 1;

        if (isError(myerr) || isnan(res) || isinf(res)) return 1e30;

        loglike += res;
    }

    plc_total_time[plc_timing_calls % 100] = get_time() - plc_start_time;
    plc_timing_calls++;

    if (rank == 1 && plc_timing_calls % 100 == 0) {
        printf("\n============================================================\n");
        printf("Likelihood Timing Profile: run_plc calls %ld - %ld\n", plc_timing_calls - 99, plc_timing_calls);
        printf("============================================================\n");
        double sum_total = 0.0;
        for (int i = 0; i < 100; i++) sum_total += plc_total_time[i];

        for (int k = 0; k < cache->num_liks; k++) {
            double sum_lik = 0.0;
            int count_lik = 0;
            for (int i = 0; i < 100; i++) {
                if (plc_lik_valid[k][i]) { sum_lik += plc_lik_time[k][i]; count_lik++; }
            }
            if (count_lik > 0) printf("Average LIK_%d [%s] : %.6f s (%d evaluations)\n", k+1, cache->liks[k].name, sum_lik / count_lik, count_lik);
        }
        printf("Average TOTAL Time      : %.6f s\n", sum_total / 100.0);
        printf("============================================================\n");
        fflush(stdout);
        memset(plc_lik_valid, 0, sizeof(plc_lik_valid));
    }

    return -2.0 * loglike;
}

void cleanup_clik_objects()
{
    ClikCache *cache = get_clik_cache();
    for (int i = 0; i < cache->num_liks; i++) {
        if (cache->liks[i].lik) clik_cleanup(&(cache->liks[i].lik));
        if (cache->liks[i].clvec_buffer) free(cache->liks[i].clvec_buffer);
        if (cache->liks[i].nuis_buffer) free(cache->liks[i].nuis_buffer);
        if (cache->liks[i].map) free(cache->liks[i].map);
    }
    cache->initialized = 0;
}

// use this in the main mcmc loop to calculate the likelihood for a given set of parameters and power spectra 
// so if someone wants to add other likelihood eg BAO, supernova they can add here instead of modifying the run_plc function
double calc_likelihood(double *task, double *cl_tt_in, double *cl_te_in, double *cl_ee_in, double *cl_bb_in, double *matt, int fast_mode, double *cached_slow_chi2) {
    return run_plc(0, task, cl_tt_in, cl_te_in, cl_ee_in, cl_bb_in, fast_mode, cached_slow_chi2);
}

void print_and_validate_parameters(int myrank) {
    error *myerr = initError();
    if (!get_clik_cache()->initialized) {
        if (!initialize_clik_objects(&myerr)) {
            if (myrank == 0) printf("[ERROR] Could not initialize CLIK to read parameters.\n");
            return;
        }
    }

    ClikCache *cache = get_clik_cache();
    int missing_count = 0;
    
    if (myrank == 0) {
        printf("\n========================================================================\n");
        printf("             CLIK LIKELIHOOD NUISANCE REQUIREMENTS                      \n");
        printf("========================================================================\n");
    }
    
    for (int i = 0; i < cache->num_liks; i++) {
        LikelihoodData *ld = &cache->liks[i];
        if (myrank == 0) {
            printf("Likelihood %d: %s\n", i + 1, ld->name);
            printf("  -> Total Required Nuisance Parameters: %d\n", ld->num_nuis);
        }
        
        if (ld->num_nuis > 0) {
            for (int k = 0; k < ld->num_nuis; k++) {
                int found_in_ini = 0;
                int is_sampled = 0;
                for (int p = 0; p < global_config.param_count; p++) {
                    if (strcmp(ld->nuis_names[k], global_config.params[p].name) == 0) {
                        found_in_ini = 1;
                        global_config.params[p].usage = USAGE_NUISANCE;
                        is_sampled = (global_config.params[p].state == PARAM_SAMPLED);
                        break;
                    }
                }
                
                if (myrank == 0) {
                    if (found_in_ini) {
                        printf("      [%02d] %-25s [FOUND - %s]\n", k, ld->nuis_names[k], is_sampled ? "SAMPLED" : "FIXED");
                    } else {
                        printf("      [%02d] %-25s [MISSING - FATAL]\n", k, ld->nuis_names[k]);
                    }
                }
                if (!found_in_ini) missing_count++;
            }
        } else if (myrank == 0) {
            printf("      (No nuisance parameters required)\n");
        }
        if (myrank == 0) printf("------------------------------------------------------------------------\n");
    }
    if (myrank == 0) printf("========================================================================\n\n");

    if (missing_count > 0) {
        if (myrank == 0) {
            printf("[FATAL ERROR] %d nuisance parameters are missing from param_reader.ini!\n", missing_count);
            printf("Please read the table above, add the missing parameters, and re-run. Exiting MCMC...\n");
        }
        exit(1);
    }
    
    const char* camb_names[] = {
        "Omega_m_h2", "Omega_b_h2", "h", "tau", "n_s", "A_s", 
        "w0", "wa", "m_nu", "N_nu", "alpha_s", "r", "n_t", "alpha_t", "Y_he", "Omega_k"
    };
    int n_camb_names = 16;
    int unmapped_count = 0;
    
    for (int p = 0; p < global_config.param_count; p++) {
        ParameterConfig *param = &global_config.params[p];
        if (param->usage == USAGE_NUISANCE) continue; 
        
        int is_camb = 0;
        for (int c = 0; c < n_camb_names; c++) {
            if (strcmp(param->name, camb_names[c]) == 0) {
                is_camb = 1;
                param->usage = USAGE_CAMB;
                param->target_index = c;
                break;
            }
        }
        
        if (!is_camb) {
            if (param->state == PARAM_SAMPLED) {
                if (myrank == 0) printf("[FATAL ERROR] Parameter '%s' is marked 'sample' but is not required by any configured likelihood or CAMB.\n", param->name);
                unmapped_count++;
            } else if (myrank == 0) {
                printf("[WARNING] Fixed parameter '%s' is not required by any configured likelihood. It will be ignored.\n", param->name);
            }
        }
    }
    
    if (unmapped_count > 0) {
        if (myrank == 0) printf("[FATAL ERROR] %d sampled parameters are unmapped! Exiting...\n", unmapped_count);
        exit(1);
    }
}
#endif
