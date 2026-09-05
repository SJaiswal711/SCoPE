// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// //                                                                     Version inculdes timming and fast_mode

// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef _NRUNPLC_H_
#define _NRUNPLC_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "clik.h"
#include "param_config_single.h"

extern Config global_config;

#define MAX_CLVEC_SIZE 8000   
#define MAX_NUIS_BUF   50

typedef struct {
    clik_object *obj;
    int has_cl[6];
    int lmax[6];
    int n_nuis;
    int cl_offset[6];   
    int cl_total_len;   
    int dimension;       
} ClikLikelihoodInfo;

typedef struct {
    ClikLikelihoodInfo camspec;
    ClikLikelihoodInfo commander;
    ClikLikelihoodInfo lowlike;
    int initialized;
    int max_lmax;        
} ClikCache;

static ClikCache *get_clik_cache() {
    static ClikCache cache = {0};
    return &cache;
}

static void cache_likelihood_info(ClikLikelihoodInfo *info, clik_object *obj, error **err) {
    info->obj = obj;
    info->dimension = 0;
    info->n_nuis = 0;
    info->cl_total_len = 0;
    if (!obj) return;

    clik_get_has_cl(obj, info->has_cl, err);
    clik_get_lmax(obj, info->lmax, err);

    parname *names = NULL;
    info->n_nuis = clik_get_extra_parameter_names(obj, &names, err);
    if (info->n_nuis > 0 && names) free(names);

    int offset = 0;
    for (int i = 0; i < 6; i++) {
        info->cl_offset[i] = offset;
        if (info->has_cl[i] && info->lmax[i] >= 0) {
            offset += (info->lmax[i] + 1);
        }
    }
    info->cl_total_len = offset;
    info->dimension = offset + (info->n_nuis > 0 ? info->n_nuis : 0);
}

static int initialize_clik_objects(error **err) {
    ClikCache *cache = get_clik_cache();
    if (cache->initialized) return 1;
    if (global_config.likelihood_count == 0) return 0;

    clik_object *camspec_obj = NULL, *commander_obj = NULL, *lowlike_obj = NULL;

    if (global_config.likelihood_count > 0)
        camspec_obj = clik_init(global_config.likelihood_paths[0], err);
    if (global_config.likelihood_count > 1)
        commander_obj = clik_init(global_config.likelihood_paths[1], err);
    if (global_config.likelihood_count > 2)
        lowlike_obj = clik_init(global_config.likelihood_paths[2], err);

    if (isError(*err)) return 0;

    cache_likelihood_info(&cache->camspec, camspec_obj, err);
    cache_likelihood_info(&cache->commander, commander_obj, err);
    cache_likelihood_info(&cache->lowlike, lowlike_obj, err);

    if (cache->camspec.obj) map_parameters(&global_config, cache->camspec.obj);

    int max_lmax = 0;
    ClikLikelihoodInfo *all[3] = { &cache->camspec, &cache->commander, &cache->lowlike };
    for (int k = 0; k < 3; k++) {
        if (!all[k]->obj) continue;
        for (int i = 0; i < 4; i++) {
            if (all[k]->has_cl[i] && all[k]->lmax[i] > max_lmax) max_lmax = all[k]->lmax[i];
        }
    }
    if (max_lmax > 2600) max_lmax = 2600; 
    cache->max_lmax = max_lmax;

    cache->initialized = 1;
    return 1;
}

static void fill_clik_vector_cached(
    ClikLikelihoodInfo *info,
    double *clvec,
    const double *TT, const double *TE, const double *EE, const double *BB,
    const double *nuisance_vector, int n_nuis_provided)
{
    for (int i = 0; i < 6; i++) {
        if (info->has_cl[i] && info->lmax[i] >= 0) {
            int len = info->lmax[i] + 1;
            const double *src = NULL;
            if (i == 0) src = TT;
            if (i == 1) src = EE;
            if (i == 2) src = BB;
            if (i == 3) src = TE;
            if (src) {
                int copy = (len <= 2601) ? len : 2601;
                memcpy(clvec + info->cl_offset[i], src, copy * sizeof(double));
            }
        }
    }
    if (info->n_nuis > 0 && n_nuis_provided >= info->n_nuis) {
        memcpy(clvec + info->cl_total_len, nuisance_vector, info->n_nuis * sizeof(double));
    }
}

// -------------------------------------------------------------------------
// NEW RUN_PLC: Now features fast_mode to bypass Commander and Lowlike!
// -------------------------------------------------------------------------

// -------------------------------------------------------------------------
// Timing profile: every 100 run_plc() calls
// -------------------------------------------------------------------------
static long plc_timing_calls = 0;
static double plc_call_time = 0.0;

static double plc_camspec_time[100];
static double plc_commander_time[100];
static double plc_lowlike_time[100];
static double plc_total_time[100];

static int plc_camspec_valid[100];
static int plc_commander_valid[100];
static int plc_lowlike_valid[100];

static double run_plc(
    int rank, double *task,
    double *cl_tt_in, double *cl_te_in, double *cl_ee_in, double *cl_bb_in,
    int fast_mode, double *cached_other_chi2)
{
    error *myerr = initError();
    if (!initialize_clik_objects(&myerr)) return 1e30;

    double plc_start_time = get_time();

    ClikCache *cache = get_clik_cache();

    static double norm[2601];
    static int norm_ready = 0;
    if (!norm_ready) {
        norm[0] = 0.0; norm[1] = 0.0;
        for (int l = 2; l <= 2600; l++) norm[l] = 2.0 * M_PI / (l * (l + 1.0));
        norm_ready = 1;
    }

    static double TT[2601] = {0}, TE[2601] = {0}, EE[2601] = {0}, BB[2601] = {0};
    
    int lmax_needed = cache->max_lmax;
    
    // IF IN FAST MODE, THE SPECTRA HAVEN'T CHANGED! SKIP THIS LOOP!
    if (!fast_mode) {
        #pragma omp simd
        for (int l = 2; l <= lmax_needed; l++) {
            TT[l] = cl_tt_in[l] * norm[l];
            TE[l] = cl_te_in[l] * norm[l];
            EE[l] = cl_ee_in[l] * norm[l];
            BB[l] = cl_bb_in[l] * norm[l];
        }
    }

    static double nuis[MAX_NUIS_BUF];
    int n_nuis = cache->camspec.n_nuis;
    if (n_nuis <= 0) n_nuis = 1;
    if (n_nuis > MAX_NUIS_BUF) return 1e30;
    memset(nuis, 0, n_nuis * sizeof(double));

    int idx = 0;
    for (int i = 0; i < global_config.param_count; i++) {
        ParameterConfig *p = &global_config.params[i];
        double val = p->is_estimated ? task[idx++] : p->lower_bound;
        if (p->usage == USAGE_NUISANCE && p->target_index >= 0 && p->target_index < n_nuis) {
            nuis[p->target_index] = val;
        }
    }

    double loglike_camspec = 0.0;
    double loglike_other = 0.0;
    ClikLikelihoodInfo *infos[3] = { &cache->camspec, &cache->commander, &cache->lowlike };
    static double clvec[MAX_CLVEC_SIZE];

    for (int k = 0; k < 3; k++) {
        ClikLikelihoodInfo *info = infos[k];
        if (!info->obj) continue;

        int is_camspec = (info == &cache->camspec);

        // MASSIVE OPTIMIZATION: Skip evaluating Commander & Lowlike in fast mode!
        if (fast_mode && !is_camspec) continue; 

        int ndim = info->dimension;
        if (ndim > MAX_CLVEC_SIZE) return 1e30;
        memset(clvec, 0, ndim * sizeof(double));

        fill_clik_vector_cached(info, clvec, TT, TE, EE, BB, is_camspec ? nuis : NULL, is_camspec ? n_nuis : 0);

        // Timing the clik_compute call

        double t0 = get_time();

        double res = clik_compute(info->obj, clvec, &myerr);

        double dt = get_time() - t0;

        if (k == 0) {
            plc_camspec_time[plc_timing_calls % 100] = dt;
            plc_camspec_valid[plc_timing_calls % 100] = 1;
        }
        else if (k == 1) {
            plc_commander_time[plc_timing_calls % 100] = dt;
            plc_commander_valid[plc_timing_calls % 100] = 1;
        }
        else if (k == 2) {
            plc_lowlike_time[plc_timing_calls % 100] = dt;
            plc_lowlike_valid[plc_timing_calls % 100] = 1;
        }

        // timing end

        if (isError(myerr) || isnan(res) || isinf(res)) return 1e30;

        if (is_camspec) loglike_camspec += res;
        else loglike_other += res;
    }

    plc_call_time = get_time() - plc_start_time;
    plc_total_time[plc_timing_calls % 100] = plc_call_time;
    plc_timing_calls++;

    if (rank == 1 && plc_timing_calls % 100 == 0) {

        printf("\n");
        printf("============================================================\n");
        printf("Likelihood Timing Profile: run_plc calls %ld - %ld\n",
               plc_timing_calls - 99,
               plc_timing_calls);
        printf("============================================================\n");

        double sum_camspec = 0.0, sum_commander = 0.0, sum_lowlike = 0.0, sum_total = 0.0;
        int count_camspec = 0, count_commander = 0, count_lowlike = 0;

        for (int i = 0; i < 100; i++) {
            if (plc_camspec_valid[i]) { sum_camspec += plc_camspec_time[i]; count_camspec++; }
            if (plc_commander_valid[i]) { sum_commander += plc_commander_time[i]; count_commander++; }
            if (plc_lowlike_valid[i]) { sum_lowlike += plc_lowlike_time[i]; count_lowlike++; }
            sum_total += plc_total_time[i];
        }

        printf("Average CAMspec   : %.6f s (%d evaluations)\n", 
               count_camspec > 0 ? sum_camspec / count_camspec : 0.0, count_camspec);
        printf("Average Commander : %.6f s (%d evaluations)\n", 
               count_commander > 0 ? sum_commander / count_commander : 0.0, count_commander);
        printf("Average Lowlike   : %.6f s (%d evaluations)\n", 
               count_lowlike > 0 ? sum_lowlike / count_lowlike : 0.0, count_lowlike);
        printf("Average Total     : %.6f s\n", sum_total / 100.0);
        
        printf("============================================================\n");
        fflush(stdout);

        // Reset tracking arrays for the next 100 calls
        memset(plc_camspec_valid, 0, sizeof(plc_camspec_valid));
        memset(plc_commander_valid, 0, sizeof(plc_commander_valid));
        memset(plc_lowlike_valid, 0, sizeof(plc_lowlike_valid));
    }

    // Return logic based on mode
    if (fast_mode) {
        return -2.0 * loglike_camspec + (*cached_other_chi2); // Add cached value!
    } else {
        if (cached_other_chi2) *cached_other_chi2 = -2.0 * loglike_other; // Save value for later!
        return -2.0 * (loglike_camspec + loglike_other);
    }
}
#endif

