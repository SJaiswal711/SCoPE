#ifndef _NRUNPLC_H_
#define _NRUNPLC_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "clik.h"
#include "param_config_single.h"

extern Config global_config;

/* =======================
   Clik cache (process-wide)
   ======================= */
typedef struct
{
    clik_object *camspec;
    clik_object *commander;
    clik_object *lowlike;
    int initialized;
    int n_nuis_total;
} ClikCache;

static ClikCache *get_clik_cache()
{
    static ClikCache cache = {0};
    return &cache;
}

/* =======================
   Initialize clik - ONCE PER PROCESS
   ======================= */
int initialize_clik_objects(error **err)
{
    ClikCache *cache = get_clik_cache();

    if (cache->initialized)
    {
        // printf("[DEBUG] clik already initialized\n");
        return 1;
    }
    // if (cache->camspec) {
    // map_parameters(&global_config, cache->camspec);   // <<< ADD THIS LINE

    // printf("[DEBUG] initialize_clik_objects() entered\n");
    // printf("[DEBUG] likelihood_count = %d\n", global_config.likelihood_count);

    if (global_config.likelihood_count == 0)
    {
        // printf("[ERROR] likelihood_count == 0\n");
        return 0;
    }

    for (int i = 0; i < global_config.likelihood_count && i < 3; i++)
    {
        // printf("[DEBUG] Likelihood[%d] path = %s\n",
        //        i, global_config.likelihood_paths[i]);
    }

    // Initialize ONLY if not already done
    if (!cache->initialized)
    {
        if (global_config.likelihood_count > 0)
        {
            cache->camspec = clik_init(global_config.likelihood_paths[0], err);
            // printf("[DEBUG] camspec ptr = %p\n", (void *)cache->camspec);
        }

        if (global_config.likelihood_count > 1)
        {
            cache->commander = clik_init(global_config.likelihood_paths[1], err);
            // printf("[DEBUG] commander ptr = %p\n", (void *)cache->commander);
        }

        if (global_config.likelihood_count > 2)
        {
            cache->lowlike = clik_init(global_config.likelihood_paths[2], err);
            // printf("[DEBUG] lowlike ptr = %p\n", (void *)cache->lowlike);
        }

        if (isError(*err))
        {
            printf("[ERROR] clik_init failed\n");
            printError(stderr, *err);
            return 0;
        }

        if (cache->camspec)
        {
            parname *nuis_names = NULL;
            cache->n_nuis_total =
                clik_get_extra_parameter_names(cache->camspec, &nuis_names, err);

            // printf("[DEBUG] CAMspec n_nuis_total = %d\n", cache->n_nuis_total);

            if (!isError(*err) && nuis_names)
                free(nuis_names);
            map_parameters(&global_config, cache->camspec);
        }

        cache->initialized = 1;
        // printf("[DEBUG] clik initialization DONE\n");
    }

    return 1;
}

/* =======================
   Dimension helper
   ======================= */
int get_clik_dimension(clik_object *clikid, error **err)
{
    if (!clikid)
        return 0;

    int has_cl[6], lmax[6];
    clik_get_has_cl(clikid, has_cl, err);
    clik_get_lmax(clikid, lmax, err);

    // printf("[DEBUG] has_cl: ");
    // for (int i = 0; i < 6; i++)
    //     printf("%d ", has_cl[i]);
    // printf("\n");

    // printf("[DEBUG] lmax: ");
    // for (int i = 0; i < 6; i++)
    //     printf("%d ", lmax[i]);
    // printf("\n");

    parname *names = NULL;
    int n_nuis = clik_get_extra_parameter_names(clikid, &names, err);
    if (n_nuis > 0 && names)
        free(names);

    int dim = n_nuis;
    for (int i = 0; i < 6; i++)
        if (has_cl[i] && lmax[i] >= 0)
            dim += (lmax[i] + 1);

    // printf("[DEBUG] clik dimension = %d (nuis=%d)\n", dim, n_nuis);
    return dim;
}

/* =======================
   Fill clik vector
   ======================= */
int fill_clik_vector_generic(
    clik_object *clikid,
    double *clvec,
    const double *TT,
    const double *TE,
    const double *EE,
    const double *BB,
    const double *nuisance_vector,
    int n_nuis_provided,
    error **err)
{

    int has_cl[6], lmax[6];
    int offset = 0;

    clik_get_has_cl(clikid, has_cl, err);
    clik_get_lmax(clikid, lmax, err);

    for (int i = 0; i < 6; i++)
    {
        if (has_cl[i] && lmax[i] >= 0)
        {
            int len = lmax[i] + 1;
            const double *src = NULL;

            if (i == 0)
                src = TT;
            if (i == 1)
                src = EE;
            if (i == 2)
                src = BB;
            if (i == 3)
                src = TE;

            memset(clvec + offset, 0, len * sizeof(double));

            if (src)
            {
                int copy = (len <= 2501) ? len : 2501;
                memcpy(clvec + offset, src, copy * sizeof(double));
            }

            // printf("[DEBUG] filled CL[%d] block, offset=%d len=%d\n",
                //    i, offset, len);

            offset += len;
        }
    }

    parname *names = NULL;
    int n_expected = clik_get_extra_parameter_names(clikid, &names, err);
    if (n_expected > 0 && names)
        free(names);

    // printf("[DEBUG] nuisance expected=%d provided=%d offset=%d\n",
        //    n_expected, n_nuis_provided, offset);

    if (n_expected > 0)
    {
        if (n_nuis_provided >= n_expected)
            memcpy(clvec + offset, nuisance_vector, n_expected * sizeof(double));
        else
            memset(clvec + offset, 0, n_expected * sizeof(double));
    }

    return 1;
}

/* =======================
   Main likelihood
   ======================= */
double run_plc(
    int rank,
    double *task,
    double *cl_tt_in,
    double *cl_te_in,
    double *cl_ee_in,
    double *cl_bb_in)
{

    error *myerr = initError();
    /* ---- ADD THESE DECLARATIONS AT THE TOP ---- */
    double loglike_sum = 0.0;
    double *dynamic_nuisance = NULL;

    // printf("\n[DEBUG][Rank %d] run_plc() entered\n", rank);

    if (!initialize_clik_objects(&myerr))
    {
        // printf("[ERROR][Rank %d] clik init failed\n", rank);
        return 1e30;
    }

    double TT[2601] = {0}, TE[2601] = {0},
           EE[2601] = {0}, BB[2601] = {0};

    for (int l = 2; l <= 2600; l++)
    {
        double norm = 2.0 * M_PI / (l * (l + 1.0));
        // norm = 1;
        TT[l] = cl_tt_in[l] * norm;
        TE[l] = cl_te_in[l] * norm;
        EE[l] = cl_ee_in[l] * norm;
        BB[l] = cl_bb_in[l] * norm;
    }
    if (!isfinite(TT[2]) || TT[2] <= 0.0)
    {
        printf("[FATAL][Rank %d] TT[2]=%e invalid\n", rank, TT[2]);
    }

    // printf("[DEBUG][Rank %d] TT[2]=%.3e TT[1000]=%.3e\n",
        //    rank, TT[2], TT[1000]);

    ClikCache *cache = get_clik_cache();
    int n_nuis = (cache->n_nuis_total > 0) ? cache->n_nuis_total : 1;

    double *nuis = calloc(n_nuis, sizeof(double));

    int idx = 0;
    for (int i = 0; i < global_config.param_count; i++)
    {
        ParameterConfig *p = &global_config.params[i];
        double val = p->is_estimated ? task[idx++] : p->lower_bound;
        if (p->usage == USAGE_NUISANCE &&
            p->target_index >= 0 &&
            p->target_index < n_nuis)
        {
            nuis[p->target_index] = val;
            // printf("[DEBUG] nuisance[%d]=%g\n", p->target_index, val);
        }
    }

    double loglike = 0.0;
    clik_object *liks[3] = {
        cache->camspec,
        cache->commander,
        cache->lowlike};

    for (int k = 0; k < 3; k++)
    {
        if (!liks[k])
            continue;

        int ndim = get_clik_dimension(liks[k], &myerr);
        double *clvec = calloc(ndim, sizeof(double));

        if (liks[k] == cache->camspec)
        {
            fill_clik_vector_generic(
                liks[k], clvec, TT, TE, EE, BB,
                nuis, n_nuis, &myerr);
        }
        else
        {
            fill_clik_vector_generic(
                liks[k], clvec, TT, TE, EE, BB,
                NULL, 0, &myerr); // <<< NO NUISANCE
        }

        // double res = clik_compute(liks[k], clvec, &myerr);
        double res = clik_compute(liks[k], clvec, &myerr);

        if (isError(myerr) || isnan(res) || isinf(res))
        {
            printf("[ERROR][Rank %d] lik[%d] returned NaN/Inf\n", rank, k);
            free(clvec);
            free(dynamic_nuisance);
            return 1e30; // HARD reject
        }

        // printf("[DEBUG][Rank %d] lik[%d]=%.6f\n", rank, k, res);
        loglike_sum += res;

        // printf("[DEBUG][Rank %d] lik[%d]=%f\n", rank, k, res);

        if (isError(myerr))
        {
            printError(stderr, myerr);
            free(clvec);
            free(nuis);
            return 1e30;
        }

        loglike += res;
        free(clvec);
    }

    free(nuis);

    double chi2 = -2.0 * loglike;
    // printf("[DEBUG][Rank %d] TOTAL chi2 = %f\n", rank, chi2);

    return chi2;
}

void cleanup_clik_objects()
{
    ClikCache *cache = get_clik_cache();
    if (cache->camspec)
        clik_cleanup(&cache->camspec);
    if (cache->commander)
        clik_cleanup(&cache->commander);
    if (cache->lowlike)
        clik_cleanup(&cache->lowlike);
    cache->initialized = 0;
}

#endif