#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "clik.h"

// Thread-local storage for clik objects
typedef struct {
    clik_object *camspec;
    clik_object *commander; 
    clik_object *lowlike;
    int initialized;
} ClikCache;

// Function to get thread-local cache
ClikCache* get_clik_cache() {
    static __thread ClikCache cache = {0};
    return &cache;
}

// Proper dimension calculation
int get_clik_dimension(clik_object *clikid, error **err) {
    if (!clikid || !err) return -1;
    
    int has_cl[6], lmax[6];
    
    clik_get_has_cl(clikid, has_cl, err);
    if (isError(*err)) return -1;
    
    clik_get_lmax(clikid, lmax, err);
    if (isError(*err)) return -1;
    
    parname *nuis_names = NULL;
    int n_nuis = clik_get_extra_parameter_names(clikid, &nuis_names, err);
    if (isError(*err)) {
        if (nuis_names) free(nuis_names);
        return -1;
    }
    if (nuis_names) free(nuis_names);
    
    int ndim = n_nuis;
    for (int i = 0; i < 6; i++) {
        if (has_cl[i] && lmax[i] >= 0) {
            ndim += (lmax[i] + 1);
        }
    }
    
    return ndim;
}

// Initialize clik objects for this thread
int initialize_clik_objects(error **err) {
    ClikCache *cache = get_clik_cache();
    
    if (cache->initialized) return 1;
    
    // Initialize CAMspec
    cache->camspec = clik_init("/Users/deathmac/Documents/scope/mcmc/clik/CAMspec_v6.2TN_2013_02_26_dist.clik/", err);
    if (isError(*err)) {
        fprintf(stderr, "Error initializing CAMspec\n");
        return 0;
    }
    
    // Initialize Commander
    cache->commander = clik_init("/Users/deathmac/Documents/scope/mcmc/clik/commander_v4.1_lm49.clik/", err);
    if (isError(*err)) {
        fprintf(stderr, "Error initializing Commander\n");
        clik_cleanup(&cache->camspec);
        return 0;
    }
    
    // Initialize LowLike
    cache->lowlike = clik_init("/Users/deathmac/Documents/scope/mcmc/clik/lowlike_v222.clik", err);
    if (isError(*err)) {
        fprintf(stderr, "Error initializing Lowlike\n");
        clik_cleanup(&cache->camspec);
        clik_cleanup(&cache->commander);
        return 0;
    }
    
    cache->initialized = 1;
    return 1;
}

// Properly fill clik input vector
int fill_clik_vector(clik_object *clikid, double *clvec, 
                     const double *TT, const double *TE, 
                     const double *EE, const double *BB,
                     const double *nuisance, int n_nuis, error **err) {
    if (!clikid || !clvec) return 0;
    
    int has_cl[6], lmax[6];
    int offset = 0;
    
    clik_get_has_cl(clikid, has_cl, err);
    if (isError(*err)) return 0;
    
    clik_get_lmax(clikid, lmax, err);
    if (isError(*err)) return 0;
    
    // Fill spectra in the order clik expects
    for (int i = 0; i < 6; i++) {
        if (has_cl[i] && lmax[i] >= 0) {
            int length = lmax[i] + 1;
            const double *source = NULL;
            
            switch(i) {
                case 0: source = TT; break; // TT
                case 1: source = EE; break; // EE  
                case 2: source = BB; break; // BB
                case 3: source = TE; break; // TE
                default: source = NULL; break;
            }
            
            if (source) {
                // The source arrays (TT, TE, etc.) are size 2501.
                // clik_get_lmax will return the max l needed (e.g., 2500).
                // length will be lmax+1 (e.g., 2501).
                // This safeguard handles cases where clik needs *less* than l=2500
                // or (more importantly) if it needs *more* (up to 2501).
                int copy_length = (length <= 2501) ? length : 2501;
                memcpy(clvec + offset, source, copy_length * sizeof(double));
                if (copy_length < length) {
                    // Zero-pad if clik expects lmax > 2500 (which our source array doesn't have)
                    memset(clvec + offset + copy_length, 0, (length - copy_length) * sizeof(double));
                }
            } else {
                memset(clvec + offset, 0, length * sizeof(double));
            }
            offset += length;
        }
    }
    
    // Add nuisance parameters
    if (nuisance && n_nuis > 0) {
        memcpy(clvec + offset, nuisance, n_nuis * sizeof(double));
    }
    
    return 1;
}

// MODIFIED: Function signature changed to accept Cl arrays directly
double  run_plc_from_arrays(int rank1, double *cl_tt_in, double *cl_te_in, double *cl_ee_in, double *cl_bb_in) {
    double A[14] = {
        0.2035919e03, 0.7221782e02, 0.6017222e02,
        0.3253835e01, 0.5231600e02, 0.4644192e01, 
        0.8141186e00, 0.1000000e01, 0.6563661e00,
        0.1000592e01, 0.9973067e00, 0.0000000e00,
        0.1139657e01, 0.3847442e00
    };

    // --- Read input file section REMOVED ---
    /*
    FILE *readfile = fopen(filename, "r");
    ...
    fclose(readfile);
    */

    // Initialize spectra arrays (for C_l)
    double TT[2501] = {0.0};
    double TE[2501] = {0.0};
    double EE[2501] = {0.0}; 
    double BB[2501] = {0.0};

    // --- NEW: Normalize spectra from input arrays ---
    // The input arrays (cl_tt_in, etc.) are the D_l values from CAMB.
    // We convert them to C_l = D_l * 2*pi / (l*(l+1))
    // The Planck likelihood code only requires up to l=2500.
    for (int l = 2; l <= 2500; l++) {
        double norm = 3.14159256 / (l * (l + 1) / 2.0); // 2*pi / (l*(l+1))
        TT[l] = cl_tt_in[l] * norm;
        TE[l] = cl_te_in[l] * norm; 
        EE[l] = cl_ee_in[l] * norm;
        BB[l] = cl_bb_in[l] * norm;
    }
    // --- End of new section ---


    // Initialize error handling
    error *myerr = initError();
    
    // Initialize clik objects for this thread
    if (!initialize_clik_objects(&myerr)) {
        fprintf(stderr, "Rank %d: Failed to initialize clik objects\n", rank1);
        return 1e30;
    }
    
    ClikCache *cache = get_clik_cache();
    
    // Get dimensions
    int ndim_cam = get_clik_dimension(cache->camspec, &myerr);
    int ndim_comm = get_clik_dimension(cache->commander, &myerr);
    int ndim_low = get_clik_dimension(cache->lowlike, &myerr);
    
    if (ndim_cam <= 0 || ndim_comm <= 0 || ndim_low <= 0) {
        fprintf(stderr, "Rank %d: Error getting dimensions: %d,%d,%d\n", 
                rank1, ndim_cam, ndim_comm, ndim_low);
        return 1e30;
    }

    // Debug output
    printf("Rank %d: Dimensions - CAMspec: %d, Commander: %d, Lowlike: %d\n", 
           rank1, ndim_cam, ndim_comm, ndim_low);

    // Allocate input vectors
    double *clvec_cam = calloc(ndim_cam, sizeof(double));
    double *clvec_comm = calloc(ndim_comm, sizeof(double));
    double *clvec_low = calloc(ndim_low, sizeof(double));
    
    if (!clvec_cam || !clvec_comm || !clvec_low) {
        fprintf(stderr, "Rank %d: Memory allocation failed\n", rank1);
        if (clvec_cam) free(clvec_cam);
        if (clvec_comm) free(clvec_comm);
        if (clvec_low) free(clvec_low);
        return 1e30;
    }

    // Fill input vectors
    if (!fill_clik_vector(cache->camspec, clvec_cam, TT, TE, EE, BB, A, 14, &myerr)) {
        fprintf(stderr, "Rank %d: Error filling CAMspec vector\n", rank1);
        free(clvec_cam);
        free(clvec_comm);
        free(clvec_low);
        return 1e30;
    }
    
    if (!fill_clik_vector(cache->commander, clvec_comm, TT, TE, EE, BB, NULL, 0, &myerr)) {
        fprintf(stderr, "Rank %d: Error filling Commander vector\n", rank1);
        free(clvec_cam);
        free(clvec_comm);
        free(clvec_low);
        return 1e30;
    }
    
    if (!fill_clik_vector(cache->lowlike, clvec_low, TT, TE, EE, BB, NULL, 0, &myerr)) {
        fprintf(stderr, "Rank %d: Error filling Lowlike vector\n", rank1);
        free(clvec_cam);
        free(clvec_comm);
        free(clvec_low);
        return 1e30;
    }

    // Compute likelihoods
    double result1 = 0.0, result11 = 0.0, result2 = 0.0;
    int success = 1;
    
    result1 = clik_compute(cache->camspec, clvec_cam, &myerr);
    if (isError(myerr)) {
        fprintf(stderr, "Rank %d: CAMspec computation error\n", rank1);
        success = 0;
    }
    
    result11 = clik_compute(cache->commander, clvec_comm, &myerr);
    if (isError(myerr)) {
        fprintf(stderr, "Rank %d: Commander computation error\n", rank1);
        success = 0;
    }
    
    result2 = clik_compute(cache->lowlike, clvec_low, &myerr);
    if (isError(myerr)) {
        fprintf(stderr, "Rank %d: Lowlike computation error\n", rank1);
        success = 0;
    }

    // Cleanup
    free(clvec_cam);
    free(clvec_comm);
    free(clvec_low);

    if (!success) {
        return 1e30;
    }

    double total = -2.0 * (result1 + result11 + result2);
    
    printf("Rank %d: likelihoods = (%f, %f, %f) total = %f\n", 
           rank1, result1, result11, result2, total);

    return total;
}

// File-based likelihood function (for backward compatibility)
double run_plc(char filename[], int rank1) {
    double A[14] = {
        0.2035919e03, 0.7221782e02, 0.6017222e02,
        0.3253835e01, 0.5231600e02, 0.4644192e01, 
        0.8141186e00, 0.1000000e01, 0.6563661e00,
        0.1000592e01, 0.9973067e00, 0.0000000e00,
        0.1139657e01, 0.3847442e00
    };

    // Read input file
    FILE *readfile = fopen(filename, "r");
    if (!readfile) {
        fprintf(stderr, "Rank %d: Could not open input file '%s'\n", rank1, filename);
        return 1e30;
    }

    // Initialize spectra arrays
    double TT[2501] = {0.0};
    double TE[2501] = {0.0};
    double EE[2501] = {0.0}; 
    double BB[2501] = {0.0};

    // Read and normalize spectra
    int l;
    double TTd, TEd, EEd, BBd;
    
    for (int i = 2; i <= 2500; i++) {
        if (fscanf(readfile, "%d %lf %lf %lf %lf", &l, &TTd, &EEd, &BBd, &TEd) != 5) {
            fprintf(stderr, "Rank %d: Error reading at l=%d\n", rank1, i);
            fclose(readfile);
            return 1e30;
        }
        if (l != i) {
            fprintf(stderr, "Rank %d: Multipole mismatch: expected %d got %d\n", rank1, i, l);
            fclose(readfile);
            return 1e30;
        }
        
        double norm = 3.14159256 / (l * (l + 1) / 2.0);
        TT[l] = TTd * norm;
        TE[l] = TEd * norm; 
        EE[l] = EEd * norm;
        BB[l] = BBd * norm;
    }
    fclose(readfile);

    // Initialize error handling
    error *myerr = initError();
    
    // Initialize clik objects for this thread
    if (!initialize_clik_objects(&myerr)) {
        fprintf(stderr, "Rank %d: Failed to initialize clik objects\n", rank1);
        return 1e30;
    }
    
    ClikCache *cache = get_clik_cache();
    
    // Get dimensions
    int ndim_cam = get_clik_dimension(cache->camspec, &myerr);
    int ndim_comm = get_clik_dimension(cache->commander, &myerr);
    int ndim_low = get_clik_dimension(cache->lowlike, &myerr);
    
    if (ndim_cam <= 0 || ndim_comm <= 0 || ndim_low <= 0) {
        fprintf(stderr, "Rank %d: Error getting dimensions: %d,%d,%d\n", 
                rank1, ndim_cam, ndim_comm, ndim_low);
        return 1e30;
    }

    // Debug output
    printf("Rank %d: Dimensions - CAMspec: %d, Commander: %d, Lowlike: %d\n", 
           rank1, ndim_cam, ndim_comm, ndim_low);

    // Allocate input vectors
    double *clvec_cam = calloc(ndim_cam, sizeof(double));
    double *clvec_comm = calloc(ndim_comm, sizeof(double));
    double *clvec_low = calloc(ndim_low, sizeof(double));
    
    if (!clvec_cam || !clvec_comm || !clvec_low) {
        fprintf(stderr, "Rank %d: Memory allocation failed\n", rank1);
        if (clvec_cam) free(clvec_cam);
        if (clvec_comm) free(clvec_comm);
        if (clvec_low) free(clvec_low);
        return 1e30;
    }

    // Fill input vectors
    if (!fill_clik_vector(cache->camspec, clvec_cam, TT, TE, EE, BB, A, 14, &myerr)) {
        fprintf(stderr, "Rank %d: Error filling CAMspec vector\n", rank1);
        free(clvec_cam);
        free(clvec_comm);
        free(clvec_low);
        return 1e30;
    }
    
    if (!fill_clik_vector(cache->commander, clvec_comm, TT, TE, EE, BB, NULL, 0, &myerr)) {
        fprintf(stderr, "Rank %d: Error filling Commander vector\n", rank1);
        free(clvec_cam);
        free(clvec_comm);
        free(clvec_low);
        return 1e30;
    }
    
    if (!fill_clik_vector(cache->lowlike, clvec_low, TT, TE, EE, BB, NULL, 0, &myerr)) {
        fprintf(stderr, "Rank %d: Error filling Lowlike vector\n", rank1);
        free(clvec_cam);
        free(clvec_comm);
        free(clvec_low);
        return 1e30;
    }

    // Compute likelihoods
    double result1 = 0.0, result11 = 0.0, result2 = 0.0;
    int success = 1;
    
    result1 = clik_compute(cache->camspec, clvec_cam, &myerr);
    if (isError(myerr)) {
        fprintf(stderr, "Rank %d: CAMspec computation error\n", rank1);
        success = 0;
    }
    
    result11 = clik_compute(cache->commander, clvec_comm, &myerr);
    if (isError(myerr)) {
        fprintf(stderr, "Rank %d: Commander computation error\n", rank1);
        success = 0;
    }
    
    result2 = clik_compute(cache->lowlike, clvec_low, &myerr);
    if (isError(myerr)) {
        fprintf(stderr, "Rank %d: Lowlike computation error\n", rank1);
        success = 0;
    }

    // Cleanup
    free(clvec_cam);
    free(clvec_comm);
    free(clvec_low);

    if (!success) {
        return 1e30;
    }

    double total = -2.0 * (result1 + result11 + result2);
    
    printf("Rank %d: likelihoods = (%f, %f, %f) total = %f\n", 
           rank1, result1, result11, result2, total);

    return total;
}
// Cleanup function
void cleanup_clik_objects() {
    ClikCache *cache = get_clik_cache();
    
    if (cache->camspec) {
        clik_cleanup(&cache->camspec);
        cache->camspec = NULL;
    }
    if (cache->commander) {
        clik_cleanup(&cache->commander);
        cache->commander = NULL;
    }
    if (cache->lowlike) {
        clik_cleanup(&cache->lowlike);
        cache->lowlike = NULL;
    }
    
    cache->initialized = 0;
}
