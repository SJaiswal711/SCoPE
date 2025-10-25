#ifndef _NEWPAR_H_
#define _NEWPAR_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
================================================================================
 Modern CAMB Interface for MCMC
================================================================================
 This header provides the `param_iface` function to act as a modern
 replacement for the old file-based CAMB calling system. It calls the
 `camb_from_params_` Fortran subroutine directly for high performance.

 Workflow:
 1. The MCMC code calls `param_iface()` with the process ID (`mid`) and
    the cosmological parameters (`task[]`).
 2. It declares the external Fortran subroutine `camb_from_params_`.
 3. It dynamically allocates memory (using malloc) for the Cl arrays.
    This is essential for stability in a multi-process MPI environment.
 4. It calls the Fortran subroutine, which updates parameters and reruns CAMB.
 5. The lensed Cls are returned from Fortran into the C arrays.
 6. This function writes the Cls into the output file that the `run_plc()`
    likelihood function expects.
 7. It frees the allocated memory and returns a success/failure code.
================================================================================
*/


// --- Declare the external Fortran subroutine ---
// This tells C about the function from your `interface2.f90` module.
extern void camb_from_params_(double* task, int* lmax, double* scale_in,
                              double* cl_tt, double* cl_te, double* cl_ee, double* cl_bb);


// --- Define the main interface function ---
// This function will be called by your MCMC code.
int param_iface(int mid, double task[])
{
    // --- Define CAMB settings ---
    int l_max = 2700; // Maximum multipole. Must match `l_max_scalar` in .ini
    double scale_in = 1.0; // Dummy value. Scaling is now correctly handled by `CMB_outputscale` in the .ini file.

    // --- Allocate memory for the output Cl arrays ---
    // Using malloc is thread-safe for MPI, as seen in `test_camb_2.c`.
    double* cl_tt = (double*)malloc((l_max + 1) * sizeof(double));
    double* cl_te = (double*)malloc((l_max + 1) * sizeof(double));
    double* cl_ee = (double*)malloc((l_max + 1) * sizeof(double));
    double* cl_bb = (double*)malloc((l_max + 1) * sizeof(double));

    // --- Robust error checking for memory allocation ---
    if (!cl_tt || !cl_te || !cl_ee || !cl_bb) {
        fprintf(stderr, "MCMC-ERROR (rank %d): Failed to allocate memory for Cl arrays.\n", mid);
        free(cl_tt); free(cl_te); free(cl_ee); free(cl_bb); // Free any that succeeded
        return 0; // Return 0 to indicate failure
    }

    // --- Call the Fortran CAMB engine ---
    // The `task` array with new cosmological parameters is passed directly.
    camb_from_params_(task, &l_max, &scale_in, cl_tt, cl_te, cl_ee, cl_bb);

    // --- Write results to the output file for the likelihood code ---
    char filename[128];
    sprintf(filename, "test_%d_lensedCls.dat", mid);
    FILE* outfile = fopen(filename, "w");
    if (outfile == NULL) {
        fprintf(stderr, "MCMC-ERROR (rank %d): Could not open file '%s' for writing.\n", mid, filename);
        free(cl_tt); free(cl_te); free(cl_ee); free(cl_bb);
        return 0; // Indicate failure
    }

    // Write the lensed spectra, starting from l=2 as expected by the likelihood code.
    for (int l = 2; l <= l_max; ++l) {
        fprintf(outfile, "%5d   %.8e   %.8e   %.8e   %.8e\n",
                l,
                cl_tt[l],
                cl_ee[l],
                cl_bb[l],
                cl_te[l]);
    }
    fclose(outfile);

    // --- Clean up allocated memory ---
    free(cl_tt);
    free(cl_te);
    free(cl_ee);
    free(cl_bb);

    // --- Return 1 to indicate success ---
    return 1;
}

#endif // _NEWPAR_H_
