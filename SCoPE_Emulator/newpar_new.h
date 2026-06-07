#ifndef _NEWPAR_H_
#define _NEWPAR_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "param_config_single.h"

// Access the global configuration defined in mcmc.c
extern Config global_config;

/*
================================================================================
 Modern CAMB Interface for MCMC (Auto-Mapping Version)
================================================================================
 1. Creates a standard input vector for Fortran (size 12).
 2. Fills it with defaults.
 3. Overwrites values using the Global Config map.
 4. Calls Fortran.
================================================================================
*/

// --- Declare the external Fortran subroutine ---
// Must match the signature in interface.f90
extern void camb_from_params_(double* task, int* lmax, double* scale_in,
                              double* cl_tt, double* cl_te, double* cl_ee, double* cl_bb);


int param_iface(int mid, double task[], double* cl_tt, double* cl_te, double* cl_ee, double* cl_bb)
{
    // --- 1. Define Standard CAMB Settings ---
    int l_max = 2600; 
    double scale_in = 7.4311e12; // Standard scaling (often ignored if CAMB_outputscale is set in .ini)
    
    // --- 2. Initialize the "Standard Vector" with Defaults ---
    // This ensures that if a parameter is missing from param.ini, CAMB gets a safe default.
    // Order matches 'interface.f90': 
    // [0:ombh2, 1:omch2, 2:h, 3:tau, 4:ns, 5:As, 6:w, 7:wa, 8:mnu, 9:nrun, 10:Neff, 11:omk]
    double camb_vector[12];
    camb_vector[0] = 0.0226;  // Omega_b h^2
    camb_vector[1] = 0.112;   // Omega_c h^2
    camb_vector[2] = 0.70;    // h
    camb_vector[3] = 0.09;    // tau
    camb_vector[4] = 0.96;    // n_s
    camb_vector[5] = 3.0;     // ln(10^10 A_s)
    camb_vector[6] = -1.0;    // w (Dark Energy)
    camb_vector[7] = 0.0;     // wa
    camb_vector[8] = 0.0;     // sum(m_nu)
    camb_vector[9] = 0.0;     // alpha_s (running)
    camb_vector[10] = 3.046;  // N_eff
    camb_vector[11] = 0.0;    // Omega_k

    // --- 3. The "Smart Loop": Fill Vector from MCMC Task ---
    int task_index = 0; // Tracks position in the MCMC chain array

    for(int i = 0; i < global_config.param_count; i++) {
        ParameterConfig* p = &global_config.params[i];
        double val;

        // A. Retrieve Value (either from MCMC chain or Fixed value)
        if (p->is_estimated) {
            val = task[task_index]; // Grab current estimated value
            task_index++;
        } else {
            val = p->lower_bound;   // Use fixed value from config
        }

        // B. Route to CAMB if mapped
        if (p->usage == USAGE_CAMB && p->target_index >= 0 && p->target_index < 12) {
            camb_vector[p->target_index] = val;
            
            // Debug print for rank 0 only (optional)
            // if(mid==0) printf("DEBUG: CAMB[%d] (%s) <--- %f\n", p->target_index, p->name, val);
        }
    }

    // --- 4. Call Fortran CAMB Engine ---
    // We pass the constructed 'camb_vector' instead of the raw 'task'
    camb_from_params_(camb_vector, &l_max, &scale_in, cl_tt, cl_te, cl_ee, cl_bb);

    // --- 5. Sanity Check ---
    if (!isfinite(cl_tt[2])) {
        fprintf(stderr,"[param_iface] Error: Rank %d produced NaN Cls.\n", mid);
        return 0; // Failure
    }

    return 1; // Success
}

#endif // _NEWPAR_H_