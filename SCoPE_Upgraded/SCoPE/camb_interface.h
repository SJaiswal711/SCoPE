#ifndef _CAMB_INTERFACE_H_
#define _CAMB_INTERFACE_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "param_config_reader.h"

extern Config global_config;

/* Full perturbations (CMB Cls) */
extern void camb_from_params_(double *task_f, int *lmax_in, double *scale_in,
                              int *num_k, double *k_in, double *z_pk,
                              double *cl_tt, double *cl_te, double *cl_ee,
                              double *cl_bb, double *cl_pp, double *pk_out);

/* Background only (H(z), D_M(z), r_d) */
// extern void camb_background_(double *task_f, int *num_z, double *z_in,
//                              double *rd_out, double *zdrag_out,
//                              double *H_out, double *DM_out, double *DH_out);

static inline void build_camb_vector(double task[], double camb_vector[16]) 
{
    /* Defaults matching standard CAMB if parameter is missing from .ini */
    camb_vector[ 0] = 0.1346;  /* Omega_m h^2  */
    camb_vector[ 1] = 0.0226;  /* Omega_b h^2  */
    camb_vector[ 2] = 0.70;    /* h            */
    camb_vector[ 3] = 0.09;    /* tau          */
    camb_vector[ 4] = 0.96;    /* n_s          */
    camb_vector[ 5] = 3.0;     /* ln(1e10 A_s) */
    camb_vector[ 6] = -1.0;    /* w            */
    camb_vector[ 7] = 0.0;     /* wa           */
    camb_vector[ 8] = 0.0;     /* m_nu         */
    camb_vector[ 9] = 3.046;   /* N_nu         */
    camb_vector[10] = 0.0;     /* alpha_s      */
    camb_vector[11] = 0.0;     /* r            */
    camb_vector[12] = 0.0;     /* n_t          */
    camb_vector[13] = 0.0;     /* alpha_t      */
    camb_vector[14] = 0.245;   /* Y_he         */
    camb_vector[15] = 0.0;     /* Omega_k      */

    int task_index = 0;
    for(int i = 0; i < global_config.param_count; i++) {
        ParameterConfig* p = &global_config.params[i];
        double val;

        if (p->is_estimated) {
            val = task[task_index];
            task_index++;
        } else {
            val = p->fixed_value;
        }

        if (p->usage == USAGE_CAMB && p->target_index >= 0 && p->target_index < 16) {
            camb_vector[p->target_index] = val;
        }
    }
}

/* ============================================================================
   1. CMB Perturbations (Slow, ~1 sec)
   ============================================================================ */
int param_iface(int mid, double task[], double* cl_tt, double* cl_te, double* cl_ee, double* cl_bb)
{
    int    lmax_in  = 2600; 
    double scale_in = 7.4311e12;
    int    num_k    = 0;
    double z_pk     = 0.0;
    double dummy_k[1] = {0.0};

    /* FIX: Valid buffers for Fortran (NULL was crashing) */
    static double cl_pp_dummy[2601] = {0};
    static double pk_dummy[1] = {0};
    
    double camb_vector[16];
    build_camb_vector(task, camb_vector);

    double o_m = camb_vector[0]; 
    double o_b = camb_vector[1]; 
    double h   = camb_vector[2]; 
    double tau = camb_vector[3]; 

    if (o_b >= o_m) return 0;               
    if (o_m <= 0.0 || o_b <= 0.0) return 0; 
    if (h <= 0.2 || h >= 1.5) return 0;     
    if (tau < 0.0 || tau > 0.5) return 0;

    camb_from_params_(camb_vector, &lmax_in, &scale_in, 
                      &num_k, dummy_k, &z_pk, 
                      cl_tt, cl_te, cl_ee, cl_bb, 
                      cl_pp_dummy, pk_dummy);  /* NOT NULL anymore */

    if (!isfinite(cl_tt[3]) || isnan(cl_tt[0])) {
        return 0;
    }
    return 1;
}

/* ============================================================================
   2. Background Only — Pure Analytic (Fast, ~50 µs, no CAMB call)
   ============================================================================
   Computes H(z), D_M(z), D_H(z) directly from the cosmological parameters.
   No CAMB_GetResults call → no NULL-pointer segfault, no 200 ms overhead.

   Physics:
       E²(z) = Ω_m (1+z)³ + Ω_r (1+z)⁴ + Ω_k (1+z)² + Ω_DE · f(z)
       f(z)  = (1+z)^[3(1+w0+wa)] · exp(-3 wa z / (1+z))
       H(z)  = H0 · E(z)
       D_M(z) = (c/H0) · ∫₀ᶻ dz'/E(z')

   rdrag / zdrag use Eisenstein & Hu (1998) fitting formulas (~1% accurate).
   ============================================================================ */
int param_iface_bg(double task[], int num_z, double *z_in,
                   double *rd_out, double *zdrag_out,
                   double *H_out, double *DM_out, double *DH_out)
{
    double camb_vector[16];
    build_camb_vector(task, camb_vector);

    double ombh2 = camb_vector[1];                 /* Ω_b h² */
    double omch2 = camb_vector[0] - ombh2;         /* Ω_c h² */
    double h     = camb_vector[2];
    double mnu   = camb_vector[8];                 /* Σ m_ν [eV] */
    double neff  = camb_vector[9];
    double w0    = camb_vector[6];
    double wa    = camb_vector[7];
    double omk   = camb_vector[15];

    /* Physical bounds */
    if (h <= 0.2 || h >= 1.5)         return 0;
    if (ombh2 <= 0.0)                 return 0;
    if (omch2 <  0.0)                 return 0;
    if (mnu   <  0.0)                 return 0;
    if (neff  <  0.0 || neff > 10.0)  return 0;
    if (omk   < -0.5 || omk > 0.5)    return 0;

    double h2 = h * h;
    double H0 = 100.0 * h;                         /* km/s/Mpc */
    const double c_km_s = 299792.458;

    /* Density parameters today (fraction of critical density) */
    double Om_b  = ombh2 / h2;
    double Om_c  = omch2 / h2;
    double Om_nu = (mnu / 93.14) / h2;
    double Om_m  = Om_b + Om_c + Om_nu;

    double Tcmb = 2.7255;
    double Og   = 2.469e-5 * pow(Tcmb / 2.7255, 4.0) / h2;
    double Onu  = 0.2271 * neff * Og;
    double Om_r = Og + Onu;

    double Om_k  = omk;
    double Om_DE = 1.0 - Om_m - Om_r - Om_k;
    if (Om_DE < 0.0) return 0;

    /* rdrag / zdrag via Eisenstein & Hu (1998) */
    double Ob_h2 = ombh2;
    double Om_h2 = ombh2 + omch2;
    double z_d = 1291.0 * pow(Om_h2, 0.251)
               / (1.0 + 0.659 * pow(Om_h2, 0.828))
               * (1.0 + 0.0063 * (1.0 / pow(Ob_h2, 0.44) - 1.0));
    double rdrag = 44.5 * log(9.83 / Om_h2)
                 / sqrt(1.0 + 10.0 * pow(Ob_h2, 0.75));
    *rd_out    = rdrag;
    *zdrag_out = z_d;
    if (!isfinite(rdrag) || rdrag <= 0.0) return 0;

    /* --- H(z), D_H(z) --- */
    int    wa_is_zero = (fabs(wa) < 1e-10);
    for (int i = 0; i < num_z; i++) {
        double z  = z_in[i];
        double op = 1.0 + z;

        double DE_f = wa_is_zero
            ? pow(op, 3.0 * (1.0 + w0))
            : pow(op, 3.0 * (1.0 + w0 + wa)) * exp(-3.0 * wa * z / op);

        double Ez2 = Om_m * op * op * op
                   + Om_r * op * op * op * op
                   + Om_k * op * op
                   + Om_DE * DE_f;
        if (Ez2 <= 0.0) return 0;

        H_out[i]  = H0 * sqrt(Ez2);
        DH_out[i] = c_km_s / H_out[i];
    }

    /* --- D_M(z) via cumulative trapezoidal integration (assumes sorted z) --- */
    double cumul = 0.0;
    for (int i = 0; i < num_z; i++) {
        if (i > 0 && z_in[i] > z_in[i-1]) {
            double zm = 0.5 * (z_in[i-1] + z_in[i]);
            double op = 1.0 + zm;

            double DE_f = wa_is_zero
                ? pow(op, 3.0 * (1.0 + w0))
                : pow(op, 3.0 * (1.0 + w0 + wa)) * exp(-3.0 * wa * zm / op);

            double Ez2 = Om_m * op * op * op
                       + Om_r * op * op * op * op
                       + Om_k * op * op
                       + Om_DE * DE_f;
            if (Ez2 <= 0.0) return 0;

            cumul += (z_in[i] - z_in[i-1]) / sqrt(Ez2);
        }
        DM_out[i] = (c_km_s / H0) * cumul;
    }

    return 1;
}

#endif // _CAMB_INTERFACE_H_