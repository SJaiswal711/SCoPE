#ifndef _LIKELIHOOD_INTERFACE_H_
#define _LIKELIHOOD_INTERFACE_H_

#include "cmb_likelihood.h"
#include "sn_likelihood.h"
#include "camb_interface.h"

static inline double calc_likelihood(int rank, double *task,
                                     double *tt, double *te, double *ee, double *bb,
                                     int fast_mode, double *cached_slow_chi2,
                                     double *ind_chi2_out) 
{
    double chi2 = run_plc(rank, task, tt, te, ee, bb, fast_mode, 0, cached_slow_chi2, ind_chi2_out);
    if (chi2 >= 1e29) return 1e30;

    if (global_config.use_sn) {
        int num_z = 500;
        double z_grid[500], H[500], DM[500], DH[500];
        double rd, zdrag;
        for (int i = 0; i < num_z; i++) z_grid[i] = i * 3.0 / (num_z - 1);

        if (!param_iface_bg(task, num_z, z_grid, &rd, &zdrag, H, DM, DH)) return 1e30;
        chi2 += sn_chi2(task, num_z, z_grid, DM);
    }

    return isfinite(chi2) ? chi2 : 1e30;
}

static inline double calc_likelihood_drag(int rank, double *task,
                                          double *tt, double *te, double *ee, double *bb,
                                          int num_z, const double *z_grid, const double *DM,
                                          double ind_chi2_fixed) 
{
    double chi2 = run_plc(rank, task, tt, te, ee, bb, 0, 1, NULL, NULL);
    if (chi2 >= 1e29) return 1e30;

    chi2 += ind_chi2_fixed;

    if (global_config.use_sn) {
        chi2 += sn_chi2(task, num_z, z_grid, DM);
    }
    return isfinite(chi2) ? chi2 : 1e30;
}

#endif