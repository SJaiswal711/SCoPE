#ifndef _SN_LIKELIHOOD_H_
#define _SN_LIKELIHOOD_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_errno.h>
#include "param_config_reader.h"

extern Config global_config;

typedef struct {
    double zHD;
    double m_b_corr;
    double CEPH_DIST;
    int    is_calibrator;
} Supernova;

static Supernova   *g_sn_data      = NULL;
static int         *g_sn_mask      = NULL;
static int          g_sn_mask_total= 0;
static int          g_num_sn       = 0;
static gsl_matrix  *g_sn_chol      = NULL;
static int          g_sn_ready     = 0;

static int g_idx_omegam_h2 = -1;
static int g_idx_h         = -1;
static int g_idx_M_B       = -1;
static double g_M_B_fixed  = -19.3; /* Fallback for fixed M_B */

static const double SN_Z_MIN = 0.01;

static void sn_resolve_param_indices(void) {
    int task_idx = 0;
    int found_M_B = 0;
    for (int i = 0; i < global_config.param_count; i++) {
        ParameterConfig *p = &global_config.params[i];
        
        if (strcmp(p->name, "M_B") == 0) {
            p->usage = USAGE_NUISANCE; 
            if (p->state == PARAM_FIXED) {
                g_M_B_fixed = p->fixed_value;
                found_M_B = 1;
            }
        }
        
        if (p->state != PARAM_SAMPLED) continue;

        if      (strcmp(p->name, "Omega_m_h2") == 0) g_idx_omegam_h2 = task_idx;
        else if (strcmp(p->name, "h")          == 0) g_idx_h         = task_idx;
        else if (strcmp(p->name, "M_B")        == 0) {
            g_idx_M_B = task_idx;
            found_M_B = 1;
        }

        task_idx++;
    }

    if (g_idx_omegam_h2 < 0 || g_idx_h < 0 || !found_M_B) {
        fprintf(stderr, "[SNe] FATAL: param_reader.ini must declare 'Omega_m_h2', 'h', and 'M_B'.\n");
        exit(1);
    }
}

static int sn_load_data(const char *sn_file) {
    FILE *fp = fopen(sn_file, "r");
    if (!fp) {
        fprintf(stderr, "[SNe] WARNING: cannot open '%s' — SN disabled\n", sn_file);
        return 0;
    }

    char buffer[1024];
    fgets(buffer, sizeof(buffer), fp); 

    Supernova *tmp = NULL;
    int total = 0;
    double zHD, m_b_corr, CEPH_DIST;
    int is_calibrator;

    while (fscanf(fp, "%*s %*d %lf %*lf %*lf %*lf %*lf %*lf %lf %*lf %*lf %*lf %lf %d %*[^\n]",
                  &zHD, &m_b_corr, &CEPH_DIST, &is_calibrator) == 4) {
        tmp = realloc(tmp, (total + 1) * sizeof(Supernova));
        tmp[total].zHD = zHD;
        tmp[total].m_b_corr = m_b_corr;
        tmp[total].CEPH_DIST = CEPH_DIST;
        tmp[total].is_calibrator = is_calibrator;
        total++;
    }
    fclose(fp);

    g_sn_mask = malloc(total * sizeof(int));
    g_sn_mask_total = total;
    int count = 0;
    for (int i = 0; i < total; i++) {
        g_sn_mask[i] = (tmp[i].is_calibrator == 1 || tmp[i].zHD > SN_Z_MIN) ? 1 : 0;
        if (g_sn_mask[i]) count++;
    }

    g_sn_data = malloc(count * sizeof(Supernova));
    int idx = 0;
    for (int i = 0; i < total; i++) {
        if (g_sn_mask[i]) g_sn_data[idx++] = tmp[i];
    }
    g_num_sn = count;
    free(tmp);
    return 1;
}

static int sn_load_cov(const char *cov_file) {
    FILE *fp = fopen(cov_file, "r");
    if (!fp) {
        fprintf(stderr, "[SNe] WARNING: cannot open '%s' — SN disabled\n", cov_file);
        return 0;
    }

    int matrix_size = 0;
    if (fscanf(fp, "%d", &matrix_size) != 1 || matrix_size != g_sn_mask_total) {
        fprintf(stderr, "[SNe] FATAL: cov size %d != data rows %d\n", matrix_size, g_sn_mask_total);
        fclose(fp); return 0;
    }

    double *full = malloc((size_t)matrix_size * matrix_size * sizeof(double));
    for (size_t i = 0; i < (size_t)matrix_size * matrix_size; i++) {
        if (fscanf(fp, "%lf", &full[i]) != 1) { free(full); fclose(fp); return 0; }
    }
    fclose(fp);

    g_sn_chol = gsl_matrix_alloc(g_num_sn, g_num_sn);
    int ri = 0;
    for (int i = 0; i < matrix_size; i++) {
        if (!g_sn_mask[i]) continue;
        int ci = 0;
        for (int j = 0; j < matrix_size; j++) {
            if (g_sn_mask[j]) gsl_matrix_set(g_sn_chol, ri, ci++, full[i * matrix_size + j]);
        }
        ri++;
    }
    free(full);

    gsl_error_handler_t *old = gsl_set_error_handler_off();
    int status = gsl_linalg_cholesky_decomp1(g_sn_chol);
    gsl_set_error_handler(old);
    
    if (status != GSL_SUCCESS) {
        fprintf(stderr, "[SNe] FATAL: Cholesky decomposition failed\n");
        gsl_matrix_free(g_sn_chol); g_sn_chol = NULL;
        return 0;
    }
    return 1;
}

static int sn_init(const char *sn_file, const char *cov_file) {
    if (g_sn_ready) return 1;
    sn_resolve_param_indices();
    if (!sn_load_data(sn_file)) return 0;
    if (!sn_load_cov(cov_file)) return 0;
    g_sn_ready = 1;
    printf("[SNe] Loaded %d Supernovae from %s\n", g_num_sn, sn_file);
    return 1;
}

static void sn_free(void) {
    free(g_sn_data); g_sn_data = NULL;
    free(g_sn_mask); g_sn_mask = NULL;
    if (g_sn_chol) { gsl_matrix_free(g_sn_chol); g_sn_chol = NULL; }
    g_num_sn = 0; g_sn_ready = 0;
}

static double sn_chi2(const double *task, int num_z, const double *z_grid, const double *DM_grid) {
    if (!g_sn_ready) return 0.0;

    double M_B = (g_idx_M_B >= 0) ? task[g_idx_M_B] : g_M_B_fixed;
    double *resid = malloc(g_num_sn * sizeof(double));

    for (int i = 0; i < g_num_sn; i++) {
        double mu_th;
        if (g_sn_data[i].is_calibrator) {
            mu_th = g_sn_data[i].CEPH_DIST;
        } else {
            double z = g_sn_data[i].zHD;
            double DC = 0.0;
            if (z <= z_grid[0]) DC = DM_grid[0];
            else if (z >= z_grid[num_z-1]) DC = DM_grid[num_z-1];
            else {
                int j = 1;
                while (j < num_z && z_grid[j] < z) j++;
                double t = (z - z_grid[j-1]) / (z_grid[j] - z_grid[j-1]);
                DC = (1.0 - t) * DM_grid[j-1] + t * DM_grid[j];
            }
            
            double DL = (1.0 + z) * DC;
            if (DL <= 0.0) { free(resid); return 1e30; }
            mu_th = 5.0 * log10(DL) + 25.0;
        }
        resid[i] = g_sn_data[i].m_b_corr - (mu_th + M_B);
    }

    gsl_vector_view r_view = gsl_vector_view_array(resid, g_num_sn);
    gsl_blas_dtrsv(CblasLower, CblasNoTrans, CblasNonUnit, g_sn_chol, &r_view.vector);

    double chi2 = 0.0;
    for (int i = 0; i < g_num_sn; i++) {
        chi2 += resid[i] * resid[i];
    }
    free(resid);
    return chi2;
}

#endif