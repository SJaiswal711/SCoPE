#include "transforms.h"
#include <math.h>
#include <omp.h>

void transform_log10(double *data, size_t n, double eps) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        data[i] = log10(data[i] + eps);
    }
}

void inverse_log10(double *data, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        data[i] = pow(10.0, data[i]);
    }
}

void transform_atanh_rho(const double *TT, const double *EE, const double *TE,
                         double *rho_out, size_t N, int D, double eps) {
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; i++) {
        for (int l = 0; l < D; l++) {
            size_t idx = i * D + l;
            double tt = TT[idx];
            double ee = EE[idx];
            double te = TE[idx];
            double denom = sqrt((tt + eps) * (ee + eps));
            double rho = te / denom;
            if (rho > 0.9999) rho = 0.9999;
            if (rho < -0.9999) rho = -0.9999;
            rho_out[idx] = atanh(rho);
        }
    }
}

void inverse_atanh_te(const double *rho_z, double *TE,
                      const double *TT, const double *EE, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        double rho = tanh(rho_z[i]);
        if (rho > 0.999) rho = 0.999;
        if (rho < -0.999) rho = -0.999;
        TE[i] = rho * sqrt(TT[i] * EE[i]);
    }
}