#ifndef TRANSFORMS_H
#define TRANSFORMS_H

#include <stddef.h>

/* Apply log10(x + eps) in-place */
void transform_log10(double *data, size_t n, double eps);

/* Inverse: x = pow(10, y) */
void inverse_log10(double *data, size_t n);

/* Compute atanh(rho) from TT, EE, TE spectra */
void transform_atanh_rho(const double *TT, const double *EE, const double *TE,
                         double *rho_out, size_t N, int D, double eps);

/* Reconstruct TE from rho_z = atanh(rho) and TT, EE */
void inverse_atanh_te(const double *rho_z, double *TE,
                      const double *TT, const double *EE, size_t n);

#endif /* TRANSFORMS_H */