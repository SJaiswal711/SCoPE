#include "transforms.h"
#include <math.h>
#include <omp.h>

void transform_log10(double *data, size_t n, double eps) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        double v = data[i];
        if (!isfinite(v) || v < 0.0) v = 0.0;
        data[i] = log10(v + eps);
    }
}

void inverse_log10(double *data, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        double v = data[i];
        if (!isfinite(v)) v = -12.0;
        data[i] = pow(10.0, v);
    }
}

/* ============================================================
   transform_atanh_rho

   Two problems this fixes:

   1. ZERO-CROSSING ARTIFACT
      At ℓ where EE (or TT) is at the numerical noise floor (~1e-15),
      TE / sqrt(TT·EE) can blow up (numerator noise / sqrt(noise)).
      Old code clamped rho to ±0.9999 then atanh gave ±4.95, which the
      polynomial cannot fit (rv ≈ 0.9-1.0 per RHO mode).

      Fix: SIGNAL_FLOOR = 1e-10. If sqrt(TT·EE) < floor, set rho = 0
      (no correlation detected). This kills the 0/0 amplification
      before atanh sees it.

   2. LARGE-SIGNAL SATURATION
      Even physical rho near ±1 (e.g. reionization TE spike at low ℓ)
      produces atanh ≈ ±4.95, making y_std_RHO ≈ 5 per mode.

      Fix: atanh(0.8 · rho). Range becomes [-1.10, +1.10], a 4.5×
      reduction. Physical values |rho| ≤ 0.9 become |atanh(0.8·rho)|
      ≤ 1.0, still well-represented by the polynomial.

   Both fixes are inside the loop. No change to the function signature.
   ============================================================ */
void transform_atanh_rho(const double *TT, const double *EE, const double *TE,
                         double *rho_out, size_t N, int D, double eps) {
    const double SIGNAL_FLOOR = 1e-10;

    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < N; i++) {
        for (int l = 0; l < D; l++) {
            size_t idx = i * D + l;
            double tt = TT[idx];
            double ee = EE[idx];
            double te = TE[idx];

            if (!isfinite(tt) || tt < 0.0) tt = 0.0;
            if (!isfinite(ee) || ee < 0.0) ee = 0.0;
            if (!isfinite(te))             te = 0.0;

            /* Signal floor: rho is undefined when signal is below noise */
            double signal = sqrt(tt * ee);
            if (!isfinite(signal) || signal < SIGNAL_FLOOR) {
                rho_out[idx] = 0.0;
                continue;
            }

            double rho = te / (signal + eps);
            if (!isfinite(rho)) rho = 0.0;
            if (rho >  1.0) rho =  1.0;
            if (rho < -1.0) rho = -1.0;

            /* Soft saturation: rho ∈ [-1,1] → 0.8·rho ∈ [-0.8,0.8]
               → atanh ∈ [-1.099, +1.099]. No hard clip needed. */
            rho_out[idx] = atanh(0.8 * rho);
        }
    }
}

/* ============================================================
   inverse_atanh_te

   Inverse of the forward transform. Given rho_z = atanh(0.8·rho),
   recover rho = tanh(rho_z) / 0.8, then TE = rho · sqrt(TT·EE).

   Within the training range (rho_z ∈ [-1.099, +1.099]), the result
   is in [-1, +1] after division, so the clamp only fires on extreme
   extrapolation (|rho_z| > 1.099), where the emulator was never
   intended to operate.
   ============================================================ */
void inverse_atanh_te(const double *rho_z, double *TE,
                      const double *TT, const double *EE, size_t n) {
    #pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        double rho = tanh(rho_z[i]) / 0.8;
        if (!isfinite(rho)) rho = 0.0;
        if (rho >  1.0) rho =  1.0;
        if (rho < -1.0) rho = -1.0;
        TE[i] = rho * sqrt(TT[i] * EE[i]);
    }
}