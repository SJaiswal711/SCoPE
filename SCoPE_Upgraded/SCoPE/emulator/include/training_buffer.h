#ifndef TRAINING_BUFFER_H
#define TRAINING_BUFFER_H

#include <stddef.h>

#define N_SPECTRA 2601   /* Number of multipoles (l=0..2600) */

/* Training point – stores pointers to heap arrays (no copy) */
typedef struct {
    double *params;          /* Cosmological parameters (copied) */
    double *cl_tt;
    double *cl_te;
    double *cl_ee;
    double *cl_bb;
    double loglike;
    int step;
    int valid;               /* Whether this slot contains valid data */
} TrainingPoint;

/* Circular buffer */
typedef struct {
    TrainingPoint *buffer;   /* Pre-allocated array of slots */
    int capacity;
    int size;                /* Number of valid points */
    int head;                /* Next write index */
    int tail;                /* Oldest valid index */
    int n_params;            /* Number of cosmological parameters */
    int total_added;
    int n_freed;
} TrainingBuffer;

/* Create buffer */
TrainingBuffer* training_buffer_create(int capacity, int n_params);

/* Destroy buffer (frees all stored arrays) */
void training_buffer_destroy(TrainingBuffer *buf);

/* Add a point – takes ownership of Cl arrays */
int training_buffer_add(TrainingBuffer *buf,
                        double *params,
                        double *cl_tt, double *cl_te,
                        double *cl_ee, double *cl_bb,
                        double loglike, int step);

/* Get point by logical index (0 = oldest) */
TrainingPoint* training_buffer_get(TrainingBuffer *buf, int idx);

/* Copy all points into contiguous arrays (caller must free) */
int training_buffer_copy_all(TrainingBuffer *buf,
                             double **params_out,
                             double **cl_tt_out,
                             double **cl_te_out,
                             double **cl_ee_out,
                             double **cl_bb_out);

/* Statistics */
int training_buffer_size(TrainingBuffer *buf);
int training_buffer_capacity(TrainingBuffer *buf);

#endif /* TRAINING_BUFFER_H */