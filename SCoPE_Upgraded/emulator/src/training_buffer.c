#include "training_buffer.h"
#include <stdlib.h>
#include <string.h>

TrainingBuffer* training_buffer_create(int capacity, int n_params) {
    TrainingBuffer *buf = calloc(1, sizeof(TrainingBuffer));
    if (!buf) return NULL;
    buf->capacity = capacity;
    buf->n_params = n_params;
    buf->buffer = calloc(capacity, sizeof(TrainingPoint));
    if (!buf->buffer) { free(buf); return NULL; }
    buf->head = 0;
    buf->tail = 0;
    buf->size = 0;
    return buf;
}

void training_buffer_destroy(TrainingBuffer *buf) {
    if (!buf) return;
    for (int i = 0; i < buf->capacity; i++) {
        if (buf->buffer[i].valid) {
            free(buf->buffer[i].cl_tt);
            free(buf->buffer[i].cl_te);
            free(buf->buffer[i].cl_ee);
            free(buf->buffer[i].cl_bb);
            free(buf->buffer[i].params);
        }
    }
    free(buf->buffer);
    free(buf);
}

int training_buffer_add(TrainingBuffer *buf,
                        double *params,
                        double *cl_tt, double *cl_te,
                        double *cl_ee, double *cl_bb,
                        double loglike, int step) {
    if (!buf) return 0;
    int idx = buf->head;
    TrainingPoint *pt = &buf->buffer[idx];
    if (pt->valid) {
        free(pt->cl_tt);
        free(pt->cl_te);
        free(pt->cl_ee);
        free(pt->cl_bb);
        free(pt->params);
        buf->n_freed++;
    }
    pt->cl_tt = cl_tt;
    pt->cl_te = cl_te;
    pt->cl_ee = cl_ee;
    pt->cl_bb = cl_bb;
    pt->params = malloc(buf->n_params * sizeof(double));
    if (!pt->params) return 0;
    memcpy(pt->params, params, buf->n_params * sizeof(double));
    pt->loglike = loglike;
    pt->step = step;
    pt->valid = 1;
    buf->head = (buf->head + 1) % buf->capacity;
    if (buf->size < buf->capacity) {
        buf->size++;
    } else {
        buf->tail = (buf->tail + 1) % buf->capacity;
    }
    buf->total_added++;
    return 1;
}

TrainingPoint* training_buffer_get(TrainingBuffer *buf, int idx) {
    if (!buf || idx < 0 || idx >= buf->size) return NULL;
    int physical = (buf->tail + idx) % buf->capacity;
    return &buf->buffer[physical];
}

int training_buffer_copy_all(TrainingBuffer *buf,
                             double **params_out,
                             double **cl_tt_out,
                             double **cl_te_out,
                             double **cl_ee_out,
                             double **cl_bb_out) {
    if (!buf || buf->size == 0) return 0;
    int n = buf->size;
    int nf = N_SPECTRA;
    int np = buf->n_params;
    *params_out = malloc(n * np * sizeof(double));
    *cl_tt_out = malloc(n * nf * sizeof(double));
    *cl_te_out = malloc(n * nf * sizeof(double));
    *cl_ee_out = malloc(n * nf * sizeof(double));
    *cl_bb_out = malloc(n * nf * sizeof(double));
    if (!*params_out || !*cl_tt_out || !*cl_te_out || !*cl_ee_out || !*cl_bb_out) {
        free(*params_out); free(*cl_tt_out); free(*cl_te_out);
        free(*cl_ee_out); free(*cl_bb_out);
        return 0;
    }
    for (int i = 0; i < n; i++) {
        TrainingPoint *pt = training_buffer_get(buf, i);
        memcpy(&(*params_out)[i*np], pt->params, np * sizeof(double));
        memcpy(&(*cl_tt_out)[i*nf], pt->cl_tt, nf * sizeof(double));
        memcpy(&(*cl_te_out)[i*nf], pt->cl_te, nf * sizeof(double));
        memcpy(&(*cl_ee_out)[i*nf], pt->cl_ee, nf * sizeof(double));
        memcpy(&(*cl_bb_out)[i*nf], pt->cl_bb, nf * sizeof(double));
    }
    return n;
}

int training_buffer_size(TrainingBuffer *buf) { return buf ? buf->size : 0; }
int training_buffer_capacity(TrainingBuffer *buf) { return buf ? buf->capacity : 0; }