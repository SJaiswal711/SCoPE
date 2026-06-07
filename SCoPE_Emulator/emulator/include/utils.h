#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdint.h>

/* Save double array to binary file */
int save_binary_double(const char *fname, const double *data, size_t n);

/* Load double array from binary file */
int load_binary_double(const char *fname, double *data, size_t n);

/* Save int64 array */
int save_binary_int64(const char *fname, const int64_t *data, size_t n);

int load_binary_int64(const char *fname, int64_t *data, size_t n);

/* Get file size in bytes */
long get_file_size(const char *fname);

#endif /* UTILS_H */