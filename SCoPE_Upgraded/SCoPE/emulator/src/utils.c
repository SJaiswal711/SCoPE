#include "utils.h"
#include <stdio.h>
#include <stdlib.h>

int save_binary_double(const char *fname, const double *data, size_t n) {
    FILE *fp = fopen(fname, "wb");
    if (!fp) return 0;
    size_t written = fwrite(data, sizeof(double), n, fp);
    fclose(fp);
    return written == n;
}

int load_binary_double(const char *fname, double *data, size_t n) {
    FILE *fp = fopen(fname, "rb");
    if (!fp) return 0;
    size_t read = fread(data, sizeof(double), n, fp);
    fclose(fp);
    return read == n;
}

int save_binary_int64(const char *fname, const int64_t *data, size_t n) {
    FILE *fp = fopen(fname, "wb");
    if (!fp) return 0;
    size_t written = fwrite(data, sizeof(int64_t), n, fp);
    fclose(fp);
    return written == n;
}

int load_binary_int64(const char *fname, int64_t *data, size_t n) {
    FILE *fp = fopen(fname, "rb");
    if (!fp) return 0;
    size_t read = fread(data, sizeof(int64_t), n, fp);
    fclose(fp);
    return read == n;
}

long get_file_size(const char *fname) {
    FILE *fp = fopen(fname, "rb");
    if (!fp) return -1;
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fclose(fp);
    return size;
}