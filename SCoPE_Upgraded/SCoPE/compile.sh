#!/bin/bash
set -e

module load gcc/13.1.0
module load mpi/openmpi-cuda

ROOT=/scratch/shambhavij.sps.iitmandi
SCOPE=$ROOT/SCoPE
CAMB=$SCOPE/CAMB
GCC=/home/apps/gcc/gcc1120
GSL=$ROOT/DIR/gsl-2.8
LAPACK=$ROOT/DIR/lapack-3.12.1/install
CFITSIO=$ROOT/DIR/cfitsio-4.5.0
CLIK=$SCOPE/data/clik

export PATH="$GCC/bin:$PATH"

CFITSIO_LIB=$CFITSIO/.libs
RPATH="$GCC/lib64:$CFITSIO_LIB:$LAPACK/lib64:$GSL/lib:$CLIK/lib:$CAMB/fortran/Releaselib"

export LD_LIBRARY_PATH="$RPATH:${LD_LIBRARY_PATH:-}"

mkdir -p build

# Compile Fortran interface
"$GCC/bin/gfortran" -O3 -fopenmp \
    -I"$CAMB/fortran/Releaselib" \
    -I"$CAMB/forutils/Releaselib" \
    -Jbuild -c camb_iface.f90 -o build/camb_iface.o

# Compile and link
mpicc -O3 -fopenmp -o nmcmc \
    mcmc_scope.c \
    emulator/src/emulator.c \
    emulator/src/training_buffer.c \
    emulator/src/pca_online.c \
    emulator/src/gp_online.c \
    emulator/src/poly_online.c \
    emulator/src/transforms.c \
    emulator/src/utils.c \
    build/camb_iface.o \
    -I"$GSL/include" \
    -I"$CLIK/include" \
    -I"$CLIK/src" \
    -I"$CLIK/src/minipmc" \
    -Iemulator/include \
    -I"$LAPACK/include" \
    -I"$CFITSIO" \
    -L"$GSL/lib" -L"$CLIK/lib" \
    -L"$LAPACK/lib64" -L"$CFITSIO_LIB" \
    -Wl,--disable-new-dtags \
    -Wl,-rpath,"$RPATH" \
    "$CAMB/fortran/Releaselib/camblib.so" \
    -lclik -lgsl -lgslcblas -llapacke -llapack -lblas \
    -lgfortran -lm

echo "Build completed: nmcmc"