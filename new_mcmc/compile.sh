#!/bin/bash

# Exit immediately if a command fails
#!/bin/bash
module load gcc/13.1.0
module load mpi/openmpi-cuda
export PATH=/home/apps/gcc/gcc1120/bin:$PATH

# Set paths for LAPACK and CFITSIO
export GSL_PATH=/scratch/shambhavij.sps.iitmandi/DIR/gsl-2.8
export CPATH=$GSL_PATH/include:$CPATH
export LIBRARY_PATH=$GSL_PATH/lib:$LIBRARY_PATH
export LDFLAGS="-L/scratch/shambhavij.sps.iitmandi/DIR/lapack-3.12.1/install/lib64 -L/scratch/shambhavij.sps.iitmandi/DIR/cfitsio-4.5.0/.libs $LDFLAGS"
export CFLAGS="-I/scratch/shambhavij.sps.iitmandi/DIR/lapack-3.12.1/install/include -I/scratch/shambhavij.sps.iitmandi/DIR/cfitsio-4.5.0 $CFLAGS"
export LD_LIBRARY_PATH=/scratch/shambhavij.sps.iitmandi/DIR/lapack-3.12.1/install/lib64:$GSL_PATH/lib:/home/apps/gcc/gcc1120/lib64:/scratch/shambhavij.sps.iitmandi/DIR/cfitsio-4.5.0/.libs:/scratch/shambhavij.sps.iitmandi/SCoPE/data/clik/lib:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=/scratch/shambhavij.sps.iitmandi/SCoPE/CAMB/fortran/Releaselib:$LD_LIBRARY_PATH


# 4. Compile Command
# Note: We pass the full path to camblib.so because it is not named 'libcamblib.so',
# so -lcamblib would not find it.
mpicc -o mcmc mcmc.c \
  -I$GSL_PATH/include \
  -I/scratch/shambhavij.sps.iitmandi/SCoPE/data/clik/include \
  -I/scratch/shambhavij.sps.iitmandi/SCoPE/data/clik/src \
  -I/scratch/shambhavij.sps.iitmandi/SCoPE/data/clik/src/minipmc \
  -L$GSL_PATH/lib \
  -L/scratch/shambhavij.sps.iitmandi/SCoPE/data/clik/lib \
  -Wl,-rpath,/scratch/shambhavij.sps.iitmandi/SCoPE/data/clik/lib \
  -Wl,-rpath,/scratch/shambhavij.sps.iitmandi/SCoPE/CAMB/fortran/Releaselib \
  /scratch/shambhavij.sps.iitmandi/SCoPE/CAMB/fortran/Releaselib/camblib.so \
  -lclik -lgsl -lgslcblas -lgfortran -lm

echo "--- Compile successful! ---"