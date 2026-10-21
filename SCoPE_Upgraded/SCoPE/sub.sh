#!/bin/bash
#SBATCH --job-name=new
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --partition=standard
#SBATCH --output=polycheck.%j.out
#SBATCH --error=polycheck.%j.err



echo "Starting SCoPE"
hostname
date

module load openmpi/4.1.4
echo "MPI loaded"

ulimit -s unlimited
# export LD_LIBRARY_PATH=/home/apps/gcc/gcc1120/lib64:$LD_LIBRARY_PATH
# mpirun --mca btl ^openib -np 3 ./nmcmc -o noemu000
# DUMP_TRACE=trace_noemu.bin DUMP_CAMB=camb_noemu.bin \
mpirun --mca btl ^openib -np 3 ./nmcmc -o poly004
status=$?
echo "SCoPE exit code: $status"
date
exit $status