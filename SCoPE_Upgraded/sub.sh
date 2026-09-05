#!/bin/bash
#SBATCH --job-name=scope
#SBATCH -N 1
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=1
#SBATCH --error=scope.%J.err
#SBATCH --output=scope.%J.out
#SBATCH --time=48:00:00
#SBATCH --partition=standard

module load openmpi/4.1.4

set -e
set -x
ulimit -s unlimited

outdir="CHECK_dr000"

mpirun --mca btl ^openib -np 7 ./nmcmc -o "$outdir"
