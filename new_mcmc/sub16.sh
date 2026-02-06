#!/bin/bash
#SBATCH -N 1                     # number of nodes
#SBATCH --ntasks=21          # Total number of MPI tasks
#SBATCH --cpus-per-task=1        # CPU cores per MPI task (adjust if needed)
#SBATCH --error=m.%J.err	 # name of output file
#SBATCH --output=m.%J.out	 # name of error file
#SBATCH --time=48:00:00          # time required to execute the program
#SBATCH --partition=standard     # specifies queue name 

module load openmpi/4.1.4
module load valgrind/3.15.0 
set -e
set -x
ulimit -s unlimited
# mpirun --mca btl ^openib -np 3 ./nmcmc
# ASAN_OPTIONS=detect_leaks=1:abort_on_error=1:halt_on_error=1 mpirun --mca btl ^openib -np 3 ./nmcmc
mpirun --mca btl ^openib -np 21 ./mcmc

# mpirun --mca btl ^openib -np 6 valgrind --leak-check=full  --mca pml ob1 --mca btl tcp,self ./mcmc16
# mpirun -np 6 \
#   valgrind --track-origins=yes ./nmcmc

