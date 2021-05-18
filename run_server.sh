#!/bin/bash

#SBATCH --job-name=bbaruffaldi-tinymc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28

### Enviroment setup
. /etc/profile
source /opt/intel/oneapi/compiler/latest/env/vars.sh

export OMP_NUM_THREADS=28
export OMP_WAIT_POLICY=active
export OMP_DYNAMIC=false 
export OMP_PROC_BIND=true

#./run_schedule.sh
#./run_critical.sh
#./run_threads.sh
#./run_fotones.sh
#./run_fotones_big.sh
