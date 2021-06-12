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

sleep 60

#./scripts/run_block_size.sh
#./scripts/run_fotones_1.sh
#./scripts/run_fotones_2.sh
#./scripts/run_fotones_3.sh
#./scripts/run_fotones_4.sh
#./scripts/run_fotones_5.sh
#./scripts/run_fotones_6.sh
#./scripts/run_fotones_7.sh
