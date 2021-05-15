#!/bin/bash

#SBATCH --job-name=bbaruffaldi-tinymc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28

### Enviroment setup
. /etc/profile
source /opt/intel/oneapi/compiler/latest/env/vars.sh

srun ./run_compiler.sh
srun ./run_fotones.sh
srun ./run_target.sh
