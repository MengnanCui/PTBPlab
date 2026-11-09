#!/bin/bash
#SBATCH -J gc_quick
#SBATCH -p dg50
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 1 -c 16
#SBATCH -t 00:30:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
source ~/software/miniforge3/etc/profile.d/conda.sh
conda activate pydftb
cd ~/PTBPlab/gpurun
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo CPUS $(nproc)
/usr/bin/time -v pydftb gpu-check --quick --dtype both --json gc_quick.json
