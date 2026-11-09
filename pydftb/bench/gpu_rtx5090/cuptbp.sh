#!/bin/bash
#SBATCH -J gc_cuptbp
#SBATCH -p dg50
#SBATCH --gres=gpu:1
#SBATCH -N 1 -n 1 -c 32
#SBATCH -t 02:00:00
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err
source ~/software/miniforge3/etc/profile.d/conda.sh
conda activate pydftb
export PYTHONUNBUFFERED=1
cd ~/PTBPlab/gpurun
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
echo CPUS $(nproc)
/usr/bin/time -v pydftb gpu-check --dtype both --timings cu,ptbp --cases h2o --json gc_cuptbp.json
