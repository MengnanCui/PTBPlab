#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
#SBATCH -J test_slurm
#SBATCH --partition=medium
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
# Wall clock limit:
#SBATCH --time=23:50:00
#SBATCH --mem=96000 
                                         
module purge
# For FHIAIMS
module load intel/19.1.1 impi/2019.7 qt/5.9 mkl/2020.1
module load parallel/201807

#for i in {0..4}; do
#  cd "aims_${i}"
#  srun /u/mncui/software/fhiaims_code/bin/aims.201231.scalapack.mpi.x > aims.out
#  cd ..
#done
#cd "aims_${SLURM_ARRAY_TASK_ID}"
#srun /u/mncui/software/fhiaims_code/bin/aims.201231.scalapack.mpi.x > aims.out

#python aims_calc_v2.py tmp.test.xyz 
export OMP_NUM_THREADS=1
python run_parallel.py data.xyz 39
