#!/bin/bash
# Run VASP in every directory given (or every subdirectory of $PWD) that has INCAR and POSCAR,
# one after another, with $SLURM_NTASKS ranks.  POTCAR is concatenated from the names in
# POTCARS (same order as POSCAR line 6).  Finished directories (OUTCAR with
# "General timing") are skipped, so a resubmission continues where the last one stopped.
module purge
module load intel/2022.1
export PATH=/data/home/dg002434/software/vasp.6.6.0/bin:$PATH
PP=/data/home/dg002434/software/potpaw_PBE.54
NP=${SLURM_NTASKS:-4}
dirs=("$@"); [ ${#dirs[@]} -eq 0 ] && dirs=(*/)
for d in "${dirs[@]}"; do
  d=${d%/}
  [ -f "$d/INCAR" ] && [ -f "$d/POSCAR" ] || continue
  if grep -qs "General timing" "$d/OUTCAR"; then echo "skip $d"; continue; fi
  ( cd "$d" || exit
    cat $(for p in $(cat POTCARS); do echo $PP/$p/POTCAR; done) > POTCAR
    exe=vasp_std; [ -f USE_GAM ] && exe=vasp_gam
    t0=$(date +%s)
    mpirun -np $NP $exe > vasp.log 2>&1
    echo "done $d $(( $(date +%s) - t0 )) s $(grep -c 'F=' OSZICAR) ionic"
  )
done
