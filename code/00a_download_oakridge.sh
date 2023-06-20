#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -l walltime=24:00:00
#PBS -l pmem=10gb
#PBS -A kaf26_c_g_sc_default
#PBS -j oe
#PBS -o download_oakridge.out
#PBS -l feature=rhel7

echo "Job started on `hostname` at `date`"

# Go to the correct place
cd $PBS_O_WORKDIR

export STORAGE_DIR=/gpfs/group/kaf26/default/public/OakRidgeCMIP6

wget --no-check-certificate -i ../utils/oakridge_cmip6_temp_precip_all.txt -P $STORAGE_DIR

echo "Job Ended at `date`"
