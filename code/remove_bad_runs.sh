#!/bin/bash

#PBS -l nodes=1:ppn=1
#PBS -l walltime=12:00:00
#PBS -l pmem=10gb
#PBS -A kaf26_c_g_sc_default
#PBS -j oe
#PBS -l feature=rhel7

 echo "Job started on `hostname` at `date`"

# Remove wbm dirs
cd /gpfs/group/kaf26/default/dcl5300/wbm_soilM_crop_uc_lafferty-etal-2024-tbd_DATA/wbm/

rm -rf NEX-GDDP_HadGEM3-GC31-MM_r1i1p1f3_ssp126
rm -rf NEX-GDDP_HadGEM3-GC31-MM_r1i1p1f3_historical
rm -rf NEX-GDDP_HadGEM3-GC31-LL_r1i1p1f3_ssp126
rm -rf NEX-GDDP_HadGEM3-GC31-LL_r1i1p1f3_ssp245
rm -rf NEX-GDDP_HadGEM3-GC31-LL_r1i1p1f3_ssp585
rm -rf NEX-GDDP_HadGEM3-GC31-LL_r1i1p1f3_historical


# Remove spool dirs
cd /gpfs/group/kaf26/default/dcl5300/wbm_soilM_crop_uc_lafferty-etal-2024-tbd_DATA/wbm/wbm_spool/flowdirection206_us

rm -rf NEX-GDDP_HadGEM3-GC31-MM_r1i1p1f3_ssp126_*
rm -rf NEX-GDDP_HadGEM3-GC31-MM_r1i1p1f3_historical_*
rm -rf NEX-GDDP_HadGEM3-GC31-LL_r1i1p1f3_ssp126_*
rm -rf NEX-GDDP_HadGEM3-GC31-LL_r1i1p1f3_ssp245_*
rm -rf NEX-GDDP_HadGEM3-GC31-LL_r1i1p1f3_ssp585_*
rm -rf NEX-GDDP_HadGEM3-GC31-LL_r1i1p1f3_historical_*

echo "Job Ended at `date`"

