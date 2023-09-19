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

rm -rf NEX-GDDP_IPSL-CM6A-LR_r1i1p1f1_historical
rm -rf NEX-GDDP_GFDL-CM4_gr2_r1i1p1f1_ssp245
rm -rf NEX-GDDP_GFDL-CM4_gr2_r1i1p1f1_historical
rm -rf NEX-GDDP_GFDL-CM4_gr2_r1i1p1f1_ssp585
rm -rf NEX-GDDP_INM-CM5-0_r1i1p1f1_historical
rm -rf NEX-GDDP_KACE-1-0-G_r1i1p1f1_ssp245
rm -rf NEX-GDDP_KACE-1-0-G_r1i1p1f1_ssp585
rm -rf NEX-GDDP_KACE-1-0-G_r1i1p1f1_ssp370
rm -rf NEX-GDDP_KACE-1-0-G_r1i1p1f1_ssp126
rm -rf NEX-GDDP_MPI-ESM1-2-LR_r1i1p1f1_historical
rm -rf NEX-GDDP_UKESM1-0-LL_r1i1p1f2_ssp245
rm -rf NEX-GDDP_UKESM1-0-LL_r1i1p1f2_ssp585
rm -rf NEX-GDDP_UKESM1-0-LL_r1i1p1f2_ssp370
rm -rf NEX-GDDP_UKESM1-0-LL_r1i1p1f2_ssp126
rm -rf NEX-GDDP_GFDL-ESM4_r1i1p1f1_historical
rm -rf NEX-GDDP_ACCESS-ESM1-5_r1i1p1f1_historical
rm -rf NEX-GDDP_MIROC-ES2L_r1i1p1f2_historical
rm -rf NEX-GDDP_GISS-E2-1-G_r1i1p1f2_historical
rm -rf NEX-GDDP_INM-CM4-8_r1i1p1f1_historical
rm -rf NEX-GDDP_EC-Earth3_r1i1p1f1_historical
rm -rf NEX-GDDP_BCC-CSM2-MR_r1i1p1f1_historical
rm -rf NEX-GDDP_TaiESM1_r1i1p1f1_ssp585
rm -rf NEX-GDDP_HadGEM3-GC31-LL_r1i1p1f3_historical
rm -rf NEX-GDDP_CNRM-CM6-1_r1i1p1f2_historical
rm -rf NEX-GDDP_HadGEM3-GC31-MM_r1i1p1f3_historical
rm -rf NEX-GDDP_HadGEM3-GC31-MM_r1i1p1f3_ssp585
rm -rf NEX-GDDP_MIROC6_r1i1p1f1_ssp585
rm -rf LOCA2_INM-CM5-0_r2i1p1f1_ssp370_mid
rm -rf LOCA2_INM-CM5-0_r5i1p1f1_ssp370_mid
rm -rf LOCA2_INM-CM5-0_r1i1p1f1_ssp585_mid
rm -rf LOCA2_INM-CM5-0_r4i1p1f1_ssp370_mid
rm -rf LOCA2_INM-CM5-0_r3i1p1f1_ssp370_mid
rm -rf LOCA2_FGOALS-g3_r1i1p1f1_ssp585_mid
rm -rf LOCA2_FGOALS-g3_r4i1p1f1_ssp585_mid
rm -rf LOCA2_FGOALS-g3_r3i1p1f1_ssp585_mid

# Remove spool dirs
cd /gpfs/group/kaf26/default/dcl5300/wbm_soilM_crop_uc_lafferty-etal-2024-tbd_DATA/wbm/wbm_spool/flowdirection206_us

rm -rf NEX-GDDP_IPSL-CM6A-LR_r1i1p1f1_historical_*
rm -rf NEX-GDDP_GFDL-CM4_gr2_r1i1p1f1_ssp245_*
rm -rf NEX-GDDP_GFDL-CM4_gr2_r1i1p1f1_historical_*
rm -rf NEX-GDDP_GFDL-CM4_gr2_r1i1p1f1_ssp585_*
rm -rf NEX-GDDP_INM-CM5-0_r1i1p1f1_historical_*
rm -rf NEX-GDDP_KACE-1-0-G_r1i1p1f1_ssp245_*
rm -rf NEX-GDDP_KACE-1-0-G_r1i1p1f1_ssp585_*
rm -rf NEX-GDDP_KACE-1-0-G_r1i1p1f1_ssp370_*
rm -rf NEX-GDDP_KACE-1-0-G_r1i1p1f1_ssp126_*
rm -rf NEX-GDDP_MPI-ESM1-2-LR_r1i1p1f1_historical_*
rm -rf NEX-GDDP_UKESM1-0-LL_r1i1p1f2_ssp245_*
rm -rf NEX-GDDP_UKESM1-0-LL_r1i1p1f2_ssp585_*
rm -rf NEX-GDDP_UKESM1-0-LL_r1i1p1f2_ssp370_*
rm -rf NEX-GDDP_UKESM1-0-LL_r1i1p1f2_ssp126_*
rm -rf NEX-GDDP_GFDL-ESM4_r1i1p1f1_historical_*
rm -rf NEX-GDDP_ACCESS-ESM1-5_r1i1p1f1_historical_*
rm -rf NEX-GDDP_MIROC-ES2L_r1i1p1f2_historical_*
rm -rf NEX-GDDP_GISS-E2-1-G_r1i1p1f2_historical_*
rm -rf NEX-GDDP_INM-CM4-8_r1i1p1f1_historical_*
rm -rf NEX-GDDP_EC-Earth3_r1i1p1f1_historical_*
rm -rf NEX-GDDP_BCC-CSM2-MR_r1i1p1f1_historical_*
rm -rf NEX-GDDP_TaiESM1_r1i1p1f1_ssp585_*
rm -rf NEX-GDDP_HadGEM3-GC31-LL_r1i1p1f3_historical_*
rm -rf NEX-GDDP_CNRM-CM6-1_r1i1p1f2_historical_*
rm -rf NEX-GDDP_HadGEM3-GC31-MM_r1i1p1f3_historical_*
rm -rf NEX-GDDP_HadGEM3-GC31-MM_r1i1p1f3_ssp585_*
rm -rf NEX-GDDP_MIROC6_r1i1p1f1_ssp585_*
rm -rf LOCA2_INM-CM5-0_r2i1p1f1_ssp370_mid_*
rm -rf LOCA2_INM-CM5-0_r5i1p1f1_ssp370_mid_*
rm -rf LOCA2_INM-CM5-0_r1i1p1f1_ssp585_mid_*
rm -rf LOCA2_INM-CM5-0_r4i1p1f1_ssp370_mid_*
rm -rf LOCA2_INM-CM5-0_r3i1p1f1_ssp370_mid_*
rm -rf LOCA2_FGOALS-g3_r1i1p1f1_ssp585_mid_*
rm -rf LOCA2_FGOALS-g3_r4i1p1f1_ssp585_mid_*
rm -rf LOCA2_FGOALS-g3_r3i1p1f1_ssp585_mid_*

 echo "Job Ended at `date`"

