import os
import numpy as np
import pandas as pd

##########################
# Get all climate combos #
##########################
run_info = []

# NEX-GDDP-CMIP6
nex_path = "/gpfs/group/kaf26/default/public/NEX-GDDP-CMIP6/models"

ensemble = "NEX-GDDP"

nex_models = os.listdir(nex_path)
for model in nex_models:
    ssps = os.listdir(f"{nex_path}/{model}")
    for ssp in ssps:
        # Get info
        member = os.listdir(f"{nex_path}/{model}/{ssp}/tas/")[0].split("_")[-3]
        grid = os.listdir(f"{nex_path}/{model}/{ssp}/tas/")[0].split("_")[-2]
        # Append
        run_info.append({"ensemble":ensemble, "model": model, "member": member, "grid": grid, "ssp":ssp, "method":0})
    
# LOCA2
loca_path = "/gpfs/group/kaf26/default/public/LOCA2"

ensemble = "LOCA2"

loca_models = os.listdir(loca_path)
for model in loca_models:
    if model == "training_data":
        continue
    members = os.listdir(f"{loca_path}/{model}/0p0625deg/")
    for member in members:
        # Get info
        ssps = os.listdir(f"{loca_path}/{model}/0p0625deg/{member}")
        # Append
        for ssp in ssps:
            run_info.append({"ensemble":ensemble, "model": model, "member": member, "ssp":ssp, "grid":0, "method":0})
            
# OakRidge (manual)
ensemble = "OakRidge"
ssps = ["ssp585", "historical"]
vars_ = ["tmax","tmin","prcp"]
methods = ["DBCCA_Daymet", "RegCM_Daymet", "DBCCA_Livneh", "RegCM_Livneh"]

oknl_infos = [{"model": "ACCESS-CM2", "member": "r1i1p1f1", "grid":0},
               {"model": "BCC-CSM2-MR", "member": "r1i1p1f1", "grid":0},
               {"model": "CNRM-ESM2-1", "member": "r1i1p1f2", "grid":0},
               {"model": "MPI-ESM1-2-HR", "member": "r1i1p1f1", "grid":0},
               {"model": "MRI-ESM2-0", "member": "r1i1p1f1", "grid":0},
              {"model": "NorESM2-MM", "member": "r1i1p1f1", "grid":0}]

for method in methods:
    for ssp in ssps:
        for info in oknl_infos:
            info_tmp = info.copy()
            info_tmp["ensemble"] = ensemble
            info_tmp["method"] = method
            info_tmp["ssp"] = ssp
            run_info.append(info_tmp)
        
#########
# Store #
#########
df = pd.DataFrame(run_info)
df.to_csv("./climate_drivers.csv", index=False)