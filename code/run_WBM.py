import json
import os
import shutil
import subprocess
import time

import numpy as np
import pandas as pd

def run_WBM(
    run_type,
    main_path,
    sim_ID,
    project_ID,
    run_start,
    run_end,
    spinup_start,
    spinup_end,
    spinup_loops,
    output_vars,
    model_info,
    network,
    pet_method,
    airT_primary,
    airT_secondary,
    airT_min,
    airT_max,
    precip_primary,
    precip_secondary,
    cloudFr_primary,
    windU_primary,
    windV_primary,
    humidity_primary,
    albedo_primary,
    rootingDepth,
    soilAWC,
    canopyHt,
    lai,
    land_collapse,
    crop_area_frac,
    crop_area_frac_patch,
    crop_par,
    crop_par_patch,
):
    """
    Inputs:
        run_type: determines whether climate init files need to be updated ("obs" or "model")
        main_path: complete path to main folder for all results (str)
        sim_ID: path to simulation folder relative to main path (str)
        project_ID: project identifier (str)
        run_start: date of run start (str)
        run_end: date of run end (str)
        spinup_start: date of spinup start (str)
        spinup_end: date of spinup end (str)
        spinup_loops: number of spinup loops (int)
        output_vars: list of output vars (list of str)
        model_info: list of climate model information, only used for run_type = "model" (dict)
        network: path to network file (str)
        pet_method: PET method ("Hamon" or "Penmann-Monteith")
        airT_primary: name of primary airT init file (str)
        airT_secondary: name of patch airT init file (str)
        airT_min: name of primary airTmin init file (str)
        airT_max: name of primary airTmax init file (str)
        precip_primary: name of primary precip init file (str)
        precip_secondary: name of patch precip init file (str)
        cloudFr_primary: name of primary cloudFr init file (str)
        windU_primary: name of primary windU init file (str)
        windV_primary: name of primary windV init file (str)
        humidity_primary: name of primary humidity init file (str)
        albedo_primary: name of primary albedo init file (str)
        rootingDepth: name of rootingDepth init file (str)
        soilAWC: path to soilAWC tif file (str)
        canopyHt: path to canopyHt tif (file)
        lai: name of lai init file (str)
        land_collapse: land collapse method ("average" or empty)
        crop_area_frac: name of crop area fraction timeseries init file (str)
        crop_area_frac_patch: name of patch crop area fraction timeseries init file (str)
        crop_par: name of crop parameter file (str)
        crop_par_patch: name of patch crop parameter file (str)
    
    Outputs:
        This function copies a blank WBM simulation directory template, updates the init files based on function
        arguments, and initiates the simulation. Checks for errors in setup with -noRun flag. If no errors spool
        job will be submitted and run job will be submitted dependent on 0 exit of spool job.
    """
    # Returns a blank string if parameter is empty
    def data_init_path(par):
        if par == "":
            return ""
        else:
            return f"{sim_path}/data_init/{par}"

    #################
    # Set up folders
    #################
    sim_path = main_path + sim_ID

    if os.path.isdir(sim_path):
        print("Directory already exists!")
    else:
        shutil.copytree(f"{main_path}/wbm_storage_template", sim_path)

    ###################
    # Update init file
    ###################
    with open("../utils/wbm_init_template.json") as f:
        wbm_init = json.load(f)

    # These are typically the settings we care about but there are more!
    # Organization
    wbm_init["ID"] = sim_ID
    wbm_init["Comment"] = ""
    wbm_init["Project"] = project_ID

    wbm_init["MT_Code_Name"]["MT_ID"] = sim_ID
    wbm_init["MT_Code_Name"]["Output_dir"] = f"{sim_path}/wbm_output"
    wbm_init["MT_Code_Name"]["Run_Start"] = run_start
    wbm_init["MT_Code_Name"]["Run_End"] = run_end

    wbm_init["Spinup"]["Start"] = spinup_start
    wbm_init["Spinup"]["End"] = spinup_end
    wbm_init["Spinup"]["Loops"] = spinup_loops

    wbm_init["Network"] = "/gpfs/group/kaf26/default/private/WBM_data/squam.sr.unh.edu/US_CDL_v3_data/network/flowdirection206_us.asc"
    
    # Output variables
    wbm_init['Output_vars'] =  '\n '.join(output_vars)

    # Params
    wbm_init["PET"] = pet_method

    # Climate
    wbm_init["MT_airT"]["Primary"] = data_init_path(airT_primary)
    wbm_init["MT_airT"]["Secondary"] = data_init_path(airT_secondary)
    wbm_init["MT_airT"]["airTmin"] = data_init_path(airT_min)
    wbm_init["MT_airT"]["airTmax"] = data_init_path(airT_max)

    wbm_init["MT_Precip"]["Primary"] = f"{sim_path}/data_init/{precip_primary}"
    wbm_init["MT_Precip"]["Secondary"] = data_init_path(precip_secondary)

    wbm_init["MT_windU"]["Primary"] = data_init_path(windU_primary)
    wbm_init["MT_windV"]["Primary"] = data_init_path(windV_primary)
    wbm_init["MT_humidity"]["Primary"] = data_init_path(humidity_primary)
    wbm_init["MT_albedo"]["Primary"] = data_init_path(albedo_primary)

    wbm_init["MT_cloudFr"]["Primary"] = f"{sim_path}/data_init/{cloudFr_primary}"
    
    # Soil
    wbm_init["rootingDepth"] = f"{sim_path}/data_init/{rootingDepth}"
    wbm_init["soilAWCapacity"] = soilAWC
    
    # Canopy
    wbm_init["canopyHt"] = canopyHt

    # Crop
    wbm_init["MT_LAI"] = data_init_path(lai)

    wbm_init["landCollapse"] = land_collapse

    wbm_init["Irrigation"]["CropAreaFrac"] = f"{sim_path}/data_init/{crop_area_frac}"
    wbm_init["Irrigation"]["CropAreaFracPatch"] = data_init_path(crop_area_frac_patch)

    wbm_init["Irrigation"]["CropParFile"] = f"{sim_path}/data/{crop_par}"
    wbm_init["Irrigation"]["CropParFilePatch"] = data_init_path(crop_par_patch)

    # Save init file
    f = open(f"{sim_path}/wbm_init/wbm.init", "w")
    f.write("{ \n")
    for key in wbm_init:
        # Subkeys
        if type(wbm_init[key]) == dict:
            f.write(key + " => {\n")
            for subkey in wbm_init[key]:
                # Skip writing if entry = XXX
                if str(wbm_init[key][subkey])[-3:] == "XXX":
                    continue
                f.write(subkey + " => '" + str(wbm_init[key][subkey]) + "',\n")
            f.write("},\n")
        # Regalar keys
        else:
            # Skip writing if entry = XXX
            if str(wbm_init[key])[-3:] == "XXX":
                continue
            f.write(key + " => '" + str(wbm_init[key]) + "',\n")
    f.write("} \n")
    f.close()
    
    ####################
    # Update crop files
    ####################
    df_crop = pd.read_csv(f"{main_path}{sim_ID}/data/{crop_par}")
    df_crop = df_crop.applymap(lambda x: x.replace("PATH_HERE", f"{main_path}{sim_ID}") if type(x)==str else x)
    df_crop.to_csv(f"{main_path}{sim_ID}/data/{crop_par}", index=False)

    ##############################
    # Update climate init files 
    ##############################
    if run_type == "model":
        for var in model_info["vars"]:
            # Read and update each init file
            with open(f"{sim_path}/data_init/{model_info['ensemble']}_{var}_daily.init", "r") as f:
                newlines = []
                for line in f.readlines():      
                    newline = line
                    newline = newline.replace("xMODELx", model_info["model"])
                    newline = newline.replace("xSSPx", model_info["ssp"])
                    newline = newline.replace("xSTART_DATEx", run_start)
                    newline = newline.replace("xEND_DATEx", run_end)
                    if "member" in model_info.keys():
                        newline = newline.replace("xMEMBERx", model_info["member"])
                    if "grid" in model_info.keys():
                        newline = newline.replace("xGRIDx", model_info["grid"])
                    newlines.append(newline)
            # Write new init file
            with open(f"{sim_path}/data_init/{model_info['ensemble']}_{var}_daily.init", "w") as f:
                for line in newlines:
                    f.writelines(line)

    ################
    # Run jobs
    ################
    ##### -noRun
    args = f"sim_dir={sim_ID}"
    out = f"{sim_path}/check_run_WBM.out"

    if os.path.isfile(f"{sim_path}/check_run_WBM.out"):
        check_job_out = subprocess.run(
            ["tail", "-n", "5", f"{sim_path}/check_run_WBM.out"],
            capture_output=True,
            text=True,
        ).stdout.split("\n")[-4]
        if check_job_out == "All Done!":
            print("Check already completed.")
        else:
            os.remove(f"{sim_path}/check_run_WBM.out")

            check_jobid = subprocess.run(
                ["qsub", "-v", args, "-o", out, f"{sim_path}/check_run_WBM.pbs"],
                capture_output=True,
                text=True,
            ).stdout.split(".")[0]
            print("check jobid: " + check_jobid)
    else:
        check_jobid = subprocess.run(
            ["qsub", "-v", args, "-o", out, f"{sim_path}/check_run_WBM.pbs"],
            capture_output=True,
            text=True,
        ).stdout.split(".")[0]
        print("check jobid: " + check_jobid)

    # check for errors in setup
    while not os.path.isfile(f"{sim_path}/check_run_WBM.out"):
        time.sleep(60)
    check_job_out = subprocess.run(
        ["tail", "-n", "5", f"{sim_path}/check_run_WBM.out"],
        capture_output=True,
        text=True,
    ).stdout.split("\n")[-4]
    if check_job_out != "All Done!":
        print("Something went wrong!")
        return None

    ####### Spool and run
    args = "sim_dir=" + sim_ID
    out = f"{sim_path}/spool_WBM.out"

    # check if spooling already done
    spooling_complete = False
    if os.path.isfile(f"{sim_path}/spool_WBM.out"):
        check_job_out = subprocess.run(
            ["tail", "-n", "5", f"{sim_path}/spool_WBM.out"],
            capture_output=True,
            text=True,
        ).stdout.split("\n")[-4]
        if check_job_out == "All Done!":
            print("Spooling already completed.")
            spooling_complete = True
        else:
            os.remove(f"{sim_path}/spool_WBM.out")
            spool_jobid = subprocess.run(
                ["qsub", "-v", args, "-o", out, f"{sim_path}/spool_WBM.pbs"],
                capture_output=True,
                text=True,
            ).stdout.split(".")[0]
            print("spool jobid: " + spool_jobid)
    else:
        spool_jobid = subprocess.run(
            ["qsub", "-v", args, "-o", out, f"{sim_path}/spool_WBM.pbs"],
            capture_output=True,
            text=True,
        ).stdout.split(".")[0]
        print("spool jobid: " + spool_jobid)

    ########## Run WBM (when spooling is complete)
    args = f"sim_dir={sim_ID}"
    out = f"{sim_path}/run_WBM.out"

    if spooling_complete:
        run_jobid = subprocess.run(
            ["qsub", "-v", args, "-o", out, f"{sim_path}/run_WBM.pbs"],
            capture_output=True,
            text=True,
        ).stdout.split(".")[0]
        print("run jobid: " + run_jobid)
    else:
        run_jobid = subprocess.run(
            ["qsub", "-W", "depend=afterok:" + spool_jobid, "-v", args, "-o", out, f"{sim_path}/run_WBM.pbs"],
            capture_output=True,
            text=True,
        ).stdout.split(".")[0]
        print("run jobid: " + run_jobid)