import json

import jax.numpy as jnp
import numpy as np
import pandas as pd
from SALib import ProblemSpec
from src.water_balance_jax import (
    construct_Kpet_crop,
    construct_Kpet_gen,
    wbm_jax,
)
from utils.global_paths import project_data_path


def wbm_sobol(
    ix,
    iy,
    forcing,
    eval,
    tas_delta,
    prcp_factor,
    Kpet_name,
    experiment_name,
    N,
    save_name,
):
    """
    Perform a single gridpoint Sobol SA using parameter file in `experiment_name`.
        - ix, iy assuming CONUS (whole domain)
        - Metrics include mean, SD, range, and RMSE against obs if `eval` = NLDAS, SMAP
        - CC can be imposed by `tas_delta` and `prcp_factor`
    """
    ##################
    # Get forcing
    ##################
    # Read data
    if forcing == "SMAP":
        forcing_data = np.load(
            f"{project_data_path}/WBM/calibration/CONUS/{forcing}/inputs.npz"
        )
    else:
        forcing_data = np.load(
            f"{project_data_path}/WBM/calibration/CONUS/VIC/inputs.npz"
        )  # identical for SA purposes

    # Select gridpoint
    tas_in = forcing_data["tas"][ix, iy, :]
    prcp_in = forcing_data["prcp"][ix, iy, :]
    lai_in = forcing_data["lai"][ix, iy, :]
    phi = forcing_data["lats"][iy]

    ###################
    # Get observations
    ###################
    if eval == "SMAP":
        obs = np.load(
            f"{project_data_path}/WBM/calibration/CONUS/{forcing}/{forcing}_validation.npy"
        )
        obs = obs[ix, iy, :]
        obs_centered = obs - jnp.mean(obs)
    elif eval == "NLDAS":
        obs_list = ["VIC", "NOAH", "MOSAIC"]
        obs = [
            np.load(
                f"{project_data_path}/WBM/calibration/CONUS/{obs}/{obs}_validation.npy"
            )
            for obs in obs_list
        ]
        obs = [obs_tmp[ix, iy, :] for obs_tmp in obs]
        obs_centered = [obs_tmp - jnp.mean(obs_tmp) for obs_tmp in obs]

    ################
    # Get params
    ################
    params = np.loadtxt(
        f"{project_data_path}/WBM/SA/{experiment_name}_{str(N)}_params.txt",
        float,
    )
    n_params = len(params)

    ####################
    # Loop through all
    ####################
    out_mean = np.zeros(n_params)
    out_sd = np.zeros(n_params)
    out_range = np.zeros(n_params)
    if eval == "SMAP":
        out_rmse = np.zeros(n_params)
        out_ubrmse = np.zeros(n_params)
    elif eval == "NLDAS":
        out_rmse = [np.zeros(n_params) for _ in range(3)]
        out_ubrmse = [np.zeros(n_params) for _ in range(3)]

    for iparam in range(n_params):
        # Read in correct order!
        Ts = params[iparam][0]
        Tm = params[iparam][1]
        wiltingp = params[iparam][2]
        awCap = params[iparam][3]
        alpha = params[iparam][4]
        betaHBV = params[iparam][5]
        if Kpet_name == "crop":
            GS_start = int(params[iparam][6])
            GS_length = int(params[iparam][7])
            L_ini = params[iparam][8]
            L_dev = params[iparam][9]
            L_mid = params[iparam][10]
            L_late = 1.0 - (L_ini + L_dev + L_mid)
            Kc_ini = params[iparam][11]
            Kc_mid = params[iparam][12]
            Kc_end = params[iparam][13]
            Kmin = params[iparam][14]
            Kmax = params[iparam][15]
            c_lai = params[iparam][16]
            # Construct Kc timeseries
            Kpet_in = construct_Kpet_crop(
                GS_start,
                GS_length,
                L_ini,
                L_dev,
                L_mid,
                L_late,
                Kc_ini,
                Kc_mid,
                Kc_end,
                Kmin,
                Kmax,
                c_lai,
                lai_in,
            )
        elif Kpet_name == "gen":
            Kmin = params[iparam][6]
            Kmax = params[iparam][7]
            c_lai = params[iparam][8]
            # Construct Kc timeseries
            Kpet_in = construct_Kpet_gen(Kmin, Kmax, c_lai, lai_in)

        # Initial conditions
        Ws_init = awCap / 2.0  # Initial soil moisture
        Wi_init = 0.0  # Canopy water storage
        Sp_init = 0.0  # Snowpack

        # Assume 1m root depth
        rootDepth = 1.0

        # Run it
        out = wbm_jax(
            tas=tas_in + tas_delta,
            prcp=prcp_in * prcp_factor,
            Kpet=Kpet_in,
            Ws_init=Ws_init,
            Wi_init=Wi_init,
            Sp_init=Sp_init,
            lai=lai_in,
            phi=phi,
            params=(Ts, Tm, awCap, wiltingp, rootDepth, alpha, betaHBV),
        )

        # Store metrics
        out_mean[iparam] = jnp.mean(out)
        out_sd[iparam] = jnp.std(out)
        out_range[iparam] = jnp.max(out) - jnp.min(out)

        out_centered = out - jnp.mean(out)
        if eval == "SMAP":
            out_rmse[iparam] = jnp.sqrt(jnp.mean((out - obs) ** 2))
            out_ubrmse[iparam] = jnp.sqrt(
                jnp.mean((out_centered - obs_centered) ** 2)
            )
        elif eval == "NLDAS":
            for io in range(len(obs_list)):
                out_rmse[io][iparam] = jnp.sqrt(jnp.mean((out - obs[io]) ** 2))
                out_ubrmse[io][iparam] = jnp.sqrt(
                    jnp.mean((out_centered - obs_centered[io]) ** 2)
                )

    ####################
    # Calculate indices
    ####################
    # Problem spec
    with open(f"{project_data_path}/WBM/SA/{experiment_name}.json") as f:
        params_dict = json.load(f)

    param_names = list(params_dict.keys())
    sp = ProblemSpec(
        {
            "num_vars": n_params,
            "names": param_names,
            "bounds": [params_dict[param] for param in param_names],
        }
    ).set_samples(params)

    # Analyze all
    df_out = []

    # Set up list
    if eval == "SMAP":
        metrics = [out_mean, out_sd, out_range, out_rmse, out_ubrmse]
        metric_names = ["mean", "sd", "range", "rmse_SMAP", "ubrmse_SMAP"]
    elif eval == "NLDAS":
        metrics = (
            [out_mean, out_sd, out_range]
            + [out for out in out_rmse]
            + [out for out in out_ubrmse]
        )
        metric_names = (
            ["mean", "sd", "range"]
            + [f"rmse_{obs}" for obs in obs_list]
            + [f"ubrmse_{obs}" for obs in obs_list]
        )
    else:
        metrics = [out_mean, out_sd, out_range]
        metric_names = ["mean", "sd", "range"]

    # Loop and calculate
    for metric, metric_name in zip(metrics, metric_names):
        # Calculate
        sp.set_results(metric)
        sp.analyze_sobol()
        # Store
        total, first, second = sp.to_df()
        df_tmp = pd.merge(total, first, left_index=True, right_index=True)
        df_tmp["metric"] = metric_name
        df_out.append(df_tmp.reset_index().rename(columns={"index": "param"}))

    # Save
    df_out = pd.concat(df_out)
    df_out.to_csv(
        f"{project_data_path}/WBM/SA/{save_name}_res.csv", index=False
    )
