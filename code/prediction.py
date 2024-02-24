import jax.numpy as jnp
from water_balance_jax import construct_Kpet_vec, wbm_jax


# Prediction function: non-VIC
def make_prediction_main(
    theta, constants, x_forcing_nt, x_forcing_nyrs, x_maps
):
    # Read inputs
    tas, prcp = x_forcing_nt
    lai = x_forcing_nyrs

    (
        awCap,
        wiltingp,
        Ws_init,
        clayfrac,
        sandfrac,
        siltfrac,
        rootDepth,
        lats,
        elev_std,
        corn,
        cotton,
        rice,
        sorghum,
        soybeans,
        wheat,
    ) = x_maps

    # Define all constants
    Ts, Tm, Wi_init, Sp_init = constants

    # Define all params
    (
        awCap_scalar,
        wiltingp_scalar,
        alpha_claycoef,
        alpha_sandcoef,
        alpha_siltcoef,
        betaHBV_claycoef,
        betaHBV_sandcoef,
        betaHBV_siltcoef,
        betaHBV_elevcoef,
        GS_start_corn,
        GS_end_corn,
        L_ini_corn,
        L_dev_corn,
        L_mid_corn,
        Kc_ini_corn,
        Kc_mid_corn,
        Kc_end_corn,
        K_min_corn,
        K_max_corn,
        GS_start_cotton,
        GS_end_cotton,
        L_ini_cotton,
        L_dev_cotton,
        L_mid_cotton,
        Kc_ini_cotton,
        Kc_mid_cotton,
        Kc_end_cotton,
        K_min_cotton,
        K_max_cotton,
        GS_start_rice,
        GS_end_rice,
        L_ini_rice,
        L_dev_rice,
        L_mid_rice,
        Kc_ini_rice,
        Kc_mid_rice,
        Kc_end_rice,
        K_min_rice,
        K_max_rice,
        GS_start_sorghum,
        GS_end_sorghum,
        L_ini_sorghum,
        L_dev_sorghum,
        L_mid_sorghum,
        Kc_ini_sorghum,
        Kc_mid_sorghum,
        Kc_end_sorghum,
        K_min_sorghum,
        K_max_sorghum,
        GS_start_soybeans,
        GS_end_soybeans,
        L_ini_soybeans,
        L_dev_soybeans,
        L_mid_soybeans,
        Kc_ini_soybeans,
        Kc_mid_soybeans,
        Kc_end_soybeans,
        K_min_soybeans,
        K_max_soybeans,
        GS_start_wheat,
        GS_end_wheat,
        L_ini_wheat,
        L_dev_wheat,
        L_mid_wheat,
        Kc_ini_wheat,
        Kc_mid_wheat,
        Kc_end_wheat,
        K_min_wheat,
        K_max_wheat,
    ) = jnp.exp(theta)

    # Construct Kpet as weighted average
    Kpet_corn = construct_Kpet_vec(
        GS_start_corn,
        GS_end_corn,
        L_ini_corn,
        L_dev_corn,
        L_mid_corn,
        1.0 - (L_ini_corn + L_dev_corn + L_mid_corn),
        Kc_ini_corn,
        Kc_mid_corn,
        Kc_end_corn,
        K_min_corn,
        K_max_corn,
        lai,
    )
    Kpet_cotton = construct_Kpet_vec(
        GS_start_cotton,
        GS_end_cotton,
        L_ini_cotton,
        L_dev_cotton,
        L_mid_cotton,
        1.0 - (L_ini_cotton + L_dev_cotton + L_mid_cotton),
        Kc_ini_cotton,
        Kc_mid_cotton,
        Kc_end_cotton,
        K_min_cotton,
        K_max_cotton,
        lai,
    )
    Kpet_rice = construct_Kpet_vec(
        GS_start_rice,
        GS_end_rice,
        L_ini_rice,
        L_dev_rice,
        L_mid_rice,
        1.0 - (L_ini_rice + L_dev_rice + L_mid_rice),
        Kc_ini_rice,
        Kc_mid_rice,
        Kc_end_rice,
        K_min_rice,
        K_max_rice,
        lai,
    )
    Kpet_sorghum = construct_Kpet_vec(
        GS_start_sorghum,
        GS_end_sorghum,
        L_ini_sorghum,
        L_dev_sorghum,
        L_mid_sorghum,
        1.0 - (L_ini_sorghum + L_dev_sorghum + L_mid_sorghum),
        Kc_ini_sorghum,
        Kc_mid_sorghum,
        Kc_end_sorghum,
        K_min_sorghum,
        K_max_sorghum,
        lai,
    )
    Kpet_soybeans = construct_Kpet_vec(
        GS_start_soybeans,
        GS_end_soybeans,
        L_ini_soybeans,
        L_dev_soybeans,
        L_mid_soybeans,
        1.0 - (L_ini_soybeans + L_dev_soybeans + L_mid_soybeans),
        Kc_ini_soybeans,
        Kc_mid_soybeans,
        Kc_end_soybeans,
        K_min_soybeans,
        K_max_soybeans,
        lai,
    )
    Kpet_wheat = construct_Kpet_vec(
        GS_start_wheat,
        GS_end_wheat,
        L_ini_wheat,
        L_dev_wheat,
        L_mid_wheat,
        1.0 - (L_ini_wheat + L_dev_wheat + L_mid_wheat),
        Kc_ini_wheat,
        Kc_mid_wheat,
        Kc_end_wheat,
        K_min_wheat,
        K_max_wheat,
        lai,
    )

    other = 1.0 - (corn + cotton + rice + sorghum + soybeans + wheat)
    weights = jnp.array([corn, cotton, rice, sorghum, soybeans, wheat, other])
    Kpets = jnp.array(
        [
            Kpet_corn,
            Kpet_cotton,
            Kpet_rice,
            Kpet_sorghum,
            Kpet_soybeans,
            Kpet_wheat,
            jnp.ones(365),
        ]
    )
    Kpet = jnp.average(Kpets, weights=weights, axis=0)

    # params that WBM sees
    awCap_scaled = awCap_scalar * awCap
    wiltingp_scaled = wiltingp_scalar * wiltingp

    alpha = (
        1.0
        + (alpha_claycoef * clayfrac)
        + (alpha_sandcoef * sandfrac)
        + (alpha_siltcoef * siltfrac)
    )

    betaHBV = (
        1.0
        + (betaHBV_claycoef * clayfrac)
        + (betaHBV_sandcoef * sandfrac)
        + (betaHBV_siltcoef * siltfrac)
        + (betaHBV_elevcoef * elev_std)
    )

    params = (Ts, Tm, wiltingp_scaled, awCap_scaled, rootDepth, alpha, betaHBV)

    # Make prediction
    prediction = wbm_jax(
        tas, prcp, Kpet, Ws_init, Wi_init, Sp_init, lai, lats, params
    )

    return prediction


# Prediction functions: VIC
def make_prediction_vic(
    theta, constants, x_forcing_nt, x_forcing_nyrs, x_maps
):
    # Read inputs
    tas, prcp = x_forcing_nt
    lai = x_forcing_nyrs

    (
        awCap_frac,
        wiltingp_frac,
        sand,
        loamy_sand,
        sandy_loam,
        silt_loam,
        silt,
        loam,
        sandy_clay_loam,
        silty_clay_loam,
        clay_loam,
        sandy_clay,
        silty_clay,
        clay,
        Ws_init,
        clayfrac,
        sandfrac,
        siltfrac,
        rootDepth,
        lats,
        elev_std,
        corn,
        cotton,
        rice,
        sorghum,
        soybeans,
        wheat,
    ) = x_maps

    # Define all constants
    Ts, Tm, Wi_init, Sp_init = constants

    # Define all params
    (
        awCap_sand,
        awCap_loamy_sand,
        awCap_sandy_loam,
        awCap_silt_loam,
        awCap_silt,
        awCap_loam,
        awCap_sandy_clay_loam,
        awCap_silty_clay_loam,
        awCap_clay_loam,
        awCap_sandy_clay,
        awCap_silty_clay,
        awCap_clay,
        wiltingp_sand,
        wiltingp_loamy_sand,
        wiltingp_sandy_loam,
        wiltingp_silt_loam,
        wiltingp_silt,
        wiltingp_loam,
        wiltingp_sandy_clay_loam,
        wiltingp_silty_clay_loam,
        wiltingp_clay_loam,
        wiltingp_sandy_clay,
        wiltingp_silty_clay,
        wiltingp_clay,
        # awCap_claycoef,
        # awCap_sandcoef,
        # awCap_siltcoef,
        # wiltingp_claycoef,
        # wiltingp_sandcoef,
        # wiltingp_siltcoef,
        alpha_claycoef,
        alpha_sandcoef,
        alpha_siltcoef,
        betaHBV_claycoef,
        betaHBV_sandcoef,
        betaHBV_siltcoef,
        betaHBV_elevcoef,
        GS_start_corn,
        GS_end_corn,
        L_ini_corn,
        L_dev_corn,
        L_mid_corn,
        Kc_ini_corn,
        Kc_mid_corn,
        Kc_end_corn,
        K_min_corn,
        K_max_corn,
        GS_start_cotton,
        GS_end_cotton,
        L_ini_cotton,
        L_dev_cotton,
        L_mid_cotton,
        Kc_ini_cotton,
        Kc_mid_cotton,
        Kc_end_cotton,
        K_min_cotton,
        K_max_cotton,
        GS_start_rice,
        GS_end_rice,
        L_ini_rice,
        L_dev_rice,
        L_mid_rice,
        Kc_ini_rice,
        Kc_mid_rice,
        Kc_end_rice,
        K_min_rice,
        K_max_rice,
        GS_start_sorghum,
        GS_end_sorghum,
        L_ini_sorghum,
        L_dev_sorghum,
        L_mid_sorghum,
        Kc_ini_sorghum,
        Kc_mid_sorghum,
        Kc_end_sorghum,
        K_min_sorghum,
        K_max_sorghum,
        GS_start_soybeans,
        GS_end_soybeans,
        L_ini_soybeans,
        L_dev_soybeans,
        L_mid_soybeans,
        Kc_ini_soybeans,
        Kc_mid_soybeans,
        Kc_end_soybeans,
        K_min_soybeans,
        K_max_soybeans,
        GS_start_wheat,
        GS_end_wheat,
        L_ini_wheat,
        L_dev_wheat,
        L_mid_wheat,
        Kc_ini_wheat,
        Kc_mid_wheat,
        Kc_end_wheat,
        K_min_wheat,
        K_max_wheat,
    ) = jnp.exp(theta)

    # Construct Kpet as weighted average
    Kpet_corn = construct_Kpet_vec(
        GS_start_corn,
        GS_end_corn,
        L_ini_corn,
        L_dev_corn,
        L_mid_corn,
        1.0 - (L_ini_corn + L_dev_corn + L_mid_corn),
        Kc_ini_corn,
        Kc_mid_corn,
        Kc_end_corn,
        K_min_corn,
        K_max_corn,
        lai,
    )
    Kpet_cotton = construct_Kpet_vec(
        GS_start_cotton,
        GS_end_cotton,
        L_ini_cotton,
        L_dev_cotton,
        L_mid_cotton,
        1.0 - (L_ini_cotton + L_dev_cotton + L_mid_cotton),
        Kc_ini_cotton,
        Kc_mid_cotton,
        Kc_end_cotton,
        K_min_cotton,
        K_max_cotton,
        lai,
    )
    Kpet_rice = construct_Kpet_vec(
        GS_start_rice,
        GS_end_rice,
        L_ini_rice,
        L_dev_rice,
        L_mid_rice,
        1.0 - (L_ini_rice + L_dev_rice + L_mid_rice),
        Kc_ini_rice,
        Kc_mid_rice,
        Kc_end_rice,
        K_min_rice,
        K_max_rice,
        lai,
    )
    Kpet_sorghum = construct_Kpet_vec(
        GS_start_sorghum,
        GS_end_sorghum,
        L_ini_sorghum,
        L_dev_sorghum,
        L_mid_sorghum,
        1.0 - (L_ini_sorghum + L_dev_sorghum + L_mid_sorghum),
        Kc_ini_sorghum,
        Kc_mid_sorghum,
        Kc_end_sorghum,
        K_min_sorghum,
        K_max_sorghum,
        lai,
    )
    Kpet_soybeans = construct_Kpet_vec(
        GS_start_soybeans,
        GS_end_soybeans,
        L_ini_soybeans,
        L_dev_soybeans,
        L_mid_soybeans,
        1.0 - (L_ini_soybeans + L_dev_soybeans + L_mid_soybeans),
        Kc_ini_soybeans,
        Kc_mid_soybeans,
        Kc_end_soybeans,
        K_min_soybeans,
        K_max_soybeans,
        lai,
    )
    Kpet_wheat = construct_Kpet_vec(
        GS_start_wheat,
        GS_end_wheat,
        L_ini_wheat,
        L_dev_wheat,
        L_mid_wheat,
        1.0 - (L_ini_wheat + L_dev_wheat + L_mid_wheat),
        Kc_ini_wheat,
        Kc_mid_wheat,
        Kc_end_wheat,
        K_min_wheat,
        K_max_wheat,
        lai,
    )

    other = 1.0 - (corn + cotton + rice + sorghum + soybeans + wheat)
    weights = jnp.array([corn, cotton, rice, sorghum, soybeans, wheat, other])
    Kpets = jnp.array(
        [
            Kpet_corn,
            Kpet_cotton,
            Kpet_rice,
            Kpet_sorghum,
            Kpet_soybeans,
            Kpet_wheat,
            jnp.ones(365),
        ]
    )
    Kpet = jnp.average(Kpets, weights=weights, axis=0)

    # params that WBM sees
    awCap_scaled = 1 * (
        (awCap_sand * sand)
        + (awCap_loamy_sand * loamy_sand)
        + (awCap_sandy_loam * sandy_loam)
        + (awCap_silt_loam * silt_loam)
        + (awCap_silt * silt)
        + (awCap_loam * loam)
        + (awCap_sandy_clay_loam * sandy_clay_loam)
        + (awCap_silty_clay_loam * silty_clay_loam)
        + (awCap_clay_loam * clay_loam)
        + (awCap_sandy_clay * sandy_clay)
        + (awCap_silty_clay * silty_clay)
        + (awCap_clay * clay)
    )
    wiltingp_scaled = 1 * (
        (wiltingp_sand * sand)
        + (wiltingp_loamy_sand * loamy_sand)
        + (wiltingp_sandy_loam * sandy_loam)
        + (wiltingp_silt_loam * silt_loam)
        + (wiltingp_silt * silt)
        + (wiltingp_loam * loam)
        + (wiltingp_sandy_clay_loam * sandy_clay_loam)
        + (wiltingp_silty_clay_loam * silty_clay_loam)
        + (wiltingp_clay_loam * clay_loam)
        + (wiltingp_sandy_clay * sandy_clay)
        + (wiltingp_silty_clay * silty_clay)
        + (wiltingp_clay * clay)
    )

    alpha = (
        1.0
        + (alpha_claycoef * clayfrac)
        + (alpha_sandcoef * sandfrac)
        + (alpha_siltcoef * siltfrac)
    )
    betaHBV = (
        1.0
        + (betaHBV_claycoef * clayfrac)
        + (betaHBV_sandcoef * sandfrac)
        + (betaHBV_siltcoef * siltfrac)
        + (betaHBV_elevcoef * elev_std)
    )

    params = (Ts, Tm, wiltingp_scaled, awCap_scaled, rootDepth, alpha, betaHBV)

    # Make prediction
    prediction = wbm_jax(
        tas, prcp, Kpet, Ws_init, Wi_init, Sp_init, lai, lats, params
    )

    return prediction
