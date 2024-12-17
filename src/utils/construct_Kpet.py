import jax.numpy as jnp
import numpy as np
import xarray as xr

from water_balance_jax import construct_Kpet_crop, construct_Kpet_gen


def construct_Kpet_map(df_params, ds_land, ds_lai):
    # Grab all required parameters
    GS_start_corn = jnp.exp(df_params["GS_start_corn"].iloc[0])
    GS_end_corn = jnp.exp(df_params["GS_end_corn"].iloc[0])
    L_ini_corn = jnp.exp(df_params["L_ini_corn"].iloc[0])
    L_dev_corn = jnp.exp(df_params["L_dev_corn"].iloc[0])
    L_mid_corn = jnp.exp(df_params["L_mid_corn"].iloc[0])
    Kc_ini_corn = jnp.exp(df_params["Kc_ini_corn"].iloc[0])
    Kc_mid_corn = jnp.exp(df_params["Kc_mid_corn"].iloc[0])
    Kc_end_corn = jnp.exp(df_params["Kc_end_corn"].iloc[0])
    Kmin_corn = jnp.exp(df_params["Kmin_corn"].iloc[0])
    Kmax_corn = jnp.exp(df_params["Kmax_corn"].iloc[0])
    c_lai_corn = jnp.exp(df_params["c_lai_corn"].iloc[0])
    GS_start_cotton = jnp.exp(df_params["GS_start_cotton"].iloc[0])
    GS_end_cotton = jnp.exp(df_params["GS_end_cotton"].iloc[0])
    L_ini_cotton = jnp.exp(df_params["L_ini_cotton"].iloc[0])
    L_dev_cotton = jnp.exp(df_params["L_dev_cotton"].iloc[0])
    L_mid_cotton = jnp.exp(df_params["L_mid_cotton"].iloc[0])
    Kc_ini_cotton = jnp.exp(df_params["Kc_ini_cotton"].iloc[0])
    Kc_mid_cotton = jnp.exp(df_params["Kc_mid_cotton"].iloc[0])
    Kc_end_cotton = jnp.exp(df_params["Kc_end_cotton"].iloc[0])
    Kmin_cotton = jnp.exp(df_params["Kmin_cotton"].iloc[0])
    Kmax_cotton = jnp.exp(df_params["Kmax_cotton"].iloc[0])
    c_lai_cotton = jnp.exp(df_params["c_lai_cotton"].iloc[0])
    GS_start_rice = jnp.exp(df_params["GS_start_rice"].iloc[0])
    GS_end_rice = jnp.exp(df_params["GS_end_rice"].iloc[0])
    L_ini_rice = jnp.exp(df_params["L_ini_rice"].iloc[0])
    L_dev_rice = jnp.exp(df_params["L_dev_rice"].iloc[0])
    L_mid_rice = jnp.exp(df_params["L_mid_rice"].iloc[0])
    Kc_ini_rice = jnp.exp(df_params["Kc_ini_rice"].iloc[0])
    Kc_mid_rice = jnp.exp(df_params["Kc_mid_rice"].iloc[0])
    Kc_end_rice = jnp.exp(df_params["Kc_end_rice"].iloc[0])
    Kmin_rice = jnp.exp(df_params["Kmin_rice"].iloc[0])
    Kmax_rice = jnp.exp(df_params["Kmax_rice"].iloc[0])
    c_lai_rice = jnp.exp(df_params["c_lai_rice"].iloc[0])
    GS_start_sorghum = jnp.exp(df_params["GS_start_sorghum"].iloc[0])
    GS_end_sorghum = jnp.exp(df_params["GS_end_sorghum"].iloc[0])
    L_ini_sorghum = jnp.exp(df_params["L_ini_sorghum"].iloc[0])
    L_dev_sorghum = jnp.exp(df_params["L_dev_sorghum"].iloc[0])
    L_mid_sorghum = jnp.exp(df_params["L_mid_sorghum"].iloc[0])
    Kc_ini_sorghum = jnp.exp(df_params["Kc_ini_sorghum"].iloc[0])
    Kc_mid_sorghum = jnp.exp(df_params["Kc_mid_sorghum"].iloc[0])
    Kc_end_sorghum = jnp.exp(df_params["Kc_end_sorghum"].iloc[0])
    Kmin_sorghum = jnp.exp(df_params["Kmin_sorghum"].iloc[0])
    Kmax_sorghum = jnp.exp(df_params["Kmax_sorghum"].iloc[0])
    c_lai_sorghum = jnp.exp(df_params["c_lai_sorghum"].iloc[0])
    GS_start_soybeans = jnp.exp(df_params["GS_start_soybeans"].iloc[0])
    GS_end_soybeans = jnp.exp(df_params["GS_end_soybeans"].iloc[0])
    L_ini_soybeans = jnp.exp(df_params["L_ini_soybeans"].iloc[0])
    L_dev_soybeans = jnp.exp(df_params["L_dev_soybeans"].iloc[0])
    L_mid_soybeans = jnp.exp(df_params["L_mid_soybeans"].iloc[0])
    Kc_ini_soybeans = jnp.exp(df_params["Kc_ini_soybeans"].iloc[0])
    Kc_mid_soybeans = jnp.exp(df_params["Kc_mid_soybeans"].iloc[0])
    Kc_end_soybeans = jnp.exp(df_params["Kc_end_soybeans"].iloc[0])
    Kmin_soybeans = jnp.exp(df_params["Kmin_soybeans"].iloc[0])
    Kmax_soybeans = jnp.exp(df_params["Kmax_soybeans"].iloc[0])
    c_lai_soybeans = jnp.exp(df_params["c_lai_soybeans"].iloc[0])
    GS_start_wheat = jnp.exp(df_params["GS_start_wheat"].iloc[0])
    GS_end_wheat = jnp.exp(df_params["GS_end_wheat"].iloc[0])
    L_ini_wheat = jnp.exp(df_params["L_ini_wheat"].iloc[0])
    L_dev_wheat = jnp.exp(df_params["L_dev_wheat"].iloc[0])
    L_mid_wheat = jnp.exp(df_params["L_mid_wheat"].iloc[0])
    Kc_ini_wheat = jnp.exp(df_params["Kc_ini_wheat"].iloc[0])
    Kc_mid_wheat = jnp.exp(df_params["Kc_mid_wheat"].iloc[0])
    Kc_end_wheat = jnp.exp(df_params["Kc_end_wheat"].iloc[0])
    Kmin_wheat = jnp.exp(df_params["Kmin_wheat"].iloc[0])
    Kmax_wheat = jnp.exp(df_params["Kmax_wheat"].iloc[0])
    c_lai_wheat = jnp.exp(df_params["c_lai_wheat"].iloc[0])
    Kmin_cropland_other = jnp.exp(df_params["Kmin_cropland_other"].iloc[0])
    Kmax_cropland_other = jnp.exp(df_params["Kmax_cropland_other"].iloc[0])
    c_lai_cropland_other = jnp.exp(df_params["c_lai_cropland_other"].iloc[0])
    Kmin_evergreen_needleleaf = jnp.exp(
        df_params["Kmin_evergreen_needleleaf"].iloc[0]
    )
    Kmax_evergreen_needleleaf = jnp.exp(
        df_params["Kmax_evergreen_needleleaf"].iloc[0]
    )
    c_lai_evergreen_needleleaf = jnp.exp(
        df_params["c_lai_evergreen_needleleaf"].iloc[0]
    )
    Kmin_evergreen_broadleaf = jnp.exp(
        df_params["Kmin_evergreen_broadleaf"].iloc[0]
    )
    Kmax_evergreen_broadleaf = jnp.exp(
        df_params["Kmax_evergreen_broadleaf"].iloc[0]
    )
    c_lai_evergreen_broadleaf = jnp.exp(
        df_params["c_lai_evergreen_broadleaf"].iloc[0]
    )
    Kmin_deciduous_needleleaf = jnp.exp(
        df_params["Kmin_deciduous_needleleaf"].iloc[0]
    )
    Kmax_deciduous_needleleaf = jnp.exp(
        df_params["Kmax_deciduous_needleleaf"].iloc[0]
    )
    c_lai_deciduous_needleleaf = jnp.exp(
        df_params["c_lai_deciduous_needleleaf"].iloc[0]
    )
    Kmin_deciduous_broadleaf = jnp.exp(
        df_params["Kmin_deciduous_broadleaf"].iloc[0]
    )
    Kmax_deciduous_broadleaf = jnp.exp(
        df_params["Kmax_deciduous_broadleaf"].iloc[0]
    )
    c_lai_deciduous_broadleaf = jnp.exp(
        df_params["c_lai_deciduous_broadleaf"].iloc[0]
    )
    Kmin_mixed_forest = jnp.exp(df_params["Kmin_mixed_forest"].iloc[0])
    Kmax_mixed_forest = jnp.exp(df_params["Kmax_mixed_forest"].iloc[0])
    c_lai_mixed_forest = jnp.exp(df_params["c_lai_mixed_forest"].iloc[0])
    Kmin_woodland = jnp.exp(df_params["Kmin_woodland"].iloc[0])
    Kmax_woodland = jnp.exp(df_params["Kmax_woodland"].iloc[0])
    c_lai_woodland = jnp.exp(df_params["c_lai_woodland"].iloc[0])
    Kmin_wooded_grassland = jnp.exp(df_params["Kmin_wooded_grassland"].iloc[0])
    Kmax_wooded_grassland = jnp.exp(df_params["Kmax_wooded_grassland"].iloc[0])
    c_lai_wooded_grassland = jnp.exp(
        df_params["c_lai_wooded_grassland"].iloc[0]
    )
    Kmin_closed_shurbland = jnp.exp(df_params["Kmin_closed_shurbland"].iloc[0])
    Kmax_closed_shurbland = jnp.exp(df_params["Kmax_closed_shurbland"].iloc[0])
    c_lai_closed_shurbland = jnp.exp(
        df_params["c_lai_closed_shurbland"].iloc[0]
    )
    Kmin_open_shrubland = jnp.exp(df_params["Kmin_open_shrubland"].iloc[0])
    Kmax_open_shrubland = jnp.exp(df_params["Kmax_open_shrubland"].iloc[0])
    c_lai_open_shrubland = jnp.exp(df_params["c_lai_open_shrubland"].iloc[0])
    Kmin_grassland = jnp.exp(df_params["Kmin_grassland"].iloc[0])
    Kmax_grassland = jnp.exp(df_params["Kmax_grassland"].iloc[0])
    c_lai_grassland = jnp.exp(df_params["c_lai_grassland"].iloc[0])
    Kmin_barren = jnp.exp(df_params["Kmin_barren"].iloc[0])
    Kmax_barren = jnp.exp(df_params["Kmax_barren"].iloc[0])
    c_lai_barren = jnp.exp(df_params["c_lai_barren"].iloc[0])
    Kmin_urban = jnp.exp(df_params["Kmin_urban"].iloc[0])
    Kmax_urban = jnp.exp(df_params["Kmax_urban"].iloc[0])
    c_lai_urban = jnp.exp(df_params["c_lai_urban"].iloc[0])

    # Transform LAI to numpy array
    lai = np.transpose(ds_lai["LAI"].to_numpy(), (2, 1, 0))

    # Construct Kpet as weighted average
    Kpet_corn = construct_Kpet_crop(
        GS_start_corn,
        GS_end_corn,
        L_ini_corn,
        L_dev_corn,
        L_mid_corn,
        1.0 - (L_ini_corn + L_dev_corn + L_mid_corn),
        Kc_ini_corn,
        Kc_mid_corn,
        Kc_end_corn,
        Kmin_corn,
        Kmax_corn,
        c_lai_corn,
        lai,
    )
    Kpet_cotton = construct_Kpet_crop(
        GS_start_cotton,
        GS_end_cotton,
        L_ini_cotton,
        L_dev_cotton,
        L_mid_cotton,
        1.0 - (L_ini_cotton + L_dev_cotton + L_mid_cotton),
        Kc_ini_cotton,
        Kc_mid_cotton,
        Kc_end_cotton,
        Kmin_cotton,
        Kmax_cotton,
        c_lai_cotton,
        lai,
    )
    Kpet_rice = construct_Kpet_crop(
        GS_start_rice,
        GS_end_rice,
        L_ini_rice,
        L_dev_rice,
        L_mid_rice,
        1.0 - (L_ini_rice + L_dev_rice + L_mid_rice),
        Kc_ini_rice,
        Kc_mid_rice,
        Kc_end_rice,
        Kmin_rice,
        Kmax_rice,
        c_lai_rice,
        lai,
    )
    Kpet_sorghum = construct_Kpet_crop(
        GS_start_sorghum,
        GS_end_sorghum,
        L_ini_sorghum,
        L_dev_sorghum,
        L_mid_sorghum,
        1.0 - (L_ini_sorghum + L_dev_sorghum + L_mid_sorghum),
        Kc_ini_sorghum,
        Kc_mid_sorghum,
        Kc_end_sorghum,
        Kmin_sorghum,
        Kmax_sorghum,
        c_lai_sorghum,
        lai,
    )
    Kpet_soybeans = construct_Kpet_crop(
        GS_start_soybeans,
        GS_end_soybeans,
        L_ini_soybeans,
        L_dev_soybeans,
        L_mid_soybeans,
        1.0 - (L_ini_soybeans + L_dev_soybeans + L_mid_soybeans),
        Kc_ini_soybeans,
        Kc_mid_soybeans,
        Kc_end_soybeans,
        Kmin_soybeans,
        Kmax_soybeans,
        c_lai_soybeans,
        lai,
    )
    Kpet_wheat = construct_Kpet_crop(
        GS_start_wheat,
        GS_end_wheat,
        L_ini_wheat,
        L_dev_wheat,
        L_mid_wheat,
        1.0 - (L_ini_wheat + L_dev_wheat + L_mid_wheat),
        Kc_ini_wheat,
        Kc_mid_wheat,
        Kc_end_wheat,
        Kmin_wheat,
        Kmax_wheat,
        c_lai_wheat,
        lai,
    )

    Kpet_cropland_other = construct_Kpet_gen(
        Kmin_cropland_other, Kmax_cropland_other, c_lai_cropland_other, lai
    )
    Kpet_evergreen_needleleaf = construct_Kpet_gen(
        Kmin_evergreen_needleleaf,
        Kmax_evergreen_needleleaf,
        c_lai_evergreen_needleleaf,
        lai,
    )
    Kpet_evergreen_broadleaf = construct_Kpet_gen(
        Kmin_evergreen_broadleaf,
        Kmax_evergreen_broadleaf,
        c_lai_evergreen_broadleaf,
        lai,
    )
    Kpet_deciduous_needleleaf = construct_Kpet_gen(
        Kmin_deciduous_needleleaf,
        Kmax_deciduous_needleleaf,
        c_lai_deciduous_needleleaf,
        lai,
    )
    Kpet_deciduous_broadleaf = construct_Kpet_gen(
        Kmin_deciduous_broadleaf,
        Kmax_deciduous_broadleaf,
        c_lai_deciduous_broadleaf,
        lai,
    )
    Kpet_mixed_forest = construct_Kpet_gen(
        Kmin_mixed_forest, Kmax_mixed_forest, c_lai_mixed_forest, lai
    )
    Kpet_woodland = construct_Kpet_gen(
        Kmin_woodland, Kmax_woodland, c_lai_woodland, lai
    )
    Kpet_wooded_grassland = construct_Kpet_gen(
        Kmin_wooded_grassland,
        Kmax_wooded_grassland,
        c_lai_wooded_grassland,
        lai,
    )
    Kpet_closed_shurbland = construct_Kpet_gen(
        Kmin_closed_shurbland,
        Kmax_closed_shurbland,
        c_lai_closed_shurbland,
        lai,
    )
    Kpet_open_shrubland = construct_Kpet_gen(
        Kmin_open_shrubland, Kmax_open_shrubland, c_lai_open_shrubland, lai
    )
    Kpet_grassland = construct_Kpet_gen(
        Kmin_grassland, Kmax_grassland, c_lai_grassland, lai
    )
    Kpet_barren = construct_Kpet_gen(
        Kmin_barren, Kmax_barren, c_lai_barren, lai
    )
    Kpet_urban = construct_Kpet_gen(Kmin_urban, Kmax_urban, c_lai_urban, lai)

    weights = jnp.array(
        [
            np.transpose(ds_land["corn"].to_numpy()),
            np.transpose(ds_land["cotton"].to_numpy()),
            np.transpose(ds_land["rice"].to_numpy()),
            np.transpose(ds_land["sorghum"].to_numpy()),
            np.transpose(ds_land["soybeans"].to_numpy()),
            np.transpose(ds_land["durum_wheat"].to_numpy())
            + np.transpose(ds_land["spring_wheat"].to_numpy()),
            np.transpose(ds_land["cropland_other"].to_numpy()),
            np.transpose(ds_land["evergreen_needleleaf"].to_numpy()),
            np.transpose(ds_land["evergreen_broadleaf"].to_numpy()),
            np.transpose(ds_land["deciduous_needleleaf"].to_numpy()),
            np.transpose(ds_land["deciduous_broadleaf"].to_numpy()),
            np.transpose(ds_land["mixed_forest"].to_numpy()),
            np.transpose(ds_land["woodland"].to_numpy()),
            np.transpose(ds_land["wooded_grassland"].to_numpy()),
            np.transpose(ds_land["closed_shurbland"].to_numpy()),
            np.transpose(ds_land["open_shrubland"].to_numpy()),
            np.transpose(ds_land["grassland"].to_numpy()),
            np.transpose(ds_land["barren"].to_numpy()),
            np.transpose(ds_land["urban"].to_numpy()),
        ]
    )
    Kpets = jnp.array(
        [
            Kpet_corn,
            Kpet_cotton,
            Kpet_rice,
            Kpet_sorghum,
            Kpet_soybeans,
            Kpet_wheat,
            Kpet_cropland_other,
            Kpet_evergreen_needleleaf,
            Kpet_evergreen_broadleaf,
            Kpet_deciduous_needleleaf,
            Kpet_deciduous_broadleaf,
            Kpet_mixed_forest,
            Kpet_woodland,
            Kpet_wooded_grassland,
            Kpet_closed_shurbland,
            Kpet_open_shrubland,
            Kpet_grassland,
            Kpet_barren,
            Kpet_urban,
        ]
    )
    Kpet = jnp.average(
        Kpets,
        weights=np.tile(weights[..., np.newaxis], (1, 1, 1, 365)),
        axis=0,
    )

    # Transform back to xarray
    ds_Kpet = xr.Dataset(
        data_vars=dict(
            Kpet=(
                ["doy", "lat", "lon"],
                np.transpose(Kpet, (2, 1, 0)).astype(np.float32),
            )
        ),
        coords=dict(
            lon=ds_land["lon"].astype(np.float32),
            lat=ds_land["lat"].astype(np.float32),
            doy=np.arange(365),
        ),
    ).drop_vars(["time", "spatial_ref"])

    return ds_Kpet
