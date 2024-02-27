# Reads inputs for WBM calibration from npz files and prepares them for vmap
import jax.numpy as jnp
import numpy as np
from utils.global_paths import project_data_path


def read_inputs(subset_name, obs_name, remove_nans):
    ######################
    # Read obs
    obs = np.load(
        f"{project_data_path}/WBM/calibration/{subset_name}/{obs_name}/{obs_name}_validation.npy"
    )

    ######################
    # Read and extract inputs
    npz = np.load(
        f"{project_data_path}/WBM/calibration/{subset_name}/{obs_name}/inputs.npz"
    )

    # Meteo forcing
    tas = npz["tas"]
    prcp = npz["prcp"]

    # LAI
    lai = npz["lai"]

    # Soil properties
    awCap = npz["awCap"]
    wiltingp = npz["wiltingp"]

    clayfrac = npz["clayfrac"]
    sandfrac = npz["sandfrac"]
    siltfrac = npz["siltfrac"]

    rootDepth = npz["rootDepth"]

    # Land use
    corn = npz["corn"]
    cotton = npz["cotton"]
    rice = npz["rice"]
    sorghum = npz["sorghum"]
    soybeans = npz["soybeans"]
    durum_wheat = npz["durum_wheat"]
    spring_wheat = npz["spring_wheat"]
    wheat = durum_wheat + spring_wheat

    cropland_other = npz["cropland_other"]
    evergreen_needleleaf = npz["evergreen_needleleaf"]
    evergreen_broadleaf = npz["evergreen_broadleaf"]
    deciduous_needleleaf = npz["deciduous_needleleaf"]
    deciduous_broadleaf = npz["deciduous_broadleaf"]
    mixed_forest = npz["mixed_forest"]
    woodland = npz["woodland"]
    wooded_grassland = npz["wooded_grassland"]
    closed_shurbland = npz["closed_shurbland"]
    open_shrubland = npz["open_shrubland"]
    grassland = npz["grassland"]
    barren = npz["barren"]
    urban = npz["urban"]

    # Geophysical
    elev_std = npz["elev_std"]

    lats = npz["lats"]

    # Initial conditions
    Ws_init = npz["soilMoist_init"]

    ##########################
    # Prepare inputs for vmap:
    # spatial dimensions need to be collapsed and first
    # NaN gridpoints need to be removed
    nx = tas.shape[0]
    ny = tas.shape[1]
    nt = tas.shape[2]

    assert nt % 365 == 0

    ## Obs
    ys = obs.reshape(nx * ny, nt)
    nan_inds_obs = jnp.isnan(ys).any(axis=1)

    ## Forcing: all days
    tas_in = tas.reshape(nx * ny, nt)
    prcp_in = prcp.reshape(nx * ny, nt)

    x_forcing_nt = jnp.stack([tas_in, prcp_in], axis=1)
    nan_inds_forcing_nt = jnp.isnan(x_forcing_nt).any(axis=(1, 2))

    ## Forcing: yearly
    lai_in = lai.reshape(nx * ny, 365)
    x_forcing_nyrs = lai_in
    nan_inds_forcing_nyrs = jnp.isnan(x_forcing_nyrs).any(axis=1)

    ## Maps
    awCap_in = awCap.reshape(nx * ny)
    wiltingp_in = wiltingp.reshape(nx * ny)

    Ws_init_in = Ws_init.reshape(nx * ny)

    clayfrac_in = clayfrac.reshape(nx * ny)
    sandfrac_in = sandfrac.reshape(nx * ny)
    siltfrac_in = siltfrac.reshape(nx * ny)

    rootDepth_in = rootDepth.reshape(nx * ny)

    lats_in = np.tile(lats, nx)
    elev_std_in = elev_std.reshape(nx * ny)

    corn_in = corn.reshape(nx * ny)
    cotton_in = cotton.reshape(nx * ny)
    rice_in = rice.reshape(nx * ny)
    sorghum_in = sorghum.reshape(nx * ny)
    soybeans_in = soybeans.reshape(nx * ny)
    wheat_in = wheat.reshape(nx * ny)

    cropland_other_in = cropland_other.reshape(nx * ny)
    evergreen_needleleaf_in = evergreen_needleleaf.reshape(nx * ny)
    evergreen_broadleaf_in = evergreen_broadleaf.reshape(nx * ny)
    deciduous_needleleaf_in = deciduous_needleleaf.reshape(nx * ny)
    deciduous_broadleaf_in = deciduous_broadleaf.reshape(nx * ny)
    mixed_forest_in = mixed_forest.reshape(nx * ny)
    woodland_in = woodland.reshape(nx * ny)
    wooded_grassland_in = wooded_grassland.reshape(nx * ny)
    closed_shurbland_in = closed_shurbland.reshape(nx * ny)
    open_shrubland_in = open_shrubland.reshape(nx * ny)
    grassland_in = grassland.reshape(nx * ny)
    barren_in = barren.reshape(nx * ny)
    urban_in = urban.reshape(nx * ny)

    x_maps = jnp.stack(
        [
            awCap_in,
            wiltingp_in,
            Ws_init_in,
            clayfrac_in,
            sandfrac_in,
            siltfrac_in,
            rootDepth_in,
            lats_in,
            elev_std_in,
            corn_in,
            cotton_in,
            rice_in,
            sorghum_in,
            soybeans_in,
            wheat_in,
            cropland_other_in,
            evergreen_needleleaf_in,
            evergreen_broadleaf_in,
            deciduous_needleleaf_in,
            deciduous_broadleaf_in,
            mixed_forest_in,
            woodland_in,
            wooded_grassland_in,
            closed_shurbland_in,
            open_shrubland_in,
            grassland_in,
            barren_in,
            urban_in,
        ],
        axis=1,
    )
    nan_inds_maps = jnp.isnan(x_maps).any(axis=1)

    # Remove NaNs if desired
    if remove_nans:
        nan_inds = (
            nan_inds_obs
            + nan_inds_forcing_nt
            + nan_inds_forcing_nyrs
            + nan_inds_maps
        )
        ys = ys[~nan_inds]
        x_forcing_nt = x_forcing_nt[~nan_inds]
        x_forcing_nyrs = x_forcing_nyrs[~nan_inds]
        x_maps = x_maps[~nan_inds]

    # Return
    return ys, x_forcing_nt, x_forcing_nyrs, x_maps


#########################
# OLD
#####################
# if obs_name == "VICx":
#     awCap_frac_in = awCap_frac.reshape(nx * ny)
#     wiltingp_frac_in = wiltingp_frac.reshape(nx * ny)
#     sand_in = sand.reshape(nx * ny)
#     loamy_sand_in = loamy_sand.reshape(nx * ny)
#     sandy_loam_in = sandy_loam.reshape(nx * ny)
#     silt_loam_in = silt_loam.reshape(nx * ny)
#     silt_in = silt.reshape(nx * ny)
#     loam_in = loam.reshape(nx * ny)
#     sandy_clay_loam_in = sandy_clay_loam.reshape(nx * ny)
#     silty_clay_loam_in = silty_clay_loam.reshape(nx * ny)
#     clay_loam_in = clay_loam.reshape(nx * ny)
#     sandy_clay_in = sandy_clay.reshape(nx * ny)
#     silty_clay_in = silty_clay.reshape(nx * ny)
#     clay_in = clay.reshape(nx * ny)
# else:
# if obs_name == "VICx":
#         awCap_frac = npz["awCap_frac"]
#         wiltingp_frac = npz["wiltingp_frac"]
#         sand = npz["sand"]
#         loamy_sand = npz["loamy_sand"]
#         sandy_loam = npz["sandy_loam"]
#         silt_loam = npz["silt_loam"]
#         silt = npz["silt"]
#         loam = npz["loam"]
#         sandy_clay_loam = npz["sandy_clay_loam"]
#         silty_clay_loam = npz["silty_clay_loam"]
#         clay_loam = npz["clay_loam"]
#         sandy_clay = npz["sandy_clay"]
#         silty_clay = npz["silty_clay"]
#         clay = npz["clay"]
#     else:
# all_other_in = all_other.reshape(nx * ny)
# if obs_name == "VICx":
#     x_maps = jnp.stack(
#         [
#             awCap_frac_in,
#             wiltingp_frac_in,
#             sand_in,
#             loamy_sand_in,
#             sandy_loam_in,
#             silt_loam_in,
#             silt_in,
#             loam_in,
#             sandy_clay_loam_in,
#             silty_clay_loam_in,
#             clay_loam_in,
#             sandy_clay_in,
#             silty_clay_in,
#             clay_in,
#             Ws_init_in,
#             clayfrac_in,
#             sandfrac_in,
#             siltfrac_in,
#             rootDepth_in,
#             lats_in,
#             elev_std_in,
#             corn_in,
#             cotton_in,
#             rice_in,
#             sorghum_in,
#             soybeans_in,
#             wheat_in,
#         ],
#         axis=1,
#     )
# else:
