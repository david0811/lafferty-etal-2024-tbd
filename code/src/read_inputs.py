# Reads inputs for WBM calibration from npz files and prepares them for vmap
import jax.numpy as jnp
import numpy as np
import xarray as xr
from utils.global_paths import project_data_path


def read_projection_inputs(subset_name, obs_name, projection_id, remove_nans):
    # For initial conditions
    Ws_init = np.load(f"{project_data_path}/WBM/calibration/{subset_name}/{obs_name}/{obs_name}_validation.npy")
    Ws_init = Ws_init[:, :, 0]

    # Obs inputs
    npz = np.load(
        f"{project_data_path}/WBM/calibration/{subset_name}/{obs_name}/inputs.npz"
    )

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

    # Memory management
    del npz

    ######################
    # Read and extract projection inputs
    ds = xr.open_dataset(f"{project_data_path}/projections/{subset_name}/forcing/{projection_id}.zarr",
                        engine='zarr')
    ds = ds.convert_calendar(calendar="noleap", dim="time")
    ds = ds.sel(time=slice("2016-01-01", "2100-12-31"))  # initial conditions for 2016-01-01

    tas = np.transpose(ds["tas"].to_numpy(), (2, 1, 0))
    prcp = np.transpose(ds["pr"].to_numpy(), (2, 1, 0))

    # Memory management
    del ds

    ##########################
    # Prepare inputs for vmap:
    # spatial dimensions need to be collapsed and first
    # NaN gridpoints need to be removed
    nx = tas.shape[0]
    ny = tas.shape[1]
    nt = tas.shape[2]

    assert nt % 365 == 0

    ## Forcing: all days
    tas = tas.reshape(nx * ny, nt)
    prcp = prcp.reshape(nx * ny, nt)

    x_forcing_nt = jnp.stack([tas, prcp], axis=1)
    nan_inds_forcing_nt = jnp.isnan(x_forcing_nt).any(axis=(1, 2))

    ## Forcing: yearly
    lai = lai.reshape(nx * ny, 365)
    x_forcing_nyrs = lai
    nan_inds_forcing_nyrs = jnp.isnan(x_forcing_nyrs).any(axis=1)

    ## Maps
    awCap = awCap.reshape(nx * ny)
    wiltingp = wiltingp.reshape(nx * ny)

    Ws_init = Ws_init.reshape(nx * ny)

    clayfrac = clayfrac.reshape(nx * ny)
    sandfrac = sandfrac.reshape(nx * ny)
    siltfrac = siltfrac.reshape(nx * ny)

    rootDepth = rootDepth.reshape(nx * ny)

    lats = np.tile(lats, nx)
    elev_std = elev_std.reshape(nx * ny)

    corn = corn.reshape(nx * ny)
    cotton = cotton.reshape(nx * ny)
    rice = rice.reshape(nx * ny)
    sorghum = sorghum.reshape(nx * ny)
    soybeans = soybeans.reshape(nx * ny)
    wheat = wheat.reshape(nx * ny)

    cropland_other = cropland_other.reshape(nx * ny)
    evergreen_needleleaf = evergreen_needleleaf.reshape(nx * ny)
    evergreen_broadleaf = evergreen_broadleaf.reshape(nx * ny)
    deciduous_needleleaf = deciduous_needleleaf.reshape(nx * ny)
    deciduous_broadleaf = deciduous_broadleaf.reshape(nx * ny)
    mixed_forest = mixed_forest.reshape(nx * ny)
    woodland = woodland.reshape(nx * ny)
    wooded_grassland = wooded_grassland.reshape(nx * ny)
    closed_shurbland = closed_shurbland.reshape(nx * ny)
    open_shrubland = open_shrubland.reshape(nx * ny)
    grassland = grassland.reshape(nx * ny)
    barren = barren.reshape(nx * ny)
    urban = urban.reshape(nx * ny)

    x_maps = jnp.stack(
        [
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
            cropland_other,
            evergreen_needleleaf,
            evergreen_broadleaf,
            deciduous_needleleaf,
            deciduous_broadleaf,
            mixed_forest,
            woodland,
            wooded_grassland,
            closed_shurbland,
            open_shrubland,
            grassland,
            barren,
            urban,
        ],
        axis=1,
    )
    nan_inds_maps = jnp.isnan(x_maps).any(axis=1)

    # Remove NaNs if desired
    if remove_nans:
        nan_inds = nan_inds_forcing_nt + nan_inds_forcing_nyrs + nan_inds_maps
        x_forcing_nt = x_forcing_nt[~nan_inds]
        x_forcing_nyrs = x_forcing_nyrs[~nan_inds]
        x_maps = x_maps[~nan_inds]
        valid_inds = ~nan_inds

    # Return
    return x_forcing_nt, x_forcing_nyrs, x_maps, valid_inds


def read_hindcast_inputs(subset_name, obs_name, remove_nans):
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

    # Memory management
    del npz

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
    tas = tas.reshape(nx * ny, nt)
    prcp = prcp.reshape(nx * ny, nt)

    x_forcing_nt = jnp.stack([tas, prcp], axis=1)
    nan_inds_forcing_nt = jnp.isnan(x_forcing_nt).any(axis=(1, 2))

    ## Forcing: yearly
    lai = lai.reshape(nx * ny, 365)
    x_forcing_nyrs = lai
    nan_inds_forcing_nyrs = jnp.isnan(x_forcing_nyrs).any(axis=1)

    ## Maps
    awCap = awCap.reshape(nx * ny)
    wiltingp = wiltingp.reshape(nx * ny)

    Ws_init = Ws_init.reshape(nx * ny)

    clayfrac = clayfrac.reshape(nx * ny)
    sandfrac = sandfrac.reshape(nx * ny)
    siltfrac = siltfrac.reshape(nx * ny)

    rootDepth = rootDepth.reshape(nx * ny)

    lats = np.tile(lats, nx)
    elev_std = elev_std.reshape(nx * ny)

    corn = corn.reshape(nx * ny)
    cotton = cotton.reshape(nx * ny)
    rice = rice.reshape(nx * ny)
    sorghum = sorghum.reshape(nx * ny)
    soybeans = soybeans.reshape(nx * ny)
    wheat = wheat.reshape(nx * ny)

    cropland_other = cropland_other.reshape(nx * ny)
    evergreen_needleleaf = evergreen_needleleaf.reshape(nx * ny)
    evergreen_broadleaf = evergreen_broadleaf.reshape(nx * ny)
    deciduous_needleleaf = deciduous_needleleaf.reshape(nx * ny)
    deciduous_broadleaf = deciduous_broadleaf.reshape(nx * ny)
    mixed_forest = mixed_forest.reshape(nx * ny)
    woodland = woodland.reshape(nx * ny)
    wooded_grassland = wooded_grassland.reshape(nx * ny)
    closed_shurbland = closed_shurbland.reshape(nx * ny)
    open_shrubland = open_shrubland.reshape(nx * ny)
    grassland = grassland.reshape(nx * ny)
    barren = barren.reshape(nx * ny)
    urban = urban.reshape(nx * ny)

    x_maps = jnp.stack(
        [
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
            cropland_other,
            evergreen_needleleaf,
            evergreen_broadleaf,
            deciduous_needleleaf,
            deciduous_broadleaf,
            mixed_forest,
            woodland,
            wooded_grassland,
            closed_shurbland,
            open_shrubland,
            grassland,
            barren,
            urban,
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