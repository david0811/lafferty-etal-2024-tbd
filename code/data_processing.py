import os
from glob import glob

import numpy as np
import regionmask
import xarray as xr
from global_paths import nldas_path, project_data_path, smap_path


# Subsetting function
def _subset_states(ds, list_of_states):
    """
    Subsets a netCDF file to a list of states using regionmask
    """
    if list_of_states is None:
        return ds
    # Subset
    subset_index = (
        regionmask.defined_regions.natural_earth_v5_0_0.us_states_50.map_keys(
            list_of_states
        )
    )
    subset_mask = (
        regionmask.defined_regions.natural_earth_v5_0_0.us_states_50.mask(ds)
    )
    ds_subset = ds.where(subset_mask.isin(subset_index), drop=True)
    # Return
    return ds_subset


# SMAP processing
def process_smap(subset_name, list_of_states):
    """
    Grabs SMAP outputs and stores as one netCDF file with after subsetting to list_of_states.
    """
    if not os.path.isfile(
        f"{project_data_path}/WBM/calibration/{subset_name}/SMAP/SMAP_validation.nc"
    ):
        # Read all
        files = glob(
            f"{smap_path}/processed_nldas_grid/SMAP_L4_SM_gph_all_nldas_*.nc"
        )
        ds_smap = xr.concat(
            [
                _subset_states(
                    xr.open_dataset(file)["sm_rootzone"], list_of_states
                )
                for file in files
            ],
            dim="time",
        )

        # 365 day calendar
        ds_smap = ds_smap.convert_calendar(calendar="noleap", dim="time")

        # Merge and store (and change units to kg/m3)
        ds_out = xr.Dataset({"soilMoist": 1000 * ds_smap})
        ds_out.attrs["units"] = "kg/m3"
        ds_out.to_netcdf(
            f"{project_data_path}/WBM/calibration/{subset_name}/SMAP/SMAP_validation.nc"
        )

        # Also store numpy array for quicker evaluations
        npy_out = np.transpose(ds_out["soilMoist"].to_numpy(), (2, 1, 0))
        np.save(
            f"{project_data_path}/WBM/calibration/{subset_name}/SMAP/SMAP_validation.npy",
            npy_out,
        )
    else:
        print("SMAP already processed")


# NLDAS processing
def process_nldas(subset_name, list_of_states):
    """
    Grabs NDLAS outputs and stores as one netCDF file with after subsetting to list_of_states.
    """
    nldas_dict = {"VIC": "SOILM0_100cm", "NOAH": "SOILM", "MOSAIC": "SOILM"}

    # Loop through each
    for model, var_id in nldas_dict.items():
        if not os.path.isfile(
            f"{project_data_path}/WBM/calibration/{subset_name}/{model}/{model}_validation.nc"
        ):
            # Read all
            files = glob(f"{nldas_path}/{model}/daily/*.nc")
            ds_nldas = xr.concat(
                [
                    _subset_states(
                        xr.open_dataset(file)[var_id], list_of_states
                    )
                    for file in files
                ],
                dim="time",
            )

            # 365 day calendar
            ds_nldas = ds_nldas.convert_calendar(calendar="noleap", dim="time")

            # Select correct depth
            if model in ["MOSAIC", "NOAH"]:
                ds_nldas = ds_nldas.isel(depth=1)
            else:
                ds_nldas = ds_nldas.isel(depth=0)

            # Merge and store
            ds_out = xr.Dataset({"soilMoist": ds_nldas})
            ds_out.attrs["units"] = "kg/m3"
            ds_out.to_netcdf(
                f"{project_data_path}/WBM/calibration/{subset_name}/{model}/{model}_validation.nc"
            )

            # Also store numpy array for quicker evaluations
            npy_out = np.transpose(ds_out["soilMoist"].to_numpy(), (2, 1, 0))
            np.save(
                f"{project_data_path}/WBM/calibration/{subset_name}/{model}/{model}_validation.npy",
                npy_out,
            )


# Forcing processing
def process_forcing(subset_name, list_of_states):
    """
    Grabs all forcing inputs are stores as single numpy npz file.
    SMAP and NLDAS handled separately since meteo forcing is different.
    """
    for obs_name in ["MOSAIC", "NOAH", "VIC", "SMAP"]:
        if not os.path.isfile(
            f"{project_data_path}/WBM/calibration/{subset_name}/{obs_name}/inputs.npz"
        ):
            # Dict to save
            save_dict = {}
            ######### Climate drivers
            if obs_name == "SMAP":
                files = glob(
                    f"{smap_path}/processed_nldas_grid/SMAP_L4_SM_gph_all_nldas_*.nc"
                )
                ds_forcing = xr.concat(
                    [
                        _subset_states(xr.open_dataset(file), list_of_states)
                        for file in files
                    ],
                    dim="time",
                )
            else:
                files = glob(
                    f"{nldas_path}/forcing/daily/NLDAS_FORA0125_H.A*.nc"
                )
                ds_forcing = xr.concat(
                    [
                        _subset_states(xr.open_dataset(file), list_of_states)
                        for file in files
                    ],
                    dim="time",
                )

            # 365 day calendar
            ds_forcing = ds_forcing.convert_calendar(
                calendar="noleap", dim="time"
            )

            # Numpy arrays in correct order (lon, lat, time)
            if obs_name == "SMAP":
                tas = np.transpose(
                    ds_forcing["temp_lowatmmodlay"].to_numpy() - 273.15,
                    (2, 1, 0),
                )
                prcp = np.transpose(
                    ds_forcing["precipitation_total_surface_flux"].to_numpy()
                    * 86400,
                    (2, 1, 0),
                )
            else:
                tas = np.transpose(
                    ds_forcing["TMP"].to_numpy() - 273.15, (2, 1, 0)
                )
                prcp = np.transpose(ds_forcing["APCP"].to_numpy(), (2, 1, 0))

            save_dict["tas"] = tas
            save_dict["prcp"] = prcp

            ############ Geophysical inputs
            # Soil types for VIC
            if obs_name == "VIC":
                ds_sand = _subset_states(
                    xr.open_dataset(
                        f"{project_data_path}/WBM/geo_inputs/NLDAS_sand.nc"
                    ),
                    list_of_states,
                )
                sand = np.transpose(ds_sand["sand"].to_numpy())
                save_dict["sand"] = sand

                ds_loamy_sand = _subset_states(
                    xr.open_dataset(
                        f"{project_data_path}/WBM/geo_inputs/NLDAS_loamy_sand.nc"
                    ),
                    list_of_states,
                )
                loamy_sand = np.transpose(
                    ds_loamy_sand["loamy_sand"].to_numpy()
                )
                save_dict["loamy_sand"] = loamy_sand

                ds_sandy_loam = _subset_states(
                    xr.open_dataset(
                        f"{project_data_path}/WBM/geo_inputs/NLDAS_sandy_loam.nc"
                    ),
                    list_of_states,
                )
                sandy_loam = np.transpose(
                    ds_sandy_loam["sandy_loam"].to_numpy()
                )
                save_dict["sandy_loam"] = sandy_loam

                ds_silt_loam = _subset_states(
                    xr.open_dataset(
                        f"{project_data_path}/WBM/geo_inputs/NLDAS_silt_loam.nc"
                    ),
                    list_of_states,
                )
                silt_loam = np.transpose(ds_silt_loam["silt_loam"].to_numpy())
                save_dict["silt_loam"] = silt_loam

                ds_silt = _subset_states(
                    xr.open_dataset(
                        f"{project_data_path}/WBM/geo_inputs/NLDAS_silt.nc"
                    ),
                    list_of_states,
                )
                silt = np.transpose(ds_silt["silt"].to_numpy())
                save_dict["silt"] = silt

                ds_loam = _subset_states(
                    xr.open_dataset(
                        f"{project_data_path}/WBM/geo_inputs/NLDAS_loam.nc"
                    ),
                    list_of_states,
                )
                loam = np.transpose(ds_loam["loam"].to_numpy())
                save_dict["loam"] = loam

                ds_sandy_clay_loam = _subset_states(
                    xr.open_dataset(
                        f"{project_data_path}/WBM/geo_inputs/NLDAS_sandy_clay_loam.nc"
                    ),
                    list_of_states,
                )
                sandy_clay_loam = np.transpose(
                    ds_sandy_clay_loam["sandy_clay_loam"].to_numpy()
                )
                save_dict["sandy_clay_loam"] = sandy_clay_loam

                ds_silty_clay_loam = _subset_states(
                    xr.open_dataset(
                        f"{project_data_path}/WBM/geo_inputs/NLDAS_silty_clay_loam.nc"
                    ),
                    list_of_states,
                )
                silty_clay_loam = np.transpose(
                    ds_silty_clay_loam["silty_clay_loam"].to_numpy()
                )
                save_dict["silty_clay_loam"] = silty_clay_loam

                ds_clay_loam = _subset_states(
                    xr.open_dataset(
                        f"{project_data_path}/WBM/geo_inputs/NLDAS_clay_loam.nc"
                    ),
                    list_of_states,
                )
                clay_loam = np.transpose(ds_clay_loam["clay_loam"].to_numpy())
                save_dict["clay_loam"] = clay_loam

                ds_sandy_clay = _subset_states(
                    xr.open_dataset(
                        f"{project_data_path}/WBM/geo_inputs/NLDAS_sandy_clay.nc"
                    ),
                    list_of_states,
                )
                sandy_clay = np.transpose(
                    ds_sandy_clay["sandy_clay"].to_numpy()
                )
                save_dict["sandy_clay"] = sandy_clay

                ds_silty_clay = _subset_states(
                    xr.open_dataset(
                        f"{project_data_path}/WBM/geo_inputs/NLDAS_silty_clay.nc"
                    ),
                    list_of_states,
                )
                silty_clay = np.transpose(
                    ds_silty_clay["silty_clay"].to_numpy()
                )
                save_dict["silty_clay"] = silty_clay

                ds_clay = _subset_states(
                    xr.open_dataset(
                        f"{project_data_path}/WBM/geo_inputs/NLDAS_clay.nc"
                    ),
                    list_of_states,
                )
                clay = np.transpose(ds_clay["clay"].to_numpy())
                save_dict["clay"] = clay

                # Root Depth
                ds_rootDepth = _subset_states(
                    xr.open_dataset(
                        f"{project_data_path}/WBM/geo_inputs/VIC_rootDepth.nc"
                    ),
                    list_of_states,
                )
                rootDepth = np.transpose(ds_rootDepth["rootDepth"].to_numpy())
                save_dict["rootDepth"] = rootDepth

            else:
                # Wilting point and awCap
                ds_awCap = _subset_states(
                    xr.open_dataset(
                        f"{project_data_path}/WBM/geo_inputs/{obs_name}_awCap.nc"
                    ),
                    list_of_states,
                )
                awCap = np.transpose(ds_awCap["awCap"].to_numpy())
                save_dict["awCap"] = awCap

                ds_wiltingp = _subset_states(
                    xr.open_dataset(
                        f"{project_data_path}/WBM/geo_inputs/{obs_name}_wiltingp.nc"
                    ),
                    list_of_states,
                )
                wiltingp = np.transpose(ds_wiltingp["wiltingp"].to_numpy())
                save_dict["wiltingp"] = wiltingp

                rootDepth = np.ones(wiltingp.shape)
                save_dict["rootDepth"] = rootDepth

            # Content fractions
            ds_clayfrac = _subset_states(
                xr.open_dataset(
                    f"{project_data_path}/WBM/geo_inputs/clayfrac_NLDASgrid.nc"
                ),
                list_of_states,
            )
            ds_sandfrac = _subset_states(
                xr.open_dataset(
                    f"{project_data_path}/WBM/geo_inputs/sandfrac_NLDASgrid.nc"
                ),
                list_of_states,
            )
            ds_siltfrac = _subset_states(
                xr.open_dataset(
                    f"{project_data_path}/WBM/geo_inputs/siltfrac_NLDASgrid.nc"
                ),
                list_of_states,
            )

            clayfrac = np.transpose(
                ds_clayfrac["clayfrac"].to_numpy() / 100
            )  # percentage -> fraction
            sandfrac = np.transpose(
                ds_sandfrac["sandfrac"].to_numpy() / 100
            )  # percentage -> fraction
            siltfrac = np.transpose(
                ds_siltfrac["siltfrac"].to_numpy() / 100
            )  # percentage -> fraction

            save_dict["clayfrac"] = clayfrac
            save_dict["sandfrac"] = sandfrac
            save_dict["siltfrac"] = siltfrac

            # Initial conditions
            ds_init = _subset_states(
                xr.open_dataset(
                    f"{project_data_path}/WBM/calibration/{subset_name}/{obs_name}/{obs_name}_validation.nc"
                ),
                list_of_states,
            ).isel(time=0)
            soilMoist_init = np.transpose(ds_init["soilMoist"].to_numpy())
            save_dict["soilMoist_init"] = soilMoist_init

            # LAI
            ds_lai = _subset_states(
                xr.open_dataset(
                    f"{project_data_path}/WBM/geo_inputs/LAI_GLDAS_clima_NLDASgrid.nc"
                ),
                list_of_states,
            )
            lai = np.transpose(ds_lai["LAI"].to_numpy(), (2, 1, 0))
            save_dict["lai"] = lai

            # Land properties
            ds_land = _subset_states(
                xr.open_dataset(
                    f"{project_data_path}/WBM/geo_inputs/CDL-NLDAS_landtypes_NLDASgrid.nc"
                ),
                list_of_states,
            )
            corn = np.transpose(ds_land["corn"].to_numpy())
            save_dict["corn"] = corn

            cotton = np.transpose(ds_land["cotton"].to_numpy())
            save_dict["cotton"] = cotton

            rice = np.transpose(ds_land["rice"].to_numpy())
            save_dict["rice"] = rice

            sorghum = np.transpose(ds_land["sorghum"].to_numpy())
            save_dict["sorghum"] = sorghum

            soybeans = np.transpose(ds_land["soybeans"].to_numpy())
            save_dict["soybeans"] = soybeans

            durum_wheat = np.transpose(ds_land["durum_wheat"].to_numpy())
            save_dict["durum_wheat"] = durum_wheat

            spring_wheat = np.transpose(ds_land["spring_wheat"].to_numpy())
            save_dict["spring_wheat"] = spring_wheat

            cropland_other = np.transpose(ds_land["cropland_other"].to_numpy())
            save_dict["cropland_other"] = cropland_other

            water = np.transpose(ds_land["water"].to_numpy())
            save_dict["water"] = water

            evergreen_needleleaf = np.transpose(
                ds_land["evergreen_needleleaf"].to_numpy()
            )
            save_dict["evergreen_needleleaf"] = evergreen_needleleaf

            evergreen_broadleaf = np.transpose(
                ds_land["evergreen_broadleaf"].to_numpy()
            )
            save_dict["evergreen_broadleaf"] = evergreen_broadleaf

            deciduous_needleleaf = np.transpose(
                ds_land["deciduous_needleleaf"].to_numpy()
            )
            save_dict["deciduous_needleleaf"] = deciduous_needleleaf

            deciduous_broadleaf = np.transpose(
                ds_land["deciduous_broadleaf"].to_numpy()
            )
            save_dict["deciduous_broadleaf"] = deciduous_broadleaf

            mixed_forest = np.transpose(ds_land["mixed_forest"].to_numpy())
            save_dict["mixed_forest"] = mixed_forest

            woodland = np.transpose(ds_land["woodland"].to_numpy())
            save_dict["woodland"] = woodland

            wooded_grassland = np.transpose(
                ds_land["wooded_grassland"].to_numpy()
            )
            save_dict["wooded_grassland"] = wooded_grassland

            closed_shurbland = np.transpose(
                ds_land["closed_shurbland"].to_numpy()
            )
            save_dict["closed_shurbland"] = closed_shurbland

            open_shrubland = np.transpose(ds_land["open_shrubland"].to_numpy())
            save_dict["open_shrubland"] = open_shrubland
            grassland = np.transpose(ds_land["grassland"].to_numpy())
            save_dict["grassland"] = grassland
            barren = np.transpose(ds_land["barren"].to_numpy())
            save_dict["barren"] = barren
            urban = np.transpose(ds_land["urban"].to_numpy())
            save_dict["urban"] = urban

            # Elevation properties
            ds_elev = _subset_states(
                xr.open_dataset(
                    f"{project_data_path}/WBM/geo_inputs/NLDAS_elev_STD_NLDASgrid.nc"
                ),
                list_of_states,
            )
            elev_std = np.transpose(ds_elev["NLDAS_elev_std"].to_numpy())
            save_dict["elev_std"] = elev_std

            # Lat, Lon
            lats = ds_lai.lat.to_numpy()
            save_dict["lats"] = lats

            # lons = ds_lai.lon.to_numpy()

            # Store numpy for easy access
            np.savez(
                f"{project_data_path}/WBM/calibration/{subset_name}/{obs_name}/inputs.npz",
                **save_dict,
            )
