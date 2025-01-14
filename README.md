# lafferty-etal-2025-tbd

**Combined climate and hydrologic uncertainties shape projections of future soil moisture in the central and eastern United States**

*David C. Lafferty<sup>1\*a</sup>, Danielle S. Grogan<sup>2</sup>, Shan Zuidema<sup>2</sup>, Iman Haqiqi<sup>3</sup>, Atieh Alipour<sup>4b</sup>, Ryan L. Sriver<sup>1</sup>, Klaus Keller <sup>4</sup>*

<sup>1 </sup>Department of Climate, Meteorology, \& Atmospheric Sciences, University of Illinois Urbana-Champaign\
<sup>2 </sup>Institute for the Study of Earth, Oceans, and Space, University of New Hampshire\
<sup>3 </sup>Department of Agricultural Economics, Purdue University\
<sup>3 </sup>Thayer School of Engineering, Dartmouth College

\* corresponding author:  `dcl257@cornell.edu`\
<sup>a </sup> Current address: Department of Biological & Environmental Engineering, Cornell University\
<sup>b </sup> Current address: National Oceanic & Atmospheric Administration

## Abstract
Climate change is altering the frequency and intensity of physical hazards worldwide, magnifying the risks to critical systems such as agriculture and water resources. Designing adaptive measures to mitigate these risks requires accounting for diverse climate futures, but this is challenging due to the large uncertainties in modeling both future climate change and the associated sectoral impacts. Here, we address this challenge in a hydrologic context by examining the combined role of climate and hydrologic uncertainties in shaping projections of future soil moisture. We focus on the eastern United States given its importance in global maize and soybean production. By encoding a simple conceptual water balance model in a differentiable programming framework, we facilitate fast runtimes and an efficient calibration procedure that enable an improved uncertainty analysis. We characterize and analyze uncertainty in model parameters by calibrating against different target datasets, including satellite- and reanalysis-derived products such as SMAP and NLDAS-2, as well as using several loss functions. We then convolve the resulting parameter ensemble with a set of downscaled and bias-corrected climate projections to produce a large ensemble (2340 members) of daily soil moisture simulations at approximately 12.5 km resolution over the domain. For annual average soil moisture, we find that most ensemble members project drying trends across most of the region, although some simulate a wetting of soils throughout this century. Our ensemble shows an increase in the frequency and intensity of dry extremes while there is less agreement for changes to wet extremes. We conduct sensitivity analyses on a variety of soil moisture metrics to measure the relative influence of climate and hydrologic uncertainties across space and time. Both climate and hydrologic factors contribute non-negligible uncertainty to long-term trends, but hydrologic uncertainty is dominant for projecting changes in the extremes. Our results underscore the need to account for combined hydrologic and climate uncertainties when developing actionable hydroclimatic projections.

## Journal reference
_Coming soon_

## Code reference
_your software reference here_

## Data reference

### Input data
| Dataset | Link | DOI | Notes |
|---------|------|-----|-------|
| NLDAS-2 forcing inputs | https://disc.gsfc.nasa.gov/datasets/NLDAS_FORA0125_H_002/summary | https://doi.org/10.5067/6J5LHHOHZHN4 | We use `TMP` and `APCP` as weather inputs. |
| NLDAS-2 model outputs | VIC: https://disc.gsfc.nasa.gov/datasets/NLDAS_VIC0125_H_002/summary <br> Noah: https://disc.gsfc.nasa.gov/datasets/NLDAS_NOAH0125_H_002/summary <br> Mosaic: https://disc.gsfc.nasa.gov/datasets/NLDAS_NOAH0125_H_002/summary | VIC: https://doi.org/10.5067/ELBDAPAKNGJ9 <br> Noah: https://doi.org/10.5067/EN4MBWTCENE5 <br> Mosaic: https://doi.org/10.5067/47Z13FNQODKV | We use `SOILM0_100cm` from VIC, and `SOILM` from Noah and Mosaic. | 
| NLDAS-2 auxiliary data | Elevation: https://ldas.gsfc.nasa.gov/nldas/elevation <br> Vegetation class: https://ldas.gsfc.nasa.gov/nldas/vegetation-class | - | - |
| SMAP Level 4 (Version 7) | https://nsidc.org/data/spl4smlm/versions/7 | https://doi.org/10.5067/KN96XNPZM4EG | We calibrate against `sm_rootzone` using `temp_lowatmmodlay` and `precipitation_total_surface_flux` as weather inputs. |   
| USDA Cropland Data Layer | https://www.nass.usda.gov/Research_and_Science/Cropland/Release/index.php | - | Accessed February 2022 |
| GLDAS Leaf Area Index | https://ldas.gsfc.nasa.gov/gldas/lai-greenness | - | Accessed May 2023 | 
| UC Davis soil properties | https://casoilresource.lawr.ucdavis.edu/soil-properties/download.php | - | Accessed November 2023. We use `Sand`, `Silt`, `Clay` (percent by weight). | 

### Output data
_Coming soon_

## Contributing modeling software
| Model | Version | Repository Link | DOI |
|-------|---------|-----------------|-----|
| pyWBM | - | https://github.com/david0811/pyWBM | - |

## Reproduce my experiment
Project dependencies are specified in `pyproject.toml`. You can clone this directory and install via pip by running `pip install -e .` from the root directory. You'll also need to download all of the input data sets and update the paths appropriately in `utils/global_paths.py`.

The following scripts can then be used to reproduce the experiment:

| Script | Description |
|--------|-------------|
| 00a_data_processing.ipynb | Processes all input data, including resampling hourly to daily resolution and re-gridding. |
| 00b_data_preparation.ipynb | Prepares all input data as .npz files for the calibration script. |
| 01_initial_sa.ipynb | Runs the initial Sobol' sensitivity analysis for select locations. |
| 02a_calibration.ipynb | Runs the calibration. |
| 02b_calibration_plots.ipynb | Plots the calibration results. |
| 03_projections.ipynb | Constructs the climate change projections. |
| 04a_forward_sa.ipynb | Runs the forward sensitivity analysis. |
| 04b_forward_sa_plots.ipynb | Plots the forward sensitivity analysis results. |
| 99_misc_plots.ipynb | Miscellaneous plots. | 
| 99_format_outputs.ipynb | Formats soil moisture outputs and forcing inputs for public sharing. |