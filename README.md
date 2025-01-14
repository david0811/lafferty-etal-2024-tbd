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
Climate change is altering the frequency and intensity of physical hazards worldwide, magnifying the risks to many systems that provide critical services to humanity such as agriculture and water resources. In a hydrologic context, quantifying these risks is challenging due to large uncertainties in modeling the future climate change and the associated hydrologic response. Here, we examine the combined role of climate and hydrologic uncertainties in shaping projections of future soil moisture. We focus on the central and eastern United States given its significance to global maize and soybean production. We encode a conceptual model of soil moisture in a differentiable programming framework to facilitate faster runtimes and a more efficient calibration. We characterize and analyze uncertainty in model parameters by calibrating against different targets, including satellite- and reanalysis-derived products such as SMAP and NLDAS-2, as well as using several error metric functions. We then convolve the resulting parameter ensemble with a set of downscaled and bias-corrected climate projections to produce a large ensemble ($\sim$2200 members) of daily soil moisture simulations at $\sim$12.5 km resolution over the domain. We conduct sensitivity analyses on a variety of soil moisture metrics, some targeting long-term trends and others short-lived extremes, to measure the relative influence of climate and hydrologic-parameter uncertainties across space and time. Across most of the region, we find weak but statistically significant drying trends in mean and extreme soil moisture metrics, with both climate and parametric factors contributing non-negligible uncertainty. Our sensitivity analyses also reveal a distinct spatial pattern, where hydrologic uncertainty is more important (relative to climate) for projecting dry extremes in dry areas, and wet extremes in wet areas. Our results highlight the importance of considering combined hydrologic and climate uncertainties when constructing projections of decision-relevant hydroclimatic outcomes.

## Journal reference
_Coming soon_

## Code reference
_your software reference here_

## Data reference

### Input data
| Dataset | Link | DOI | Description |
|---------|------|-----|-------------|
| NLDAS-2 forcing inputs | https://disc.gsfc.nasa.gov/datasets/NLDAS_FORA0125_H_002/summary | https://doi.org/10.5067/6J5LHHOHZHN4 | - |
| NLDAS-2 model outputs | - | - | - | 
| SMAP forcing inputs | - | - | - |
| SMAP model outputs | - | - | - | 
| SMAP model outputs | - | - | - | 
| NLDAS-2 auxiliary data | - | - | - | 
| USDA Cropland Data Layer | - | - | - |
| GLDAS Leaf Area Index | - | - | - | 

### Output data
_coming soon_

## Contributing modeling software
| Model | Version | Repository Link | DOI |
|-------|---------|-----------------|-----|
| pyWBM | - | https://github.com/david0811/pyWBM | - |

## Reproduce my experiment
_instructions here_
