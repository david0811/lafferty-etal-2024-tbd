import jax.numpy as jnp 

# Constants
Ts = -1. # Snowfall threshold
Tm = 1.  # Snowmelt threshold
Wi_init = 0. # Initial canopy storage (could spinup to find this but miniscule difference)
Sp_init = 0. # Initial snowpack storage (could spinup to find this but miniscule difference)

constants = jnp.array([Ts, Tm, Wi_init, Sp_init])

# Parameters
awCap_scalar = jnp.log(1.)
wiltingp_scalar = jnp.log(1.)

alpha_claycoef = jnp.log(0.5)
alpha_sandcoef = jnp.log(0.5)
alpha_siltcoef = jnp.log(0.5)

betaHBV_claycoef = jnp.log(0.5)
betaHBV_sandcoef = jnp.log(0.5)
betaHBV_siltcoef = jnp.log(0.5)
betaHBV_elevcoef = jnp.log(0.5)

GS_start_corn = jnp.log(91) # April 1st
GS_length_corn = jnp.log(274 - 91) # April 1st -> October 1st
L_ini_corn = jnp.log(0.17)
L_dev_corn = jnp.log(0.28)
L_mid_corn = jnp.log(0.33)
Kc_ini_corn = jnp.log(0.3)
Kc_mid_corn = jnp.log(1.2)
Kc_end_corn = jnp.log(0.4)
K_min_corn = jnp.log(0.3)
K_max_corn = jnp.log(1.2)

GS_start_cotton = jnp.log(91) # April 1st
GS_length_cotton = jnp.log(274 - 91) # April 1st -> October 1st
L_ini_cotton = jnp.log(0.17)
L_dev_cotton = jnp.log(0.33)
L_mid_cotton = jnp.log(0.25)
Kc_ini_cotton = jnp.log(0.35)
Kc_mid_cotton = jnp.log(1.18)
Kc_end_cotton = jnp.log(0.6)
K_min_cotton = jnp.log(0.35)
K_max_cotton = jnp.log(1.18)

# Rice growing season: https://www.ers.usda.gov/topics/crops/rice/rice-sector-at-a-glance/
GS_start_rice = jnp.log(91) # April 1st
GS_length_rice = jnp.log(244 - 91) # April 1st -> September 1st
L_ini_rice = jnp.log(0.17)
L_dev_rice = jnp.log(0.28)
L_mid_rice = jnp.log(0.44)
Kc_ini_rice = jnp.log(1.05)
Kc_mid_rice = jnp.log(1.2)
Kc_end_rice = jnp.log(0.75)
K_min_rice = jnp.log(0.75)
K_max_rice = jnp.log(1.2)

GS_start_sorghum = jnp.log(91) # April 1st
GS_length_sorghum = jnp.log(274 - 91) # April 1st -> October 1st
L_ini_sorghum = jnp.log(0.15)
L_dev_sorghum = jnp.log(0.28)
L_mid_sorghum = jnp.log(0.33)
Kc_ini_sorghum = jnp.log(0.3)
Kc_mid_sorghum = jnp.log(1.1)
Kc_end_sorghum = jnp.log(0.55)
K_min_sorghum = jnp.log(0.3)
K_max_sorghum = jnp.log(1.1)

GS_start_soybeans = jnp.log(91) # April 1st
GS_length_soybeans = jnp.log(274 - 91) # April 1st -> October 1st 
L_ini_soybeans = jnp.log(0.15)
L_dev_soybeans = jnp.log(0.2)
L_mid_soybeans = jnp.log(0.45)
Kc_ini_soybeans = jnp.log(0.4)
Kc_mid_soybeans = jnp.log(1.15)
Kc_end_soybeans = jnp.log(0.5)
K_min_soybeans = jnp.log(0.4)
K_max_soybeans = jnp.log(1.15)

# Assume spring wheat
GS_start_wheat = jnp.log(91) # April 1st
GS_length_wheat = jnp.log(244 - 91) # April 1st -> September 1st
L_ini_wheat = jnp.log(0.15)
L_dev_wheat = jnp.log(0.25)
L_mid_wheat = jnp.log(0.4)
Kc_ini_wheat = jnp.log(0.4)
Kc_mid_wheat = jnp.log(1.15)
Kc_end_wheat = jnp.log(0.3)
K_min_wheat = jnp.log(0.3)
K_max_wheat = jnp.log(1.15)

initial_params = jnp.array([awCap_scalar, wiltingp_scalar, \
                       alpha_claycoef, alpha_sandcoef, alpha_siltcoef, \
                       betaHBV_claycoef, betaHBV_sandcoef, betaHBV_siltcoef, betaHBV_elevcoef, \
                       GS_start_corn, GS_length_corn, L_ini_corn, L_dev_corn, L_mid_corn, Kc_ini_corn, Kc_mid_corn, Kc_end_corn, K_min_corn, K_max_corn, \
                       GS_start_cotton, GS_length_cotton, L_ini_cotton, L_dev_cotton, L_mid_cotton, Kc_ini_cotton, Kc_mid_cotton, Kc_end_cotton, K_min_cotton, K_max_cotton, \
                       GS_start_rice, GS_length_rice, L_ini_rice, L_dev_rice, L_mid_rice, Kc_ini_rice, Kc_mid_rice, Kc_end_rice, K_min_rice, K_max_rice,  \
                       GS_start_sorghum, GS_length_sorghum, L_ini_sorghum, L_dev_sorghum, L_mid_sorghum, Kc_ini_sorghum, Kc_mid_sorghum, Kc_end_sorghum, K_min_sorghum, K_max_sorghum, \
                       GS_start_soybeans, GS_length_soybeans, L_ini_soybeans, L_dev_soybeans, L_mid_soybeans, Kc_ini_soybeans, Kc_mid_soybeans, Kc_end_soybeans, K_min_soybeans, K_max_soybeans, \
                       GS_start_wheat, GS_length_wheat, L_ini_wheat, L_dev_wheat, L_mid_wheat, Kc_ini_wheat, Kc_mid_wheat, Kc_end_wheat, K_min_wheat, K_max_wheat])