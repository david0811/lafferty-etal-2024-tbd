import jax.numpy as jnp 

# Constants
Ts = -1. # Snowfall threshold
Tm = 1.  # Snowmelt threshold
Wi_init = 0. # Initial canopy storage (could spinup to find this but miniscule difference)
Sp_init = 0. # Initial snowpack storage (could spinup to find this but miniscule difference)

constants = jnp.array([Ts, Tm, Wi_init, Sp_init])

# Parameters
awCap_scalar = 1.
awCap_claycoef = 200.
awCap_sandcoef = 200.
awCap_siltcoef = 200.

wiltingp_scalar = 1.
wiltingp_claycoef = 200.
wiltingp_sandcoef = 200.
wiltingp_siltcoef = 200.

alpha_claycoef = 0.5
alpha_sandcoef = 0.5
alpha_siltcoef = 0.5

betaHBV_claycoef = 0.5
betaHBV_sandcoef = 0.5
betaHBV_siltcoef = 0.5
betaHBV_elevcoef = 0.5

GS_start_corn = 91 # April 1st
GS_end_corn = 274 # October 1st
L_ini_corn = 0.17
L_dev_corn = 0.28
L_mid_corn = 0.33
Kc_ini_corn = 0.3
Kc_mid_corn = 1.2
Kc_end_corn = 0.4
K_min_corn = 0.3
K_max_corn = 1.2

GS_start_cotton = 91 # April 1st
GS_end_cotton = 274 # October 1st
L_ini_cotton = 0.17
L_dev_cotton = 0.33
L_mid_cotton = 0.25
Kc_ini_cotton = 0.35
Kc_mid_cotton = 1.18
Kc_end_cotton = 0.6
K_min_cotton = 0.35
K_max_cotton = 1.18

# Rice growing season: https://www.ers.usda.gov/topics/crops/rice/rice-sector-at-a-glance/
GS_start_rice = 91 # April 1st
GS_end_rice = 244 # September 1st
L_ini_rice = 0.17
L_dev_rice = 0.28
L_mid_rice = 0.44
Kc_ini_rice = 1.05
Kc_mid_rice = 1.2
Kc_end_rice = 0.75
K_min_rice = 0.75
K_max_rice = 1.2

GS_start_sorghum = 91 # April 1st
GS_end_sorghum = 274 # October 1st
L_ini_sorghum = 0.15
L_dev_sorghum = 0.28
L_mid_sorghum = 0.33
Kc_ini_sorghum = 0.3
Kc_mid_sorghum = 1.1
Kc_end_sorghum = 0.55
K_min_sorghum = 0.3
K_max_sorghum = 1.1

GS_start_soybeans = 91 # April 1st
GS_end_soybeans = 274 # October 1st 
L_ini_soybeans = 0.15
L_dev_soybeans = 0.2
L_mid_soybeans = 0.45
Kc_ini_soybeans = 0.4
Kc_mid_soybeans = 1.15
Kc_end_soybeans = 0.5
K_min_soybeans = 0.4
K_max_soybeans = 1.15

# Assume spring wheat
GS_start_wheat = 91 # April 1st
GS_end_wheat = 244 # September 1st
L_ini_wheat = 0.15
L_dev_wheat = 0.25
L_mid_wheat = 0.4
Kc_ini_wheat = 0.4
Kc_mid_wheat = 1.15
Kc_end_wheat = 0.3
K_min_wheat = 0.4
K_max_wheat = 1.15

initial_params = jnp.array([
    # awCap_scalar, wiltingp_scalar, \
                            awCap_claycoef, awCap_sandcoef, awCap_siltcoef, \
                            wiltingp_claycoef, wiltingp_sandcoef, wiltingp_siltcoef, \
                            alpha_claycoef, alpha_sandcoef, alpha_siltcoef, \
                            betaHBV_claycoef, betaHBV_sandcoef, betaHBV_siltcoef, betaHBV_elevcoef, \
                            GS_start_corn, GS_end_corn, L_ini_corn, L_dev_corn, L_mid_corn, Kc_ini_corn, Kc_mid_corn, Kc_end_corn, K_min_corn, K_max_corn, \
                            GS_start_cotton, GS_end_cotton, L_ini_cotton, L_dev_cotton, L_mid_cotton, Kc_ini_cotton, Kc_mid_cotton, Kc_end_cotton, K_min_cotton, K_max_cotton, \
                            GS_start_rice, GS_end_rice, L_ini_rice, L_dev_rice, L_mid_rice, Kc_ini_rice, Kc_mid_rice, Kc_end_rice, K_min_rice, K_max_rice,  \
                            GS_start_sorghum, GS_end_sorghum, L_ini_sorghum, L_dev_sorghum, L_mid_sorghum, Kc_ini_sorghum, Kc_mid_sorghum, Kc_end_sorghum, K_min_sorghum, K_max_sorghum, \
                            GS_start_soybeans, GS_end_soybeans, L_ini_soybeans, L_dev_soybeans, L_mid_soybeans, Kc_ini_soybeans, Kc_mid_soybeans, Kc_end_soybeans, K_min_soybeans, K_max_soybeans, \
                            GS_start_wheat, GS_end_wheat, L_ini_wheat, L_dev_wheat, L_mid_wheat, Kc_ini_wheat, Kc_mid_wheat, Kc_end_wheat, K_min_wheat, K_max_wheat])