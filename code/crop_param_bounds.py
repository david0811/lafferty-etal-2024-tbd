import jax.numpy as jnp 

# Crop parameter bounds
GS_start_corn_lower, GS_start_corn_upper = jnp.log(60), jnp.log(182) # March 1st, July 1st
GS_length_corn_lower, GS_length_corn_upper = jnp.log(), jnp.log(274 - 91) # April 1st -> October 1st
L_ini_corn_lower, L_ini_corn_upper = jnp.log(), jnp.log(0.17)
L_dev_corn_lower, L_dev_corn_upper = jnp.log(), jnp.log(0.28)
L_mid_corn_lower, L_mid_corn_upper = jnp.log(), jnp.log(0.33)
Kc_ini_corn_lower, Kc_ini_corn_upper = jnp.log(), jnp.log(0.3)
Kc_mid_corn_lower, Kc_mid_corn_upper = jnp.log(), jnp.log(1.2)
Kc_end_corn_lower, Kc_end_corn_upper = jnp.log(), jnp.log(0.4)
K_min_corn_lower, K_min_corn_upper = jnp.log(), jnp.log(0.3)
K_max_corn_lower, K_max_corn_upper = jnp.log(), jnp.log(1.2)

GS_start_cotton_lower, GS_start_cotton_upper = jnp.log(60), jnp.log(182) # March 1st, July 1st
GS_length_cotton_lower, GS_length_cotton_upper = jnp.log(), jnp.log(274 - 91) # April 1st -> October 1st
L_ini_cotton_lower, L_ini_cotton_upper = jnp.log(), jnp.log(0.17)
L_dev_cotton_lower, L_dev_cotton_upper = jnp.log(), jnp.log(0.33)
L_mid_cotton_lower, L_mid_cotton_upper = jnp.log(), jnp.log(0.25)
Kc_ini_cotton_lower, Kc_ini_cotton_upper = jnp.log(), jnp.log(0.35)
Kc_mid_cotton_lower, Kc_mid_cotton_upper = jnp.log(), jnp.log(1.18)
Kc_end_cotton_lower, Kc_end_cotton_upper = jnp.log(), jnp.log(0.6)
K_min_cotton_lower, K_min_cotton_upper = jnp.log(), jnp.log(0.35)
K_max_cotton_lower, K_max_cotton_upper = jnp.log(), jnp.log(1.18)

# Rice growing season: https://www.ers.usda.gov/topics/crops/rice/rice-sector-at-a-glance/
GS_start_rice_lower, GS_start_rice_upper = jnp.log(60), jnp.log(182) # March 1st, July 1st
GS_length_rice_lower, GS_length_rice_upper = jnp.log(), jnp.log(244 - 91) # April 1st -> September 1st
L_ini_rice_lower, L_ini_rice_upper = jnp.log(), jnp.log(0.17)
L_dev_rice_lower, L_dev_rice_upper = jnp.log(), jnp.log(0.28)
L_mid_rice_lower, L_mid_rice_upper = jnp.log(), jnp.log(0.44)
Kc_ini_rice_lower, Kc_ini_rice_upper = jnp.log(), jnp.log(1.05)
Kc_mid_rice_lower, Kc_mid_rice_upper = jnp.log(), jnp.log(1.2)
Kc_end_rice_lower, Kc_end_rice_upper = jnp.log(), jnp.log(0.75)
K_min_rice_lower, K_min_rice_upper = jnp.log(), jnp.log(0.75)
K_max_rice_lower, K_max_rice_upper = jnp.log(), jnp.log(1.2)

GS_start_sorghum_lower, GS_start_sorghum_upper = jnp.log(60), jnp.log(182) # March 1st, July 1st
GS_length_sorghum_lower, GS_length_sorghum_upper = jnp.log(), jnp.log(274 - 91) # April 1st -> October 1st
L_ini_sorghum_lower, L_ini_sorghum_upper = jnp.log(), jnp.log(0.15)
L_dev_sorghum_lower, L_dev_sorghum_upper = jnp.log(), jnp.log(0.28)
L_mid_sorghum_lower, L_mid_sorghum_upper = jnp.log(), jnp.log(0.33)
Kc_ini_sorghum_lower, Kc_ini_sorghum_upper = jnp.log(), jnp.log(0.3)
Kc_mid_sorghum_lower, Kc_mid_sorghum_upper = jnp.log(), jnp.log(1.1)
Kc_end_sorghum_lower, Kc_end_sorghum_upper = jnp.log(), jnp.log(0.55)
K_min_sorghum_lower, K_min_sorghum_upper = jnp.log(), jnp.log(0.3)
K_max_sorghum_lower, K_max_sorghum_upper = jnp.log(), jnp.log(1.1)

GS_start_soybeans_lower, GS_start_soybeans_upper = jnp.log(60), jnp.log(182) # March 1st, July 1st
GS_length_soybeans_lower, GS_length_soybeans_upper = jnp.log(), jnp.log(274 - 91) # April 1st -> October 1st 
L_ini_soybeans_lower, L_ini_soybeans_upper = jnp.log(), jnp.log(0.15)
L_dev_soybeans_lower, L_dev_soybeans_upper = jnp.log(), jnp.log(0.2)
L_mid_soybeans_lower, L_mid_soybeans_upper = jnp.log(), jnp.log(0.45)
Kc_ini_soybeans_lower, Kc_ini_soybeans_upper = jnp.log(), jnp.log(0.4)
Kc_mid_soybeans_lower, Kc_mid_soybeans_upper = jnp.log(), jnp.log(1.15)
Kc_end_soybeans_lower, Kc_end_soybeans_upper = jnp.log(), jnp.log(0.5)
K_min_soybeans_lower, K_min_soybeans_upper = jnp.log(), jnp.log(0.4)
K_max_soybeans_lower, K_max_soybeans_upper = jnp.log(), jnp.log(1.15)

# Assume spring wheat
GS_start_wheat_lower, GS_start_wheat_upper = jnp.log(60), jnp.log(182) # March 1st, July 1st
GS_length_wheat_lower, GS_length_wheat_upper = jnp.log(), jnp.log(244 - 91) # April 1st -> September 1st
L_ini_wheat_lower, L_ini_wheat_upper = jnp.log(), jnp.log(0.15)
L_dev_wheat_lower, L_dev_wheat_upper = jnp.log(), jnp.log(0.25)
L_mid_wheat_lower, L_mid_wheat_upper = jnp.log(), jnp.log(0.4)
Kc_ini_wheat_lower, Kc_ini_wheat_upper = jnp.log(), jnp.log(0.4)
Kc_mid_wheat_lower, Kc_mid_wheat_upper = jnp.log(), jnp.log(1.15)
Kc_end_wheat_lower, Kc_end_wheat_upper = jnp.log(), jnp.log(0.3)
K_min_wheat_lower, K_min_wheat_upper = jnp.log(), jnp.log(0.3)
K_max_wheat_lower, K_max_wheat_upper = jnp.log(), jnp.log(1.15)

crop_params_lower = jnp.array([
                       GS_start_corn_lower, GS_length_corn_lower, L_ini_corn_lower, L_dev_corn_lower, L_mid_corn_lower, Kc_ini_corn_lower, Kc_mid_corn_lower, Kc_end_corn_lower, K_min_corn_lower, K_max_corn_lower, \
                       GS_start_cotton_lower, GS_length_cotton_lower, L_ini_cotton_lower, L_dev_cotton_lower, L_mid_cotton_lower, Kc_ini_cotton_lower, Kc_mid_cotton_lower, Kc_end_cotton_lower, K_min_cotton_lower, K_max_cotton_lower, \
                       GS_start_rice_lower, GS_length_rice_lower, L_ini_rice_lower, L_dev_rice_lower, L_mid_rice_lower, Kc_ini_rice_lower, Kc_mid_rice_lower, Kc_end_rice_lower, K_min_rice_lower, K_max_rice_lower,  \
                       GS_start_sorghum_lower, GS_length_sorghum_lower, L_ini_sorghum_lower, L_dev_sorghum_lower, L_mid_sorghum_lower, Kc_ini_sorghum_lower, Kc_mid_sorghum_lower, Kc_end_sorghum_lower, K_min_sorghum_lower, K_max_sorghum_lower, \
                       GS_start_soybeans_lower, GS_length_soybeans_lower, L_ini_soybeans_lower, L_dev_soybeans_lower, L_mid_soybeans_lower, Kc_ini_soybeans_lower, Kc_mid_soybeans_lower, Kc_end_soybeans_lower, K_min_soybeans_lower, K_max_soybeans_lower, \
                       GS_start_wheat_lower, GS_length_wheat_lower, L_ini_wheat_lower, L_dev_wheat_lower, L_mid_wheat_lower, Kc_ini_wheat_lower, Kc_mid_wheat_lower, Kc_end_wheat_lower, K_min_wheat_lower, K_max_wheat_lower])

crop_params_upper = jnp.array([
                       GS_start_corn_upper, GS_length_corn_upper, L_ini_corn_upper, L_dev_corn_upper, L_mid_corn_upper, Kc_ini_corn_upper, Kc_mid_corn_upper, Kc_end_corn_upper, K_min_corn_upper, K_max_corn_upper, \
                       GS_start_cotton_upper, GS_length_cotton_upper, L_ini_cotton_upper, L_dev_cotton_upper, L_mid_cotton_upper, Kc_ini_cotton_upper, Kc_mid_cotton_upper, Kc_end_cotton_upper, K_min_cotton_upper, K_max_cotton_upper, \
                       GS_start_rice_upper, GS_length_rice_upper, L_ini_rice_upper, L_dev_rice_upper, L_mid_rice_upper, Kc_ini_rice_upper, Kc_mid_rice_upper, Kc_end_rice_upper, K_min_rice_upper, K_max_rice_upper,  \
                       GS_start_sorghum_upper, GS_length_sorghum_upper, L_ini_sorghum_upper, L_dev_sorghum_upper, L_mid_sorghum_upper, Kc_ini_sorghum_upper, Kc_mid_sorghum_upper, Kc_end_sorghum_upper, K_min_sorghum_upper, K_max_sorghum_upper, \
                       GS_start_soybeans_upper, GS_length_soybeans_upper, L_ini_soybeans_upper, L_dev_soybeans_upper, L_mid_soybeans_upper, Kc_ini_soybeans_upper, Kc_mid_soybeans_upper, Kc_end_soybeans_upper, K_min_soybeans_upper, K_max_soybeans_upper, \
                       GS_start_wheat_upper, GS_length_wheat_upper, L_ini_wheat_upper, L_dev_wheat_upper, L_mid_wheat_upper, Kc_ini_wheat_upper, Kc_mid_wheat_upper, Kc_end_wheat_upper, K_min_wheat_upper, K_max_wheat_upper])