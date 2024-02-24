import jax.numpy as jnp

### Parameter bounds
# awCap
awCap_scalar_lower, awCap_scalar_upper = (
    jnp.log(0.5),
    jnp.log(5.0),
)  # Central 1.
awCap_claycoef_lower, awCap_claycoef_upper = (
    jnp.log(10),
    jnp.log(500.0),
)  # Central 100
awCap_sandcoef_lower, awCap_sandcoef_upper = (
    jnp.log(10),
    jnp.log(500.0),
)  # Central 100
awCap_siltcoef_lower, awCap_siltcoef_upper = (
    jnp.log(10),
    jnp.log(500.0),
)  # Central 100

awCap_sand_lower, awCap_sand_upper = jnp.log(10), jnp.log(1000)
awCap_loamy_sand_lower, awCap_loamy_sand_upper = jnp.log(10), jnp.log(1000)
awCap_sandy_loam_lower, awCap_sandy_loam_upper = jnp.log(10), jnp.log(1000)
awCap_silt_loam_lower, awCap_silt_loam_upper = jnp.log(10), jnp.log(1000)
awCap_silt_lower, awCap_silt_upper = jnp.log(10), jnp.log(1000)
awCap_loam_lower, awCap_loam_upper = jnp.log(10), jnp.log(1000)
awCap_sandy_clay_loam_lower, awCap_sandy_clay_loam_upper = (
    jnp.log(10),
    jnp.log(1000),
)
awCap_silty_clay_loam_lower, awCap_silty_clay_loam_upper = (
    jnp.log(10),
    jnp.log(1000),
)
awCap_clay_loam_lower, awCap_clay_loam_upper = jnp.log(10), jnp.log(1000)
awCap_sandy_clay_lower, awCap_sandy_clay_upper = jnp.log(10), jnp.log(1000)
awCap_silty_clay_lower, awCap_silty_clay_upper = jnp.log(10), jnp.log(1000)
awCap_clay_lower, awCap_clay_upper = jnp.log(10), jnp.log(1000)

# wiltingp
wiltingp_scalar_lower, wiltingp_scalar_upper = (
    jnp.log(0.5),
    jnp.log(5.0),
)  # Central 1.
wiltingp_claycoef_lower, wiltingp_claycoef_upper = (
    jnp.log(10),
    jnp.log(500.0),
)  # Central 100
wiltingp_sandcoef_lower, wiltingp_sandcoef_upper = (
    jnp.log(10),
    jnp.log(500.0),
)  # Central 100
wiltingp_siltcoef_lower, wiltingp_siltcoef_upper = (
    jnp.log(10),
    jnp.log(500.0),
)  # Central 100

wiltingp_sand_lower, wiltingp_sand_upper = jnp.log(10), jnp.log(500)
wiltingp_loamy_sand_lower, wiltingp_loamy_sand_upper = (
    jnp.log(10),
    jnp.log(500),
)
wiltingp_sandy_loam_lower, wiltingp_sandy_loam_upper = (
    jnp.log(10),
    jnp.log(500),
)
wiltingp_silt_loam_lower, wiltingp_silt_loam_upper = jnp.log(10), jnp.log(500)
wiltingp_silt_lower, wiltingp_silt_upper = jnp.log(10), jnp.log(500)
wiltingp_loam_lower, wiltingp_loam_upper = jnp.log(10), jnp.log(500)
wiltingp_sandy_clay_loam_lower, wiltingp_sandy_clay_loam_upper = (
    jnp.log(10),
    jnp.log(500),
)
wiltingp_silty_clay_loam_lower, wiltingp_silty_clay_loam_upper = (
    jnp.log(10),
    jnp.log(500),
)
wiltingp_clay_loam_lower, wiltingp_clay_loam_upper = jnp.log(10), jnp.log(500)
wiltingp_sandy_clay_lower, wiltingp_sandy_clay_upper = (
    jnp.log(10),
    jnp.log(500),
)
wiltingp_silty_clay_lower, wiltingp_silty_clay_upper = (
    jnp.log(10),
    jnp.log(500),
)
wiltingp_clay_lower, wiltingp_clay_upper = jnp.log(10), jnp.log(500)

# alpha
alpha_claycoef_lower, alpha_claycoef_upper = (
    jnp.log(0.001),
    jnp.log(100.0),
)  # Central 0.5
alpha_sandcoef_lower, alpha_sandcoef_upper = (
    jnp.log(0.001),
    jnp.log(100.0),
)  # Central 0.5
alpha_siltcoef_lower, alpha_siltcoef_upper = (
    jnp.log(0.001),
    jnp.log(100.0),
)  # Central 0.5

# betaHBV
betaHBV_claycoef_lower, betaHBV_claycoef_upper = (
    jnp.log(0.001),
    jnp.log(100.0),
)  # Central 0.5
betaHBV_sandcoef_lower, betaHBV_sandcoef_upper = (
    jnp.log(0.001),
    jnp.log(100.0),
)  # Central 0.5
betaHBV_siltcoef_lower, betaHBV_siltcoef_upper = (
    jnp.log(0.001),
    jnp.log(100.0),
)  # Central 0.5
betaHBV_elevcoef_lower, betaHBV_elevcoef_upper = (
    jnp.log(0.001),
    jnp.log(100.0),
)  # Central 0.5

# Corn
GS_start_corn_lower, GS_start_corn_upper = (
    jnp.log(60),
    jnp.log(152),
)  # March 1st, June 1st
GS_end_corn_lower, GS_end_corn_upper = (
    jnp.log(244),
    jnp.log(334),
)  # Sep 1st, latest Nov 30th
L_ini_corn_lower, L_ini_corn_upper = (
    jnp.log(0.07),
    jnp.log(0.22),
)  # central 0.17
L_dev_corn_lower, L_dev_corn_upper = (
    jnp.log(0.18),
    jnp.log(0.33),
)  # central 0.28
L_mid_corn_lower, L_mid_corn_upper = (
    jnp.log(0.13),
    jnp.log(0.38),
)  # central 0.33
Kc_ini_corn_lower, Kc_ini_corn_upper = (
    jnp.log(0.1),
    jnp.log(0.5),
)  # central 0.3
Kc_mid_corn_lower, Kc_mid_corn_upper = (
    jnp.log(1.0),
    jnp.log(1.5),
)  # central 1.2
Kc_end_corn_lower, Kc_end_corn_upper = (
    jnp.log(0.2),
    jnp.log(0.6),
)  # central 0.4
K_min_corn_lower, K_min_corn_upper = jnp.log(0.1), jnp.log(0.5)  # central 0.3
K_max_corn_lower, K_max_corn_upper = jnp.log(1.0), jnp.log(1.5)  # central 1.2

# Cotton
GS_start_cotton_lower, GS_start_cotton_upper = (
    jnp.log(60),
    jnp.log(152),
)  # March 1st, June 1st
GS_end_cotton_lower, GS_end_cotton_upper = (
    jnp.log(244),
    jnp.log(334),
)  # Sep 1st, Nov 30th
L_ini_cotton_lower, L_ini_cotton_upper = (
    jnp.log(0.07),
    jnp.log(0.25),
)  # central 0.17
L_dev_cotton_lower, L_dev_cotton_upper = (
    jnp.log(0.23),
    jnp.log(0.4),
)  # central 0.33
L_mid_cotton_lower, L_mid_cotton_upper = (
    jnp.log(0.15),
    jnp.log(0.3),
)  # central 0.25
Kc_ini_cotton_lower, Kc_ini_cotton_upper = (
    jnp.log(0.15),
    jnp.log(0.65),
)  # central 0.35
Kc_mid_cotton_lower, Kc_mid_cotton_upper = (
    jnp.log(1.0),
    jnp.log(1.5),
)  # central 1.18
Kc_end_cotton_lower, Kc_end_cotton_upper = (
    jnp.log(0.4),
    jnp.log(0.8),
)  # central 0.6
K_min_cotton_lower, K_min_cotton_upper = (
    jnp.log(0.1),
    jnp.log(0.6),
)  # central 0.35
K_max_cotton_lower, K_max_cotton_upper = (
    jnp.log(1.0),
    jnp.log(1.5),
)  # central 1.18

# Rice growing season: https://www.ers.usda.gov/topics/crops/rice/rice-sector-at-a-glance/
GS_start_rice_lower, GS_start_rice_upper = (
    jnp.log(60),
    jnp.log(182),
)  # March 1st, July 1st
GS_end_rice_lower, GS_end_rice_upper = (
    jnp.log(214),
    jnp.log(334),
)  # August 1st, Nov 30th
L_ini_rice_lower, L_ini_rice_upper = (
    jnp.log(0.07),
    jnp.log(0.21),
)  # central 0.17
L_dev_rice_lower, L_dev_rice_upper = (
    jnp.log(0.18),
    jnp.log(0.32),
)  # central 0.28
L_mid_rice_lower, L_mid_rice_upper = (
    jnp.log(0.34),
    jnp.log(0.48),
)  # central 0.44
Kc_ini_rice_lower, Kc_ini_rice_upper = (
    jnp.log(0.95),
    jnp.log(1.1),
)  # central 1.05
Kc_mid_rice_lower, Kc_mid_rice_upper = (
    jnp.log(1.1),
    jnp.log(1.3),
)  # central 1.2
Kc_end_rice_lower, Kc_end_rice_upper = (
    jnp.log(0.65),
    jnp.log(0.85),
)  # central 0.75
K_min_rice_lower, K_min_rice_upper = (
    jnp.log(0.65),
    jnp.log(0.85),
)  # central 0.75
K_max_rice_lower, K_max_rice_upper = jnp.log(1.0), jnp.log(1.4)  # central 1.2

# Sorghum
GS_start_sorghum_lower, GS_start_sorghum_upper = (
    jnp.log(60),
    jnp.log(182),
)  # March 1st, July 1st
GS_end_sorghum_lower, GS_end_sorghum_upper = (
    jnp.log(214),
    jnp.log(334),
)  # August 1st, Nov 30th
L_ini_sorghum_lower, L_ini_sorghum_upper = (
    jnp.log(0.05),
    jnp.log(0.20),
)  # central 0.15
L_dev_sorghum_lower, L_dev_sorghum_upper = (
    jnp.log(0.18),
    jnp.log(0.33),
)  # central 0.28
L_mid_sorghum_lower, L_mid_sorghum_upper = (
    jnp.log(0.23),
    jnp.log(0.38),
)  # central 0.33
Kc_ini_sorghum_lower, Kc_ini_sorghum_upper = (
    jnp.log(0.1),
    jnp.log(0.5),
)  # central 0.3
Kc_mid_sorghum_lower, Kc_mid_sorghum_upper = (
    jnp.log(1.0),
    jnp.log(1.2),
)  # central 1.1
Kc_end_sorghum_lower, Kc_end_sorghum_upper = (
    jnp.log(0.35),
    jnp.log(0.75),
)  # central 0.55
K_min_sorghum_lower, K_min_sorghum_upper = (
    jnp.log(0.1),
    jnp.log(0.5),
)  # central 0.3
K_max_sorghum_lower, K_max_sorghum_upper = (
    jnp.log(1.0),
    jnp.log(1.2),
)  # central 1.1

# Soybeans
GS_start_soybeans_lower, GS_start_soybeans_upper = (
    jnp.log(60),
    jnp.log(182),
)  # March 1st, July 1st
GS_end_soybeans_lower, GS_end_soybeans_upper = (
    jnp.log(244),
    jnp.log(334),
)  # Sep 1st, Nov 30th
L_ini_soybeans_lower, L_ini_soybeans_upper = (
    jnp.log(0.05),
    jnp.log(0.2),
)  # central 0.15
L_dev_soybeans_lower, L_dev_soybeans_upper = (
    jnp.log(0.1),
    jnp.log(0.25),
)  # central 0.2
L_mid_soybeans_lower, L_mid_soybeans_upper = (
    jnp.log(0.35),
    jnp.log(0.5),
)  # central 0.45
Kc_ini_soybeans_lower, Kc_ini_soybeans_upper = (
    jnp.log(0.2),
    jnp.log(0.6),
)  # central 0.4
Kc_mid_soybeans_lower, Kc_mid_soybeans_upper = (
    jnp.log(1.0),
    jnp.log(1.3),
)  # central 1.15
Kc_end_soybeans_lower, Kc_end_soybeans_upper = (
    jnp.log(0.3),
    jnp.log(0.7),
)  # central 0.5
K_min_soybeans_lower, K_min_soybeans_upper = (
    jnp.log(0.2),
    jnp.log(0.6),
)  # central 0.4
K_max_soybeans_lower, K_max_soybeans_upper = (
    jnp.log(1.0),
    jnp.log(1.3),
)  # central 1.15

# Assume spring wheat
GS_start_wheat_lower, GS_start_wheat_upper = (
    jnp.log(60),
    jnp.log(152),
)  # March 1st, June 1st
GS_end_wheat_lower, GS_end_wheat_upper = (
    jnp.log(214),
    jnp.log(274),
)  # August 1st, Oct 1st
L_ini_wheat_lower, L_ini_wheat_upper = (
    jnp.log(0.05),
    jnp.log(0.2),
)  # central 0.15
L_dev_wheat_lower, L_dev_wheat_upper = (
    jnp.log(0.15),
    jnp.log(0.3),
)  # central 0.25
L_mid_wheat_lower, L_mid_wheat_upper = (
    jnp.log(0.3),
    jnp.log(0.45),
)  # central 0.4
Kc_ini_wheat_lower, Kc_ini_wheat_upper = (
    jnp.log(0.2),
    jnp.log(0.6),
)  # central 0.4
Kc_mid_wheat_lower, Kc_mid_wheat_upper = (
    jnp.log(1.0),
    jnp.log(1.3),
)  # central 1.15
Kc_end_wheat_lower, Kc_end_wheat_upper = (
    jnp.log(0.1),
    jnp.log(0.5),
)  # central 0.3
K_min_wheat_lower, K_min_wheat_upper = (
    jnp.log(0.2),
    jnp.log(0.6),
)  # central 0.4
K_max_wheat_lower, K_max_wheat_upper = (
    jnp.log(1.0),
    jnp.log(1.3),
)  # central 1.15


params_main_lower = jnp.array(
    [
        awCap_scalar_lower,
        wiltingp_scalar_lower,
        alpha_claycoef_lower,
        alpha_sandcoef_lower,
        alpha_siltcoef_lower,
        betaHBV_claycoef_lower,
        betaHBV_sandcoef_lower,
        betaHBV_siltcoef_lower,
        betaHBV_elevcoef_lower,
        GS_start_corn_lower,
        GS_end_corn_lower,
        L_ini_corn_lower,
        L_dev_corn_lower,
        L_mid_corn_lower,
        Kc_ini_corn_lower,
        Kc_mid_corn_lower,
        Kc_end_corn_lower,
        K_min_corn_lower,
        K_max_corn_lower,
        GS_start_cotton_lower,
        GS_end_cotton_lower,
        L_ini_cotton_lower,
        L_dev_cotton_lower,
        L_mid_cotton_lower,
        Kc_ini_cotton_lower,
        Kc_mid_cotton_lower,
        Kc_end_cotton_lower,
        K_min_cotton_lower,
        K_max_cotton_lower,
        GS_start_rice_lower,
        GS_end_rice_lower,
        L_ini_rice_lower,
        L_dev_rice_lower,
        L_mid_rice_lower,
        Kc_ini_rice_lower,
        Kc_mid_rice_lower,
        Kc_end_rice_lower,
        K_min_rice_lower,
        K_max_rice_lower,
        GS_start_sorghum_lower,
        GS_end_sorghum_lower,
        L_ini_sorghum_lower,
        L_dev_sorghum_lower,
        L_mid_sorghum_lower,
        Kc_ini_sorghum_lower,
        Kc_mid_sorghum_lower,
        Kc_end_sorghum_lower,
        K_min_sorghum_lower,
        K_max_sorghum_lower,
        GS_start_soybeans_lower,
        GS_end_soybeans_lower,
        L_ini_soybeans_lower,
        L_dev_soybeans_lower,
        L_mid_soybeans_lower,
        Kc_ini_soybeans_lower,
        Kc_mid_soybeans_lower,
        Kc_end_soybeans_lower,
        K_min_soybeans_lower,
        K_max_soybeans_lower,
        GS_start_wheat_lower,
        GS_end_wheat_lower,
        L_ini_wheat_lower,
        L_dev_wheat_lower,
        L_mid_wheat_lower,
        Kc_ini_wheat_lower,
        Kc_mid_wheat_lower,
        Kc_end_wheat_lower,
        K_min_wheat_lower,
        K_max_wheat_lower,
    ]
)


params_main_upper = jnp.array(
    [
        awCap_scalar_upper,
        wiltingp_scalar_upper,
        alpha_claycoef_upper,
        alpha_sandcoef_upper,
        alpha_siltcoef_upper,
        betaHBV_claycoef_upper,
        betaHBV_sandcoef_upper,
        betaHBV_siltcoef_upper,
        betaHBV_elevcoef_upper,
        GS_start_corn_upper,
        GS_end_corn_upper,
        L_ini_corn_upper,
        L_dev_corn_upper,
        L_mid_corn_upper,
        Kc_ini_corn_upper,
        Kc_mid_corn_upper,
        Kc_end_corn_upper,
        K_min_corn_upper,
        K_max_corn_upper,
        GS_start_cotton_upper,
        GS_end_cotton_upper,
        L_ini_cotton_upper,
        L_dev_cotton_upper,
        L_mid_cotton_upper,
        Kc_ini_cotton_upper,
        Kc_mid_cotton_upper,
        Kc_end_cotton_upper,
        K_min_cotton_upper,
        K_max_cotton_upper,
        GS_start_rice_upper,
        GS_end_rice_upper,
        L_ini_rice_upper,
        L_dev_rice_upper,
        L_mid_rice_upper,
        Kc_ini_rice_upper,
        Kc_mid_rice_upper,
        Kc_end_rice_upper,
        K_min_rice_upper,
        K_max_rice_upper,
        GS_start_sorghum_upper,
        GS_end_sorghum_upper,
        L_ini_sorghum_upper,
        L_dev_sorghum_upper,
        L_mid_sorghum_upper,
        Kc_ini_sorghum_upper,
        Kc_mid_sorghum_upper,
        Kc_end_sorghum_upper,
        K_min_sorghum_upper,
        K_max_sorghum_upper,
        GS_start_soybeans_upper,
        GS_end_soybeans_upper,
        L_ini_soybeans_upper,
        L_dev_soybeans_upper,
        L_mid_soybeans_upper,
        Kc_ini_soybeans_upper,
        Kc_mid_soybeans_upper,
        Kc_end_soybeans_upper,
        K_min_soybeans_upper,
        K_max_soybeans_upper,
        GS_start_wheat_upper,
        GS_end_wheat_upper,
        L_ini_wheat_upper,
        L_dev_wheat_upper,
        L_mid_wheat_upper,
        Kc_ini_wheat_upper,
        Kc_mid_wheat_upper,
        Kc_end_wheat_upper,
        K_min_wheat_upper,
        K_max_wheat_upper,
    ]
)

params_vic_lower = jnp.array(
    [
        awCap_sand_lower,
        awCap_loamy_sand_lower,
        awCap_sandy_loam_lower,
        awCap_silt_loam_lower,
        awCap_silt_lower,
        awCap_loam_lower,
        awCap_sandy_clay_loam_lower,
        awCap_silty_clay_loam_lower,
        awCap_clay_loam_lower,
        awCap_sandy_clay_lower,
        awCap_silty_clay_lower,
        awCap_clay_lower,
        wiltingp_sand_lower,
        wiltingp_loamy_sand_lower,
        wiltingp_sandy_loam_lower,
        wiltingp_silt_loam_lower,
        wiltingp_silt_lower,
        wiltingp_loam_lower,
        wiltingp_sandy_clay_loam_lower,
        wiltingp_silty_clay_loam_lower,
        wiltingp_clay_loam_lower,
        wiltingp_sandy_clay_lower,
        wiltingp_silty_clay_lower,
        wiltingp_clay_lower,
        # awCap_claycoef_lower,
        # awCap_sandcoef_lower,
        # awCap_siltcoef_lower,
        # wiltingp_claycoef_lower,
        # wiltingp_sandcoef_lower,
        # wiltingp_siltcoef_lower,
        alpha_claycoef_lower,
        alpha_sandcoef_lower,
        alpha_siltcoef_lower,
        betaHBV_claycoef_lower,
        betaHBV_sandcoef_lower,
        betaHBV_siltcoef_lower,
        betaHBV_elevcoef_lower,
        GS_start_corn_lower,
        GS_end_corn_lower,
        L_ini_corn_lower,
        L_dev_corn_lower,
        L_mid_corn_lower,
        Kc_ini_corn_lower,
        Kc_mid_corn_lower,
        Kc_end_corn_lower,
        K_min_corn_lower,
        K_max_corn_lower,
        GS_start_cotton_lower,
        GS_end_cotton_lower,
        L_ini_cotton_lower,
        L_dev_cotton_lower,
        L_mid_cotton_lower,
        Kc_ini_cotton_lower,
        Kc_mid_cotton_lower,
        Kc_end_cotton_lower,
        K_min_cotton_lower,
        K_max_cotton_lower,
        GS_start_rice_lower,
        GS_end_rice_lower,
        L_ini_rice_lower,
        L_dev_rice_lower,
        L_mid_rice_lower,
        Kc_ini_rice_lower,
        Kc_mid_rice_lower,
        Kc_end_rice_lower,
        K_min_rice_lower,
        K_max_rice_lower,
        GS_start_sorghum_lower,
        GS_end_sorghum_lower,
        L_ini_sorghum_lower,
        L_dev_sorghum_lower,
        L_mid_sorghum_lower,
        Kc_ini_sorghum_lower,
        Kc_mid_sorghum_lower,
        Kc_end_sorghum_lower,
        K_min_sorghum_lower,
        K_max_sorghum_lower,
        GS_start_soybeans_lower,
        GS_end_soybeans_lower,
        L_ini_soybeans_lower,
        L_dev_soybeans_lower,
        L_mid_soybeans_lower,
        Kc_ini_soybeans_lower,
        Kc_mid_soybeans_lower,
        Kc_end_soybeans_lower,
        K_min_soybeans_lower,
        K_max_soybeans_lower,
        GS_start_wheat_lower,
        GS_end_wheat_lower,
        L_ini_wheat_lower,
        L_dev_wheat_lower,
        L_mid_wheat_lower,
        Kc_ini_wheat_lower,
        Kc_mid_wheat_lower,
        Kc_end_wheat_lower,
        K_min_wheat_lower,
        K_max_wheat_lower,
    ]
)

params_vic_upper = jnp.array(
    [
        awCap_sand_upper,
        awCap_loamy_sand_upper,
        awCap_sandy_loam_upper,
        awCap_silt_loam_upper,
        awCap_silt_upper,
        awCap_loam_upper,
        awCap_sandy_clay_loam_upper,
        awCap_silty_clay_loam_upper,
        awCap_clay_loam_upper,
        awCap_sandy_clay_upper,
        awCap_silty_clay_upper,
        awCap_clay_upper,
        wiltingp_sand_upper,
        wiltingp_loamy_sand_upper,
        wiltingp_sandy_loam_upper,
        wiltingp_silt_loam_upper,
        wiltingp_silt_upper,
        wiltingp_loam_upper,
        wiltingp_sandy_clay_loam_upper,
        wiltingp_silty_clay_loam_upper,
        wiltingp_clay_loam_upper,
        wiltingp_sandy_clay_upper,
        wiltingp_silty_clay_upper,
        wiltingp_clay_upper,
        # awCap_claycoef_upper,
        # awCap_sandcoef_upper,
        # awCap_siltcoef_upper,
        # wiltingp_claycoef_upper,
        # wiltingp_sandcoef_upper,
        # wiltingp_siltcoef_upper,
        alpha_claycoef_upper,
        alpha_sandcoef_upper,
        alpha_siltcoef_upper,
        betaHBV_claycoef_upper,
        betaHBV_sandcoef_upper,
        betaHBV_siltcoef_upper,
        betaHBV_elevcoef_upper,
        GS_start_corn_upper,
        GS_end_corn_upper,
        L_ini_corn_upper,
        L_dev_corn_upper,
        L_mid_corn_upper,
        Kc_ini_corn_upper,
        Kc_mid_corn_upper,
        Kc_end_corn_upper,
        K_min_corn_upper,
        K_max_corn_upper,
        GS_start_cotton_upper,
        GS_end_cotton_upper,
        L_ini_cotton_upper,
        L_dev_cotton_upper,
        L_mid_cotton_upper,
        Kc_ini_cotton_upper,
        Kc_mid_cotton_upper,
        Kc_end_cotton_upper,
        K_min_cotton_upper,
        K_max_cotton_upper,
        GS_start_rice_upper,
        GS_end_rice_upper,
        L_ini_rice_upper,
        L_dev_rice_upper,
        L_mid_rice_upper,
        Kc_ini_rice_upper,
        Kc_mid_rice_upper,
        Kc_end_rice_upper,
        K_min_rice_upper,
        K_max_rice_upper,
        GS_start_sorghum_upper,
        GS_end_sorghum_upper,
        L_ini_sorghum_upper,
        L_dev_sorghum_upper,
        L_mid_sorghum_upper,
        Kc_ini_sorghum_upper,
        Kc_mid_sorghum_upper,
        Kc_end_sorghum_upper,
        K_min_sorghum_upper,
        K_max_sorghum_upper,
        GS_start_soybeans_upper,
        GS_end_soybeans_upper,
        L_ini_soybeans_upper,
        L_dev_soybeans_upper,
        L_mid_soybeans_upper,
        Kc_ini_soybeans_upper,
        Kc_mid_soybeans_upper,
        Kc_end_soybeans_upper,
        K_min_soybeans_upper,
        K_max_soybeans_upper,
        GS_start_wheat_upper,
        GS_end_wheat_upper,
        L_ini_wheat_upper,
        L_dev_wheat_upper,
        L_mid_wheat_upper,
        Kc_ini_wheat_upper,
        Kc_mid_wheat_upper,
        Kc_end_wheat_upper,
        K_min_wheat_upper,
        K_max_wheat_upper,
    ]
)
