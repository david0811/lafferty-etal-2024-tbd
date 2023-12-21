import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
def update_state(state, forcing, params):
    """
    Stateless update function for WBM soil moisture simulation.
    """

    # Retrieve previous iteration values
    Ws_prev, Wi, Sp = state

    # Retrieve parameter values
    Ts, Tm, Clai, wiltingp, alpha, betaHBV, Wcap, phi = params

    # Retrieve forcing
    tas, prcp, lai, Kpet, doy = forcing

    # PET function
    def calculate_potential_evapotranspiration(tas, doy, phi):
        """
        Inputs:
         - tas: daily mean temperature [C]
         - N:   day of year
         - phi: latitude [deg]
        Outputs:
         - daily potential evapotranspiration calculated via the Hamon method [mm]
        Notes: (e.g.) http://data.snap.uaf.edu/data/Base/AK_2km/PET/Hamon_PET_equations.pdf
        """

        # Calculate solar declination (delta)
        delta = -23.44 * jnp.cos(jnp.radians((360 / 365) * (doy + 10)))

        # Calculate fractional day length (Lambda)
        pi = 3.14159265359
        Lambda = (1 / pi) * jnp.arccos(
            -jnp.tan(jnp.radians(phi)) * jnp.tan(jnp.radians(delta))
        )

        # Calculate saturation vapor pressure
        tas_gt_zero = tas > 0
        Psat = (tas_gt_zero * (0.61078 * jnp.exp((17.26939 * tas) / (tas + 237.3)))) + (
            ((1 - tas_gt_zero)) * (0.61078 * jnp.exp((21.87456 * tas) / (tas + 265.5)))
        )

        # Calculate saturation vapor density (rho_sat)
        rho_sat = (2.167 * Psat) / (tas + 273.15)

        # Calculate potential evapotranspiration (PET)
        PET = 330.2 * Lambda * rho_sat

        return PET

    ######################################
    # Begin simulation
    ######################################

    ################
    # Snowfall
    ################
    # Precipitation is assumed to be entirely snow/rain
    # if temperature is below/above threshold (Ts)
    is_snowfall = tas < Ts

    Ps = (is_snowfall * prcp) + ((1 - is_snowfall) * 0.0)
    Pa = (is_snowfall * 0.0) + ((1 - is_snowfall) * prcp)

    Sp = Sp + Ps

    ################
    # Snowmelt
    ################
    # Snowmelt is assumed to occur if temperature
    # is above a threshold (Tm), but is limited to
    # the volume of the snowpack
    is_snowmelt = tas > Tm
    Ms = is_snowmelt * (2.63 + 2.55 * tas + 0.0912 * tas * Pa) + (
        (1 - is_snowmelt) * 0.0
    )

    snowmelt_gt_snowpack = Ms > Sp
    Ms = (snowmelt_gt_snowpack * Sp) + ((1 - snowmelt_gt_snowpack) * Ms)
    Sp = (snowmelt_gt_snowpack * 0.0) + ((1 - snowmelt_gt_snowpack) * (Sp - Ms))

    #########################
    # Canopy & throughfall
    #########################
    # Maximum canopy storage scales with LAI
    Wi_max = Clai * lai

    # Open water evaporation rate assumed to be PET
    Eow = calculate_potential_evapotranspiration(tas, doy, phi)
    # Canopy evaporation
    Ec = Eow * ((Wi / Wi_max) ** 0.6666667)

    # Throughfall is rainfall minus (canopy storage plus canopy evaporation)
    # Throughfall if zero if all rainfall goes to canopy
    canopy_full = Wi_max < Wi + Pa - Ec
    Pt = (canopy_full * (Pa - Ec - (Wi_max - Wi))) + ((1 - canopy_full) * 0.0)

    # Update canopy storage
    canopy_space = Wi + (Pa - Pt) - Ec <= Wi_max
    canopy_leftover = Wi + (Pa - Pt) - Ec > 0.0

    Wi = (
        ((canopy_space * canopy_leftover) * (Wi + (Pa - Pt) - Ec))
        + ((canopy_space * (1 - canopy_leftover)) * 0.0)
        + ((1 - canopy_space) * Wi_max)
    )

    ########################
    # Evapotranspiration
    ########################
    # Potential ET scales with (annual) coefficient
    PET = Kpet * calculate_potential_evapotranspiration(tas, doy, phi)

    # Calculate actual evapotranspiration
    # Actual ET is limited by water availability (throughfall + snowmelt)
    # otherwise the difference is scaled by drying function
    avail_water = (Pt + Ms) >= PET
    AET = (avail_water * PET) + (
        (1 - avail_water)
        * ((1 - jnp.exp(-alpha * Ws_prev / Wcap)) / (1 - jnp.exp(-alpha)))
        * ((PET - Pt - Ms))
    )

    ################
    # Runoff
    ################
    # HBV direct groundwater recharge (can also be thought of as runoff)
    # scales nonlinearly with saturation in the active zone
    # Direct groundwater recharge (HBV)
    Id = (Pt + Ms) * (Ws_prev / Wcap) ** betaHBV

    ################
    # Soil moisture
    ################
    # Soil surplus is the leftover water after saturating soils
    # It gets partitioned to more runoff and groundwater recharge
    soil_full = Wcap < Ws_prev + (Pt + Ms - Id) - AET
    S = (soil_full * (Ws_prev + (Pt + Ms - Id) - AET - Wcap)) + ((1 - soil_full) * 0.0)

    # Update soil moisture
    Ws_new = Ws_prev + (Pt + Ms - Id) - AET - S

    # Soil moisture must be positive
    Ws_new = jnp.maximum(Ws_new, 0.0)

    # Soil moisture out (+ wilting point)
    Ws_out = Ws_prev + wiltingp

    return (Ws_new, Wi, Sp), Ws_out


@jax.jit
def run_wbm_jax(
    initial_conditions,
    forcing_data,
    parameters,
    constants,
):
    """
    MAIN SOIL MOISTURE SIMULATION CODE

    ** All arguments should be dicts with correct names! **

    initial_conditions: Ws_init, Wi_init, Sp_init
    forcing_data: prcp, tas
    params: Ts, Tm, rootDepth, awCap, wiltingp, GS_start, GS_length, alpha, betaHBV
    constants: lai, Kpet, phi, nt, doy
    """

    ##################################
    # Read arguments
    ##################################
    Ws_init, Wi_init, Sp_init = initial_conditions

    tas, prcp = forcing_data

    Ts, Tm, Clai, awCap, wiltingp, alpha, betaHBV, Kmin, Kmax, Klai = parameters

    rootDepth, lai, phi, doy = constants

    # Soil moisture capacity
    Wcap = awCap * rootDepth / 1000

    # PET coefficicient timeseries
    Kpet = jnp.array([Kmin + (Kmax - Kmin) * (1 - jnp.exp(-Klai * l)) for l in lai])

    # Prepare passing to jax lax scan
    forcing = jnp.stack([tas, prcp, lai, Kpet, doy], axis=1)[1:,]
    params = Ts, Tm, Clai, wiltingp, alpha, betaHBV, Wcap, phi

    update_fn = partial(update_state, params=params)

    # Initial conditions
    init = (Ws_init, Wi_init, Sp_init)

    outs, Ws_out = jax.lax.scan(update_fn, init, forcing)

    return Ws_out
