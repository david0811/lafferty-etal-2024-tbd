import numpy as np
import math
from numba import njit

################
## Model code ##
################


def run_wbm(
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

    Inputs:
     - INPUT NAME       DIMENSIONS     DESCRIPTION [UNITS]
     - Ws_frac_init     (x,y)          initial soil moisture content [mm]
     - Wi_init          (x,y)          initial canopy water storage [mm]
     - Sp_init          (x,y)          initial snowpack [mm]
     - Wg_init          (x,y)          initial groundwater [mm]
     - prcp             (x,y,t)        daily precipitation timeseries [mm]
     - tas              (x,y,t)        daily mean temperature timeseries [C]
     - Ts               (x,y)          snowfall threshold [C]
     - Tm               (x,y)          snowmelt threshold [C]
     - rootDepth        (x,y)          rooting depth / depth of active layer [mm]
     - awCap            (x,y)          available water capacity [mm/m]
     - wiltingp
     - GS_start         (x,y,doy)      planting day / start of growing season [day of year]
     - GS_length        (x,y,doy)      length of growing season [days]
     - Kpet               (x,y,doy)      crop scalar factor timeseries []
     - lai              (x,y,doy)      leaf area index timeseries []
     - alpha            (x,y)          drying function scale parameter []
     - betaHBV          (x,y)          HBV direct recharge parameter []
     - phi              (y)            latitude [deg]
     - doy              (t)            day of year []

    Outputs:
     - Ws: daily soil moisture content timeseries [mm]

    NOTES:
     -
    """

    ##################################
    # Read arguments
    ##################################
    Ws_init = initial_conditions["Ws_init"]
    Wi_init = initial_conditions["Wi_init"]
    Sp_init = initial_conditions["Sp_init"]

    prcp = forcing_data["prcp"]
    tas = forcing_data["tas"]

    # Ts = parameters["Ts"]
    # Tm = parameters["Tm"]
    # Clai = parameters["Clai"]
    # awCap = parameters["awCap"]
    # wiltingp = parameters["wiltingp"]
    # # GS_start = parameters["GS_start"]
    # # GS_length = parameters["GS_length"]
    # alpha = parameters["alpha"]
    # betaHBV = parameters["betaHBV"]
    # Kmin = parameters["Kmin"]
    # Kmax = parameters["Kmax"]
    # Klai = parameters["Klai"]
    Ts, Tm, Clai, awCap, wiltingp, alpha, betaHBV, Kmin, Kmax, Klai = parameters

    rootDepth = constants["rootDepth"]
    lai = constants["lai"]
    phi = constants["phi"]
    nt = constants["nt"]
    doy = constants["doy"]

    #################################
    # Setup arrays
    #################################
    # Soil moisture capacity
    Wcap = awCap * rootDepth / 1000

    # Soil moisture [mm]
    Ws = np.empty(nt)
    Ws[0] = Ws_init

    # Soil moisture out [mm]
    Ws_out = np.empty(nt)
    Ws_out[0] = Ws[0] + wiltingp

    # Canopy water storage
    Wi = np.empty(2)
    Wi[0] = Wi_init

    # Snowpack
    Sp = np.empty(2)
    Sp[0] = Sp_init

    # PET coefficicient timeseries
    Kpet = np.array([Kmin + (Kmax - Kmin) * (1 - np.exp(-Klai * l)) for l in lai])

    #################################
    # Run simulation
    #################################
    return _simulate_water_balance(
        Ws=Ws,
        Ws_out=Ws_out,
        Wi=Wi,
        Sp=Sp,
        Wcap=Wcap,
        prcp=prcp,
        tas=tas,
        Ts=Ts,
        Tm=Tm,
        Clai=Clai,
        wiltingp=wiltingp,
        alpha=alpha,
        betaHBV=betaHBV,
        lai=lai,
        Kpet=Kpet,
        phi=phi,
        nt=nt,
        doy=doy,
    )


# @njit(parallel=False)
def _simulate_water_balance(
    Ws,
    Ws_out,
    Wi,
    Sp,
    Wcap,
    prcp,
    tas,
    Ts,
    Tm,
    Clai,
    wiltingp,
    alpha,
    betaHBV,
    lai,
    Kpet,
    phi,
    nt,
    doy,
):
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
        delta = -23.44 * math.cos(math.radians((360 / 365) * (doy + 10)))

        # Calculate fractional day length (Lambda)
        Lambda = (1 / math.pi) * math.acos(
            -math.tan(math.radians(phi)) * math.tan(math.radians(delta))
        )

        # Calculate saturation vapor pressure
        if tas > 0:
            Psat = 0.61078 * np.exp((17.26939 * tas) / (tas + 237.3))
        else:
            Psat = 0.61078 * np.exp((21.87456 * tas) / (tas + 265.5))

        # Calculate saturation vapor density (rho_sat)
        rho_sat = (2.167 * Psat) / (tas + 273.15)

        # Calculate potential evapotranspiration (PET)
        PET = 330.2 * Lambda * rho_sat

        return PET

    ######################################
    # Begin simulation
    ######################################
    for t in range(1, nt):
        # t runs from 1 to n_sim
        # t2 and t2o are 0 or 1 oppositely (used for untracked variables)
        t2 = t % 2
        t2o = int(not (t2))

        ################
        # Snowfall
        ################
        # Precipitation is assumed to be entirely snow/rain
        # if temperature is below/above threshold (Ts)
        if tas[t] < Ts:
            Ps = prcp[t]
            Pa = 0
            Sp[t2] = Sp[t2o] + Ps
        else:
            Pa = prcp[t]
            Sp[t2] = Sp[t2o]

        ################
        # Snowmelt
        ################
        # Snowmelt is assumed to occur if temperature
        # is above a threshold (Tm), but is limited to
        # the volume of the snowpack
        if tas[t] > Tm:
            Ms = 2.63 + 2.55 * tas[t] + 0.0912 * tas[t] * Pa
            if Ms > Sp[t2]:
                Ms = Sp[t2]
                Sp[t2] = 0
            else:
                Sp[t2] = Sp[t2] - Ms
        else:
            Ms = 0.0

        #########################
        # Canopy & throughfall
        #########################
        # Maximum canopy storage scales with LAI
        Wi_max = Clai * lai[doy[t]]

        # Open water evaporation rate assumed to be PET
        Eow = calculate_potential_evapotranspiration(tas[t], doy[t], phi)
        # Canopy evaporation
        Ec = Eow * ((Wi[t2o] / Wi_max) ** 0.6666667)

        # Throughfall is rainfall minus (canopy storage plus canopy evaporation)
        # Throughfall if zero if all rainfall goes to canopy
        if Wi_max < Pa + Wi[t2o] - Ec:
            Pt = Pa - Ec - (Wi_max - Wi[t2o])
        else:
            Pt = 0

        # Update canopy storage
        if Wi[t2o] + (Pa - Pt) - Ec <= Wi_max:
            if Wi[t2o] + (Pa - Pt) - Ec > 0.0:
                Wi[t2] = Wi[t2o] + (Pa - Pt) - Ec
            else:
                Wi[t2] = 0.0
        else:
            Wi[t2] = Wi_max

        ########################
        # Evapotranspiration
        ########################
        # Potential ET scales with (annual) coefficient
        PET = Kpet[doy[t]] * calculate_potential_evapotranspiration(tas[t], doy[t], phi)

        # Calculate actual evapotranspiration
        # Actual ET is limited by water availability (throughfall + snowmelt)
        # otherwise the difference is scaled by drying function
        if (Pt + Ms) >= PET:
            AET = PET
        else:
            g = (1 - np.exp(-alpha * Ws[t - 1] / Wcap)) / (1 - np.exp(-alpha))
            AET = g * (PET - Pt - Ms)

        ################
        # Runoff
        ################
        # HBV direct groundwater recharge (can also be thought of as runoff)
        # scales nonlinearly with saturation in the active zone
        # Direct groundwater recharge (HBV)
        Id = (Pt + Ms) * (Ws[t - 1] / Wcap) ** betaHBV

        ################
        # Soil moisture
        ################
        # Soil surplus is the leftover water after saturating soils
        # It gets partitioned to more runoff and groundwater recharge
        if Wcap < Ws[t - 1] + (Pt + Ms - Id) - AET:
            S = Ws[t - 1] + (Pt + Ms - Id) - AET - Wcap
        else:
            S = 0

        # Update soil moisture
        Ws[t] = Ws[t - 1] + (Pt + Ms - Id) - AET - S

        # Soil moisture must be positive
        if Ws[t] < 0:
            Ws[t] = 0.0

        # Soil wetness
        # Ws_frac[ix,iy,t] = (Ws[ix,iy,t] + wiltingp

        # Soil moisture out (+ wilting point)
        Ws_out[t] = Ws[t] + wiltingp

    # Return
    return Ws_out
