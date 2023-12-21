import numpy as np
import math
from numba import njit

################
## Model code ##
################


@njit(parallel=False)
def simulate_gridpoint_water_balance(
    initial_conditions,
    forcing_data,
    params,
    constants,
):
    """
    MAIN SOIL MOISTURE SIMULATION CODE

    ** All arguments should be dicts with correct names! **

    initial_conditions: Ws_init, Wi_init, Sp_init
    forcing_data: prcp, tas
    params: Ts, Tm, rootDepth, awCap, wiltingp, GS_start, GS_length, alpha, betaHBV
    constants: lai, Kc, phi, doy

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
     - wilting_point    (x,y)          wilting point [mm/m]
     - GS_start         (x,y,doy)      planting day / start of growing season [day of year]
     - GS_length        (x,y,doy)      length of growing season [days]
     - Kc               (x,y,doy)      crop scalar factor timeseries []
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

    ##################################
    # Initialization
    ##################################
    # Simulation dimensions
    nx, ny, nt = T.shape

    # Soil moisture capacity
    Wcap = np.empty((nx, ny, 366))

    # Soil moisture capacity and initial values
    for ix in range(nx):
        for iy in range(ny):
            for t in range(366):
                if (t < GS_start[ix, iy]) or (
                    t > (GS_start[ix, iy] + GS_length[ix, iy])
                ):
                    # outside GS
                    Wcap[ix, iy, t] = awCap[ix, iy] * rootDepth_oGS[ix, iy] / 1000
                else:
                    # during GS
                    Wcap[ix, iy, t] = (
                        awCap[ix, iy]
                        * (rootDepth_GS_factor[ix, iy] * rootDepth_oGS[ix, iy])
                        / 1000
                    )

    # Soil moisture [mm]
    Ws = np.empty_like(T)
    Ws[:, :, 0] = Ws_init

    # Soil moisture out [mm]
    Ws_out = np.empty_like(T)
    Ws_out[:, :, 0] = Ws[:, :, 0] + wilting_point

    # # Soil moisture fraction
    # Ws_frac = np.empty_like(T)
    # Ws_frac[:,:,0] = Ws_frac_init

    # Canopy water storage
    Wi = np.empty((nx, ny, 2))
    Wi[:, :, 0] = Wi_init

    # Snowpack
    Sp = np.empty((nx, ny, 2))
    Sp[:, :, 0] = Sp_init

    ######################################
    # Begin simulation
    ######################################
    for ix in range(nx):
        for iy in range(ny):
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
                if T[ix, iy, t] < Ts:
                    Ps = P[ix, iy, t]
                    Pa = 0
                    Sp[ix, iy, t2] = Sp[ix, iy, t2o] + Ps
                else:
                    Pa = P[ix, iy, t]
                    Sp[ix, iy, t2] = Sp[ix, iy, t2o]

                ################
                # Snowmelt
                ################
                # Snowmelt is assumed to occur if temperature
                # is above a threshold (Tm), but is limited to
                # the volume of the snowpack
                if T[ix, iy, t] > Tm:
                    Ms = 2.63 + 2.55 * T[ix, iy, t] + 0.0912 * T[ix, iy, t] * Pa
                    if Ms > Sp[ix, iy, t2]:
                        Ms = Sp[ix, iy, t2]
                        Sp[ix, iy, t2] = 0
                    else:
                        Sp[ix, iy, t2] = Sp[ix, iy, t2] - Ms
                else:
                    Ms = 0.0

                #########################
                # Canopy & throughfall
                #########################
                # Maximum canopy storage scales with LAI
                Wi_max = 0.25 * lai[ix, iy, doy[t]]

                # Open water evaporation rate assumed to be PET
                Eow = calculate_potential_evapotranspiration(
                    T[ix, iy, t], doy[t], phi[iy]
                )
                # Canopy evaporation
                Ec = Eow * ((Wi[ix, iy, t2o] / Wi_max) ** 0.6666667)

                # Throughfall is rainfall minus (canopy storage plus canopy evaporation)
                # Throughfall if zero if all rainfall goes to canopy
                if Wi_max < Pa + Wi[ix, iy, t2o] - Ec:
                    Pt = Pa - Ec - (Wi_max - Wi[ix, iy, t2o])
                else:
                    Pt = 0

                # Update canopy storage
                if Wi[ix, iy, t2o] + (Pa - Pt) - Ec <= Wi_max:
                    if Wi[ix, iy, t2o] + (Pa - Pt) - Ec > 0.0:
                        Wi[ix, iy, t2] = Wi[ix, iy, t2o] + (Pa - Pt) - Ec
                    else:
                        Wi[ix, iy, t2] = 0.0
                else:
                    Wi[ix, iy, t2] = Wi_max

                ########################
                # Evapotranspiration
                ########################
                # Potential ET scales with (annual) crop-specific coefficient
                PET = Kc[ix, iy, doy[t]] * calculate_potential_evapotranspiration(
                    T[ix, iy, t], doy[t], phi[iy]
                )

                # Calculate actual evapotranspiration
                # Actual ET is limited by water availability (throughfall + snowmelt)
                # otherwise the difference is scaled by drying function
                if (Pt + Ms) >= PET:
                    AET = PET
                else:
                    g = (
                        1
                        - np.exp(
                            -alpha[ix, iy] * Ws[ix, iy, t - 1] / Wcap[ix, iy, doy[t]]
                        )
                    ) / (1 - np.exp(-alpha[ix, iy]))
                    AET = g * (PET - Pt - Ms)

                ################
                # Runoff
                ################
                # HBV direct groundwater recharge (can also be thought of as runoff)
                # scales nonlinearly with saturation in the active zone
                # Direct groundwater recharge (HBV)
                Id = (Pt + Ms) * (Ws[ix, iy, t - 1] / Wcap[ix, iy, doy[t]]) ** betaHBV[
                    ix, iy
                ]

                ################
                # Soil moisture
                ################
                # Soil surplus is the leftover water after saturating soils
                # It gets partitioned to more runoff and groundwater recharge
                if Wcap[ix, iy, doy[t]] < Ws[ix, iy, t - 1] + (Pt + Ms - Id) - AET:
                    S = Ws[ix, iy, t - 1] + (Pt + Ms - Id) - AET - Wcap[ix, iy, doy[t]]
                else:
                    S = 0

                # Update soil moisture
                Ws[ix, iy, t] = Ws[ix, iy, t - 1] + (Pt + Ms - Id) - AET - S

                # Soil moisture must be positive
                if Ws[ix, iy, t] < 0:
                    Ws[ix, iy, t] = 0.0

                # Soil wetness
                # Ws_frac[ix,iy,t] = (Ws[ix,iy,t] + wilting_point_mm) / (Wcap[ix,iy,doy[t]] + wilting_point_mm)

                # Soil moisture out (+ wilting point)
                Ws_out[ix, iy, t] = Ws[ix, iy, t] + wilting_point[ix, iy]

    # Return
    return Ws_out
