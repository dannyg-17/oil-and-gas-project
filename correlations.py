"""
correlations.py
---------------
Regional empirical PVT correlations for crude oil systems.

Regions / Correlations implemented
------------------------------------
  western_usa    : Standing (1947)           — Western USA (California)
  middle_east    : Al-Marhoun (1988)         — Middle East (Saudi Arabia / Gulf)
  north_sea      : Glaso (1980)              — North Sea
  global         : Vasquez & Beggs (1980)    — Global / Latin America
  gulf_of_mexico : Petrosky & Farshad (1993) — Gulf of Mexico (range-limited)

Each region provides
  Pb     — Bubble point pressure             [psia]
  Rs(P)  — Solution GOR vs. pressure         [scf/STB]
  Bo(P)  — Oil formation volume factor       [RB/STB]
  mu(P)  — Oil viscosity vs. pressure        [cp]
  co(P)  — Undersaturated oil compressibility [psi⁻¹]  (above Pb only)

Viscosity correlations by region
----------------------------------
  western_usa    : Dead oil — Beal (1946)         (Ahmed 2019 Eq. 2-71a)
                   Saturated — Chew & Connally (1959) (Ahmed 2019 Eq. 2-78)
                   Undersaturated — Vasquez & Beggs (1980)
  middle_east    : Dead oil — Beggs & Robinson (1975)
                   Saturated — Beggs & Robinson (1975)
                   Undersaturated — Vasquez & Beggs (1980)
  north_sea      : Dead oil — Beggs & Robinson (1975)
                   Saturated — Beggs & Robinson (1975)
                   Undersaturated — Vasquez & Beggs (1980)
  global         : Dead oil — Beggs & Robinson (1975)
                   Saturated — Beggs & Robinson (1975)
                   Undersaturated — Vasquez & Beggs (1980)
  gulf_of_mexico : Dead oil — Beggs & Robinson (1975)
                   Saturated — Beggs & Robinson (1975)
                   Undersaturated — Vasquez & Beggs (1980)

Undersaturated oil compressibility
------------------------------------
  Vasquez & Beggs (1980) correlation (SPE-6719):
    co = (5*Rs + 17.2*T - 1180*gas_SG + 12.61*API - 1433) / (1e5 * P)
  Source: Vasquez & Beggs (1980) SPE-6719; Ahmed (2019) Eq. 2-47.

Units
-----
  Pressure    : psia
  Temperature : °F
  GOR / Rs    : scf/STB
  API gravity : °API
  Gas SG      : dimensionless (air = 1)
  Bo          : RB/STB
  Viscosity   : cp
  co          : psi⁻¹

References
----------
  Standing (1947)         — Drill. & Prod. Prac., API
  Vasquez & Beggs (1980)  — JPT, SPE-6719-PA
  Glaso (1980)            — JPT, SPE-8016
  Al-Marhoun (1988)       — SPE-13718-PA, JPT Vol.40 No.5
  Petrosky & Farshad (1993)— SPE-26644
  Beggs & Robinson (1975) — JPT Vol.27 No.9, pp.1140-1141
  Beal (1946)             — Trans. AIME Vol.165, pp.94-115
  Chew & Connally (1959)  — Trans. AIME Vol.216, pp.23-25
  Ahmed (2019)            — Reservoir Engineering Handbook, 5th ed., Elsevier
"""

import numpy as np

# ---------------------------------------------------------------------------
# REGION REGISTRY
# ---------------------------------------------------------------------------

REGIONS = {
    "western_usa"   : "Standing (1947)          — Western USA / California",
    "middle_east"   : "Al-Marhoun (1988)        — Middle East / Saudi Arabia",
    "north_sea"     : "Glaso (1980)             — North Sea",
    "global"        : "Vasquez & Beggs (1980)   — Global / Latin America",
    "gulf_of_mexico": "Petrosky & Farshad (1993)— Gulf of Mexico (range-limited)",
}

# Applicability ranges
RANGES = {
    "western_usa"   : {"Rs": (20, 1425),   "T": (100, 258), "API": (16, 63)},
    "middle_east"   : {"Rs": (26, 1602),   "T": (74,  240), "API": (19, 44)},
    "north_sea"     : {"Rs": (90, 2637),   "T": (80,  280), "API": (22, 48)},
    "global"        : {"Rs": (0,  2199),   "T": (70,  295), "API": (15, 59)},
    "gulf_of_mexico": {"Rs": (217, 1406),  "T": (114, 288), "API": (16, 45)},
}


def list_regions():
    """Print available regions and their correlations."""
    print("\nAvailable regional correlations:")
    for key, desc in REGIONS.items():
        r = RANGES[key]
        print(f"  '{key}'")
        print(f"      {desc}")
        print(f"      Range: Rs {r['Rs']} scf/STB | T {r['T']} °F | API {r['API']} °API")
    print()


def check_range(GOR, T, API, region):
    """
    Check if inputs fall within the correlation's development range.

    Returns a list of warning strings (empty if all inputs are in range).
    Never prints; caller decides how to handle warnings.

    Parameters
    ----------
    GOR    : float — Surface GOR [scf/STB]
    T      : float — Temperature [°F]
    API    : float — API gravity [°API]
    region : str   — region key

    Returns
    -------
    warnings : list of str
    """
    r = RANGES.get(region, None)
    if r is None:
        return []
    warnings = []
    region_label = REGIONS[region].split('—')[0].strip()
    if not (r["Rs"][0] <= GOR <= r["Rs"][1]):
        warnings.append(
            f"{region_label}: GOR={GOR} scf/STB outside development range "
            f"[{r['Rs'][0]}, {r['Rs'][1]}] scf/STB."
        )
    if not (r["T"][0] <= T <= r["T"][1]):
        warnings.append(
            f"{region_label}: T={T}°F outside development range "
            f"[{r['T'][0]}, {r['T'][1]}] °F."
        )
    if not (r["API"][0] <= API <= r["API"][1]):
        warnings.append(
            f"{region_label}: API={API}°API outside development range "
            f"[{r['API'][0]}, {r['API'][1]}] °API."
        )
    return warnings


# ---------------------------------------------------------------------------
# VASQUEZ & BEGGS SEPARATOR CORRECTION
# ---------------------------------------------------------------------------

def vb_separator_correction(gas_SG, API, T_sp, P_sp):
    """
    Correct field gas specific gravity to Vasquez & Beggs reference conditions.

    Vasquez & Beggs (1980) developed their correlation using gas SG measured
    at a reference separator pressure of 100 psia (114.7 psia absolute).
    Field separators operate at varying pressures, so raw gas_SG must be
    normalized before use in V&B correlations.

    Source: Vasquez & Beggs (1980) SPE-6719; Ahmed (2019) Eq. 2-35.

    Parameters
    ----------
    gas_SG : float — Gas SG measured at field separator (air = 1.0)
    API    : float — Oil API gravity [°API]
    T_sp   : float — Separator temperature [°F]
    P_sp   : float — Separator pressure [psia]

    Returns
    -------
    gas_SG_100 : float — Gas SG corrected to 100 psia reference [dimensionless]

    Notes
    -----
    Pass P_sp=114.7 to apply no correction (identity).
    Only applies to the 'global' (Vasquez & Beggs) region.
    """
    gas_SG_100 = gas_SG * (1.0 + 5.912e-5 * API * T_sp * np.log10(P_sp / 114.7))
    return float(gas_SG_100)


# ---------------------------------------------------------------------------
# BUBBLE POINT PRESSURE   Pb  [psia]
# ---------------------------------------------------------------------------

def bubble_point(API, GOR, T, gas_SG, region="global"):
    """
    Estimate bubble point pressure Pb [psia].

    Parameters
    ----------
    API    : float — Oil API gravity [°API]
    GOR    : float — Surface GOR (= Rs at Pb) [scf/STB]
    T      : float — Reservoir temperature [°F]
    gas_SG : float — Gas specific gravity (air = 1.0)
    region : str   — Key from REGIONS dict

    Returns
    -------
    Pb : float [psia]
    """
    region   = region.lower()
    oil_SG   = 141.5 / (API + 131.5)

    if region == "western_usa":
        # Standing (1947) bubble point correlation.
        # Source: Standing, M.B. (1947) Drill. & Prod. Prac., API, p.275.
        # Ahmed (2019) Eq. 2-1.
        x  = 0.00091 * T - 0.0125 * API
        Pb = 18.2 * ((GOR / gas_SG) ** 0.83 * 10.0 ** x - 1.4)

    elif region == "middle_east":
        # Al-Marhoun (1988 JPT) bubble point correlation.
        # Source: Al-Marhoun, M.A. (1988) "PVT Correlations for Middle East
        # Crude Oils", JPT Vol.40 No.5, pp.650-666. SPE-13718-PA.
        # Coefficients are from the corrected 1988 JPT version
        # (different from the 1985 conference paper SPE-13718).
        # Ahmed (2019) Eq. 2-6.
        Pb = (1.09373e-4
              * GOR    ** 0.5502
              * gas_SG ** (-1.71956)
              * oil_SG ** 2.5486
              * (T + 460.0) ** 2.0967)

    elif region == "north_sea":
        # Glaso (1980) bubble point correlation.
        # Source: Glaso, O. (1980) "Generalized Pressure-Volume-Temperature
        # Correlations", JPT Vol.32 No.5, pp.785-795. SPE-8016.
        # Ahmed (2019) Eq. 2-10.
        F      = (GOR / gas_SG) ** 0.816 * T ** 0.172 / API ** 0.989
        log_Pb = 1.7669 + 1.7447 * np.log10(F) - 0.30218 * np.log10(F) ** 2
        Pb     = 10.0 ** log_Pb

    elif region == "global":
        # Vasquez & Beggs (1980) bubble point correlation.
        # Source: Vasquez, M. & Beggs, H.D. (1980) "Correlations for Fluid
        # Physical Property Prediction", JPT Vol.32 No.6, pp.968-970. SPE-6719.
        # gas_SG should be corrected to 100 psia reference before calling.
        # Ahmed (2019) Eq. 2-16.
        if API <= 30:
            C1, C2, C3 = 0.0362, 1.0937, 25.724
        else:
            C1, C2, C3 = 0.0178, 1.1870, 23.931
        Pb = (GOR / (C1 * gas_SG * np.exp(C3 * API / (T + 460.0)))) ** (1.0 / C2)

    elif region == "gulf_of_mexico":
        # Petrosky & Farshad (1993) bubble point correlation.
        # Source: Petrosky, G.E. & Farshad, F.F. (1993) "Pressure-Volume-
        # Temperature Correlations for Gulf of Mexico Crude Oils", SPE-26644.
        # Ahmed (2019) Eq. 2-18.
        x  = 4.561e-5 * T ** 1.3911 - 7.916e-4 * API ** 1.5410
        Pb = (GOR ** 0.8439 / (112.727 * gas_SG ** 0.5774)) * 10.0 ** x + 12.34
        if Pb < 0:
            Pb = 14.7   # clamp; caller should check check_range()

    else:
        raise ValueError(f"Unknown region '{region}'. Call list_regions() to see options.")

    return float(max(Pb, 14.7))


# ---------------------------------------------------------------------------
# SOLUTION GOR   Rs(P)  [scf/STB]
# ---------------------------------------------------------------------------

def solution_gor(P, API, T, gas_SG, Pb, GOR, region="global"):
    """
    Solution GOR Rs as a function of pressure P [psia].

    Rs = GOR  when P >= Pb  (all gas dissolved — undersaturated).
    Rs drops as P falls below Pb (gas comes out of solution).

    Parameters
    ----------
    P      : float or array — Pressure [psia]
    API    : float
    T      : float          — Temperature [°F]
    gas_SG : float
    Pb     : float          — Bubble point pressure [psia]
    GOR    : float          — Surface GOR (= Rs at Pb) [scf/STB]
    region : str

    Returns
    -------
    Rs : array [scf/STB]
    """
    P  = np.atleast_1d(np.array(P, dtype=float))
    Rs = np.where(P >= Pb, GOR, _rs_below_pb(P, API, T, gas_SG, region))
    return np.maximum(Rs, 0.0)


def _rs_below_pb(P, API, T, gas_SG, region):
    """
    Compute Rs below bubble point by inverting Pb correlations for Rs.
    Uses the same sources as bubble_point() — only the algebraic inversion differs.
    """
    region = region.lower()

    if region == "western_usa":
        # Invert Standing (1947): Rs = gas_SG * [(P/18.2 + 1.4) / 10^x]^(1/0.83)
        x  = 0.00091 * T - 0.0125 * API
        Rs = gas_SG * ((P / 18.2 + 1.4) / 10.0 ** x) ** (1.0 / 0.83)

    elif region == "middle_east":
        # Invert Al-Marhoun (1988 JPT) for Rs
        oil_SG = 141.5 / (API + 131.5)
        denom  = (1.09373e-4
                  * gas_SG ** (-1.71956)
                  * oil_SG **  2.5486
                  * (T + 460.0) ** 2.0967)
        Rs = (P / denom) ** (1.0 / 0.5502)

    elif region == "north_sea":
        # Invert Glaso (1980) numerically via F
        # logPb = 1.7669 + 1.7447*logF - 0.30218*logF²
        # Solve quadratic for logF given logPb = log10(P)
        logP  = np.log10(np.maximum(P, 1.0))
        a_q   = -0.30218
        b_q   =  1.7447
        c_q   = 1.7669 - logP
        disc  = b_q ** 2 - 4.0 * a_q * c_q
        disc  = np.maximum(disc, 0.0)
        logF  = (-b_q - np.sqrt(disc)) / (2.0 * a_q)
        F     = 10.0 ** logF
        Rs    = gas_SG * (F * API ** 0.989 / T ** 0.172) ** (1.0 / 0.816)

    elif region == "global":
        # Invert Vasquez & Beggs (1980): Rs = C1*gas_SG*P^C2*exp(C3*API/(T+460))
        if API <= 30:
            C1, C2, C3 = 0.0362, 1.0937, 25.724
        else:
            C1, C2, C3 = 0.0178, 1.1870, 23.931
        Rs = C1 * gas_SG * P ** C2 * np.exp(C3 * API / (T + 460.0))

    elif region == "gulf_of_mexico":
        # Invert Petrosky & Farshad (1993) for Rs
        x  = 4.561e-5 * T ** 1.3911 - 7.916e-4 * API ** 1.5410
        Rs = (gas_SG ** 0.5774 * (P + 12.34) * 112.727 / 10.0 ** x) ** (1.0 / 0.8439)

    else:
        raise ValueError(f"Unknown region '{region}'.")

    return np.maximum(Rs, 0.0)


# ---------------------------------------------------------------------------
# OIL FORMATION VOLUME FACTOR   Bo(P)  [RB/STB]
# ---------------------------------------------------------------------------

def formation_volume_factor(P, API, T, gas_SG, Pb, GOR, region="global"):
    """
    Oil FVF Bo as a function of pressure P [psia].

    Below Pb : Bo from saturated correlation.
    Above Pb : Bo corrected for undersaturated compression using
               Vasquez & Beggs (1980) co via exponential decay.

    Returns
    -------
    Bo : array [RB/STB]  (always >= 1.0)
    """
    P      = np.atleast_1d(np.array(P, dtype=float))
    oil_SG = 141.5 / (API + 131.5)

    Rs_arr  = solution_gor(P, API, T, gas_SG, Pb, GOR, region)
    Bo_sat  = _bo_saturated(Rs_arr, API, T, gas_SG, oil_SG, region)
    Bo_pb   = float(_bo_saturated(np.array([GOR]), API, T, gas_SG, oil_SG, region)[0])
    Bo_uns  = _bo_undersaturated(P, Pb, Bo_pb, API, T, gas_SG, GOR)

    Bo = np.where(P >= Pb, Bo_uns, Bo_sat)
    return np.maximum(Bo, 1.0)


def _bo_saturated(Rs, API, T, gas_SG, oil_SG, region):
    """
    Bo at or below bubble point — gas partially dissolved in oil.
    All equations use the same primary sources as bubble_point().
    """
    region = region.lower()

    if region == "western_usa":
        # Standing (1947) — Drill. & Prod. Prac., API; Ahmed (2019) Eq. 2-27.
        F  = Rs * (gas_SG / oil_SG) ** 0.5 + 1.25 * T
        Bo = 0.972 + 1.47e-4 * F ** 1.175

    elif region == "middle_east":
        # Al-Marhoun (1988 JPT) — SPE-13718-PA; Ahmed (2019) Eq. 2-30.
        F  = Rs * (gas_SG / oil_SG) ** 0.5
        Bo = (0.497069
              + 8.62963e-4 * (T + 460)
              + 1.82594e-3 * F
              + 3.18099e-6 * F ** 2)

    elif region == "north_sea":
        # Glaso (1980) — SPE-8016; Ahmed (2019) Eq. 2-29.
        F        = Rs * (gas_SG / oil_SG) ** 0.526 + 0.968 * T
        log_Bo_1 = -6.58511 + 2.91329 * np.log10(F) - 0.27683 * np.log10(F) ** 2
        Bo       = 1.0 + 10.0 ** log_Bo_1

    elif region == "global":
        # Vasquez & Beggs (1980) — SPE-6719; Ahmed (2019) Eq. 2-31.
        if API <= 30:
            C1, C2, C3 = 4.677e-4, 1.751e-5, -1.811e-8
        else:
            C1, C2, C3 = 4.670e-4, 1.100e-5,  1.337e-9
        Bo = (1.0
              + C1 * Rs
              + C2 * (T - 60) * (API / gas_SG)
              + C3 * Rs * (T - 60) * (API / gas_SG))

    elif region == "gulf_of_mexico":
        # Petrosky & Farshad (1993) — SPE-26644; Ahmed (2019) Eq. 2-32.
        F  = (Rs ** 0.3738 * (gas_SG ** 0.2914 / oil_SG ** 0.6265)
              + 0.24626 * T ** 0.5371)
        Bo = 1.0113 + 7.2046e-5 * F ** 3.0936

    else:
        raise ValueError(f"Unknown region '{region}'.")

    return np.maximum(Bo, 1.0)


def _bo_undersaturated(P, Pb, Bo_pb, API, T, gas_SG, Rs_pb):
    """
    Bo above bubble point using Vasquez & Beggs (1980) compressibility.

    Bo(P) = Bo_pb * exp[-co * (P - Pb)]

    where co is from the Vasquez & Beggs (1980) correlation (see oil_compressibility()).

    Source: Vasquez & Beggs (1980) SPE-6719; Ahmed (2019) Eq. 2-42.
    """
    co = _vb_co_scalar(Rs_pb, T, gas_SG, API, Pb)
    co = max(co, 1e-9)   # physical floor; must be > 0
    return Bo_pb * np.exp(-co * (np.asarray(P) - Pb))


def _vb_co_scalar(Rs, T, gas_SG, API, P):
    """
    Vasquez & Beggs (1980) undersaturated oil compressibility [psi⁻¹].

    Source: Vasquez & Beggs (1980) SPE-6719; Ahmed (2019) Eq. 2-47.

    co = (5*Rs + 17.2*T - 1180*gas_SG + 12.61*API - 1433) / (1e5 * P)

    Parameters: Rs [scf/STB], T [°F], gas_SG [-], API [°API], P [psia]
    Returns: co [psi⁻¹]
    """
    return (5.0 * Rs + 17.2 * T - 1180.0 * gas_SG + 12.61 * API - 1433.0) / (1.0e5 * P)


# ---------------------------------------------------------------------------
# OIL COMPRESSIBILITY   co(P)  [psi⁻¹]   — above Pb only
# ---------------------------------------------------------------------------

def oil_compressibility(P, API, T, gas_SG, Pb, GOR):
    """
    Vasquez & Beggs (1980) undersaturated oil compressibility [psi⁻¹].

    Defined and physically meaningful only above bubble point.
    Below Pb, returns NaN.

    Source: Vasquez & Beggs (1980) SPE-6719; Ahmed (2019) Eq. 2-47.

    co = (5*Rs + 17.2*T - 1180*gas_SG + 12.61*API - 1433) / (1e5 * P)

    Rs used here is the solution GOR at Pb (= GOR, the input surface GOR).
    Above Pb, Rs is constant at GOR.

    Returns
    -------
    co : array [psi⁻¹]  (NaN where P < Pb)
    """
    P   = np.atleast_1d(np.array(P, dtype=float))
    co  = np.full(len(P), np.nan)

    mask = P >= Pb
    if mask.any():
        co[mask] = _vb_co_scalar(GOR, T, gas_SG, API, P[mask])

    return co


# ---------------------------------------------------------------------------
# OIL VISCOSITY   mu_o(P)  [cp]
# ---------------------------------------------------------------------------

# ---- Dead oil viscosity correlations ----------------------------------------

def _beal_dead_oil(API, T):
    """
    Beal (1946) dead oil viscosity correlation.

    Source: Beal, C. (1946) "The Viscosity of Air, Water, Natural Gas, Crude
    Oil and Its Associated Gases at Oil Field Temperatures and Pressures",
    Trans. AIME Vol.165, pp.94-115.
    Ahmed (2019) Eq. 2-71a.

    Equations
    ---------
    a    = 10^(0.43 + 8.33/API)
    μod  = (0.32 + 1.8×10⁷/API^4.53) * (360/(T+200))^a

    Valid range: 10.4 ≤ API ≤ 52.5,  100 ≤ T ≤ 220 °F (Beal 1946).

    Parameters
    ----------
    API : float [°API]
    T   : float [°F]

    Returns
    -------
    mu_dead : float [cp]
    """
    a      = 10.0 ** (0.43 + 8.33 / API)
    mu_dead = (0.32 + 1.8e7 / API ** 4.53) * (360.0 / (T + 200.0)) ** a
    return float(mu_dead)


def _beggs_robinson_dead_oil(API, T):
    """
    Beggs & Robinson (1975) dead oil viscosity correlation.

    Source: Beggs, H.D. & Robinson, J.R. (1975) "Estimating the Viscosity
    of Crude Oil Systems", JPT Vol.27 No.9, pp.1140-1141.
    Ahmed (2019) Eq. 2-68.

    Equations
    ---------
    x    = 10^(3.0324 - 0.02023*API) * T^(-1.163)
    μod  = 10^x - 1.0

    Parameters
    ----------
    API : float [°API]
    T   : float [°F]

    Returns
    -------
    mu_dead : float [cp]
    """
    x = 10.0 ** (3.0324 - 0.02023 * API) * T ** (-1.163)
    return float(10.0 ** x - 1.0)


# ---- Saturated oil viscosity correlations -----------------------------------

def _chew_connally_saturated(Rs, mu_dead):
    """
    Chew & Connally (1959) saturated oil viscosity correlation.

    Source: Chew, J. & Connally, C.A. (1959) "A Viscosity Correlation for
    Gas-Saturated Crude Oils", Trans. AIME Vol.216, pp.23-25.
    Ahmed (2019) Eq. 2-78.

    Equations
    ---------
    a_exp = Rs * (2.2×10⁻⁷ * Rs - 7.4×10⁻⁴)
    b     = 0.68/10^(8.62×10⁻⁵*Rs) + 0.25/10^(1.1×10⁻³*Rs) + 0.062/10^(3.74×10⁻³*Rs)
    μo_sat = 10^a_exp * μod^b

    Note: a_exp is the exponent of base-10, so μo_sat = 10^(a_exp) * μod^b.

    Parameters
    ----------
    Rs      : float or array — solution GOR [scf/STB]
    mu_dead : float — dead oil viscosity [cp]

    Returns
    -------
    mu_sat : float or array [cp]
    """
    Rs     = np.asarray(Rs, dtype=float)
    a_exp  = Rs * (2.2e-7 * Rs - 7.4e-4)
    b      = (0.68  / 10.0 ** (8.62e-5 * Rs)
              + 0.25  / 10.0 ** (1.1e-3  * Rs)
              + 0.062 / 10.0 ** (3.74e-3 * Rs))
    mu_sat = 10.0 ** a_exp * mu_dead ** b
    return mu_sat


def _beggs_robinson_saturated(Rs, mu_dead):
    """
    Beggs & Robinson (1975) saturated oil viscosity correlation.

    Source: Beggs & Robinson (1975) JPT, Vol.27 No.9, pp.1140-1141.
    Ahmed (2019) Eqs. 2-69 to 2-70.

    Equations
    ---------
    A      = 10.715 * (Rs + 100)^(-0.515)
    B      = 5.44   * (Rs + 150)^(-0.338)
    μo_sat = A * μod^B

    Parameters
    ----------
    Rs      : float or array [scf/STB]
    mu_dead : float [cp]

    Returns
    -------
    mu_sat : float or array [cp]
    """
    Rs     = np.asarray(Rs, dtype=float)
    A      = 10.715 * (Rs + 100.0) ** (-0.515)
    B      = 5.44   * (Rs + 150.0) ** (-0.338)
    return A * mu_dead ** B


# ---- Undersaturated viscosity -----------------------------------------------

def _vasquez_beggs_undersaturated(P, Pb, mu_sat_pb):
    """
    Vasquez & Beggs (1980) undersaturated oil viscosity correction.

    Source: Vasquez & Beggs (1980) SPE-6719; Ahmed (2019) Eq. 2-72.

    Equations
    ---------
    m      = 2.6 * P^1.187 * exp(-11.513 - 8.98×10⁻⁵ * P)
    μo_uns = μo_sat(Pb) * (P / Pb)^m

    Parameters
    ----------
    P          : array  — pressure [psia]
    Pb         : float  — bubble point [psia]
    mu_sat_pb  : float  — saturated oil viscosity at Pb [cp]

    Returns
    -------
    mu_unsat : array [cp]
    """
    P  = np.asarray(P, dtype=float)
    m  = 2.6 * P ** 1.187 * np.exp(-11.513 - 8.98e-5 * P)
    return mu_sat_pb * (P / Pb) ** m


# ---- Public viscosity dispatcher --------------------------------------------

def viscosity(P, API, T, gas_SG, Pb, GOR, region="global"):
    """
    Oil viscosity mu_o as a function of pressure P [psia].

    Region-specific dead oil and saturated oil correlations:
      western_usa    : Beal (1946) dead oil + Chew & Connally (1959) saturated
      all others     : Beggs & Robinson (1975) dead oil and saturated
    Undersaturated above Pb: Vasquez & Beggs (1980) for all regions.

    Returns
    -------
    mu_o : array [cp]   (minimum 0.05 cp applied as physical floor)
    """
    P   = np.atleast_1d(np.array(P, dtype=float))
    region = region.lower()

    # --- Dead oil viscosity
    if region == "western_usa":
        mu_dead = _beal_dead_oil(API, T)
    else:
        # Beggs & Robinson (1975) for all other regions
        mu_dead = _beggs_robinson_dead_oil(API, T)

    # --- Saturated oil viscosity (function of Rs at each P)
    Rs_arr = solution_gor(P, API, T, gas_SG, Pb, GOR, region)

    if region == "western_usa":
        mu_sat = _chew_connally_saturated(Rs_arr, mu_dead)
    else:
        mu_sat = _beggs_robinson_saturated(Rs_arr, mu_dead)

    # --- Viscosity at bubble point (for undersaturated correction reference)
    if region == "western_usa":
        mu_sat_pb = float(_chew_connally_saturated(np.array([GOR]), mu_dead)[0])
    else:
        mu_sat_pb = float(_beggs_robinson_saturated(np.array([GOR]), mu_dead)[0])

    # --- Undersaturated viscosity above Pb
    mu_unsat = _vasquez_beggs_undersaturated(P, Pb, mu_sat_pb)

    mu_o = np.where(P >= Pb, mu_unsat, mu_sat)
    return np.maximum(mu_o, 0.05)


# ---------------------------------------------------------------------------
# CONVENIENCE WRAPPER — full PVT table over a pressure sweep
# ---------------------------------------------------------------------------

def pvt_table(API, GOR, T, gas_SG, region="global", P_min=50.0, P_max=None,
              n_points=200, T_sp=60.0, P_sp=114.7):
    """
    Compute Pb, Rs, Bo, mu_o, and co across a pressure range.

    Parameters
    ----------
    API, GOR, T, gas_SG : fluid description
    region   : str   — regional correlation key
    P_min    : float — minimum pressure [psia]  (default 50)
    P_max    : float — maximum pressure [psia]  (default 1.5 × Pb)
    n_points : int   — number of pressure steps
    T_sp     : float — separator temperature [°F]   (V&B correction, global only)
    P_sp     : float — separator pressure [psia]    (default 114.7 = no correction)

    Returns
    -------
    dict with keys:
        'region'       : str
        'Pb'           : float [psia]
        'P'            : array [psia]
        'Rs'           : array [scf/STB]
        'Bo'           : array [RB/STB]
        'mu_o'         : array [cp]
        'co'           : array [psi⁻¹]  (NaN below Pb)
        'gas_SG_used'  : float
        'warnings'     : list of str
    """
    # Collect range warnings
    range_warnings = check_range(GOR, T, API, region)

    # Vasquez & Beggs separator correction (global region only)
    if region == "global" and P_sp != 114.7:
        gas_SG_used = vb_separator_correction(gas_SG, API, T_sp, P_sp)
    else:
        gas_SG_used = gas_SG

    Pb = bubble_point(API, GOR, T, gas_SG_used, region)
    if P_max is None:
        P_max = Pb * 1.5

    P_arr = np.linspace(P_min, P_max, n_points)

    return {
        "region"      : region,
        "Pb"          : Pb,
        "P"           : P_arr,
        "Rs"          : solution_gor(P_arr, API, T, gas_SG_used, Pb, GOR, region),
        "Bo"          : formation_volume_factor(P_arr, API, T, gas_SG_used, Pb, GOR, region),
        "mu_o"        : viscosity(P_arr, API, T, gas_SG_used, Pb, GOR, region),
        "co"          : oil_compressibility(P_arr, API, T, gas_SG_used, Pb, GOR),
        "gas_SG_used" : gas_SG_used,
        "warnings"    : range_warnings,
    }


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  correlations.py — Self-Test")
    print("=" * 60)

    API, GOR, T, gas_SG = 35, 500, 180.0, 0.65

    for reg in REGIONS:
        warns = check_range(GOR, T, API, reg)
        tbl   = pvt_table(API, GOR, T, gas_SG, region=reg)
        print(f"\n  {reg}: Pb={tbl['Pb']:.1f} psia  warnings={len(warns)}")
        mid = len(tbl['P']) // 2
        print(f"    @ P={tbl['P'][mid]:.0f} psia: "
              f"Rs={tbl['Rs'][mid]:.1f}  Bo={tbl['Bo'][mid]:.4f}  "
              f"mu={tbl['mu_o'][mid]:.4f}  co={tbl['co'][mid]}")

    # Beal (1946) dead oil — spot check
    mu_beal = _beal_dead_oil(35, 180.0)
    print(f"\n  Beal (1946) dead oil at API=35, T=180°F: {mu_beal:.4f} cp")

    # Chew & Connally (1959) — spot check
    mu_cc = _chew_connally_saturated(np.array([500.0]), mu_beal)[0]
    print(f"  Chew & Connally (1959) saturated at Rs=500: {mu_cc:.4f} cp")

    # V&B co spot check
    co_val = _vb_co_scalar(Rs=500, T=180, gas_SG=0.65, API=35, P=2000)
    print(f"  V&B co at P=2000 psia, Rs=500: {co_val:.3e} psi^-1")

    print("\ncorrelations.py — all tests passed.")
