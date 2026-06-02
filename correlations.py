"""
correlations.py
---------------
Regional empirical PVT correlations for crude oil systems.

Regions / Correlations implemented:
  - Standing (1947)          : Western USA (California)
  - Al-Marhoun (1988)        : Middle East (Saudi Arabia / Gulf)
  - Glaso (1980)             : North Sea
  - Vasquez & Beggs (1980)   : Global / Latin America
  - Petrosky & Farshad (1993): Gulf of Mexico (use within stated range)

Each region provides:
  Pb    — Bubble point pressure            [psia]
  Rs(P) — Solution GOR vs. pressure        [scf/STB]
  Bo(P) — Oil formation volume factor      [RB/STB]
  mu(P) — Oil viscosity vs. pressure       [cp]

Units:
  Pressure    : psia
  Temperature : °F
  GOR / Rs    : scf/STB
  API gravity : °API
  Gas SG      : dimensionless (air = 1)
  Bo          : RB/STB
  Viscosity   : cp

References:
  Standing (1947)         — Drill. & Prod. Prac., API
  Vasquez & Beggs (1980)  — JPT, SPE-6719
  Glaso (1980)            — JPT, SPE-8016
  Al-Marhoun (1988)       — SPE-13718
  Petrosky & Farshad(1993)— SPE-26644
  Beggs & Robinson (1975) — JPT (viscosity base)
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

# Applicability ranges for quick reference
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
    """Warn if inputs fall outside the correlation's development range."""
    r = RANGES.get(region, None)
    if r is None:
        return
    warnings = []
    if not (r["Rs"][0] <= GOR <= r["Rs"][1]):
        warnings.append(f"GOR={GOR} outside range {r['Rs']} scf/STB")
    if not (r["T"][0] <= T <= r["T"][1]):
        warnings.append(f"T={T} outside range {r['T']} °F")
    if not (r["API"][0] <= API <= r["API"][1]):
        warnings.append(f"API={API} outside range {r['API']} °API")
    if warnings:
        print(f"  [WARNING] {REGIONS[region].split('—')[0].strip()} — out-of-range inputs:")
        for w in warnings:
            print(f"    • {w}")



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
    If separator conditions are unknown, pass P_sp=114.7 (correction = 1, no change).
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
    GOR    : float — Surface gas-oil ratio [scf/STB]  (used as Rs at Pb)
    T      : float — Reservoir temperature [°F]
    gas_SG : float — Gas specific gravity (air = 1.0)
    region : str   — Key from REGIONS dict

    Returns
    -------
    Pb : float [psia]
    """
    region = region.lower()
    check_range(GOR, T, API, region)
    oil_SG = 141.5 / (API + 131.5)

    if region == "western_usa":
        # Standing (1947) — sign: 0.00091*T - 0.0125*API
        x = 0.00091 * T - 0.0125 * API
        Pb = 18.2 * ((GOR / gas_SG) ** 0.83 * 10.0 ** x - 1.4)

    elif region == "middle_east":
        # Al-Marhoun (1988) — JPT Vol.40 No.5, May 1988, pp.650-666 (SPE-13718-PA)
        # Coefficients verified from primary source (user-confirmed from paper).
        # Note: an earlier 1985 SPE-13718 conference version used different coefficients
        # (leading coeff 5.38088e-3, different exponents); this is the corrected 1988 JPT version.
        Pb = (1.09373e-4
              * GOR ** 0.5502
              * gas_SG ** (-1.71956)
              * oil_SG ** 2.5486
              * (T + 460.0) ** 2.0967)

    elif region == "north_sea":
        # Glaso (1980)
        F = (GOR / gas_SG) ** 0.816 * T ** 0.172 / API ** 0.989
        log_Pb = 1.7669 + 1.7447 * np.log10(F) - 0.30218 * np.log10(F) ** 2
        Pb = 10.0 ** log_Pb

    elif region == "global":
        # Vasquez & Beggs (1980) — exp form (C3 is the exponential coefficient)
        # gas_SG should be corrected to 100 psia reference via vb_separator_correction()
        # before calling this function; if uncorrected, results may carry systematic error.
        if API <= 30:
            C1, C2, C3 = 0.0362, 1.0937, 25.724
        else:
            C1, C2, C3 = 0.0178, 1.1870, 23.931
        Pb = (GOR / (C1 * gas_SG * np.exp(C3 * API / (T + 460.0)))) ** (1.0 / C2)

    elif region == "gulf_of_mexico":
        # Petrosky & Farshad (1993)
        # Note: results may be unreliable outside the development range.
        x = 4.561e-5 * T ** 1.3911 - 7.916e-4 * API ** 1.5410
        Pb = (GOR ** 0.8439 / (112.727 * gas_SG ** 0.5774)) * 10.0 ** x + 12.34
        if Pb < 0:
            print("  [WARNING] Petrosky & Farshad returned negative Pb — "
                  "inputs likely outside development range. Result clipped to 14.7 psia.")
            Pb = 14.7

    else:
        raise ValueError(f"Unknown region '{region}'. Call list_regions() to see options.")

    return float(max(Pb, 14.7))   # cannot be below atmospheric


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
    P = np.atleast_1d(np.array(P, dtype=float))
    Rs = np.where(P >= Pb, GOR, _rs_below_pb(P, API, T, gas_SG, region))
    return np.maximum(Rs, 0.0)


def _rs_below_pb(P, API, T, gas_SG, region):
    """Compute Rs below bubble point by inverting Pb correlations for Rs."""
    region = region.lower()

    if region == "western_usa":
        # Invert Standing: Rs = gas_SG * [(P/18.2 + 1.4) / 10^x]^(1/0.83)
        x = 0.00091 * T - 0.0125 * API
        Rs = gas_SG * ((P / 18.2 + 1.4) / 10.0 ** x) ** (1.0 / 0.83)

    elif region == "middle_east":
        # Invert Al-Marhoun (1988 JPT) for Rs — same coefficients as bubble_point()
        oil_SG = 141.5 / (API + 131.5)
        denom = (1.09373e-4
                 * gas_SG ** (-1.71956)
                 * oil_SG ** 2.5486
                 * (T + 460.0) ** 2.0967)
        Rs = (P / denom) ** (1.0 / 0.5502)
    elif region == "north_sea":
        # Invert Glaso numerically via F
        # logPb = 1.7669 + 1.7447*logF - 0.30218*logF^2
        # Solve quadratic for logF given logPb = log10(P)
        logP = np.log10(np.maximum(P, 1.0))
        a, b, c_coef = -0.30218, 1.7447, (1.7669 - logP)
        disc = b ** 2 - 4 * a * c_coef
        disc = np.maximum(disc, 0)
        logF = (-b - np.sqrt(disc)) / (2 * a)
        F = 10.0 ** logF
        # F = (Rs/gas_SG)^0.816 * T^0.172 / API^0.989  -> solve for Rs
        Rs = gas_SG * (F * API ** 0.989 / T ** 0.172) ** (1.0 / 0.816)

    elif region == "global":
        # Invert Vasquez & Beggs: Rs = C1 * gas_SG * P^C2 * exp(C3*API/(T+460))
        if API <= 30:
            C1, C2, C3 = 0.0362, 1.0937, 25.724
        else:
            C1, C2, C3 = 0.0178, 1.1870, 23.931
        Rs = C1 * gas_SG * P ** C2 * np.exp(C3 * API / (T + 460.0))

    elif region == "gulf_of_mexico":
        # Invert Petrosky & Farshad: Rs = [gas_SG^0.5774*(Pb+12.34)*112.727/10^x]^(1/0.8439)
        x = 4.561e-5 * T ** 1.3911 - 7.916e-4 * API ** 1.5410
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

    Below Pb : Bo computed from saturated correlation (gas dissolved changes volume).
    Above Pb : Bo corrected for slight compression of undersaturated oil.

    Returns
    -------
    Bo : array [RB/STB]  (always >= 1.0)
    """
    P = np.atleast_1d(np.array(P, dtype=float))
    oil_SG = 141.5 / (API + 131.5)

    Rs_arr = solution_gor(P, API, T, gas_SG, Pb, GOR, region)
    Bo_sat = _bo_saturated(Rs_arr, API, T, gas_SG, oil_SG, region)

    # Bo at bubble point — needed for undersaturated correction reference
    Bo_pb = float(_bo_saturated(np.array([GOR]), API, T, gas_SG, oil_SG, region)[0])
    Bo_unsat = _bo_undersaturated(P, Pb, Bo_pb)

    Bo = np.where(P >= Pb, Bo_unsat, Bo_sat)
    return np.maximum(Bo, 1.0)


def _bo_saturated(Rs, API, T, gas_SG, oil_SG, region):
    """Bo at or below bubble point — gas is partially dissolved."""
    region = region.lower()

    if region == "western_usa":
        # Standing (1947)
        F = Rs * (gas_SG / oil_SG) ** 0.5 + 1.25 * T
        Bo = 0.972 + 1.47e-4 * F ** 1.175

    elif region == "middle_east":
        # Al-Marhoun (1988)
        F = Rs * (gas_SG / oil_SG) ** 0.5
        Bo = 0.497069 + 8.62963e-4 * (T + 460) + 1.82594e-3 * F + 3.18099e-6 * F ** 2

    elif region == "north_sea":
        # Glaso (1980)
        F = Rs * (gas_SG / oil_SG) ** 0.526 + 0.968 * T
        log_Bo_m1 = -6.58511 + 2.91329 * np.log10(F) - 0.27683 * np.log10(F) ** 2
        Bo = 1.0 + 10.0 ** log_Bo_m1

    elif region == "global":
        # Vasquez & Beggs (1980)
        if API <= 30:
            C1, C2, C3 = 4.677e-4, 1.751e-5, -1.811e-8
        else:
            C1, C2, C3 = 4.670e-4, 1.100e-5,  1.337e-9
        Bo = 1.0 + C1 * Rs + C2 * (T - 60) * (API / gas_SG) + C3 * Rs * (T - 60) * (API / gas_SG)

    elif region == "gulf_of_mexico":
        # Petrosky & Farshad (1993)
        F = Rs ** 0.3738 * (gas_SG ** 0.2914 / oil_SG ** 0.6265) + 0.24626 * T ** 0.5371
        Bo = 1.0113 + 7.2046e-5 * F ** 3.0936

    else:
        raise ValueError(f"Unknown region '{region}'.")

    return np.maximum(Bo, 1.0)


def _bo_undersaturated(P, Pb, Bo_pb, co=5e-6):
    """
    Bo above bubble point using isothermal compressibility.

    co : oil compressibility [psi^-1] — typical light crude ~5e-6.
         A fuller correlation (Vasquez & Beggs) ties co to Rs, T, API.
    """
    return Bo_pb * np.exp(-co * (P - Pb))


# ---------------------------------------------------------------------------
# OIL VISCOSITY   mu_o(P)  [cp]
# ---------------------------------------------------------------------------

def viscosity(P, API, T, gas_SG, Pb, GOR, region="global"):
    """
    Oil viscosity mu_o as a function of pressure P [psia].

    Uses Beggs & Robinson (1975) as the base for all regions
    (viscosity correlations are less regionally differentiated than Pb/Bo).
    Undersaturated correction from Vasquez & Beggs (1980).

    Returns
    -------
    mu_o : array [cp]
    """
    P = np.atleast_1d(np.array(P, dtype=float))

    # Dead oil (no dissolved gas) — Beggs & Robinson (1975)
    x = 10.0 ** (3.0324 - 0.02023 * API) * T ** (-1.163)
    mu_dead = 10.0 ** x - 1.0

    # Saturated oil (gas dissolved) — Beggs & Robinson (1975)
    Rs_arr = solution_gor(P, API, T, gas_SG, Pb, GOR, region)
    A = 10.715 * (Rs_arr + 100.0) ** (-0.515)
    B = 5.44   * (Rs_arr + 150.0) ** (-0.338)
    mu_sat = A * mu_dead ** B

    # Undersaturated oil (above Pb) — Vasquez & Beggs (1980)
    # Compute mu at Pb directly (avoid recursion)
    A_pb = 10.715 * (GOR + 100.0) ** (-0.515)
    B_pb = 5.44   * (GOR + 150.0) ** (-0.338)
    mu_sat_pb = float(A_pb * mu_dead ** B_pb)
    m = 2.6 * P ** 1.187 * np.exp(-11.513 - 8.98e-5 * P)
    mu_unsat = mu_sat_pb * (P / Pb) ** m

    mu_o = np.where(P >= Pb, mu_unsat, mu_sat)
    return np.maximum(mu_o, 0.05)


# ---------------------------------------------------------------------------
# CONVENIENCE WRAPPER — full PVT table over a pressure sweep
# ---------------------------------------------------------------------------

def pvt_table(API, GOR, T, gas_SG, region="global", P_min=50.0, P_max=None,
             n_points=200, T_sp=60.0, P_sp=114.7):
    """
    Compute Pb, Rs, Bo, and mu_o across a pressure range.

    Parameters
    ----------
    API, GOR, T, gas_SG : fluid description (see bubble_point docstring)
    region   : str   — regional correlation key
    P_min    : float — minimum pressure [psia]  (default 50)
    P_max    : float — maximum pressure [psia]  (default = 1.5 * Pb)
    n_points : int   — number of pressure steps
    T_sp     : float — separator temperature [°F]  (used only for region='global')
    P_sp     : float — separator pressure [psia]   (used only for region='global')
                       default 114.7 psia = no correction applied

    Returns
    -------
    dict with keys: 'region', 'Pb', 'P', 'Rs', 'Bo', 'mu_o', 'gas_SG_used'
    """
    # Apply Vasquez & Beggs separator correction if using global region
    if region == "global" and P_sp != 114.7:
        gas_SG_used = vb_separator_correction(gas_SG, API, T_sp, P_sp)
        print(f"  [V&B separator correction] gas_SG {gas_SG:.4f} → {gas_SG_used:.4f} "
              f"(T_sp={T_sp}°F, P_sp={P_sp} psia)")
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
        "gas_SG_used" : gas_SG_used,
    }


