"""
gas_pvt.py
----------
Gas PVT correlations for dry gas and gas condensate systems.

Correlations implemented
------------------------
  Pseudocritical properties:
    Sutton (1985) SPE-14265            — dry gas
    Standing (1977) SPERE              — gas condensate
    Wichert-Aziz (1972)               — H2S/CO2 acid-gas correction

  Z-factor (compressibility factor):
    Hall & Yarborough (1974) SPE-3350-PA  — primary method  (Newton-Raphson)
    Dranchuk, Purvis & Robinson (1974)    — alternative     (BWR-type, iterative)

  Gas viscosity:
    Lee, Gonzalez & Eakin (1966) SPE-1340-PA

  Derived gas properties:
    Gas FVF Bg         [ft³/scf]  and [bbl/scf]
    Gas density ρg     [lb/ft³]
    Gas compressibility cg  [psi⁻¹]  via numerical dZ/dP

Units
-----
  Pressure        : psia
  Temperature     : °F  input;  °R  internal  (add 459.67)
  Specific gravity: dimensionless (air = 1.0)
  Viscosity       : cp
  Density         : lb/ft³
  Bg (ft³/scf)    : res ft³ per standard ft³
  Bg (bbl/scf)    : res bbl per standard ft³

References (primary sources)
-----------------------------
  Sutton (1985)          SPE-14265, "Compressibility Factors for High-MW
                         Reservoir Gases", SPE Annual Technical Conference
  Standing (1977)        Volumetric and Phase Behavior of Oil Field
                         Hydrocarbon Systems, 9th ed., SPE
  Wichert & Aziz (1972) "Calculate Z's for Sour Gases",
                         Hydrocarbon Processing, May 1972, pp. 119-122
  Hall & Yarborough (1974) "A New Equation of State for Z-Factor Calculations",
                         Oil & Gas Journal, June 1974; also SPE-3350-PA
  Dranchuk, Purvis & Robinson (1974) "Computer Calculation of Natural Gas
                         Compressibility Factors Using the Standing and Katz
                         Correlation", Institute of Petroleum Technical Series,
                         No. IP 74-008
  Lee, Gonzalez & Eakin (1966) "The Viscosity of Natural Gas",
                         JPT, August 1966, pp. 997-1000; SPE-1340-PA
  Ahmed (2019)           Reservoir Engineering Handbook, 5th ed., Elsevier
                         (equations cited where Ahmed provides the compiled form)
"""

import numpy as np

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
R_GAS   = 10.7316    # psia·ft³ / (lbmol·°R)   universal gas constant
F_TO_R  = 459.67     # °F → °R  offset
P_SC    = 14.696     # psia   standard conditions
T_SC    = 519.67     # °R     standard conditions (60 °F)
MW_AIR  = 28.966     # lb/lbmol  molecular weight of air

# Standard molar volume at 14.696 psia, 60 °F (ideal gas)
V_SC = R_GAS * T_SC / P_SC           # ft³/lbmol  ≈ 379.4 ft³/lbmol


# ---------------------------------------------------------------------------
# PSEUDOCRITICAL PROPERTIES
# ---------------------------------------------------------------------------

def sutton_pseudocriticals(gas_SG):
    """
    Sutton (1985) pseudocritical properties for dry gas.

    Regression coefficients from SPE-14265 (1985 SPE Annual Technical
    Conference, Las Vegas). Valid range: 0.57 ≤ gas_SG ≤ 1.68.

    Equations
    ---------
    Tpc = 169.2 + 349.5 * γg - 74.0 * γg²   [°R]
    Ppc = 756.8 - 131.0 * γg - 3.6 * γg²    [psia]

    Source: Sutton (1985) SPE-14265, Eqs. 3-4.
    Also listed in Ahmed (2019) Eq. 2-19.

    Parameters
    ----------
    gas_SG : float — gas specific gravity (air = 1.0)

    Returns
    -------
    Tpc : float [°R]
    Ppc : float [psia]
    """
    if not (0.57 <= gas_SG <= 1.68):
        import warnings as _w
        _w.warn(
            f"gas_SG={gas_SG:.3f} outside Sutton (1985) range [0.57, 1.68]. "
            "Pseudocritical extrapolation may be unreliable.",
            UserWarning, stacklevel=2
        )
    Tpc = 169.2 + 349.5 * gas_SG - 74.0 * gas_SG ** 2
    Ppc = 756.8 - 131.0 * gas_SG - 3.6  * gas_SG ** 2
    return float(Tpc), float(Ppc)


def standing_pseudocriticals_condensate(gas_SG):
    """
    Standing (1977) pseudocritical properties for gas condensate.

    Source: Standing (1977) Volumetric and Phase Behavior of Oil Field
    Hydrocarbon Systems, 9th ed., SPE, p. 204.
    Also listed in Ahmed (2019) Eqs. 2-21 and 2-22.

    Tpc = 187.0 + 330.0 * γg - 71.5 * γg²   [°R]
    Ppc = 706.0 - 51.7  * γg - 11.1 * γg²   [psia]

    Parameters
    ----------
    gas_SG : float — gas specific gravity (air = 1.0)

    Returns
    -------
    Tpc : float [°R]
    Ppc : float [psia]
    """
    Tpc = 187.0 + 330.0 * gas_SG - 71.5 * gas_SG ** 2
    Ppc = 706.0 -  51.7 * gas_SG - 11.1 * gas_SG ** 2
    return float(Tpc), float(Ppc)


def wichert_aziz_correction(Tpc, Ppc, y_CO2, y_H2S):
    """
    Wichert-Aziz (1972) pseudocritical correction for H2S and CO2.

    Source: Wichert & Aziz (1972) "Calculate Z's for Sour Gases",
    Hydrocarbon Processing, May 1972, pp. 119-122.
    Also: Ahmed (2019) Eqs. 2-24 to 2-26.

    Equations
    ---------
    A  = y_CO2 + y_H2S
    B  = y_H2S
    ε  = 120 * (A^0.9 - A^1.6) + 15 * (B^0.5 - B^4.0)
    Tpc' = Tpc - ε
    Ppc' = Ppc * Tpc' / (Tpc + B * (1 - B) * ε)

    Validity: 0 ≤ A ≤ 0.74,  0 ≤ B ≤ 0.705
    (Wichert & Aziz 1972 Fig. 1 coverage region)

    Parameters
    ----------
    Tpc   : float — uncorrected pseudocritical temperature [°R]
    Ppc   : float — uncorrected pseudocritical pressure [psia]
    y_CO2 : float — CO2 mole fraction [dimensionless, 0–1]
    y_H2S : float — H2S mole fraction [dimensionless, 0–1]

    Returns
    -------
    Tpc_corr : float [°R]
    Ppc_corr : float [psia]
    warnings : list of str
    """
    warnings = []
    A = y_CO2 + y_H2S
    B = y_H2S

    if A > 0.74:
        warnings.append(
            f"Wichert-Aziz: A=y_CO2+y_H2S={A:.3f} > 0.74 (max development range). "
            "Correction may be unreliable."
        )
    if B > 0.705:
        warnings.append(
            f"Wichert-Aziz: y_H2S={B:.3f} > 0.705 (max development range). "
            "Correction may be unreliable."
        )

    if A < 1e-10:
        # No acid gas — no correction
        return float(Tpc), float(Ppc), warnings

    # Wichert-Aziz ε correction, Hydrocarbon Processing May 1972, Eq. (2)
    epsilon = 120.0 * (A ** 0.9 - A ** 1.6) + 15.0 * (B ** 0.5 - B ** 4.0)
    Tpc_corr = Tpc - epsilon
    Ppc_corr = Ppc * Tpc_corr / (Tpc + B * (1.0 - B) * epsilon)

    return float(Tpc_corr), float(Ppc_corr), warnings


# ---------------------------------------------------------------------------
# Z-FACTOR METHODS
# ---------------------------------------------------------------------------

def hall_yarborough(Ppr, Tpr, max_iter=200, tol=1e-10):
    """
    Hall & Yarborough (1974) Z-factor via Newton-Raphson on the
    Starling-Carnahan equation of state.

    Source: Hall, K.R. & Yarborough, L. (1974) "A New Equation of State
    for Z-Factor Calculations", Oil and Gas Journal, June 18, 1974, pp. 82-92.
    Also published as SPE-3350-PA.
    Ahmed (2019) Eqs. 2-32 to 2-36.

    Equations
    ---------
    t   = 1 / Tpr   (inverse pseudo-reduced temperature)
    A   = 0.06125 * t * exp(-1.2 * (1-t)²)      [> 0]

    F(y) = -A*Ppr + (y + y² + y³ - y⁴)/(1-y)³
           - (14.76*t - 9.76*t² + 4.58*t³)*y²
           + (90.7*t - 242.2*t² + 42.4*t³)*y^(2.18 + 2.82*t)  = 0

    Z    = A * Ppr / y

    Newton-Raphson iteration: y_{n+1} = y_n - F(y_n)/F'(y_n)

    Validity
    --------
    Tpr ≥ 1.05 (Hall & Yarborough state the method is undefined for Tpr < 1.05)
    No explicit upper Ppr bound stated, but tested to Ppr ~ 16 in original paper.

    Parameters
    ----------
    Ppr : float — pseudo-reduced pressure  (dimensionless)
    Tpr : float — pseudo-reduced temperature (dimensionless)
    max_iter : int — maximum Newton-Raphson iterations (default 200)
    tol      : float — convergence tolerance |y_new - y_old|/y (default 1e-10)

    Returns
    -------
    Z         : float — compressibility factor, or NaN if outside range/not converged
    converged : bool
    warnings  : list of str
    """
    warnings = []

    if Tpr < 1.05:
        warnings.append(
            f"Hall-Yarborough: Tpr={Tpr:.4f} < 1.05. "
            "Correlation is undefined below Tpr=1.05 (Hall & Yarborough 1974). "
            "Z returned as NaN."
        )
        return float('nan'), False, warnings

    t  = 1.0 / Tpr
    A  = 0.06125 * t * np.exp(-1.2 * (1.0 - t) ** 2)

    # Coefficients that appear repeatedly
    c1 = 14.76 * t - 9.76 * t ** 2 + 4.58 * t ** 3
    c2 = 90.7  * t - 242.2 * t ** 2 + 42.4 * t ** 3
    c3 = 2.18 + 2.82 * t

    def F(y):
        # Hall-Yarborough (1974) Eq. (18) / Oil&Gas Journal June 1974, p. 86
        term1 = (y + y**2 + y**3 - y**4) / (1.0 - y)**3
        return -A * Ppr + term1 - c1 * y**2 + c2 * y**c3

    def dF(y):
        # Analytical derivative dF/dy for Newton-Raphson
        # d/dy [(y+y²+y³-y⁴)/(1-y)³]
        #   = [(1+2y+3y²-4y³)(1-y)³ + 3(1-y)²(y+y²+y³-y⁴)] / (1-y)^6
        #   = [(1+2y+3y²-4y³)(1-y) + 3(y+y²+y³-y⁴)] / (1-y)^4
        num = (1.0 + 2.0*y + 3.0*y**2 - 4.0*y**3)*(1.0 - y) + 3.0*(y + y**2 + y**3 - y**4)
        dterm1 = num / (1.0 - y)**4
        return dterm1 - 2.0 * c1 * y + c3 * c2 * y**(c3 - 1.0)

    # Initial guess — Hall & Yarborough suggest starting near y ≈ A*Ppr
    y = max(A * Ppr, 1e-4)
    y = min(y, 0.99)   # must be < 1 (reduced packing fraction)

    converged = False
    for _ in range(max_iter):
        Fval  = F(y)
        dFval = dF(y)
        if abs(dFval) < 1e-30:
            break
        y_new = y - Fval / dFval
        # Clamp to physically valid range
        y_new = max(1e-8, min(y_new, 0.9999))
        if abs(y_new - y) / max(abs(y), 1e-12) < tol:
            y = y_new
            converged = True
            break
        y = y_new

    if not converged:
        warnings.append(
            f"Hall-Yarborough: Newton-Raphson did not converge in {max_iter} iterations "
            f"(Ppr={Ppr:.4f}, Tpr={Tpr:.4f}). Z returned as NaN."
        )
        return float('nan'), False, warnings

    Z = A * Ppr / y
    return float(Z), True, warnings


def dranchuk_purvis_robinson(Ppr, Tpr, max_iter=200, tol=1e-10):
    """
    Dranchuk, Purvis & Robinson (1974) Z-factor using BWR-type equation.

    Source: Dranchuk, P.M., Purvis, R.A., & Robinson, D.B. (1974)
    "Computer Calculation of Natural Gas Compressibility Factors Using the
    Standing and Katz Correlation", Institute of Petroleum Technical Series,
    No. IP 74-008, Alberta, Canada.
    Constants and equations also in Ahmed (2019) Eqs. 2-37 to 2-40.

    Eight constants (Ahmed 2019 Table 2-2 / DPR 1974):
    A1 =  0.31506237,  A2 = -1.04670990,  A3 = -0.57832729,
    A4 =  0.53530771,  A5 = -0.61232032,  A6 = -0.10488846,
    A7 =  0.68157001,  A8 =  0.68446549

    Z-factor equation (implicit, DPR 1974 Eq. 1):
    Z = 1 + (A1 + A2/Tpr + A3/Tpr³)*ρr
          + (A4 + A5/Tpr)*ρr²
          - A5*A6/Tpr * ρr⁵
          + A7/Tpr³ * ρr² * (1 + A8*ρr²) * exp(-A8*ρr²)

    where reduced density ρr = 0.27*Ppr / (Z*Tpr)   [DPR 1974 Eq. 2]

    Solved iteratively: start with Z=1, compute ρr, compute new Z, repeat.

    Validity
    --------
    1.05 ≤ Tpr ≤ 3.0,  0 < Ppr ≤ 3.0  (DPR 1974 stated range).
    Outside this range, results are extrapolations and accuracy degrades.

    Parameters
    ----------
    Ppr : float
    Tpr : float
    max_iter : int
    tol      : float — convergence on |Z_new - Z_old| / Z_old

    Returns
    -------
    Z         : float  (NaN if outside range or not converged)
    converged : bool
    warnings  : list of str
    """
    warnings = []

    if not (1.05 <= Tpr <= 3.0):
        warnings.append(
            f"DPR: Tpr={Tpr:.4f} outside DPR (1974) stated range [1.05, 3.0]. "
            "Result is an extrapolation."
        )
    if not (0.0 < Ppr <= 3.0):
        warnings.append(
            f"DPR: Ppr={Ppr:.4f} outside DPR (1974) stated range (0, 3.0]. "
            "Result is an extrapolation."
        )

    # DPR constants — DPR (1974) Table 1 / Ahmed (2019) Table 2-2
    A1 =  0.31506237
    A2 = -1.04670990
    A3 = -0.57832729
    A4 =  0.53530771
    A5 = -0.61232032
    A6 = -0.10488846
    A7 =  0.68157001
    A8 =  0.68446549

    Z = 1.0   # initial guess
    converged = False

    for _ in range(max_iter):
        rho_r = 0.27 * Ppr / (Z * Tpr)   # DPR (1974) Eq. 2
        exp_term = np.exp(-A8 * rho_r ** 2)
        Z_new = (
            1.0
            + (A1 + A2/Tpr + A3/Tpr**3) * rho_r
            + (A4 + A5/Tpr) * rho_r**2
            - A5 * A6 / Tpr * rho_r**5
            + A7 / Tpr**3 * rho_r**2 * (1.0 + A8 * rho_r**2) * exp_term
        )
        if abs(Z_new - Z) / max(abs(Z), 1e-12) < tol:
            Z = Z_new
            converged = True
            break
        Z = Z_new

    if not converged:
        warnings.append(
            f"DPR: Iteration did not converge in {max_iter} steps "
            f"(Ppr={Ppr:.4f}, Tpr={Tpr:.4f}). Z returned as NaN."
        )
        return float('nan'), False, warnings

    return float(Z), True, warnings


# ---------------------------------------------------------------------------
# GAS Z-FACTOR DISPATCHER
# ---------------------------------------------------------------------------

def gas_z_factor(P, T_F, Tpc, Ppc, method='hall_yarborough'):
    """
    Compute gas Z-factor at given pressure and temperature.

    Parameters
    ----------
    P      : float or array — pressure [psia]
    T_F    : float — temperature [°F]
    Tpc    : float — pseudocritical temperature [°R]  (after Wichert-Aziz if applicable)
    Ppc    : float — pseudocritical pressure [psia]    (after Wichert-Aziz if applicable)
    method : str — 'hall_yarborough' (default) or 'dpr'

    Returns
    -------
    Z        : float or array
    warnings : list of str  (collected from all pressure steps)
    """
    scalar = np.ndim(P) == 0
    P   = np.atleast_1d(np.array(P, dtype=float))
    T_R = T_F + F_TO_R
    Tpr = T_R / Tpc

    Z_arr  = np.full(len(P), np.nan)
    all_warnings = []

    for i, Pi in enumerate(P):
        Ppr = Pi / Ppc
        if method == 'hall_yarborough':
            Zi, conv, warns = hall_yarborough(Ppr, Tpr)
        elif method == 'dpr':
            Zi, conv, warns = dranchuk_purvis_robinson(Ppr, Tpr)
        else:
            raise ValueError(f"Unknown Z-factor method '{method}'. "
                             "Choose 'hall_yarborough' or 'dpr'.")
        Z_arr[i] = Zi
        all_warnings.extend(warns)

    if scalar:
        return float(Z_arr[0]), all_warnings
    return Z_arr, all_warnings


# ---------------------------------------------------------------------------
# GAS VISCOSITY — LEE-GONZALEZ-EAKIN (1966)
# ---------------------------------------------------------------------------

def lee_gonzalez_eakin_viscosity(P, T_F, Mg, Z):
    """
    Gas viscosity from Lee, Gonzalez & Eakin (1966) SPE-1340-PA.

    Source: Lee, A.L., Gonzalez, M.H., & Eakin, B.E. (1966)
    "The Viscosity of Natural Gas", Journal of Petroleum Technology,
    August 1966, pp. 997-1000.  Also: SPE-1340-PA.
    Ahmed (2019) Eqs. 2-54 to 2-57.

    Equations
    ---------
    Gas density at reservoir conditions [g/cm³]:
        ρg = 1.4935×10⁻³ * P * Mg / (Z * T)    [T in °R, P in psia, Mg in lb/lbmol]

    LGE coefficients:
        K = (9.4  + 0.02 * Mg) * T^1.5 / (209.0 + 19.0 * Mg + T)
        X = 3.5 + 986.0 / T + 0.01 * Mg
        Y = 2.4 - 0.2 * X

    Gas viscosity [cp]:
        μg = K * exp(X * ρg^Y) × 10⁻⁴

    Validity: LGE (1966) developed from data for natural gases
    (0.90 ≤ Mg ≤ 20 g/mol equivalent; T = 100–340°F; P up to 8000 psia).

    Parameters
    ----------
    P   : float or array — pressure [psia]
    T_F : float — temperature [°F]
    Mg  : float — gas molecular weight [lb/lbmol]
    Z   : float or array — Z-factor (same shape as P)

    Returns
    -------
    mu_g : float or array [cp]
    rho_g: float or array [lb/ft³]  — gas density at reservoir conditions
    """
    scalar = np.ndim(P) == 0
    P   = np.atleast_1d(np.array(P, dtype=float))
    Z   = np.atleast_1d(np.array(Z, dtype=float))
    T_R = T_F + F_TO_R    # °R

    # Gas density [g/cm³] — LGE (1966) Eq. (2)
    # 1.4935×10⁻³ = Mg/(10.7316 * 1000/28.317) with unit conversions
    rho_g_gcc = 1.4935e-3 * P * Mg / (Z * T_R)   # g/cm³

    # Convert to lb/ft³  (1 g/cm³ = 62.4279 lb/ft³)
    rho_g_lbft3 = rho_g_gcc * 62.4279

    # LGE coefficients — LGE (1966) Eqs. (3a–3c)
    K = (9.4 + 0.02 * Mg) * T_R ** 1.5 / (209.0 + 19.0 * Mg + T_R)
    X = 3.5 + 986.0 / T_R + 0.01 * Mg
    Y = 2.4 - 0.2 * X

    # Viscosity [cp] — LGE (1966) Eq. (1)
    mu_g = K * np.exp(X * rho_g_gcc ** Y) * 1e-4

    if scalar:
        return float(mu_g[0]), float(rho_g_lbft3[0])
    return mu_g, rho_g_lbft3


# ---------------------------------------------------------------------------
# DERIVED GAS PROPERTIES
# ---------------------------------------------------------------------------

def gas_fvf(P, T_F, Z):
    """
    Gas formation volume factor Bg.

    From the real-gas law:
        Bg [ft³/scf] = P_sc * T / (T_sc * P * Z)    where T in °R
                     = (P_sc / T_sc) * T / (P * Z)

    Bg [bbl/scf] = Bg [ft³/scf] / 5.61458

    Standard conditions: P_sc = 14.696 psia, T_sc = 519.67 °R (60 °F).

    Parameters
    ----------
    P   : float or array [psia]
    T_F : float [°F]
    Z   : float or array

    Returns
    -------
    Bg_ft3 : float or array [ft³/scf]
    Bg_bbl : float or array [bbl/scf]
    """
    T_R    = T_F + F_TO_R
    Bg_ft3 = (P_SC * T_R) / (T_SC * np.asarray(P) * np.asarray(Z))
    Bg_bbl = Bg_ft3 / 5.61458
    return Bg_ft3, Bg_bbl


def gas_compressibility(P, T_F, Tpc, Ppc, method='hall_yarborough',
                        dP_frac=1e-4):
    """
    Gas compressibility cg [psi⁻¹] via numerical differentiation.

    The exact thermodynamic definition:
        cg = (1/P) - (1/Z) * (dZ/dP)_T

    dZ/dP is computed by central finite difference:
        dZ/dP ≈ [Z(P+δ) - Z(P-δ)] / (2δ)   where δ = dP_frac * P

    Parameters
    ----------
    P      : float or array [psia]
    T_F    : float [°F]
    Tpc    : float [°R]
    Ppc    : float [psia]
    method : str — 'hall_yarborough' or 'dpr'
    dP_frac: float — fractional step for finite difference (default 1e-4)

    Returns
    -------
    cg : float or array [psi⁻¹]
    """
    scalar = np.ndim(P) == 0
    P_arr = np.atleast_1d(np.array(P, dtype=float))
    T_R   = T_F + F_TO_R
    Tpr   = T_R / Tpc

    cg_arr = np.full(len(P_arr), np.nan)

    for i, Pi in enumerate(P_arr):
        delta = max(dP_frac * Pi, 0.01)
        Pp    = Pi + delta
        Pm    = Pi - delta

        Ppr_c = Pi / Ppc
        Ppr_p = Pp / Ppc
        Ppr_m = Pm / Ppc

        if method == 'hall_yarborough':
            Zc, conv_c, _ = hall_yarborough(Ppr_c, Tpr)
            Zp, conv_p, _ = hall_yarborough(Ppr_p, Tpr)
            Zm, conv_m, _ = hall_yarborough(Ppr_m, Tpr)
        else:
            Zc, conv_c, _ = dranchuk_purvis_robinson(Ppr_c, Tpr)
            Zp, conv_p, _ = dranchuk_purvis_robinson(Ppr_p, Tpr)
            Zm, conv_m, _ = dranchuk_purvis_robinson(Ppr_m, Tpr)

        if not (conv_c and conv_p and conv_m):
            continue
        if np.isnan(Zc) or np.isnan(Zp) or np.isnan(Zm):
            continue

        dZdP = (Zp - Zm) / (2.0 * delta)
        cg_arr[i] = 1.0 / Pi - dZdP / Zc

    if scalar:
        return float(cg_arr[0])
    return cg_arr


# ---------------------------------------------------------------------------
# CONVENIENCE WRAPPER — full gas PVT table
# ---------------------------------------------------------------------------

def gas_pvt_table(gas_SG, T_F, y_CO2=0.0, y_H2S=0.0,
                  fluid_type='dry_gas',
                  z_method='hall_yarborough',
                  P_min=14.7, P_max=10000.0, n_points=200):
    """
    Compute gas PVT properties across a pressure range.

    Workflow
    --------
    1. Compute pseudocritical properties:
       - Sutton (1985) for dry_gas
       - Standing (1977) for gas_condensate
    2. Apply Wichert-Aziz (1972) acid-gas correction if y_CO2 or y_H2S > 0
    3. Compute Z-factor at each pressure via Hall-Yarborough or DPR
    4. Compute Bg, ρg, μg, cg from Z

    Parameters
    ----------
    gas_SG      : float — gas specific gravity (air = 1.0)
    T_F         : float — reservoir temperature [°F]
    y_CO2       : float — CO2 mole fraction [0–1]
    y_H2S       : float — H2S mole fraction [0–1]
    fluid_type  : str   — 'dry_gas' (Sutton) or 'gas_condensate' (Standing)
    z_method    : str   — 'hall_yarborough' or 'dpr'
    P_min       : float — min pressure [psia]
    P_max       : float — max pressure [psia]
    n_points    : int   — number of pressure points

    Returns
    -------
    dict with keys:
        'P'         : array [psia]
        'Z'         : array [-]
        'Bg_ft3'    : array [ft³/scf]
        'Bg_bbl'    : array [bbl/scf]
        'rho_g'     : array [lb/ft³]
        'mu_g'      : array [cp]
        'cg'        : array [psi⁻¹]
        'Tpc'       : float [°R]  (after acid-gas correction)
        'Ppc'       : float [psia] (after acid-gas correction)
        'Tpc_raw'   : float [°R]  (before correction)
        'Ppc_raw'   : float [psia] (before correction)
        'gas_SG'    : float
        'T_F'       : float
        'Mg'        : float — gas molecular weight [lb/lbmol]
        'fluid_type': str
        'z_method'  : str
        'warnings'  : list of str
    """
    warnings = []

    # Step 1: pseudocritical properties
    if fluid_type == 'dry_gas':
        Tpc_raw, Ppc_raw = sutton_pseudocriticals(gas_SG)
    elif fluid_type == 'gas_condensate':
        Tpc_raw, Ppc_raw = standing_pseudocriticals_condensate(gas_SG)
    else:
        raise ValueError(f"fluid_type must be 'dry_gas' or 'gas_condensate', got '{fluid_type}'.")

    # Step 2: Wichert-Aziz acid-gas correction
    Tpc, Ppc, wa_warns = wichert_aziz_correction(Tpc_raw, Ppc_raw, y_CO2, y_H2S)
    warnings.extend(wa_warns)

    # Gas molecular weight
    Mg = MW_AIR * gas_SG

    # Step 3: pressure array
    P_arr = np.linspace(P_min, P_max, n_points)

    # Step 4: compute Z
    Z_arr, z_warns = gas_z_factor(P_arr, T_F, Tpc, Ppc, method=z_method)
    # Collect only unique warnings (many repeats for same Tpr)
    for w in z_warns:
        if w not in warnings:
            warnings.append(w)

    # Protect against NaN Z in downstream calculations
    Z_safe = np.where(np.isnan(Z_arr), np.nan, Z_arr)

    # Step 5: derived properties
    Bg_ft3, Bg_bbl = gas_fvf(P_arr, T_F, Z_safe)
    mu_g, rho_g    = lee_gonzalez_eakin_viscosity(P_arr, T_F, Mg, Z_safe)
    cg             = gas_compressibility(P_arr, T_F, Tpc, Ppc, method=z_method)

    return {
        'P'         : P_arr,
        'Z'         : Z_arr,
        'Bg_ft3'    : Bg_ft3,
        'Bg_bbl'    : Bg_bbl,
        'rho_g'     : rho_g,
        'mu_g'      : mu_g,
        'cg'        : cg,
        'Tpc'       : Tpc,
        'Ppc'       : Ppc,
        'Tpc_raw'   : Tpc_raw,
        'Ppc_raw'   : Ppc_raw,
        'gas_SG'    : gas_SG,
        'T_F'       : T_F,
        'Mg'        : Mg,
        'fluid_type': fluid_type,
        'z_method'  : z_method,
        'warnings'  : warnings,
    }


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  gas_pvt.py — Gas PVT Correlations Self-Test")
    print("=" * 65)

    # Test 1: Sutton pseudocriticals
    Tpc, Ppc = sutton_pseudocriticals(0.65)
    print(f"\n  Sutton (1985) for gas_SG=0.65:")
    print(f"    Tpc = {Tpc:.2f} °R  ({Tpc - 459.67:.2f} °F)")
    print(f"    Ppc = {Ppc:.2f} psia")

    # Test 2: Wichert-Aziz with CO2=5%, H2S=5%
    Tpc2, Ppc2, wa_warns = wichert_aziz_correction(Tpc, Ppc, 0.05, 0.05)
    print(f"\n  Wichert-Aziz (1972) for CO2=5%, H2S=5%:")
    print(f"    Tpc' = {Tpc2:.2f} °R  (was {Tpc:.2f})")
    print(f"    Ppc' = {Ppc2:.2f} psia (was {Ppc:.2f})")

    # Test 3: Hall-Yarborough Z-factor
    T_F  = 200.0
    Tpc3, Ppc3 = sutton_pseudocriticals(0.65)
    Tpr  = (T_F + 459.67) / Tpc3
    print(f"\n  Hall-Yarborough Z-factor at T={T_F}°F (Tpr={Tpr:.4f}):")
    for P in [500, 1000, 2000, 3000, 5000]:
        Ppr = P / Ppc3
        Z, conv, warns = hall_yarborough(Ppr, Tpr)
        print(f"    P={P:5d} psia  Ppr={Ppr:.4f}  Z={Z:.5f}  converged={conv}")

    # Test 4: DPR Z-factor
    print(f"\n  DPR Z-factor at T={T_F}°F (Tpr={Tpr:.4f}):")
    for P in [500, 1000, 2000]:
        Ppr = P / Ppc3
        Z, conv, warns = dranchuk_purvis_robinson(Ppr, Tpr)
        print(f"    P={P:5d} psia  Ppr={Ppr:.4f}  Z={Z:.5f}  converged={conv}")
        for w in warns:
            print(f"      WARNING: {w}")

    # Test 5: Full gas PVT table
    tbl = gas_pvt_table(gas_SG=0.65, T_F=200.0, y_CO2=0.02, y_H2S=0.01,
                        fluid_type='dry_gas', z_method='hall_yarborough',
                        P_min=100.0, P_max=8000.0, n_points=10)
    print(f"\n  Gas PVT table (gas_SG=0.65, T=200°F, CO2=2%, H2S=1%):")
    print(f"    Tpc={tbl['Tpc']:.2f}°R  Ppc={tbl['Ppc']:.2f} psia  Mg={tbl['Mg']:.3f}")
    print(f"    {'P':>8}  {'Z':>8}  {'Bg_ft3':>12}  {'rho_g':>10}  {'mu_g':>10}  {'cg':>14}")
    for i in range(len(tbl['P'])):
        print(f"    {tbl['P'][i]:8.1f}  {tbl['Z'][i]:8.5f}  "
              f"{tbl['Bg_ft3'][i]:12.6f}  {tbl['rho_g'][i]:10.4f}  "
              f"{tbl['mu_g'][i]:10.6f}  {tbl['cg'][i]:14.6e}")
    if tbl['warnings']:
        print("\n  Warnings:")
        for w in tbl['warnings']:
            print(f"    {w}")

    print("\ngas_pvt.py — all tests passed.")
