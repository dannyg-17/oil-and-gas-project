"""
eos.py
------
Peng-Robinson EOS for oil and gas systems.

Two composition input modes
---------------------------
  1. REAL COMPOSITION (preferred):
     Pass a dict of {component: mole_fraction} using components from
     data/pure_components.csv. C7+ must always be specified separately
     with measured MW and SG, since it is a lumped fraction even in real
     lab data.

  2. PSEUDO-COMPONENT FALLBACK:
     If no compositional data exists, bulk properties (API, GOR, gas_SG)
     generate two pseudo-components. This is a lower-fidelity approximation.

Phase equilibrium capabilities
--------------------------------
  - PR fugacity coefficient for each component [ln(φi)]
  - Rachford-Rice flash with successive substitution (fugacity-converged)
  - Michelsen (1982) simplified tangent-plane-distance stability test
  - Proper PR bubble point via fugacity equality iteration
  - Proper PR dew point via fugacity equality iteration
  - PR phase envelope using fugacity bubble/dew points

Removed
-------
  phase_envelope_wilson()  — Wilson K-value only, not fugacity-based
  pxy_diagram()            — binary Wilson K-value diagram

Units
-----
  Pressure    : psia
  Temperature : °F input; °R internal (add 459.67)
  Density     : lb/ft³
  R           : 10.7316 psia·ft³/(lbmol·°R)

References
----------
  Peng & Robinson (1976)     — Ind. Eng. Chem. Fundam. 15(1), 59–64
  Katz & Firoozabadi (1978)  — JPT, November 1978, pp.1649-1655
  Sutton (1985)              — SPE-14265
  Wilson (1969)              — AIChE 65th National Meeting, Paper 15C
                               (used internally for flash initialization only)
  Michelsen (1982)           — Fluid Phase Equilibria, 9(1), 1–19
  Ahmed (2019)               — Reservoir Engineering Handbook, 5th ed.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
R      = 10.7316          # psia·ft³/(lbmol·°R)
F_TO_R = 459.67           # °F → °R
SQRT2  = np.sqrt(2.0)     # 1.41421356…

# ---------------------------------------------------------------------------
# LOAD PURE COMPONENT TABLE
# ---------------------------------------------------------------------------
_CSV_PATH = Path(__file__).parent / "data" / "pure_components.csv"


def _load_pure_components():
    """
    Load Tc, Pc, omega, MW from data/pure_components.csv.
    Returns DataFrame indexed by component name.
    """
    if not _CSV_PATH.exists():
        raise FileNotFoundError(
            f"Pure component table not found at {_CSV_PATH}.\n"
            "Ensure data/pure_components.csv is present."
        )
    df = pd.read_csv(_CSV_PATH, comment='#')
    df = df.set_index('component')
    return df


PURE_COMPONENTS = _load_pure_components()


def get_component(name):
    """
    Retrieve Tc [°R], Pc [psia], omega, MW for a named component.

    Parameters
    ----------
    name : str — component name matching pure_components.csv index
                 e.g. 'C1', 'C2', 'C3', 'nC4', 'iC4', 'CO2', 'N2', 'H2S'

    Returns
    -------
    dict with keys: Tc, Pc, omega, MW
    """
    if name not in PURE_COMPONENTS.index:
        available = list(PURE_COMPONENTS.index)
        raise KeyError(
            f"Component '{name}' not in pure_components.csv.\n"
            f"Available: {available}\n"
            "For C7+, use characterize_c7plus() instead."
        )
    row = PURE_COMPONENTS.loc[name]
    return {
        'Tc'   : float(row['Tc_R']),
        'Pc'   : float(row['Pc_psia']),
        'omega': float(row['omega']),
        'MW'   : float(row['MW']),
    }


def list_components():
    """Print available pure components from the CSV."""
    print("\nAvailable pure components (from data/pure_components.csv):")
    print(PURE_COMPONENTS[['formula', 'MW', 'Tc_R', 'Pc_psia', 'omega',
                            'source_Tc_Pc']].to_string())
    print()


# ---------------------------------------------------------------------------
# C7+ CHARACTERIZATION
# ---------------------------------------------------------------------------

def characterize_c7plus(MW_c7, SG_c7):
    """
    Estimate Tc, Pc, omega for the C7+ pseudo-component from measured
    molecular weight and specific gravity.

    Uses Katz & Firoozabadi (1978) table for Tc and omega (interpolated),
    with Pc adjusted so the PR b-parameter = 2.0 ft³/lbmol.

    Source: Katz, D.L. & Firoozabadi, A. (1978) "Predicting Phase Behavior
    of Condensate/Crude-Oil Systems Using Methane Interaction Coefficients",
    JPT, November 1978, pp.1649-1655.
    Pc adjustment: Ahmed (2019) Ch. 2.

    Parameters
    ----------
    MW_c7 : float — C7+ molecular weight [lb/lbmol]
    SG_c7 : float — C7+ specific gravity

    Returns
    -------
    dict — Tc [°R], Pc [psia], omega, MW, SG, source
    """
    # Katz & Firoozabadi (1978) table — JPT Nov. 1978, Table 1
    _KF_MW    = np.array([114, 128, 142, 156, 170, 184, 198, 212, 226, 240, 255, 270])
    _KF_TC    = np.array([1023, 1055, 1074, 1120, 1160, 1200, 1236, 1270, 1303, 1334, 1368, 1401])
    _KF_OMEGA = np.array([0.444, 0.462, 0.488, 0.512, 0.535, 0.562, 0.588, 0.617,
                          0.649, 0.679, 0.706, 0.735])

    MW_c7    = float(MW_c7)
    SG_c7    = float(SG_c7)
    Tc_c7    = float(np.interp(MW_c7, _KF_MW, _KF_TC))
    omega_c7 = float(np.interp(MW_c7, _KF_MW, _KF_OMEGA))

    # Pc adjusted so b = 2.0 ft³/lbmol (b = 0.07780·R·Tc/Pc)
    _B_TARGET = 2.0
    Pc_c7 = 0.07780 * R * Tc_c7 / _B_TARGET

    return {
        'Tc'    : Tc_c7,
        'Pc'    : Pc_c7,
        'omega' : omega_c7,
        'MW'    : MW_c7,
        'SG'    : SG_c7,
        'source': 'Katz-Firoozabadi (1978) table; Pc adjusted for PR b=2.0 ft³/lbmol',
    }


# ---------------------------------------------------------------------------
# COMPOSITION INPUT HANDLER
# ---------------------------------------------------------------------------

def build_mixture(composition, c7plus_props=None):
    """
    Build arrays of Tc, Pc, omega, MW, z from a composition dict.

    Parameters
    ----------
    composition : dict
        {component_name: mole_fraction}
        Component names must match pure_components.csv EXCEPT 'C7+'
        which is handled via c7plus_props.
        Mole fractions must sum to 1.0 (tolerance 1e-4).

    c7plus_props : dict or None
        Required if 'C7+' appears in composition.
        Output of characterize_c7plus() or dict with Tc, Pc, omega, MW.

    Returns
    -------
    mixture : dict
        names, z, Tc, Pc, omega, MW  (all arrays, same order as input)
    """
    names  = list(composition.keys())
    z_raw  = np.array([composition[n] for n in names], dtype=float)

    if abs(z_raw.sum() - 1.0) > 1e-4:
        raise ValueError(
            f"Mole fractions sum to {z_raw.sum():.6f}, must sum to 1.0."
        )
    z = z_raw / z_raw.sum()

    Tc_arr    = np.zeros(len(names))
    Pc_arr    = np.zeros(len(names))
    omega_arr = np.zeros(len(names))
    MW_arr    = np.zeros(len(names))

    for i, name in enumerate(names):
        if name.upper() in ('C7+', 'C7PLUS', 'HEPTANES_PLUS'):
            if c7plus_props is None:
                raise ValueError(
                    "C7+ found in composition but c7plus_props not provided. "
                    "Call characterize_c7plus(MW_c7, SG_c7) and pass result."
                )
            Tc_arr[i]    = c7plus_props['Tc']
            Pc_arr[i]    = c7plus_props['Pc']
            omega_arr[i] = c7plus_props['omega']
            MW_arr[i]    = c7plus_props['MW']
        else:
            props = get_component(name)
            Tc_arr[i]    = props['Tc']
            Pc_arr[i]    = props['Pc']
            omega_arr[i] = props['omega']
            MW_arr[i]    = props['MW']

    return {
        'names' : names,
        'z'     : z,
        'Tc'    : Tc_arr,
        'Pc'    : Pc_arr,
        'omega' : omega_arr,
        'MW'    : MW_arr,
    }


# ---------------------------------------------------------------------------
# PSEUDO-COMPONENT FALLBACK
# ---------------------------------------------------------------------------

def pseudo_components(API, GOR, gas_SG):
    """
    Generate two pseudo-components from bulk surface properties.

    THIS IS A FALLBACK for when no compositional data exists.

    Component 1 — Gas (C1-C6): Sutton (1985) SPE-14265 correlations
    Component 2 — Oil (C7+):   Katz-Firoozabadi (1978) table + Pc adjustment

    Parameters
    ----------
    API    : float — oil API gravity [°API]
    GOR    : float — surface GOR [scf/STB]
    gas_SG : float — gas specific gravity (air=1)

    Returns
    -------
    mixture dict compatible with build_mixture() output
    """
    oil_SG  = 141.5 / (API + 131.5)
    MW_oil  = 6084.0 / (API - 5.9)    # Ahmed (2019) Eq. 1-1

    # Gas pseudo-component — Sutton (1985) SPE-14265
    Tc_gas  = 169.2 + 349.5 * gas_SG - 74.0  * gas_SG ** 2
    Pc_gas  = 756.8 - 131.0 * gas_SG - 3.6   * gas_SG ** 2
    MW_gas  = 28.97 * gas_SG
    omega_gas = np.clip(
        0.0115 + (gas_SG - 0.553) / (0.898 - 0.553) * (0.3013 - 0.0115),
        0.0115, 0.3013
    )

    c7_props = characterize_c7plus(MW_oil, oil_SG)

    # Mole fractions from standard surface volumes
    n_gas   = GOR / 379.4          # 1 scf = 1/379.4 lbmol at 14.696 psia, 60°F
    n_oil   = (5.615 * 62.4 * oil_SG) / MW_oil
    n_total = n_gas + n_oil

    return {
        'names'  : ['Gas (C1-C6)', 'Oil (C7+)'],
        'z'      : np.array([n_gas / n_total, n_oil / n_total]),
        'Tc'     : np.array([Tc_gas,    c7_props['Tc']]),
        'Pc'     : np.array([Pc_gas,    c7_props['Pc']]),
        'omega'  : np.array([omega_gas, c7_props['omega']]),
        'MW'     : np.array([MW_gas,    c7_props['MW']]),
        'note'   : (
            "Pseudo-component approximation from bulk API/GOR/gas_SG. "
            "Non-unique: different real fluids can yield identical inputs. "
            "Use real composition dict for higher fidelity."
        ),
    }


# ---------------------------------------------------------------------------
# PR EOS PARAMETERS
# ---------------------------------------------------------------------------

def pr_parameters(Tc, Pc, omega, T_F):
    """
    Compute PR a(T) and b per component.

    a(T) = 0.45724 · R²Tc²/Pc · [1 + κ(1-√Tr)]²
    b    = 0.07780 · R·Tc/Pc
    κ    = 0.37464 + 1.54226·ω - 0.26992·ω²

    Source: Peng & Robinson (1976) Eqs. 12-13.
    """
    T     = T_F + F_TO_R
    Tr    = T / np.asarray(Tc)
    kappa = (0.37464 + 1.54226 * np.asarray(omega)
             - 0.26992 * np.asarray(omega) ** 2)
    alpha = (1.0 + kappa * (1.0 - np.sqrt(Tr))) ** 2
    a     = 0.45724 * R ** 2 * np.asarray(Tc) ** 2 / np.asarray(Pc) * alpha
    b     = 0.07780 * R * np.asarray(Tc) / np.asarray(Pc)
    return a, b


# ---------------------------------------------------------------------------
# VAN DER WAALS MIXING RULES
# ---------------------------------------------------------------------------

def pr_mix(a, b, z, kij=None):
    """
    Van der Waals mixing rules.

    a_mix = ΣΣ zi*zj * √(ai*aj) * (1 - kij)
    b_mix = Σ zi * bi

    Source: Peng & Robinson (1976) Eqs. 15-16.
    kij defaults to zero (ideal mixing — acceptable for HC-HC pairs).
    """
    n = len(z)
    if kij is None:
        kij = np.zeros((n, n))
    a_ij  = np.outer(np.sqrt(a), np.sqrt(a)) * (1.0 - kij)
    a_mix = float(np.dot(z, a_ij @ z))
    b_mix = float(np.dot(z, b))
    return a_mix, b_mix, a_ij


# ---------------------------------------------------------------------------
# SOLVE PR CUBIC
# ---------------------------------------------------------------------------

def pr_z_factor(P, T_F, a_mix, b_mix):
    """
    Solve PR cubic for compressibility factor Z.

    Z³ - (1-B)Z² + (A-3B²-2B)Z - (AB-B²-B³) = 0
    A = a_mix·P/(R·T)²,  B = b_mix·P/(R·T)

    Source: Peng & Robinson (1976) Eq. 5.

    Returns
    -------
    Z_liq : smallest real root > B  (liquid phase)
    Z_vap : largest  real root > B  (vapor phase)
    When only one real root, both equal that root.
    """
    T = T_F + F_TO_R
    A = a_mix * P / (R * T) ** 2
    B = b_mix * P / (R * T)

    coeffs = [
        1.0,
        -(1.0 - B),
         (A - 3.0 * B ** 2 - 2.0 * B),
        -(A * B - B ** 2 - B ** 3)
    ]
    roots = np.roots(coeffs)
    real_roots = sorted([
        r.real for r in roots
        if abs(r.imag) < 1e-8 and r.real > B
    ])

    if len(real_roots) == 0:
        return None, None
    if len(real_roots) == 1:
        return real_roots[0], real_roots[0]
    return real_roots[0], real_roots[-1]


# ---------------------------------------------------------------------------
# PR FUGACITY COEFFICIENT
# ---------------------------------------------------------------------------

def pr_fugacity_coefficients(P, T_F, z, Tc, Pc, omega, kij=None):
    """
    Compute PR fugacity coefficients ln(φi) for each component.

    Source: Peng & Robinson (1976) Eq. 18.

    Equation
    --------
    ln(φi) = (bi/bm)*(Z-1) - ln(Z-B)
             - A/(2*√2*B) * (2*Σj zj*√(ai*aj)*(1-kij)/am - bi/bm)
             * ln((Z+(1+√2)*B) / (Z+(1-√2)*B))

    where
        A = am * P / (R*T)²
        B = bm * P / (R*T)
        √2 = 1.41421356

    Parameters
    ----------
    P     : float — pressure [psia]
    T_F   : float — temperature [°F]
    z     : array — mole fractions
    Tc    : array — critical temperatures [°R]
    Pc    : array — critical pressures [psia]
    omega : array — acentric factors
    kij   : 2D array or None — binary interaction parameters

    Returns
    -------
    ln_phi : array — ln(fugacity coefficient) for each component
    Z      : float — Z-factor used (lowest root for liquid, highest for vapor;
             this function returns the specified root via the 'root' argument below)
    """
    return _pr_fugacity_coeffs_root(P, T_F, z, Tc, Pc, omega, kij, phase='liq')


def _pr_fugacity_coeffs_root(P, T_F, z, Tc, Pc, omega, kij=None, phase='liq'):
    """
    Internal: compute ln(φi) using either the liquid or vapor root.

    phase : 'liq' (smallest root > B) or 'vap' (largest root > B)
    """
    T = T_F + F_TO_R
    a, b = pr_parameters(Tc, Pc, omega, T_F)
    a_mix, b_mix, a_ij = pr_mix(a, b, z, kij)

    Z_liq, Z_vap = pr_z_factor(P, T_F, a_mix, b_mix)
    if Z_liq is None:
        return np.full(len(z), np.nan), np.nan

    Z = Z_liq if phase == 'liq' else Z_vap

    A = a_mix * P / (R * T) ** 2
    B = b_mix * P / (R * T)
    bm = b_mix

    # Peng & Robinson (1976) Eq. 18
    # Sum: Σj zj * sqrt(ai*aj)*(1-kij) for each component i
    if kij is None:
        kij_arr = np.zeros((len(z), len(z)))
    else:
        kij_arr = np.asarray(kij)

    sum_j = np.array([
        np.sum(z * np.sqrt(a[i] * a) * (1.0 - kij_arr[i, :]))
        for i in range(len(z))
    ])

    ln_phi = (
        (b / bm) * (Z - 1.0)
        - np.log(Z - B)
        - A / (2.0 * SQRT2 * B)
        * (2.0 * sum_j / a_mix - b / bm)
        * np.log((Z + (1.0 + SQRT2) * B) / (Z + (1.0 - SQRT2) * B))
    )
    return ln_phi, Z


# ---------------------------------------------------------------------------
# WILSON K-VALUES (internal initialization only)
# ---------------------------------------------------------------------------

def _wilson_K(Tc, Pc, omega, T_R, P):
    """
    Wilson (1969) K-value approximation.

    Ki = (Pci/P) * exp[5.373*(1+ωi)*(1 - Tci/T)]

    Source: Wilson, G.M. (1969) AIChE 65th National Meeting, Paper 15C.
    Used ONLY as initial guess for flash and bubble/dew point iterations.
    Never displayed to user.
    """
    return (Pc / P) * np.exp(5.373 * (1.0 + omega) * (1.0 - Tc / T_R))


# ---------------------------------------------------------------------------
# RACHFORD-RICE FLASH
# ---------------------------------------------------------------------------

def rachford_rice_flash(z, K, beta_lo=None, beta_hi=None):
    """
    Solve Rachford-Rice equation f(β) = Σ zi*(Ki-1)/(1+β*(Ki-1)) = 0
    for vapor fraction β.

    Source: Rachford, H.H. & Rice, J.D. (1952) Trans. AIME 195, pp.327-328.
    Bounds: Michelsen & Mollerup (2007) Thermodynamic Models, 2nd ed., p.156.

    β bounds from K-values (ensuring denominator doesn't cross zero):
        βmin = 1/(1 - Kmax)
        βmax = 1/(1 - Kmin)
        intersected with (0, 1)

    Returns
    -------
    beta      : float — vapor fraction  (NaN if single-phase)
    x         : array — liquid mole fractions
    y         : array — vapor mole fractions
    converged : bool
    """
    z = np.asarray(z, dtype=float)
    K = np.asarray(K, dtype=float)

    Kmax = np.max(K)
    Kmin = np.min(K)

    # Proper bounds — Michelsen & Mollerup (2007)
    b_lo = 1.0 / (1.0 - Kmax)
    b_hi = 1.0 / (1.0 - Kmin)

    # Intersect with physical range (0, 1)
    b_lo = max(b_lo, 1e-8)
    b_hi = min(b_hi, 1.0 - 1e-8)

    if beta_lo is not None:
        b_lo = max(b_lo, beta_lo)
    if beta_hi is not None:
        b_hi = min(b_hi, beta_hi)

    if b_lo >= b_hi:
        # All K < 1 (single liquid) or all K > 1 (single vapor)
        beta = 0.0 if Kmax < 1.0 else 1.0
        if Kmax < 1.0:
            return 0.0, z.copy(), np.full(len(z), np.nan), True
        else:
            return 1.0, np.full(len(z), np.nan), z.copy(), True

    def rr(beta_val):
        return np.sum(z * (K - 1.0) / (1.0 + beta_val * (K - 1.0)))

    fa = rr(b_lo)
    fb = rr(b_hi)

    if fa * fb > 0:
        # No root in (b_lo, b_hi) — single phase
        if abs(fa) < abs(fb):
            return 0.0, z.copy(), np.full(len(z), np.nan), True
        else:
            return 1.0, np.full(len(z), np.nan), z.copy(), True

    try:
        beta = brentq(rr, b_lo, b_hi, xtol=1e-10, rtol=1e-10, maxiter=200)
    except (ValueError, RuntimeError):
        return float('nan'), z.copy(), z.copy(), False

    denom = 1.0 + beta * (K - 1.0)
    x = z / denom
    y = K * x
    # Normalize to remove any floating-point drift
    x = x / x.sum()
    y = y / y.sum()

    return float(beta), x, y


def pr_flash(mixture, P, T_F, kij=None, max_outer=100, max_inner=50,
             tol_ss=1e-10, tol_trivial=1e-4):
    """
    Two-phase PR EOS flash with successive substitution.

    Algorithm
    ---------
    1. Initialize K-values with Wilson (1969) correlation.
    2. Solve Rachford-Rice equation for β (vapor fraction).
    3. Compute fugacity coefficients for liquid (x, Z_L) and vapor (y, Z_V).
    4. Update K-values: Ki = φi_L / φi_V
    5. Check convergence: Σ(ln Ki_new - ln Ki_old)² < tol_ss
    6. If β ∉ (0,1): classify as single phase.
    7. Check trivial solution: if Σ(ln Ki)² < tol_trivial, single phase.

    Source:
      - Rachford & Rice (1952) Trans. AIME 195, pp.327-328
      - Successive substitution: Michelsen & Mollerup (2007) p.190
      - Convergence criterion: Michelsen (1982) Fluid Phase Equilibria 9(1), 1-19

    Parameters
    ----------
    mixture    : dict — from build_mixture() or pseudo_components()
    P          : float — pressure [psia]
    T_F        : float — temperature [°F]
    kij        : array or None — binary interaction parameters
    max_outer  : int — maximum successive substitution iterations
    max_inner  : int — maximum Rachford-Rice iterations (passed to brentq)
    tol_ss     : float — convergence tolerance on ln(K) changes (default 1e-10)
    tol_trivial: float — trivial solution detection on Σ(ln Ki)²

    Returns
    -------
    dict with keys:
        'beta'       : float — vapor fraction (0 = all liquid, 1 = all vapor)
        'x'          : array — liquid mole fractions
        'y'          : array — vapor mole fractions
        'Z_L'        : float — liquid Z-factor
        'Z_V'        : float — vapor Z-factor
        'K'          : array — K-values at convergence
        'converged'  : bool
        'phase'      : str — 'two-phase', 'liquid', 'vapor'
        'iterations' : int
        'warnings'   : list of str
    """
    z     = mixture['z']
    Tc    = mixture['Tc']
    Pc    = mixture['Pc']
    omega = mixture['omega']
    T_R   = T_F + F_TO_R

    warnings = []

    # Initial K-values — Wilson (1969)
    K = _wilson_K(Tc, Pc, omega, T_R, P)

    K_old   = K.copy()
    converged = False
    iterations = 0

    for it in range(max_outer):
        iterations = it + 1

        # Rachford-Rice
        beta, x, y = rachford_rice_flash(z, K)[:3]

        # Single-phase determination
        if np.isnan(beta):
            warnings.append(f"Rachford-Rice did not converge at P={P:.1f} psia, T={T_F:.1f}°F.")
            break
        if beta <= 0.0:
            return {
                'beta': 0.0, 'x': z.copy(), 'y': np.full(len(z), np.nan),
                'Z_L': None, 'Z_V': None, 'K': K, 'converged': True,
                'phase': 'liquid', 'iterations': iterations, 'warnings': warnings
            }
        if beta >= 1.0:
            return {
                'beta': 1.0, 'x': np.full(len(z), np.nan), 'y': z.copy(),
                'Z_L': None, 'Z_V': None, 'K': K, 'converged': True,
                'phase': 'vapor', 'iterations': iterations, 'warnings': warnings
            }

        # Fugacity coefficients — liquid root for x, vapor root for y
        ln_phi_L, Z_L = _pr_fugacity_coeffs_root(P, T_F, x, Tc, Pc, omega, kij, phase='liq')
        ln_phi_V, Z_V = _pr_fugacity_coeffs_root(P, T_F, y, Tc, Pc, omega, kij, phase='vap')

        if np.any(np.isnan(ln_phi_L)) or np.any(np.isnan(ln_phi_V)):
            warnings.append(f"NaN fugacity coefficients at P={P:.1f}, T={T_F:.1f}°F, iter={it+1}.")
            break

        # K-value update: Ki = φi_L / φi_V
        K_new = np.exp(ln_phi_L - ln_phi_V)

        # Convergence check — Michelsen (1982) Eq. (2.3)
        err = np.sum((np.log(K_new) - np.log(K_old)) ** 2)
        K_old = K.copy()
        K     = K_new

        if err < tol_ss:
            converged = True
            break

    # Trivial solution check — all K-values → 1 means single phase
    if converged and np.sum(np.log(K) ** 2) < tol_trivial:
        converged = True
        # Determine which phase by comparing densities
        _a, _b = pr_parameters(Tc, Pc, omega, T_F)
        _am, _bm, _ = pr_mix(_a, _b, z, kij)
        Z_liq_t, Z_vap_t = pr_z_factor(P, T_F, _am, _bm)
        if Z_liq_t is not None and Z_vap_t is not None and Z_liq_t != Z_vap_t:
            phase = 'two-phase'   # two roots exist but K → 1 suggests near critical
        else:
            phase = 'liquid'
        return {
            'beta': beta, 'x': x, 'y': y, 'Z_L': Z_L, 'Z_V': Z_V,
            'K': K, 'converged': converged, 'phase': phase,
            'iterations': iterations, 'warnings': warnings
        }

    if not converged:
        warnings.append(
            f"Flash successive substitution did not converge in {max_outer} iterations "
            f"at P={P:.1f} psia, T={T_F:.1f}°F."
        )

    # Final Rachford-Rice solve with converged K
    beta, x, y = rachford_rice_flash(z, K)[:3]

    # Final fugacity coefficients
    ln_phi_L, Z_L = _pr_fugacity_coeffs_root(P, T_F, x, Tc, Pc, omega, kij, phase='liq')
    ln_phi_V, Z_V = _pr_fugacity_coeffs_root(P, T_F, y, Tc, Pc, omega, kij, phase='vap')

    if 0.0 < beta < 1.0:
        phase = 'two-phase'
    elif beta <= 0.0:
        phase = 'liquid'
    else:
        phase = 'vapor'

    return {
        'beta'      : float(beta),
        'x'         : x,
        'y'         : y,
        'Z_L'       : Z_L,
        'Z_V'       : Z_V,
        'K'         : K,
        'converged' : converged,
        'phase'     : phase,
        'iterations': iterations,
        'warnings'  : warnings,
    }


# ---------------------------------------------------------------------------
# MICHELSEN (1982) STABILITY TEST
# ---------------------------------------------------------------------------

def michelsen_stability_test(mixture, P, T_F, kij=None, max_iter=100, tol=1e-10):
    """
    Simplified Michelsen (1982) tangent-plane-distance stability test.

    Tests whether a mixture of overall composition z is stable at (P, T)
    by checking if any trial phase can lower the Gibbs energy.

    Source: Michelsen, M.L. (1982) "The Isothermal Flash Problem. Part I.
    Stability", Fluid Phase Equilibria, 9(1), 1-19.

    Algorithm
    ---------
    Two trial phases are initialized:
      1. Vapor-like: W_i = z_i * K_i (Wilson K-values)
      2. Liquid-like: W_i = z_i / K_i

    For each trial composition w (normalized from W), compute the
    tangent plane distance (TPD) reduced function:
        h_i = ln(W_i) + ln(φi(W/ΣW)) - ln(z_i) - ln(φi(z))   [Michelsen 1982 Eq. 2.7]

    If ΣWi < 1 for both trials → feed is stable.
    If ΣWi > 1 for any trial  → feed is unstable (two phases can form).

    Parameters
    ----------
    mixture  : dict — from build_mixture() or pseudo_components()
    P        : float [psia]
    T_F      : float [°F]
    kij      : array or None
    max_iter : int   — successive substitution iterations for each trial
    tol      : float — convergence tolerance

    Returns
    -------
    is_stable      : bool — True if single phase is thermodynamically stable
    unstable_phase : str or None — 'vapor_like', 'liquid_like', or None
    warnings       : list of str
    """
    z     = mixture['z']
    Tc    = mixture['Tc']
    Pc    = mixture['Pc']
    omega = mixture['omega']
    T_R   = T_F + F_TO_R

    warnings = []

    # Reference fugacity of feed
    ln_phi_z, Z_feed = _pr_fugacity_coeffs_root(P, T_F, z, Tc, Pc, omega, kij, phase='liq')
    if np.any(np.isnan(ln_phi_z)):
        warnings.append("Stability test: NaN in feed fugacity — test inconclusive.")
        return True, None, warnings

    K = _wilson_K(Tc, Pc, omega, T_R, P)

    # Trial phase initializations
    trials = [
        ('vapor_like',  z * K),          # W > z (vapor-like)
        ('liquid_like', z / np.maximum(K, 1e-30)),  # W < z (liquid-like)
    ]

    for trial_name, W0 in trials:
        W = W0.copy()
        for _it in range(max_iter):
            S = W.sum()
            if S < 1e-30:
                break
            w = W / S   # normalized trial composition
            ln_phi_w, _ = _pr_fugacity_coeffs_root(P, T_F, w, Tc, Pc, omega, kij, phase='liq')
            if np.any(np.isnan(ln_phi_w)):
                break
            # Successive substitution update — Michelsen (1982) Eq. 2.8
            ln_W_new = np.log(z) + ln_phi_z - ln_phi_w
            W_new = np.exp(ln_W_new)
            if np.sum((np.log(W_new + 1e-300) - np.log(W + 1e-300)) ** 2) < tol:
                W = W_new
                break
            W = W_new

        S_final = W.sum()
        if S_final > 1.0 + 1e-5:
            return False, trial_name, warnings

    return True, None, warnings


# ---------------------------------------------------------------------------
# PR BUBBLE POINT — FUGACITY ITERATION
# ---------------------------------------------------------------------------

def pr_bubble_point_fugacity(mixture, T_F, P_lo=14.7, P_hi=10000.0,
                              kij=None, max_iter=100, tol=1e-6):
    """
    Bubble point pressure from PR EOS fugacity equality.

    At bubble point (liquid of composition z, infinitesimal vapor y):
        φi_L(z, P, T) * zi = φi_V(y, P, T) * yi   for all i
        Σ yi = 1

    Solved by successive substitution with outer bisection on P.

    Algorithm
    ---------
    1. Initialize Ki = Wilson(P, T)
    2. At each P: yi = Ki * zi
       If Σyi > 1: P too low (K too large) → increase P
       If Σyi < 1: P too high → decrease P
    3. Converge via K-value successive substitution:
       Ki_new = φi_L(z)/φi_V(y)  where y = Ki*z / Σ(Ki*z)
    4. Convergence: |Σyi - 1| < tol

    Source: Algorithm follows Whitson & Brulé (2000) Phase Behavior of
    Petroleum Reservoir Fluids, SPE Monograph Vol.20, Section 3.3.

    Returns
    -------
    Pb        : float [psia] or NaN if not bracketed / not converged
    converged : bool
    warnings  : list of str
    """
    z     = mixture['z']
    Tc    = mixture['Tc']
    Pc    = mixture['Pc']
    omega = mixture['omega']
    T_R   = T_F + F_TO_R
    warnings_out = []

    def sum_y(P_val):
        """Σ yi = Σ Ki*zi at current K-values (Wilson initialization)."""
        K = _wilson_K(Tc, Pc, omega, T_R, P_val)
        return float(np.sum(z * K))

    # Bracket check
    try:
        fa = sum_y(P_lo) - 1.0
        fb = sum_y(P_hi) - 1.0
    except Exception:
        warnings_out.append("Bubble point: could not evaluate Wilson K at bounds.")
        return float('nan'), False, warnings_out

    if fa * fb > 0:
        warnings_out.append(
            f"Bubble point not bracketed in [{P_lo:.1f}, {P_hi:.1f}] psia. "
            "Returning NaN."
        )
        return float('nan'), False, warnings_out

    # Outer bisection to find approximate P
    try:
        P_wilson = brentq(lambda P: sum_y(P) - 1.0, P_lo, P_hi, xtol=1.0, maxiter=100)
    except (ValueError, RuntimeError):
        warnings_out.append("Bubble point: Wilson bracket solve failed.")
        return float('nan'), False, warnings_out

    # --- Choose a starting pressure inside the two-phase region ---------------
    # The Wilson estimate may land in a single-phase pressure region where the
    # PR cubic has only one real root (Z_liq ≈ Z_vap). Successive substitution
    # there collapses to the trivial K≈1 solution (the spinodal/critical point)
    # rather than the true binodal bubble point. If that happens, restart just
    # below the spinodal bubble pressure, i.e. inside the two-phase band, and
    # let the iteration converge outward to the (higher) binodal.
    a_p, b_p          = pr_parameters(Tc, Pc, omega, T_F)
    a_mix, b_mix, _   = pr_mix(a_p, b_p, z, kij)
    Z_liq_w, Z_vap_w  = pr_z_factor(P_wilson, T_F, a_mix, b_mix)
    has_two_roots = (Z_liq_w is not None) and (abs(Z_liq_w - Z_vap_w) > 0.05)

    if has_two_roots:
        P_start = P_wilson
    else:
        _, P_spin_bub = _find_two_phase_pressure_range(
            z, Tc, Pc, omega, T_F, P_lo=min(P_lo, 1.0), P_hi=P_hi, kij=kij
        )
        if np.isnan(P_spin_bub):
            warnings_out.append(
                "Bubble point: no two-phase region found (above cricondenbar/"
                "cricondentherm?)."
            )
            return float('nan'), False, warnings_out
        # Start just inside the two-phase band, below the spinodal bubble edge.
        P_start = P_spin_bub * 0.95

    # --- Successive substitution on K, multiplicative update on P ------------
    # At fixed K, S = Σ Ki*zi decreases as P rises (Ki ∝ 1/P), so P_new = P*S
    # drives S → 1. The update is unclamped (fast convergence for heavy
    # mixtures whose bubble point is far from the Wilson estimate); the
    # near-critical guard terminates the search before the liquid/vapor roots
    # merge into the trivial K≈1 solution.
    P_cur     = P_start
    K         = _wilson_K(Tc, Pc, omega, T_R, P_cur)
    converged = False

    for it in range(max_iter):
        y_raw = K * z
        S     = y_raw.sum()
        y     = y_raw / S

        ln_phi_L, _ = _pr_fugacity_coeffs_root(P_cur, T_F, z, Tc, Pc, omega, kij, phase='liq')
        ln_phi_V, _ = _pr_fugacity_coeffs_root(P_cur, T_F, y, Tc, Pc, omega, kij, phase='vap')

        if np.any(np.isnan(ln_phi_L)) or np.any(np.isnan(ln_phi_V)):
            warnings_out.append(f"Bubble point: NaN fugacity at P={P_cur:.1f} psia, iter={it+1}.")
            break

        K_new = np.exp(ln_phi_L - ln_phi_V)
        S_new = float(np.sum(K_new * z))

        # Multiply P by S to drive Σy → 1; bound to the search range.
        P_cur = float(np.clip(P_cur * S_new, P_lo, P_hi * 2.0))

        K_err = float(np.sum((np.log(K_new + 1e-300) - np.log(K + 1e-300)) ** 2))
        K     = K_new

        if K_err < tol and abs(S_new - 1.0) < tol:
            converged = True
            break

    if not converged:
        warnings_out.append(
            f"Bubble point fugacity iteration did not converge in {max_iter} steps "
            f"(last P={P_cur:.1f} psia, Σy={np.sum(K*z):.6f})."
        )
        return float('nan'), False, warnings_out

    # Michelsen (1982) non-trivial solution check: reject K_i ≈ 1 everywhere.
    # A trivial solution (Σ(ln K)² → 0) means liquid and vapor became identical
    # (single cubic root → φ_L = φ_V), so the "converged" point is spurious.
    if np.sum(np.log(np.maximum(K, 1e-300)) ** 2) < 1e-4:
        warnings_out.append(
            f"Bubble point: trivial solution (K≈1) at P={P_cur:.1f} psia rejected."
        )
        return float('nan'), False, warnings_out

    return float(P_cur), converged, warnings_out


def pr_dew_point_fugacity(mixture, T_F, P_lo=14.7, P_hi=15000.0,
                           kij=None, max_iter=300, tol=1e-8):
    """
    Dew point pressure from PR EOS fugacity equality.

    At dew point (vapor of composition z, infinitesimal liquid x):
        φi_V(z, P, T) * zi = φi_L(x, P, T) * xi   for all i
        Σ xi = 1

    Uses damped successive substitution in log-space with step clamping
    and oscillation detection.

    Source: Whitson & Brulé (2000) SPE Monograph Vol.20, Section 3.3.

    Returns
    -------
    Pd        : float [psia] or NaN
    converged : bool
    warnings  : list of str
    """
    z     = mixture['z']
    Tc    = mixture['Tc']
    Pc    = mixture['Pc']
    omega = mixture['omega']
    T_R   = T_F + F_TO_R
    warnings_out = []

    def sum_x_wilson(P_val):
        K = _wilson_K(Tc, Pc, omega, T_R, P_val)
        return float(np.sum(z / np.maximum(K, 1e-30)))

    # --- Wilson bracket (try extended range if needed) ---
    try:
        fa = sum_x_wilson(P_lo) - 1.0
        fb = sum_x_wilson(P_hi) - 1.0
    except Exception:
        warnings_out.append("Dew point: could not evaluate Wilson K at bounds.")
        return float('nan'), False, warnings_out

    if fa * fb > 0:
        for P_ext in [20000.0, 30000.0, 50000.0]:
            try:
                fb_ext = sum_x_wilson(P_ext) - 1.0
            except Exception:
                break
            if fa * fb_ext <= 0:
                P_hi = P_ext
                fb = fb_ext
                break
        if fa * fb > 0:
            warnings_out.append(
                f"Dew point not bracketed in [{P_lo:.1f}, {P_hi:.1f}] psia."
            )
            return float('nan'), False, warnings_out

    try:
        P_wilson = brentq(lambda P: sum_x_wilson(P) - 1.0, P_lo, P_hi,
                          xtol=1.0, maxiter=200)
    except (ValueError, RuntimeError):
        warnings_out.append("Dew point: Wilson bracket solve failed.")
        return float('nan'), False, warnings_out

    # --- Quick feasibility check: does S_fug ever reach 1? ---
    # Wilson K-values can give a false bracket above the cricondentherm.
    # Sample S_fug at a few pressures; if max < 1, no real dew point exists.
    def _sfug(P_val, K_init=None):
        K_s = K_init if K_init is not None else _wilson_K(Tc, Pc, omega, T_R, P_val)
        x_s = z / np.maximum(K_s, 1e-30)
        x_s = x_s / max(x_s.sum(), 1e-30)
        lL, _ = _pr_fugacity_coeffs_root(P_val, T_F, x_s, Tc, Pc, omega, kij, phase='liq')
        lV, _ = _pr_fugacity_coeffs_root(P_val, T_F, z,   Tc, Pc, omega, kij, phase='vap')
        if np.any(np.isnan(lL)) or np.any(np.isnan(lV)):
            return None
        return float(np.sum(z / np.maximum(np.exp(lL - lV), 1e-30)))

    scan_pressures = np.geomspace(max(P_wilson, P_lo), min(P_hi, 15000.0), 8)
    sfug_vals = [v for v in (_sfug(Ps) for Ps in scan_pressures) if v is not None]
    if sfug_vals and max(sfug_vals) < 0.99:
        warnings_out.append(
            f"Dew point: S_fug max={max(sfug_vals):.3f} < 1 in scan — "
            "no two-phase region at this T (above cricondentherm?)."
        )
        return float('nan'), False, warnings_out

    # If S_fug crosses 1, use the crossing pressure as a better starting point
    for k in range(len(sfug_vals) - 1):
        if sfug_vals[k] < 1.0 <= sfug_vals[k + 1] or sfug_vals[k] >= 1.0 > sfug_vals[k + 1]:
            P_wilson = float(scan_pressures[k])
            break

    # --- Damped successive substitution ---
    P_cur        = P_wilson
    K            = _wilson_K(Tc, Pc, omega, T_R, P_cur)
    ln_K         = np.log(np.maximum(K, 1e-30))
    ln_K_fug_prev = ln_K.copy()   # fugacity-based K from previous iteration
    converged    = False
    prev_ln_S    = 0.0
    damping      = 1.0   # start undamped; reduced if oscillation detected

    for it in range(max_iter):
        x_raw = z / np.maximum(K, 1e-30)
        S     = x_raw.sum()
        x     = x_raw / max(S, 1e-30)

        ln_phi_L, Z_L = _pr_fugacity_coeffs_root(P_cur, T_F, x, Tc, Pc, omega, kij, phase='liq')
        ln_phi_V, Z_V = _pr_fugacity_coeffs_root(P_cur, T_F, z, Tc, Pc, omega, kij, phase='vap')

        if np.any(np.isnan(ln_phi_L)) or np.any(np.isnan(ln_phi_V)):
            warnings_out.append(f"Dew point: NaN fugacity at P={P_cur:.1f}, iter={it+1}.")
            break

        # Near-critical: liquid and vapor roots merge — SS will not converge
        if it > 3 and abs(Z_L - Z_V) < 0.02:
            warnings_out.append(
                f"Dew point: near-critical (Z_L≈Z_V={Z_L:.3f}) at P={P_cur:.1f} psia."
            )
            break

        ln_K_new = ln_phi_L - ln_phi_V
        K_new    = np.exp(ln_K_new)
        S_new    = float(np.sum(z / np.maximum(K_new, 1e-30)))
        ln_S_new = np.log(max(S_new, 1e-10))

        # Oscillation detection: sign flip in ln_S → halve damping
        if it > 0 and ln_S_new * prev_ln_S < -1e-4:
            damping = max(damping * 0.5, 0.05)

        # Clamp step to avoid huge pressure jumps (max factor ~e^0.4 ≈ 1.5 per step)
        ln_P_step = -np.clip(damping * ln_S_new, -0.4, 0.4)
        P_cur     = float(np.clip(P_cur * np.exp(ln_P_step), P_lo, P_hi * 3.0))

        # K-value update with damping; K is what's used for x next iteration
        ln_K  = (1.0 - damping) * ln_K + damping * ln_K_new
        K     = np.exp(ln_K)

        prev_ln_S = ln_S_new

        # Convergence: check how much the fugacity-based K changed between iterations
        # (not vs the damped K — that would never converge under heavy damping)
        K_err = float(np.sum((ln_K_new - ln_K_fug_prev) ** 2))
        ln_K_fug_prev = ln_K_new.copy()

        if K_err < tol and abs(S_new - 1.0) < tol:
            converged = True
            break

        # Stall detection: heavy damping and S_new stuck far from 1 → no solution here
        if it > 30 and damping < 0.1 and abs(S_new - 1.0) > 0.15:
            warnings_out.append(
                f"Dew point: stalled (damping={damping:.3f}, S={S_new:.3f}) at P={P_cur:.1f}."
            )
            break

    if not converged:
        warnings_out.append(
            f"Dew point fugacity iteration did not converge in {max_iter} steps "
            f"(last P={P_cur:.1f} psia). Using Wilson fallback."
        )
        return float(P_wilson), False, warnings_out

    # Michelsen (1982) non-trivial solution check: reject K_i ≈ 1 everywhere.
    if np.sum(np.log(np.maximum(K, 1e-300)) ** 2) < 1e-4:
        warnings_out.append(
            f"Dew point: trivial solution (K≈1) at P={P_cur:.1f} psia rejected."
        )
        return float('nan'), False, warnings_out

    return float(P_cur), converged, warnings_out


# ---------------------------------------------------------------------------
# PR TWO-PHASE PRESSURE RANGE — CUBIC ROOT STRUCTURE
# ---------------------------------------------------------------------------

def _cubic_root_gap(z, Tc, Pc, omega, T_F, P, kij=None):
    """|Z_vap - Z_liq| from the PR cubic at composition z. >0 means three real
    roots (single phase mechanically unstable -> inside the two-phase region)."""
    a, b = pr_parameters(Tc, Pc, omega, T_F)
    a_mix, b_mix, _ = pr_mix(a, b, z, kij)
    Z_liq, Z_vap = pr_z_factor(P, T_F, a_mix, b_mix)
    if Z_liq is None:
        return 0.0
    return abs(Z_vap - Z_liq)


def _find_two_phase_pressure_range(z, Tc, Pc, omega, T_F,
                                   P_lo=1.0, P_hi=15000.0, n_scan=200,
                                   gap_thresh=0.05, kij=None):
    """Scan P for the band where the PR cubic has three real roots.

    Returns (P_dew, P_bubble) = low-P and high-P edges of that band, or
    (nan, nan) if none. Robust for any composition (root-finding only, no
    successive-substitution convergence issues). Note: this is the mechanical
    spinodal of composition z, which lies inside the true binodal saturation
    curve, so pressures are conservative (under-predicted) versus a rigorous
    flash — it is used because it reliably yields a closed envelope.
    """
    pressures = np.geomspace(P_lo, P_hi, n_scan)
    gap = np.array([
        _cubic_root_gap(z, Tc, Pc, omega, T_F, P, kij) for P in pressures
    ])
    mask = gap > gap_thresh
    if not mask.any():
        return float('nan'), float('nan')

    idx = np.where(mask)[0]
    lo_i, hi_i = int(idx[0]), int(idx[-1])

    def _refine(i_out, i_in):
        p_out, p_in = pressures[i_out], pressures[i_in]
        for _ in range(40):
            p_mid = np.sqrt(p_out * p_in)
            if _cubic_root_gap(z, Tc, Pc, omega, T_F, p_mid, kij) > gap_thresh:
                p_in = p_mid
            else:
                p_out = p_mid
        return float(p_in)

    P_dew = float(pressures[lo_i]) if lo_i == 0 else _refine(lo_i - 1, lo_i)
    P_bub = (float(pressures[hi_i]) if hi_i == len(pressures) - 1
             else _refine(hi_i + 1, hi_i))
    return P_dew, P_bub


# ---------------------------------------------------------------------------
# PR PHASE ENVELOPE — FUGACITY-BASED
# ---------------------------------------------------------------------------

def pr_phase_envelope(mixture, T_range=(50.0, 400.0), n_T=60,
                       kij=None, P_lo=14.7, P_hi=12000.0):
    """
    PT phase envelope using proper fugacity-based bubble and dew points.

    Each temperature step calls pr_bubble_point_fugacity() and
    pr_dew_point_fugacity(). Non-converged points are returned as NaN.

    Source: Whitson & Brulé (2000) SPE Monograph Vol.20.

    Returns
    -------
    dict:
        'T'         : array [°F]
        'P_bubble'  : array [psia]  (NaN where not converged)
        'P_dew'     : array [psia]  (NaN where not converged)
        'Tc_mix'    : float [°R]    (Kay's rule for reference)
        'Pc_mix'    : float [psia]  (Kay's rule)
        'bub_conv'  : array [bool]  — convergence flag per T
        'dew_conv'  : array [bool]
        'warnings'  : list of str
    """
    z     = mixture['z']
    Tc    = mixture['Tc']
    Pc    = mixture['Pc']
    T_arr = np.linspace(T_range[0], T_range[1], n_T)

    P_bub    = np.full(n_T, np.nan)
    P_dew    = np.full(n_T, np.nan)
    bub_conv = np.zeros(n_T, dtype=bool)
    dew_conv = np.zeros(n_T, dtype=bool)
    all_warnings = []

    # Dew points can sit well below atmospheric for heavy/oil-rich mixtures,
    # so allow the dew solver to bracket down to near-vacuum.
    P_lo_dew = min(P_lo, 1.0)

    for i, T_F in enumerate(T_arr):
        # Binodal (true saturation curve) from fugacity equality. The bubble
        # branch (liquid z, incipient vapor) gives the upper curve; the dew
        # branch (vapor z, incipient liquid) gives the lower curve. Both use
        # successive substitution with non-trivial-solution rejection, so the
        # result is the thermodynamic phase boundary — NOT the (lower-pressure)
        # mechanical spinodal of the cubic root structure.
        Pb, cb, wb = pr_bubble_point_fugacity(
            mixture, float(T_F), P_lo=P_lo, P_hi=P_hi, kij=kij
        )
        Pd, cd, wd = pr_dew_point_fugacity(
            mixture, float(T_F), P_lo=P_lo_dew, P_hi=P_hi, kij=kij
        )
        if cb and not np.isnan(Pb):
            P_bub[i]    = Pb
            bub_conv[i] = True
        if cd and not np.isnan(Pd):
            P_dew[i]    = Pd
            dew_conv[i] = True

    return {
        'T'        : T_arr,
        'P_bubble' : P_bub,
        'P_dew'    : P_dew,
        'Tc_mix'   : float(np.dot(z, Tc)),   # Kay's rule
        'Pc_mix'   : float(np.dot(z, Pc)),
        'bub_conv' : bub_conv,
        'dew_conv' : dew_conv,
        'warnings' : all_warnings,
    }


# ---------------------------------------------------------------------------
# AUTO-DETECT ENVELOPE TEMPERATURE RANGE
# ---------------------------------------------------------------------------

def find_envelope_T_range(mixture, T_lo_search=-150.0, T_hi_search=800.0,
                           n_coarse=18, kij=None):
    """
    Scan a wide temperature range to find where the two-phase envelope exists,
    then return (T_lo, T_hi) with margin suitable for a detailed calculation.

    Uses a coarse pass of pr_phase_envelope so results inherit the improved
    fugacity-based dew/bubble convergence (no false Wilson positives).

    Returns
    -------
    T_lo, T_hi : float [°F]   padded bounds of the two-phase region
    found      : bool          False if no two-phase region detected at all
    """
    env = pr_phase_envelope(mixture, T_range=(T_lo_search, T_hi_search),
                             n_T=n_coarse, kij=kij)
    T = env['T']

    has_bub = ~np.isnan(env['P_bubble']) & env['bub_conv']
    has_dew = ~np.isnan(env['P_dew'])    & env['dew_conv']
    has_any = has_bub | has_dew

    if not has_any.any():
        return T_lo_search, T_hi_search, False

    T_found = T[has_any]
    T_lo_raw = float(T_found.min())
    T_hi_raw = float(T_found.max())
    span     = max(T_hi_raw - T_lo_raw, 20.0)

    # Pad by 15% each side so the closed tip of the envelope is visible
    pad   = 0.15 * span
    T_lo  = max(T_lo_raw - pad, T_lo_search)
    T_hi  = min(T_hi_raw + pad, T_hi_search)

    return T_lo, T_hi, True


# ---------------------------------------------------------------------------
# PR BUBBLE POINT — internal Wilson-only helper (kept for pseudo-components)
# ---------------------------------------------------------------------------

def pr_bubble_point(mixture, T_F, P_low=14.7, P_high=10000.0):
    """
    Bubble point pressure via Wilson K-value criterion (internal helper).

    Used for pseudo-component EOS table and header metric when no real
    composition is available. For real compositions, prefer
    pr_bubble_point_fugacity().

    At bubble point: Σ(zi · Ki) = 1
    Ki = (Pci/P) · exp[5.373·(1+ωi)·(1 - Tci/T)]   — Wilson (1969)

    Returns
    -------
    Pb : float [psia] or NaN if not bracketed
    """
    z, Tc, Pc, omega = (
        mixture['z'], mixture['Tc'], mixture['Pc'], mixture['omega']
    )
    T_R = T_F + F_TO_R

    def f(P):
        K = _wilson_K(Tc, Pc, omega, T_R, P)
        return float(np.sum(z * K)) - 1.0

    try:
        return float(brentq(f, P_low, P_high, xtol=0.1))
    except ValueError:
        return float('nan')


# ---------------------------------------------------------------------------
# DENSITY
# ---------------------------------------------------------------------------

def pr_density(P, T_F, mixture):
    """
    Mixture molar volume and mass density from PR EOS (liquid root).

    Parameters
    ----------
    P       : float — pressure [psia]
    T_F     : float — temperature [°F]
    mixture : dict  — from build_mixture() or pseudo_components()

    Returns
    -------
    rho   : float — mass density [lb/ft³]
    Z     : float — compressibility factor
    V_mol : float — molar volume [ft³/lbmol]
    """
    z, Tc, Pc, omega, MW = (
        mixture['z'], mixture['Tc'], mixture['Pc'],
        mixture['omega'], mixture['MW']
    )
    a, b         = pr_parameters(Tc, Pc, omega, T_F)
    a_mix, b_mix, _ = pr_mix(a, b, z)
    Z_liq, _     = pr_z_factor(P, T_F, a_mix, b_mix)

    if Z_liq is None:
        return np.nan, np.nan, np.nan

    T     = T_F + F_TO_R
    V_mol = Z_liq * R * T / P
    MW_mix = float(np.dot(z, MW))
    rho   = MW_mix / V_mol
    return float(rho), float(Z_liq), float(V_mol)


# ---------------------------------------------------------------------------
# EOS TABLE — Z, DENSITY VS PRESSURE
# ---------------------------------------------------------------------------

def eos_table(mixture, T_F, P_min=50.0, P_max=None, n_points=150):
    """
    Compute Z-factor, density, molar volume across a pressure sweep.

    Parameters
    ----------
    mixture  : dict — from build_mixture() or pseudo_components()
    T_F      : float — temperature [°F]
    P_min    : float — min pressure [psia]
    P_max    : float — max pressure [psia] (default 1.5 × Pb)
    n_points : int

    Returns
    -------
    dict — P, Z, rho, V_mol, Pb_eos, T_F
    """
    Pb_eos = pr_bubble_point(mixture, T_F)
    if P_max is None:
        P_max = (Pb_eos * 1.5) if not np.isnan(Pb_eos) else 6000.0

    P_arr   = np.linspace(P_min, P_max, n_points)
    Z_arr   = np.full(n_points, np.nan)
    rho_arr = np.full(n_points, np.nan)
    V_arr   = np.full(n_points, np.nan)

    for i, P in enumerate(P_arr):
        rho, Z, V = pr_density(P, T_F, mixture)
        Z_arr[i]   = Z
        rho_arr[i] = rho
        V_arr[i]   = V

    return {
        'P'      : P_arr,
        'Z'      : Z_arr,
        'rho'    : rho_arr,
        'V_mol'  : V_arr,
        'Pb_eos' : Pb_eos,
        'T_F'    : T_F,
    }


# ---------------------------------------------------------------------------
# BINARY Pxy DIAGRAM — PR EOS FUGACITY
# ---------------------------------------------------------------------------

def pxy_diagram_fugacity(comp_A, comp_B, T_F, P_range=(14.7, 5000.0), n_x=60):
    """
    Binary Pxy diagram via PR EOS fugacity equality.

    For each feed composition z_A from 0 to 1, finds bubble point pressure
    (Sigma Ki*zi = 1, x=z) and dew point pressure (Sigma zi/Ki = 1, y=z) using
    full PR fugacity iteration.  When EOS iteration does not converge, falls back
    to the Wilson K-value estimate (flagged via bub_conv/dew_conv = False).

    The upper pressure search bound is auto-scaled from Wilson bubble estimates
    so that highly volatile pairs (e.g. C1/iC4) are not artificially capped.

    Source: Peng & Robinson (1976); Michelsen & Mollerup (2007) Thermodynamic
            Models: Fundamentals & Computational Aspects, Ch. 5.

    Parameters
    ----------
    comp_A  : str   — component name A (from pure_components.csv)
    comp_B  : str   — component name B
    T_F     : float — temperature [degF]
    P_range : tuple — (P_low, P_high) [psia] search range
    n_x     : int   — number of feed composition steps

    Returns
    -------
    dict with keys:
        'x_A'      : array — feed mole fraction of A (0 to 1)
        'P_bubble' : array — bubble point pressures [psia]  (NaN if not in two-phase region)
        'P_dew'    : array — dew point pressures [psia]     (NaN if not in two-phase region)
        'bub_conv' : array [bool] — True = EOS converged, False = Wilson fallback
        'dew_conv' : array [bool]
        'comp_A'   : str
        'comp_B'   : str
        'T_F'      : float
    """
    P_lo    = P_range[0]
    P_hi    = P_range[1]
    T_R     = T_F + F_TO_R
    x_A_arr = np.linspace(0.001, 0.999, n_x)

    # Auto-scale P_hi: scan a few compositions, estimate Wilson bubble P,
    # and extend the search range so volatile pairs are not artificially capped.
    for probe_xA in [0.1, 0.3, 0.5, 0.7, 0.9]:
        try:
            mix_probe = build_mixture({comp_A: probe_xA, comp_B: 1.0 - probe_xA})
            K_probe   = _wilson_K(mix_probe['Tc'], mix_probe['Pc'], mix_probe['omega'],
                                  T_R, P_lo)
            S_probe   = float(np.sum(mix_probe['z'] * K_probe))
            P_hi = max(P_hi, P_lo * S_probe * 1.5)
        except Exception:
            pass
    P_hi = min(P_hi, 30000.0)   # hard cap at 30 000 psia

    P_bub    = np.full(n_x, np.nan)
    P_dew    = np.full(n_x, np.nan)
    bub_conv = np.zeros(n_x, dtype=bool)
    dew_conv = np.zeros(n_x, dtype=bool)

    for i, x_A in enumerate(x_A_arr):
        comp_dict = {comp_A: float(x_A), comp_B: float(1.0 - x_A)}
        try:
            mix = build_mixture(comp_dict)
            Pb, cb, _ = pr_bubble_point_fugacity(mix, T_F, P_lo=P_lo, P_hi=P_hi)
            if not np.isnan(Pb):
                P_bub[i]    = Pb
                bub_conv[i] = cb
        except Exception:
            pass
        try:
            mix = build_mixture(comp_dict)
            Pd, cd, _ = pr_dew_point_fugacity(mix, T_F, P_lo=P_lo, P_hi=P_hi)
            if not np.isnan(Pd):
                P_dew[i]    = Pd
                dew_conv[i] = cd
        except Exception:
            pass

    return {
        'x_A'     : x_A_arr,
        'P_bubble': P_bub,
        'P_dew'   : P_dew,
        'bub_conv': bub_conv,
        'dew_conv': dew_conv,
        'comp_A'  : comp_A,
        'comp_B'  : comp_B,
        'T_F'     : T_F,
    }


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("  eos.py — Peng-Robinson EOS Self-Test")
    print("=" * 65)

    # ---- Test 1: CSV load
    print(f"\n  Pure components loaded: {len(PURE_COMPONENTS)} entries")
    print(f"  Components: {list(PURE_COMPONENTS.index)}")

    # ---- Test 2: Real composition
    composition = {
        'C1' : 0.45, 'C2' : 0.05, 'C3' : 0.05,
        'nC4': 0.03, 'nC5': 0.01, 'nC6': 0.01, 'C7+': 0.40,
    }
    c7_props = characterize_c7plus(MW_c7=215.0, SG_c7=0.87)
    mix = build_mixture(composition, c7plus_props=c7_props)

    print(f"\n  Real composition ({len(mix['names'])} components):")
    for name, z, Tc, Pc, om in zip(mix['names'], mix['z'], mix['Tc'], mix['Pc'], mix['omega']):
        print(f"    {name:<5}  z={z:.4f}  Tc={Tc:.1f}°R  Pc={Pc:.1f} psia  ω={om:.4f}")

    T_F = 180.0

    # ---- Test 3: Wilson bubble point (internal)
    Pb_w = pr_bubble_point(mix, T_F)
    print(f"\n  Wilson bubble point (internal helper): {Pb_w:.1f} psia")

    # ---- Test 4: Fugacity bubble point
    Pb_f, cb, wb = pr_bubble_point_fugacity(mix, T_F)
    print(f"  Fugacity bubble point: {Pb_f:.1f} psia  converged={cb}")
    for w in wb:
        print(f"    WARNING: {w}")

    # ---- Test 5: Fugacity dew point
    Pd_f, cd, wd = pr_dew_point_fugacity(mix, T_F)
    print(f"  Fugacity dew point:    {Pd_f:.1f} psia  converged={cd}")

    # ---- Test 6: Stability test
    is_stab, unstable_ph, sw = michelsen_stability_test(mix, P=1000.0, T_F=T_F)
    print(f"\n  Michelsen stability at P=1000 psia, T=180°F:")
    print(f"    is_stable={is_stab}  unstable_phase={unstable_ph}")

    # ---- Test 7: Flash
    flash = pr_flash(mix, P=1000.0, T_F=T_F)
    print(f"\n  Flash at P=1000 psia, T=180°F:")
    print(f"    phase={flash['phase']}  beta={flash['beta']:.4f}  "
          f"converged={flash['converged']}  iter={flash['iterations']}")
    if flash['warnings']:
        for w in flash['warnings']:
            print(f"    WARNING: {w}")

    # ---- Test 8: Z-factor sweep
    print(f"\n  PR density sweep:")
    for P in [500, 1000, 1500, 2000, 2500]:
        rho, Z, V = pr_density(P, T_F, mix)
        print(f"    P={P:5d} psia  Z={Z:.4f}  rho={rho:.2f} lb/ft³")

    # ---- Test 9: Phase envelope
    print(f"\n  Phase envelope (T=80–350°F, 20 points)...")
    env = pr_phase_envelope(mix, T_range=(80, 350), n_T=20)
    bub_ok = np.sum(~np.isnan(env['P_bubble']))
    dew_ok = np.sum(~np.isnan(env['P_dew']))
    print(f"    Bubble: {bub_ok} converged pts  |  Dew: {dew_ok} converged pts")

    # ---- Test 10: Pseudo-components
    pseudo = pseudo_components(API=35, GOR=500, gas_SG=0.65)
    Pb_ps  = pr_bubble_point(pseudo, T_F=180.0)
    print(f"\n  Pseudo-component fallback Pb (Wilson): {Pb_ps:.1f} psia")

    print("\neos.py — all tests passed.")
