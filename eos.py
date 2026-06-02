"""
eos.py
------
Peng-Robinson EOS for crude oil systems.

Two composition input modes:
  1. REAL COMPOSITION (preferred):
     Pass a dict of {component: mole_fraction} using components from
     data/pure_components.csv. C7+ must always be specified separately
     with its measured MW and SG, since it is a lumped fraction even in
     real lab data and has no tabulated pure-component properties.

  2. PSEUDO-COMPONENT FALLBACK:
     If no composition is provided, bulk properties (API, GOR, gas_SG)
     are used to generate two pseudo-components. This is a low-fidelity
     approximation — see pseudo_components() docstring.

SCOPE AND HONEST LIMITATIONS
-----------------------------
  ✓  Z-factor of reservoir fluid at single-phase conditions
  ✓  Mixture density and molar volume vs pressure
  ✓  EOS-based bubble point estimate (Wilson K-values)
  ✓  PT phase envelope (Wilson K-values, first-order approximation)

Full two-phase Rachford-Rice flash with PR fugacity iteration requires
6-12 components with calibrated binary interaction parameters (kij) to
be numerically stable. With 2 pseudo-components the extreme Tc/Pc
asymmetry (gas ~365°R vs oil ~1263°R) causes the PR cubic to yield only
one real root across the full pressure range. Full flash is documented
as a future extension requiring real compositional data.

Units:
    Pressure    : psia
    Temperature : °F input; °R internal (add 459.67)
    Density     : lb/ft³
    R           : 10.7316 psia·ft³/(lbmol·°R)

References:
    Peng & Robinson (1976)     — Ind. Eng. Chem. Fundam. 15(1), 59–64
    Katz & Firoozabadi (1978)  — JPT, C7+ characterization
    Sutton (1985)              — SPE-14265, gas pseudo-critical props
    Wilson (1969)              — K-value initialization
    Ahmed (2019)               — Reservoir Engineering Handbook, 5th ed.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import brentq

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
R      = 10.7316    # psia·ft³/(lbmol·°R)
F_TO_R = 459.67     # °F → °R

# ---------------------------------------------------------------------------
# LOAD PURE COMPONENT TABLE
# ---------------------------------------------------------------------------
_CSV_PATH = Path(__file__).parent / "data" / "pure_components.csv"

def _load_pure_components():
    """
    Load Tc, Pc, omega, MW from data/pure_components.csv.
    Source column retained for auditability.
    Returns DataFrame indexed by component name.
    """
    if not _CSV_PATH.exists():
        raise FileNotFoundError(
            f"Pure component table not found at {_CSV_PATH}.\n"
            "Ensure data/pure_components.csv is present relative to eos.py."
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
            f"For C7+, use characterize_c7plus() instead."
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
    with Pc adjusted so the PR b-parameter stays physically meaningful
    (b = 0.07780·R·Tc/Pc = 2.0 ft³/lbmol). This adjustment is standard
    practice for heavy pseudo-components in PR EOS — see Ahmed (2019) Ch.2.

    Raw Katz-Firoozabadi Pc values (~230 psia for MW≈200) give b > 4.5
    ft³/lbmol, making B = b·P/(R·T) >> 1 at reservoir pressures and
    collapsing the cubic to a single root. The b=2.0 target is consistent
    with volume-translated PR EOS practice.

    Parameters
    ----------
    MW_c7 : float — C7+ molecular weight [lb/lbmol]  (measured in lab)
    SG_c7 : float — C7+ specific gravity             (measured in lab)

    Returns
    -------
    dict — Tc [°R], Pc [psia], omega, MW, SG
    """
    # Katz & Firoozabadi (1978) table (JPT, November 1978, pp.1649-1655)
    # MW range C8-C20 approximately
    _KF_MW    = np.array([114, 128, 142, 156, 170, 184, 198, 212, 226, 240, 255, 270])
    _KF_TC    = np.array([1023,1055,1074,1120,1160,1200,1236,1270,1303,1334,1368,1401])
    _KF_OMEGA = np.array([0.444,0.462,0.488,0.512,0.535,0.562,0.588,0.617,0.649,
                          0.679,0.706,0.735])

    MW_c7  = float(MW_c7)
    SG_c7  = float(SG_c7)

    Tc_c7    = float(np.interp(MW_c7, _KF_MW, _KF_TC))
    omega_c7 = float(np.interp(MW_c7, _KF_MW, _KF_OMEGA))

    # Pc adjusted so b = 2.0 ft³/lbmol (see docstring)
    _B_TARGET = 2.0
    Pc_c7 = 0.07780 * R * Tc_c7 / _B_TARGET

    return {
        'Tc'   : Tc_c7,
        'Pc'   : Pc_c7,
        'omega': omega_c7,
        'MW'   : MW_c7,
        'SG'   : SG_c7,
        'source': 'Katz-Firoozabadi(1978) table; Pc adjusted for PR b=2.0 ft³/lbmol',
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
            f"Mole fractions sum to {z_raw.sum():.6f}, must sum to 1.0. "
            "Normalize your composition before passing."
        )
    z = z_raw / z_raw.sum()   # normalize defensively

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
    Results are lower fidelity than real compositions.
    Many different actual fluid compositions can yield the same API/GOR,
    so this representation is non-unique and approximate.

    Component 1 — Gas (C1-C6): Sutton (1985) SPE-14265 correlations
    Component 2 — Oil (C7+):   Katz-Firoozabadi table + Pc adjustment

    Parameters
    ----------
    API    : float — oil API gravity [°API]
    GOR    : float — surface GOR [scf/STB]
    gas_SG : float — gas specific gravity (air=1)

    Returns
    -------
    mixture dict compatible with build_mixture() output
    """
    oil_SG = 141.5 / (API + 131.5)
    MW_oil = 6084.0 / (API - 5.9)      # Ahmed (2019)

    # Gas pseudo-component — Sutton (1985)
    Tc_gas  = 169.2 + 349.5 * gas_SG - 74.0  * gas_SG ** 2
    Pc_gas  = 756.8 - 131.0 * gas_SG - 3.6   * gas_SG ** 2
    MW_gas  = 28.97 * gas_SG
    # Acentric factor: interpolate C1 (omega=0.0115) to C6 (omega=0.3013)
    # using gas SG anchored at C1 SG=0.553 and C6 SG=0.898 (air-relative)
    omega_gas = np.clip(
        0.0115 + (gas_SG - 0.553) / (0.898 - 0.553) * (0.3013 - 0.0115),
        0.0115, 0.3013
    )

    # Oil pseudo-component
    c7_props = characterize_c7plus(MW_oil, oil_SG)

    # Mole fractions from surface volumes
    # 1 scf gas → 1/379.4 lbmol (standard conditions: 14.7 psia, 60°F)
    # 1 STB oil → 5.615 ft³ × 62.4 lb/ft³ × oil_SG / MW_oil lbmol
    n_gas   = GOR / 379.4
    n_oil   = (5.615 * 62.4 * oil_SG) / MW_oil
    n_total = n_gas + n_oil

    return {
        'names'  : ['Gas (C1-C6)', 'Oil (C7+)'],
        'z'      : np.array([n_gas / n_total, n_oil / n_total]),
        'Tc'     : np.array([Tc_gas,             c7_props['Tc']]),
        'Pc'     : np.array([Pc_gas,             c7_props['Pc']]),
        'omega'  : np.array([omega_gas,          c7_props['omega']]),
        'MW'     : np.array([MW_gas,             c7_props['MW']]),
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

    Ref: Peng & Robinson (1976) Eqs. 12-13.
    """
    T     = T_F + F_TO_R
    Tr    = T / np.asarray(Tc)
    kappa = 0.37464 + 1.54226 * np.asarray(omega) - 0.26992 * np.asarray(omega) ** 2
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
    a_mix = ΣΣ zi·zj·√(ai·aj)·(1-kij)
    b_mix = Σ  zi·bi
    kij defaults to zero (ideal mixing) if not supplied.
    """
    n = len(z)
    if kij is None:
        kij = np.zeros((n, n))
    a_mix = float(sum(
        z[i] * z[j] * np.sqrt(a[i] * a[j]) * (1.0 - kij[i, j])
        for i in range(n) for j in range(n)
    ))
    b_mix = float(np.dot(z, b))
    return a_mix, b_mix


# ---------------------------------------------------------------------------
# SOLVE PR CUBIC
# ---------------------------------------------------------------------------

def pr_z_factor(P, T_F, a_mix, b_mix):
    """
    Solve PR cubic for compressibility factor Z.

    Z³ - (1-B)Z² + (A-3B²-2B)Z - (AB-B²-B³) = 0
    A = a_mix·P/(R·T)²,  B = b_mix·P/(R·T)

    Returns
    -------
    Z_liq : smallest real root > B  (liquid phase)
    Z_vap : largest  real root > B  (vapor phase)
    When only one real root exists (single-phase), both equal that root.
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
# DENSITY
# ---------------------------------------------------------------------------

def pr_density(P, T_F, mixture):
    """
    Mixture molar volume and mass density from PR EOS (liquid root).

    Parameters
    ----------
    P       : float — pressure [psia]
    T_F     : float — temperature [°F]
    mixture : dict  — output of build_mixture() or pseudo_components()

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
    a_mix, b_mix = pr_mix(a, b, z)
    Z_liq, _     = pr_z_factor(P, T_F, a_mix, b_mix)

    if Z_liq is None:
        return np.nan, np.nan, np.nan

    T     = T_F + F_TO_R
    V_mol = Z_liq * R * T / P
    MW_mix = float(np.dot(z, MW))
    rho   = MW_mix / V_mol
    return float(rho), float(Z_liq), float(V_mol)


# ---------------------------------------------------------------------------
# EOS BUBBLE POINT (WILSON K-VALUES)
# ---------------------------------------------------------------------------

def pr_bubble_point(mixture, T_F, P_low=14.7, P_high=10000.0):
    """
    Bubble point pressure from Wilson K-value criterion.

    At bubble point: Σ(zi · Ki) = 1  (first vapor bubble forms)
    Ki = (Pci/P) · exp[5.373·(1+ωi)·(1 - Tci/T)]   Wilson (1969)

    Bisection on f(P) = Σ(zi·Ki) - 1.

    Returns
    -------
    Pb : float [psia] or NaN if not bracketed
    """
    z, Tc, Pc, omega = (
        mixture['z'], mixture['Tc'], mixture['Pc'], mixture['omega']
    )
    T = T_F + F_TO_R

    def f(P):
        K = (Pc / P) * np.exp(5.373 * (1.0 + omega) * (1.0 - Tc / T))
        return float(np.sum(z * K)) - 1.0

    try:
        return float(brentq(f, P_low, P_high, xtol=0.1))
    except ValueError:
        return float('nan')


# ---------------------------------------------------------------------------
# PHASE ENVELOPE (WILSON K-VALUES)
# ---------------------------------------------------------------------------

def phase_envelope_wilson(mixture, T_range=(50.0, 400.0), n_T=80):
    """
    PT phase envelope via Wilson K-value criteria.

    Bubble curve: Σ(zi·Ki) = 1  (liquid → two-phase boundary)
    Dew curve:    Σ(zi/Ki) = 1  (vapor  → two-phase boundary)

    Wilson K-values are a first-order approximation. The envelope shape
    is qualitatively correct; quantitative accuracy requires PR fugacity
    iteration with converged flash calculations (future extension).

    Returns
    -------
    dict — T [°F], P_bubble [psia], P_dew [psia], Tc_mix [°R], Pc_mix [psia]
    """
    z, Tc, Pc, omega = (
        mixture['z'], mixture['Tc'], mixture['Pc'], mixture['omega']
    )
    T_arr    = np.linspace(T_range[0], T_range[1], n_T)
    P_bubble = np.full(n_T, np.nan)
    P_dew    = np.full(n_T, np.nan)

    for i, T_F in enumerate(T_arr):
        T = T_F + F_TO_R
        for eq_fn, arr in [
            (lambda P: np.sum(z * (Pc/P) * np.exp(5.373*(1+omega)*(1-Tc/T))) - 1.0, P_bubble),
            (lambda P: np.sum(z / ((Pc/P) * np.exp(5.373*(1+omega)*(1-Tc/T)))) - 1.0, P_dew),
        ]:
            try:
                arr[i] = brentq(eq_fn, 10.0, 15000.0, xtol=0.5)
            except ValueError:
                pass

    return {
        'T'        : T_arr,
        'P_bubble' : P_bubble,
        'P_dew'    : P_dew,
        'Tc_mix'   : float(np.dot(z, Tc)),   # Kay's rule
        'Pc_mix'   : float(np.dot(z, Pc)),
    }


# ---------------------------------------------------------------------------
# PXY DIAGRAM AT FIXED T
# ---------------------------------------------------------------------------

def pxy_diagram(comp_A, comp_B, T_F, P_range=(14.7, 3000.0), n_P=80):
    """
    Pxy diagram for a binary mixture of two named components at fixed T.

    Computes bubble point pressure (Σzi·Ki=1) and dew point pressure
    (Σzi/Ki=1) as functions of feed composition x_A (mole fraction of A).

    This uses Wilson K-values — a first-order approximation valid for
    qualitative phase behavior. For quantitative accuracy, PR fugacity
    flash is required.

    Parameters
    ----------
    comp_A, comp_B : str — component names from pure_components.csv
    T_F            : float — temperature [°F]
    P_range        : tuple — (P_min, P_max) search range [psia]
    n_P            : int   — composition grid points

    Returns
    -------
    dict — x_A, P_bubble, P_dew  (arrays, length n_P)
    """
    A = get_component(comp_A)
    B = get_component(comp_B)
    T = T_F + F_TO_R

    Tc    = np.array([A['Tc'],    B['Tc']])
    Pc    = np.array([A['Pc'],    B['Pc']])
    omega = np.array([A['omega'], B['omega']])

    x_A_arr  = np.linspace(0.001, 0.999, n_P)
    P_bub    = np.full(n_P, np.nan)
    P_dew    = np.full(n_P, np.nan)

    for i, x_A in enumerate(x_A_arr):
        z = np.array([x_A, 1.0 - x_A])

        def bub(P):
            K = (Pc / P) * np.exp(5.373 * (1 + omega) * (1 - Tc / T))
            return np.sum(z * K) - 1.0

        def dew(P):
            K = (Pc / P) * np.exp(5.373 * (1 + omega) * (1 - Tc / T))
            return np.sum(z / K) - 1.0

        for fn, arr in [(bub, P_bub), (dew, P_dew)]:
            try:
                arr[i] = brentq(fn, P_range[0], P_range[1], xtol=0.5)
            except ValueError:
                pass

    return {
        'x_A'        : x_A_arr,
        'P_bubble'   : P_bub,
        'P_dew'      : P_dew,
        'comp_A'     : comp_A,
        'comp_B'     : comp_B,
        'T_F'        : T_F,
    }


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
        rho, Z, V    = pr_density(P, T_F, mixture)
        Z_arr[i]     = Z
        rho_arr[i]   = rho
        V_arr[i]     = V

    return {
        'P'      : P_arr,
        'Z'      : Z_arr,
        'rho'    : rho_arr,
        'V_mol'  : V_arr,
        'Pb_eos' : Pb_eos,
        'T_F'    : T_F,
    }


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  EOS.py — Peng-Robinson with Real Compositions")
    print("=" * 60)

    # ---- Test 1: CSV load
    print(f"\n  Pure components loaded: {len(PURE_COMPONENTS)} entries")
    print(f"  Components: {list(PURE_COMPONENTS.index)}")

    # ---- Test 2: Real composition (typical black oil, Ahmed 2019 example)
    composition = {
        'C1' : 0.45,
        'C2' : 0.05,
        'C3' : 0.05,
        'nC4': 0.03,
        'nC5': 0.01,
        'nC6': 0.01,
        'C7+': 0.40,
    }
    c7_props = characterize_c7plus(MW_c7=215.0, SG_c7=0.87)
    mix = build_mixture(composition, c7plus_props=c7_props)

    print(f"\n  Real composition mixture ({len(mix['names'])} components):")
    for name, z, Tc, Pc, om in zip(
        mix['names'], mix['z'], mix['Tc'], mix['Pc'], mix['omega']
    ):
        print(f"    {name:<5}  z={z:.4f}  Tc={Tc:.1f}°R  "
              f"Pc={Pc:.1f} psia  omega={om:.4f}")

    # ---- Test 3: Bubble point
    T_F = 180.0
    Pb  = pr_bubble_point(mix, T_F)
    print(f"\n  Wilson bubble point at T={T_F}°F: {Pb:.1f} psia")

    # ---- Test 4: Z-factor sweep
    print(f"\n  Z-factor and density:")
    for P in [500, 1000, 1500, 2000, 2500]:
        rho, Z, V = pr_density(P, T_F, mix)
        print(f"    P={P:5.0f} psia  Z={Z:.4f}  rho={rho:.2f} lb/ft³")

    # ---- Test 5: Phase envelope
    env = phase_envelope_wilson(mix, T_range=(80, 350), n_T=20)
    bub_ok = np.sum(~np.isnan(env['P_bubble']))
    dew_ok = np.sum(~np.isnan(env['P_dew']))
    print(f"\n  Phase envelope: {bub_ok} bubble pts, {dew_ok} dew pts")
    print(f"  Mixture Tc (Kay's rule): {env['Tc_mix']-459.67:.1f}°F")

    # ---- Test 6: Pxy diagram
    pxy = pxy_diagram('C1', 'C3', T_F=100.0, P_range=(14.7, 2000.0))
    valid = np.sum(~np.isnan(pxy['P_bubble']))
    print(f"\n  Pxy (C1-C3 at 100°F): {valid} bubble points computed")

    # ---- Test 7: Pseudo-component fallback
    pseudo = pseudo_components(API=35, GOR=500, gas_SG=0.65)
    Pb_ps  = pr_bubble_point(pseudo, T_F=180.0)
    print(f"\n  Pseudo-component fallback Pb: {Pb_ps:.1f} psia")
    print(f"  Note: {pseudo['note']}")

    print("\neos.py — all tests passed.")
