"""
app.py
------
Streamlit interactive frontend for the PVT Analyzer.

Run with:
    streamlit run app.py

Tabs
----
  1. Correlations  — Rs, Bo, co, viscosity curves + regional comparison
  2. Gas PVT       — Gas Z-factor, Bg, μg, ρg from gas correlations
  3. EOS           — PR EOS Z-factor, density, phase envelope
  4. 3D Surfaces   — interactive Plotly 3D (Bo, viscosity, Z, PVT)
  5. Phase         — PR fugacity phase envelope
  6. Composition   — enter real mole fractions from a lab PVT report
  7. About         — methodology, sources, references
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

# Project modules
from correlations import REGIONS, pvt_table, bubble_point, check_range
from eos import (
    pseudo_components, build_mixture, characterize_c7plus,
    eos_table, pr_bubble_point, pr_flash,
    michelsen_stability_test, PURE_COMPONENTS
)
from gas_pvt import gas_pvt_table
import plotting as P

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PVT Analyzer",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
    .assumption-box {
        background: #FFF8E1; border-left: 4px solid #FFC107;
        padding: 0.6rem 1rem; border-radius: 4px; margin: 0.5rem 0;
        font-size: 0.85rem; color: #5D4037;
    }
    .source-box {
        background: #E8F5E9; border-left: 4px solid #4CAF50;
        padding: 0.4rem 0.8rem; border-radius: 4px;
        font-size: 0.78rem; color: #1B5E20;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# SIDEBAR — FLUID INPUTS
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("🛢️ Fluid Properties")

    API    = st.slider("API Gravity [°API]", 15, 60, 35, 1,
                       help="Stock-tank oil API gravity.")
    GOR    = st.slider("Surface GOR [scf/STB]", 50, 2000, 500, 25,
                       help="Gas-oil ratio at surface (stock-tank) conditions.")
    T      = st.slider("Reservoir Temperature [°F]", 80, 320, 180, 5)
    gas_SG = st.slider("Gas Specific Gravity [-]", 0.55, 0.90, 0.65, 0.01,
                       help="Gas SG relative to air = 1.0.")

    st.divider()
    st.subheader("Correlation Region")
    region = st.selectbox(
        "Regional correlation",
        list(REGIONS.keys()),
        index=3,
        format_func=lambda r: REGIONS[r].split('—')[0].strip(),
    )

    st.divider()
    st.subheader("Vasquez & Beggs Separator Correction")
    with st.expander("What is this?", expanded=False):
        st.markdown(
            "V&B correlations were developed at a reference separator pressure "
            "of 100 psia. If your gas SG was measured at a different separator "
            "pressure, enter those conditions here to correct γ_g before "
            "applying the correlation."
        )
    use_sep_corr = st.checkbox("Apply separator correction (global region only)")
    T_sp = st.number_input("Separator temperature [°F]", 60.0, 200.0, 80.0,
                            disabled=not use_sep_corr)
    P_sp = st.number_input("Separator pressure [psia]", 14.7, 500.0, 114.7,
                            disabled=not use_sep_corr)

    st.divider()
    st.caption("Bulk inputs are used for correlations and the pseudo-component "
               "EOS fallback. For real compositions, use the Composition tab.")


# ---------------------------------------------------------------------------
# CACHED COMPUTATIONS
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_pvt(API, GOR, T, gas_SG, region, T_sp, P_sp):
    return pvt_table(API, GOR, T, gas_SG, region=region, T_sp=T_sp, P_sp=P_sp)


@st.cache_data(show_spinner=False)
def compute_pseudo(API, GOR, gas_SG):
    return pseudo_components(API, GOR, gas_SG)


@st.cache_data(show_spinner=False)
def compute_gas_pvt(gas_SG, T, y_CO2, y_H2S, fluid_type, z_method):
    return gas_pvt_table(
        gas_SG=gas_SG, T_F=T,
        y_CO2=y_CO2, y_H2S=y_H2S,
        fluid_type=fluid_type,
        z_method=z_method,
        P_min=14.7, P_max=10000.0, n_points=200,
    )


@st.cache_data(show_spinner=False)
def _compute_bo_grid_cached(API, GOR, gas_SG, T_range, P_range, region, n_T, n_P):
    return P._compute_bo_grid(API, GOR, gas_SG, T_range, P_range, region, n_T, n_P)


@st.cache_data(show_spinner=False)
def _compute_visc_grid_cached(API, GOR, gas_SG, T_range, P_range, region, n_T, n_P):
    return P._compute_visc_grid(API, GOR, gas_SG, T_range, P_range, region, n_T, n_P)


@st.cache_data(show_spinner=False)
def _compute_z_grid_cached(API, GOR, gas_SG, T_range, P_range, n_T, n_P):
    return P._compute_z_grid(API, GOR, gas_SG, T_range, P_range, n_T, n_P)


@st.cache_data(show_spinner=False)
def _compute_pvt_grid_cached(API, GOR, gas_SG, T_range, P_range, n_T, n_P):
    return P._compute_pvt_grid(API, GOR, gas_SG, T_range, P_range, n_T, n_P)


# ---------------------------------------------------------------------------
# COMPUTE SHARED STATE (runs on every slider change — no button guard)
# ---------------------------------------------------------------------------

sep_kw = {'T_sp': T_sp, 'P_sp': P_sp} if use_sep_corr else {'T_sp': 60.0, 'P_sp': 114.7}

tbl    = compute_pvt(API, GOR, T, gas_SG, region, **sep_kw)
pseudo = compute_pseudo(API, GOR, gas_SG)
Pb_corr = tbl['Pb']
Pb_eos  = pr_bubble_point(pseudo, T)

# Show any range warnings in the sidebar
range_warns = check_range(GOR, T, API, region)
if range_warns:
    with st.sidebar:
        st.divider()
        for w in range_warns:
            st.warning(w)


# ---------------------------------------------------------------------------
# HEADER METRICS
# ---------------------------------------------------------------------------

st.title("🛢️ PVT Analyzer")
st.caption(
    "Empirical correlations (Standing, Al-Marhoun, Glaso, Vasquez & Beggs, "
    "Petrosky & Farshad) + Peng-Robinson EOS with real or pseudo-compositions. "
    "Gas PVT: Hall-Yarborough (1974) / DPR (1974) Z-factor, Lee-Gonzalez-Eakin (1966) viscosity."
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("API Gravity",    f"{API}°API")
c2.metric("Surface GOR",    f"{GOR} scf/STB")
c3.metric("Temperature",    f"{T}°F")
c4.metric("Pb (Corr.)",     f"{Pb_corr:.0f} psia")
c5.metric("Pb (EOS Wilson)",f"{Pb_eos:.0f} psia" if not np.isnan(Pb_eos) else "—")

st.divider()


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------

tabs = st.tabs([
    "📈 Correlations",
    "💨 Gas PVT",
    "⚗️ EOS",
    "🌐 3D Surfaces",
    "🔀 Phase",
    "🧪 Composition",
    "📖 About",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — CORRELATIONS
# ════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.subheader("Empirical Oil PVT Correlations")

    # Show any range warnings as st.warning() — not printed to console
    if tbl['warnings']:
        for w in tbl['warnings']:
            st.warning(w)

    st.markdown(
        '<div class="assumption-box">'
        '<b>Viscosity:</b> Western USA uses Beal (1946) dead oil + Chew & Connally (1959) '
        'saturated. All other regions use Beggs & Robinson (1975). '
        'Undersaturated correction: Vasquez & Beggs (1980) for all regions. '
        '<b>Compressibility:</b> Vasquez & Beggs (1980) SPE-6719 Eq. 2-47 '
        '(co = (5Rs + 17.2T − 1180γg + 12.61API − 1433) / (10⁵ P)); '
        'defined only above Pb. '
        'Petrosky & Farshad results are flagged if Pb ≤ 14.7 psia '
        '(outside development range).'
        '</div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([2, 1])

    with col_left:
        fig_pvt, _ = P.plot_pvt_curves(API, GOR, T, gas_SG,
                                        region=region, **sep_kw)
        st.pyplot(fig_pvt, use_container_width=True)

    with col_right:
        st.markdown("**Bubble Point Summary**")
        pb_data = []
        for r in REGIONS:
            pb   = bubble_point(API, GOR, T, gas_SG, region=r)
            flag = " ⚠️ out-of-range" if pb <= 14.8 else ""
            pb_data.append({
                'Region': REGIONS[r].split('—')[0].strip(),
                'Pb [psia]': f"{pb:.0f}{flag}",
            })
        st.dataframe(pd.DataFrame(pb_data), use_container_width=True,
                     hide_index=True)

        st.markdown("**Selected region Pb**")
        st.metric(REGIONS[region].split('—')[0].strip(), f"{Pb_corr:.1f} psia")

    st.divider()
    st.subheader("Regional Comparison")
    fig_reg, _ = P.plot_regional_comparison(API, GOR, T, gas_SG)
    st.pyplot(fig_reg, use_container_width=True)

    st.divider()
    st.subheader("PVT Data Table")

    # Build table including co
    co_vals = tbl['co']
    df_tbl = pd.DataFrame({
        'P [psia]'        : tbl['P'].round(1),
        'Rs [scf/STB]'    : tbl['Rs'].round(2),
        'Bo [RB/STB]'     : tbl['Bo'].round(4),
        'μo [cp]'         : tbl['mu_o'].round(4),
        'co [psi⁻¹]'      : co_vals,   # NaN below Pb is acceptable
    })
    # Format co column — NaN shown as "< Pb"
    df_display = df_tbl.copy()
    df_display['co [psi⁻¹]'] = df_tbl['co [psi⁻¹]'].apply(
        lambda v: f"{v:.3e}" if not np.isnan(v) else "< Pb"
    )
    st.dataframe(df_display, use_container_width=True, height=300)

    csv_tbl = df_tbl.copy()
    csv_tbl['co [psi⁻¹]'] = csv_tbl['co [psi⁻¹]'].fillna('')
    st.download_button(
        "⬇️ Download PVT table as CSV",
        csv_tbl.to_csv(index=False),
        file_name=f"pvt_table_API{API}_GOR{GOR}.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — GAS PVT
# ════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.subheader("Gas PVT Properties")

    st.markdown(
        '<div class="source-box">'
        '<b>Pseudocritical properties:</b> '
        'Sutton (1985) SPE-14265 (dry gas) or Standing (1977) (gas condensate). '
        'Acid-gas correction: Wichert & Aziz (1972) Hydrocarbon Processing, May 1972. '
        '<b>Z-factor:</b> Hall & Yarborough (1974) SPE-3350-PA (Newton-Raphson) '
        'or Dranchuk, Purvis & Robinson (1974) IP 74-008. '
        '<b>Viscosity:</b> Lee, Gonzalez & Eakin (1966) SPE-1340-PA. '
        '<b>Bg</b> from real-gas law. <b>cg</b> by numerical dZ/dP.'
        '</div>',
        unsafe_allow_html=True,
    )

    g_col1, g_col2, g_col3 = st.columns(3)

    with g_col1:
        st.markdown("**Gas Description**")
        gas_SG_g    = st.slider("Gas SG [-]", 0.55, 1.20, 0.65, 0.01, key='g_sg')
        T_g         = st.slider("Temperature [°F]", 80, 320, 200, 5,    key='g_T')
        fluid_type  = st.selectbox("Fluid type",
                                    ["dry_gas", "gas_condensate"],
                                    format_func=lambda x: x.replace('_', ' ').title(),
                                    key='g_ft')

    with g_col2:
        st.markdown("**Acid Gas Fractions**")
        y_CO2 = st.slider("CO2 mole fraction [-]", 0.0, 0.30, 0.0, 0.01, key='g_co2')
        y_H2S = st.slider("H2S mole fraction [-]", 0.0, 0.30, 0.0, 0.01, key='g_h2s')
        if y_CO2 + y_H2S > 0.0:
            st.info(f"Wichert-Aziz (1972) acid-gas correction will be applied "
                    f"(A = {y_CO2+y_H2S:.3f}).")

    with g_col3:
        st.markdown("**Z-Factor Method**")
        z_method = st.radio(
            "Method",
            ["hall_yarborough", "dpr"],
            format_func=lambda x: {
                "hall_yarborough": "Hall-Yarborough (1974) [primary]",
                "dpr":             "Dranchuk-Purvis-Robinson (1974) [alternative]",
            }[x],
            key='g_zmethod',
        )
        if z_method == "dpr":
            st.info("DPR validity: 1.05 ≤ Tpr ≤ 3.0, 0 < Ppr ≤ 3.0. "
                    "Values outside this range are extrapolations.")

    with st.spinner("Computing gas PVT…"):
        gas_tbl = compute_gas_pvt(gas_SG_g, T_g, y_CO2, y_H2S, fluid_type, z_method)

    # Show warnings
    for w in gas_tbl['warnings']:
        st.warning(w)

    # Pseudocritical info
    Tpc_disp = gas_tbl['Tpc']
    Ppc_disp = gas_tbl['Ppc']
    Tpr_disp = (T_g + 459.67) / Tpc_disp
    st.markdown(
        f'<div class="assumption-box">'
        f'Tpc = {Tpc_disp:.2f} °R ({Tpc_disp - 459.67:.2f} °F)  |  '
        f'Ppc = {Ppc_disp:.2f} psia  |  '
        f'Tpr = {Tpr_disp:.4f}  |  Mg = {gas_tbl["Mg"]:.3f} lb/lbmol'
        + (f'  |  Raw Tpc = {gas_tbl["Tpc_raw"]:.2f} °R (before Wichert-Aziz)'
           if abs(gas_tbl["Tpc"] - gas_tbl["Tpc_raw"]) > 0.01 else '')
        + '</div>',
        unsafe_allow_html=True,
    )

    if Tpr_disp < 1.05:
        st.error(
            f"Tpr = {Tpr_disp:.4f} < 1.05. Hall-Yarborough correlation is "
            "undefined below Tpr = 1.05 (Hall & Yarborough 1974). "
            "Increase temperature or reduce gas SG."
        )

    # 4-panel gas PVT plot
    fig_gas, _ = P.plot_gas_pvt_curves(gas_tbl)
    st.pyplot(fig_gas, use_container_width=True)

    # Table
    st.divider()
    st.subheader("Gas PVT Data Table")
    valid_mask = ~np.isnan(gas_tbl['Z'])
    df_gas = pd.DataFrame({
        'P [psia]'      : gas_tbl['P'][valid_mask].round(1),
        'Z [-]'         : gas_tbl['Z'][valid_mask].round(5),
        'Bg [ft³/scf]'  : gas_tbl['Bg_ft3'][valid_mask],
        'Bg [bbl/scf]'  : gas_tbl['Bg_bbl'][valid_mask],
        'ρg [lb/ft³]'   : gas_tbl['rho_g'][valid_mask].round(4),
        'μg [cp]'       : gas_tbl['mu_g'][valid_mask].round(6),
        'cg [psi⁻¹]'    : gas_tbl['cg'][valid_mask],
    })
    # Format small numbers
    for col in ['Bg [ft³/scf]', 'Bg [bbl/scf]', 'cg [psi⁻¹]']:
        df_gas[col] = df_gas[col].apply(lambda v: f"{v:.4e}" if not np.isnan(v) else "")

    st.dataframe(df_gas, use_container_width=True, height=300)

    csv_gas = pd.DataFrame({
        'P_psia'        : gas_tbl['P'][valid_mask],
        'Z'             : gas_tbl['Z'][valid_mask],
        'Bg_ft3_per_scf': gas_tbl['Bg_ft3'][valid_mask],
        'Bg_bbl_per_scf': gas_tbl['Bg_bbl'][valid_mask],
        'rho_g_lb_ft3'  : gas_tbl['rho_g'][valid_mask],
        'mu_g_cp'       : gas_tbl['mu_g'][valid_mask],
        'cg_psi_inv'    : gas_tbl['cg'][valid_mask],
    })
    st.download_button(
        "⬇️ Download gas PVT table as CSV",
        csv_gas.to_csv(index=False),
        file_name=f"gas_pvt_SG{gas_SG_g:.3f}_T{T_g}F.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — EOS
# ════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.subheader("Peng-Robinson EOS — Z-factor and Density")

    st.markdown(
        '<div class="assumption-box">'
        '<b>Limitations:</b> Uses bulk API/GOR/gas_SG to generate two '
        'pseudo-components (gas C1-C6 via Sutton 1985; oil C7+ via '
        'Katz-Firoozabadi 1978). C7+ critical pressure is adjusted so '
        'b = 2.0 ft³/lbmol (Ahmed 2019). '
        'Z-factors reflect the overall liquid-phase mixture. '
        'Full two-phase flash requires real compositional data '
        '(see Composition tab). '
        'Bubble point shown is from Wilson K-values (internal helper for '
        'pseudo-components only).'
        '</div>',
        unsafe_allow_html=True,
    )

    fig_eos, _ = P.plot_eos_curves(
        pseudo, T_F=T,
        label=f'API={API}°, GOR={GOR} scf/STB (pseudo-components)'
    )
    st.pyplot(fig_eos, use_container_width=True)

    st.divider()
    st.subheader("Correlations vs EOS — Bubble Point")
    fig_cmp, _ = P.plot_correlations_vs_eos(API, GOR, T, gas_SG, mixture=pseudo)
    st.pyplot(fig_cmp, use_container_width=True)

    st.divider()
    st.subheader("EOS Data Table")
    tbl_eos = eos_table(pseudo, T_F=T)
    df_eos  = pd.DataFrame({
        'P [psia]'          : tbl_eos['P'].round(1),
        'Z [-]'             : tbl_eos['Z'].round(4),
        'Density [lb/ft³]'  : tbl_eos['rho'].round(3),
        'V_mol [ft³/lbmol]' : tbl_eos['V_mol'].round(4),
    })
    st.dataframe(df_eos, use_container_width=True, height=300)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — 3D SURFACES
# ════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    st.subheader("Interactive 3D PVT Surfaces")
    st.caption(
        "Rotate: click and drag. Zoom: scroll. "
        "Hover for exact values. All surfaces computed on a P-T grid."
    )

    surface_choice = st.radio(
        "Select surface",
        ["Bo(P,T) — Oil FVF",
         "μo(P,T) — Viscosity",
         "Z(P,T)  — EOS Compressibility",
         "V(P,T)  — Molar Volume (P-V-T)"],
        horizontal=True,
    )

    col_T, col_P = st.columns(2)
    with col_T:
        T_min_3d = st.slider("T min [°F]", 80,  200, 100, 10, key='tmin3d')
        T_max_3d = st.slider("T max [°F]", 150, 350, 280, 10, key='tmax3d')
    with col_P:
        P_min_3d = st.slider("P min [psia]",  50,  500,  100, 50, key='pmin3d')
        P_max_3d = st.slider("P max [psia]", 1000, 9000, 5000, 500, key='pmax3d')

    T_range_3d = (float(T_min_3d), float(T_max_3d))
    P_range_3d = (float(P_min_3d), float(P_max_3d))

    with st.spinner("Computing surface…"):
        if surface_choice.startswith("Bo"):
            P_arr_g, T_arr_g, grid_g = _compute_bo_grid_cached(
                API, GOR, gas_SG, T_range_3d, P_range_3d, region, 30, 40
            )
            region_label = REGIONS[region].split('—')[0].strip()
            fig_3d = P.plot_3d_surface(
                P_arr_g, T_arr_g, grid_g,
                title=f'Oil FVF Bo(P,T) — {region_label}',
                z_label='Bo [RB/STB]', colorscale='Blues',
            )
        elif surface_choice.startswith("μ"):
            P_arr_g, T_arr_g, grid_g = _compute_visc_grid_cached(
                API, GOR, gas_SG, T_range_3d, P_range_3d, region, 30, 40
            )
            region_label = REGIONS[region].split('—')[0].strip()
            fig_3d = P.plot_3d_surface(
                P_arr_g, T_arr_g, grid_g,
                title=f'Oil Viscosity μo(P,T) — {region_label}',
                z_label='μo [cp]', colorscale='Reds',
            )
        elif surface_choice.startswith("Z"):
            P_arr_g, T_arr_g, grid_g = _compute_z_grid_cached(
                API, GOR, gas_SG, T_range_3d, P_range_3d, 25, 35
            )
            fig_3d = P.plot_3d_surface(
                P_arr_g, T_arr_g, grid_g,
                title=f'Z-Factor(P,T) — PR EOS — API={API}°',
                z_label='Z [-]', colorscale='Plasma',
            )
        else:
            P_arr_g, T_arr_g, grid_g = _compute_pvt_grid_cached(
                API, GOR, gas_SG, T_range_3d, P_range_3d, 25, 35
            )
            fig_3d = P.plot_3d_surface(
                P_arr_g, T_arr_g, grid_g,
                title=f'P-V-T Surface (Molar Volume) — API={API}°',
                z_label='V [ft³/lbmol]', colorscale='Turbo',
            )

    st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown(
        '<div class="assumption-box">'
        'Bo and μ surfaces use empirical correlations (selected region). '
        'Z and V surfaces use PR EOS with pseudo-components. '
        'Temperatures outside the correlation development range yield NaN '
        '(shown as gaps in the surface).'
        '</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — PHASE
# ════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.subheader("PT Phase Envelope")

    st.markdown(
        '<div class="source-box">'
        'Bubble and dew curves computed with PR EOS fugacity equality iteration '
        '(Peng & Robinson 1976 Eq. 18). Wilson K-values (Wilson 1969) used '
        'only as internal initialization; final results are fugacity-converged. '
        'Non-converged points are shown as gaps in the envelope. '
        'Source: Whitson & Brulé (2000) SPE Monograph Vol.20, Section 3.3.'
        '</div>',
        unsafe_allow_html=True,
    )

    T_env_min = st.slider("T range min [°F]", 50, 150, 80,  key='envmin')
    T_env_max = st.slider("T range max [°F]", 200, 500, 380, key='envmax')

    with st.spinner("Building phase envelope (fugacity iterations)…"):
        fig_env, _ = P.plot_phase_envelope(
            pseudo,
            T_range=(float(T_env_min), float(T_env_max)),
            label=f'API={API}°, GOR={GOR} (pseudo-components)'
        )
    st.pyplot(fig_env, use_container_width=True)

    st.markdown(
        '<div class="assumption-box">'
        '<b>Limitations:</b> Two pseudo-components (gas + C7+) yield extreme '
        'Tc/Pc asymmetry. The phase envelope for real fluids should use '
        '6–12 components with calibrated binary interaction parameters (kij). '
        'For higher accuracy, provide real composition in the Composition tab.'
        '</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — COMPOSITION
# ════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.subheader("Real Fluid Composition Input")
    st.markdown(
        "Enter mole fractions from a lab PVT compositional report. "
        "C7+ is always a lumped fraction and requires measured MW and SG. "
        "Results update automatically on every input change."
    )

    st.markdown(
        '<div class="source-box">'
        'Pure component Tc, Pc, ω from '
        '<b>data/pure_components.csv</b> — sourced from Ahmed (2019) '
        'Table 1-1 and Peng & Robinson (1976) Table 1. '
        'No values were estimated or assumed.'
        '</div>',
        unsafe_allow_html=True,
    )

    comp_cols = st.columns(3)
    comp_inputs = {}
    components_available = [c for c in PURE_COMPONENTS.index]

    default_comp = {
        'C1': 0.45, 'C2': 0.05, 'C3': 0.05,
        'nC4': 0.03, 'nC5': 0.01, 'nC6': 0.01,
    }

    with comp_cols[0]:
        st.markdown("**Hydrocarbon components**")
        for comp in ['C1', 'C2', 'C3', 'nC4', 'iC4', 'nC5', 'iC5', 'nC6']:
            val = default_comp.get(comp, 0.0)
            comp_inputs[comp] = st.number_input(
                f"{comp} (mol%)", 0.0, 100.0, val * 100, 0.1,
                key=f'comp_{comp}'
            ) / 100.0

    with comp_cols[1]:
        st.markdown("**Non-hydrocarbons**")
        for comp in ['CO2', 'N2', 'H2S']:
            comp_inputs[comp] = st.number_input(
                f"{comp} (mol%)", 0.0, 100.0, 0.0, 0.1,
                key=f'comp_{comp}'
            ) / 100.0

        st.markdown("**C7+ fraction**")
        c7_frac = st.number_input("C7+ (mol%)", 0.0, 100.0, 40.0, 0.1) / 100.0
        MW_c7   = st.number_input("C7+ MW [lb/lbmol]", 90.0, 600.0, 215.0, 1.0)
        SG_c7   = st.number_input("C7+ SG [-]", 0.70, 1.10, 0.87, 0.01)

    with comp_cols[2]:
        st.markdown("**Composition summary**")
        known_sum = sum(comp_inputs.values())
        total_sum = known_sum + c7_frac

        st.metric("Known components sum", f"{known_sum * 100:.2f} mol%")
        st.metric("C7+ fraction",         f"{c7_frac  * 100:.2f} mol%")
        st.metric("Total",                f"{total_sum * 100:.2f} mol%",
                  delta=f"{(total_sum - 1.0) * 100:+.2f}%",
                  delta_color="off" if abs(total_sum - 1.0) < 0.01 else "inverse")

        if abs(total_sum - 1.0) > 0.01:
            st.warning(
                f"Mole fractions sum to {total_sum*100:.2f}%. "
                "Must sum to 100% to run EOS."
            )

    # Auto-update when composition is valid (no button guard)
    if abs(total_sum - 1.0) <= 0.01:
        comp_dict = {k: v for k, v in comp_inputs.items() if v > 1e-6}
        if c7_frac > 1e-6:
            comp_dict['C7+'] = c7_frac

        # Normalize
        total = sum(comp_dict.values())
        comp_dict_norm = {k: v / total for k, v in comp_dict.items()}

        c7_props = characterize_c7plus(MW_c7, SG_c7)
        mixture  = build_mixture(comp_dict_norm, c7plus_props=c7_props)

        st.success("Mixture built from composition. EOS results below.")

        # Component table
        mix_df = pd.DataFrame({
            'Component': mixture['names'],
            'z [-]'    : np.round(mixture['z'], 5),
            'Tc [°R]'  : np.round(mixture['Tc'], 1),
            'Pc [psia]': np.round(mixture['Pc'], 1),
            'ω [-]'    : np.round(mixture['omega'], 4),
            'MW'       : np.round(mixture['MW'], 3),
        })
        st.dataframe(mix_df, use_container_width=True, hide_index=True)

        # Bubble point via fugacity
        from eos import pr_bubble_point_fugacity
        Pb_fug, conv_bub, wb = pr_bubble_point_fugacity(mixture, T)
        if conv_bub:
            st.metric("EOS Bubble Point (PR Fugacity)", f"{Pb_fug:.1f} psia")
        else:
            st.metric("EOS Bubble Point (PR Fugacity)", "Not converged")
        for w in wb:
            st.warning(w)

        # Flash result at a user-chosen pressure
        st.divider()
        st.markdown("**Flash Calculation**")
        P_flash = st.slider("Flash pressure [psia]", 100, 8000, 1500, 50,
                            key='p_flash_comp')
        with st.spinner("Running flash…"):
            flash_res = pr_flash(mixture, P_flash, T)

        for w in flash_res['warnings']:
            st.warning(w)

        fig_flash, _ = P.plot_flash_result(flash_res, mixture, P_flash, T)
        st.pyplot(fig_flash, use_container_width=True)

        # Phase envelope
        st.divider()
        T_env_c_min = st.slider("Phase envelope T min [°F]", 50, 150, 80,  key='envmin_comp')
        T_env_c_max = st.slider("Phase envelope T max [°F]", 200, 500, 380, key='envmax_comp')
        with st.spinner("Building phase envelope…"):
            fig_env_real, _ = P.plot_phase_envelope(
                mixture,
                T_range=(float(T_env_c_min), float(T_env_c_max)),
                label='Real composition'
            )
        st.pyplot(fig_env_real, use_container_width=True)

        # EOS curves
        fig_eos_real, _ = P.plot_eos_curves(mixture, T_F=T, label='Real composition')
        st.pyplot(fig_eos_real, use_container_width=True)

        st.markdown(
            '<div class="assumption-box">'
            f'C7+ Tc={c7_props["Tc"]:.1f}°R, Pc={c7_props["Pc"]:.1f} psia '
            f'from Katz-Firoozabadi (1978) table at MW={MW_c7:.0f}. '
            f'Pc adjusted so PR b = 2.0 ft³/lbmol. '
            f'Source: {c7_props["source"]}'
            '</div>',
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════════
# TAB 7 — ABOUT
# ════════════════════════════════════════════════════════════════════════════

with tabs[6]:
    st.subheader("Methodology and References")

    st.markdown("""
### Oil PVT Correlations

| Correlation | Region | Source |
|---|---|---|
| Standing (1947) | Western USA / California | *Drill. & Prod. Prac.*, API |
| Al-Marhoun (1988) | Middle East / Saudi Arabia | *JPT* Vol.40 No.5; SPE-13718-PA |
| Glaso (1980) | North Sea | *JPT* SPE-8016 |
| Vasquez & Beggs (1980) | Global / Latin America | *JPT* SPE-6719 |
| Petrosky & Farshad (1993) | Gulf of Mexico | SPE-26644 |

#### Viscosity Correlations (by region)

| Region | Dead Oil | Saturated | Undersaturated |
|---|---|---|---|
| Western USA | Beal (1946) Trans. AIME Vol.165 | Chew & Connally (1959) Trans. AIME Vol.216 | Vasquez & Beggs (1980) |
| Middle East | Beggs & Robinson (1975) JPT Vol.27 | Beggs & Robinson (1975) | Vasquez & Beggs (1980) |
| North Sea | Beggs & Robinson (1975) | Beggs & Robinson (1975) | Vasquez & Beggs (1980) |
| Global | Beggs & Robinson (1975) | Beggs & Robinson (1975) | Vasquez & Beggs (1980) |
| Gulf of Mexico | Beggs & Robinson (1975) | Beggs & Robinson (1975) | Vasquez & Beggs (1980) |

Beal (1946) equations from Ahmed (2019) Eq. 2-71a.
Chew & Connally (1959) equations from Ahmed (2019) Eq. 2-78.

#### Undersaturated Oil Compressibility

Vasquez & Beggs (1980) SPE-6719; Ahmed (2019) Eq. 2-47:

    co = (5·Rs + 17.2·T − 1180·γg + 12.61·API − 1433) / (10⁵ · P)   [psi⁻¹]

Defined only above bubble point (P ≥ Pb). Below Pb, co is shown as "< Pb".

---

### Gas PVT Correlations

#### Pseudocritical Properties
- **Dry gas**: Sutton (1985) SPE-14265, Eqs. 3-4
  - Tpc = 169.2 + 349.5·γg − 74.0·γg²  [°R]
  - Ppc = 756.8 − 131.0·γg − 3.6·γg²   [psia]
- **Gas condensate**: Standing (1977) 9th ed. SPE, p. 204
- **Acid-gas correction**: Wichert & Aziz (1972) *Hydrocarbon Processing*, May 1972, pp. 119-122

#### Z-Factor Methods
- **Hall-Yarborough (1974)** SPE-3350-PA — primary method
  - Newton-Raphson on Starling-Carnahan EOS; valid Tpr ≥ 1.05
  - Convergence: |Δy/y| < 1×10⁻¹⁰; max 200 iterations
  - Returns NaN with warning if Tpr < 1.05 (correlation undefined)
- **DPR (1974)** — Dranchuk, Purvis & Robinson; BWR-type, 8 constants
  - Stated validity: 1.05 ≤ Tpr ≤ 3.0, 0 < Ppr ≤ 3.0
  - Outside range: result flagged as extrapolation

#### Gas Viscosity
- **Lee, Gonzalez & Eakin (1966)** SPE-1340-PA
  - μg = K·exp(X·ρg^Y)×10⁻⁴ [cp]; ρg in g/cm³

#### Gas Derived Properties
- Bg [ft³/scf] = P_sc · T / (T_sc · P · Z)  — from real-gas law
- cg [psi⁻¹] = 1/P − (1/Z)·(dZ/dP)_T  — numerical dZ/dP by central finite difference

---

### Equation of State

**Peng-Robinson (1976)** *Ind. Eng. Chem. Fundam.* 15(1), 59–64.
Van der Waals mixing rules. Binary interaction parameters kij = 0
(ideal mixing — acceptable for HC-HC pairs; introduce error for CO₂/H₂S-HC mixtures).

#### Phase Equilibrium (PR Fugacity)
All phase calculations use PR fugacity coefficients from Peng & Robinson (1976) Eq. 18:

    ln(φi) = (bi/bm)·(Z−1) − ln(Z−B)
             − A/(2√2·B) · (2·Σj zj·√(ai·aj)·(1−kij)/am − bi/bm)
             · ln((Z+(1+√2)·B) / (Z+(1−√2)·B))

#### Flash Calculation
Rachford-Rice (1952) equation solved by Brent method.
Successive substitution with Ki = φi_L/φi_V.
Convergence criterion: Σ(ln Ki_new − ln Ki_old)² < 1×10⁻¹⁰ (Michelsen 1982).
Wilson K-values used only as initialization (never displayed).

#### Stability Test
Michelsen (1982) simplified tangent-plane-distance test.
Two trial phases (vapor-like and liquid-like Wilson K-value initializations).

#### Bubble/Dew Points
Proper fugacity equality iteration — not Wilson K-values.
Algorithm: Whitson & Brulé (2000) SPE Monograph Vol.20, Section 3.3.

---

### Pure Component Data

| Property | Source |
|---|---|
| Tc, Pc (C1–C6) | Ahmed (2019) Table 1-1 (GPA Engineering Data Book) |
| Tc, Pc, ω (CO₂, N₂, H₂S) | Peng & Robinson (1976) Table 1 |
| ω (all hydrocarbons) | Peng & Robinson (1976) Table 1 |
| MW | IUPAC standard atomic weights |

### C7+ Characterization

Critical temperature and acentric factor from **Katz & Firoozabadi (1978)**
(*JPT*, November 1978, pp. 1649-1655), interpolated by molecular weight.
Critical pressure adjusted so PR b = 2.0 ft³/lbmol (Ahmed 2019, Ch. 2).

---

### References

1. Ahmed, T. (2019). *Reservoir Engineering Handbook*, 5th ed. Elsevier.
2. Al-Marhoun, M.A. (1988). PVT Correlations for Middle East Crude Oils. *JPT* 40(5):650–666. SPE-13718-PA.
3. Beal, C. (1946). The Viscosity of Air, Water, Natural Gas, Crude Oil and Its Associated Gases. *Trans. AIME* 165:94–115.
4. Beggs, H.D. & Robinson, J.R. (1975). Estimating the Viscosity of Crude Oil Systems. *JPT* 27(9):1140–1141.
5. Chew, J. & Connally, C.A. (1959). A Viscosity Correlation for Gas-Saturated Crude Oils. *Trans. AIME* 216:23–25.
6. Dranchuk, P.M., Purvis, R.A. & Robinson, D.B. (1974). Computer Calculation of Natural Gas Compressibility Factors. IP 74-008.
7. Glaso, O. (1980). Generalized Pressure-Volume-Temperature Correlations. *JPT* 32(5):785–795. SPE-8016.
8. Hall, K.R. & Yarborough, L. (1974). A New Equation of State for Z-Factor Calculations. *Oil & Gas Journal*, June 18 1974, pp. 82–92. SPE-3350-PA.
9. Katz, D.L. & Firoozabadi, A. (1978). Predicting Phase Behavior of Condensate/Crude-Oil Systems. *JPT* 30(11):1649–1655.
10. Lee, A.L., Gonzalez, M.H. & Eakin, B.E. (1966). The Viscosity of Natural Gas. *JPT* Aug. 1966:997–1000. SPE-1340-PA.
11. Michelsen, M.L. (1982). The Isothermal Flash Problem. Part I. Stability. *Fluid Phase Equilibria* 9(1):1–19.
12. Peng, D.Y. & Robinson, D.B. (1976). A New Two-Constant Equation of State. *Ind. Eng. Chem. Fundam.* 15(1):59–64.
13. Petrosky, G.E. & Farshad, F.F. (1993). PVT Correlations for Gulf of Mexico Crude Oils. SPE-26644.
14. Rachford, H.H. & Rice, J.D. (1952). Procedure for Use of Electronic Digital Computers in Calculating Flash Vaporization. *Trans. AIME* 195:327–328.
15. Standing, M.B. (1947). A Pressure-Volume-Temperature Correlation for Mixtures of California Oils and Gases. *Drill. & Prod. Prac.*, API.
16. Standing, M.B. (1977). *Volumetric and Phase Behavior of Oil Field Hydrocarbon Systems*, 9th ed. SPE.
17. Sutton, R.P. (1985). Compressibility Factors for High-Molecular-Weight Reservoir Gases. SPE-14265.
18. Vasquez, M. & Beggs, H.D. (1980). Correlations for Fluid Physical Property Prediction. *JPT* 32(6):968–970. SPE-6719.
19. Whitson, C.H. & Brulé, M.R. (2000). *Phase Behavior of Petroleum Reservoir Fluids*. SPE Monograph Vol. 20.
20. Wichert, E. & Aziz, K. (1972). Calculate Z's for Sour Gases. *Hydrocarbon Processing*, May 1972, pp. 119–122.
21. Wilson, G.M. (1969). A Modified Redlich-Kwong EOS. AIChE 65th National Meeting, Paper 15C.
""")
