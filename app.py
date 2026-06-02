"""
app.py
------
Streamlit interactive frontend for the PVT project.

Run with:
    streamlit run app.py

Tabs
----
  1. Correlations  — Rs, Bo, viscosity curves + regional comparison
  2. EOS           — Z-factor, density, phase envelope
  3. 3D Surfaces   — interactive Plotly 3D (Bo, viscosity, Z, PVT)
  4. Phase / Pxy   — PT phase envelope + binary Pxy diagram
  5. Composition   — enter real mole fractions from lab report
  6. About         — methodology, assumptions, references

All assumptions and limitations are displayed inline so the tool is
self-documenting for any user without prior knowledge of the codebase.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')

# Project modules
from correlations import REGIONS, pvt_table, bubble_point
from eos import (
    pseudo_components, build_mixture, characterize_c7plus,
    eos_table, phase_envelope_wilson, pxy_diagram,
    pr_bubble_point, PURE_COMPONENTS
)
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

    API = st.slider("API Gravity [°API]", 15, 60, 35, 1,
                    help="Stock-tank oil API gravity. Higher = lighter oil.")
    GOR = st.slider("Surface GOR [scf/STB]", 50, 2000, 500, 25,
                    help="Gas-oil ratio at surface (stock-tank) conditions.")
    T   = st.slider("Reservoir Temperature [°F]", 80, 320, 180, 5)
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
    with st.expander("ℹ️ What is this?", expanded=False):
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
               "EOS fallback. For real compositions, use the **Composition** tab.")


# ---------------------------------------------------------------------------
# COMPUTE SHARED STATE
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compute_pvt(API, GOR, T, gas_SG, region, T_sp, P_sp):
    return pvt_table(API, GOR, T, gas_SG, region=region, T_sp=T_sp, P_sp=P_sp)

@st.cache_data(show_spinner=False)
def compute_pseudo(API, GOR, gas_SG):
    return pseudo_components(API, GOR, gas_SG)

sep_kw = {'T_sp': T_sp, 'P_sp': P_sp} if use_sep_corr else {'T_sp': 60.0, 'P_sp': 114.7}

tbl    = compute_pvt(API, GOR, T, gas_SG, region, **sep_kw)
pseudo = compute_pseudo(API, GOR, gas_SG)
Pb_corr = tbl['Pb']
Pb_eos  = pr_bubble_point(pseudo, T)

# ---------------------------------------------------------------------------
# HEADER METRICS
# ---------------------------------------------------------------------------

st.title("🛢️ PVT Analyzer")
st.caption(
    "Empirical correlations (Standing, Al-Marhoun, Glaso, Vasquez & Beggs, "
    "Petrosky & Farshad) + Peng-Robinson EOS with real or pseudo-compositions."
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
    "⚗️ EOS",
    "🌐 3D Surfaces",
    "🔀 Phase / Pxy",
    "🧪 Composition",
    "📖 About",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — CORRELATIONS
# ════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.subheader("Empirical PVT Correlations")

    st.markdown(
        '<div class="assumption-box">'
        '<b>Assumptions:</b> GOR is used as R_s at P_b (saturated reservoir). '
        'Viscosity uses Beggs & Robinson (1975) dead-oil base for all regions. '
        'Undersaturated compressibility uses a fixed c_o = 5×10⁻⁶ psi⁻¹ '
        '(typical light crude; see Ahmed 2019 for full correlation). '
        'Petrosky & Farshad results are flagged if P_b ≤ 14.7 psia '
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
            pb = bubble_point(API, GOR, T, gas_SG, region=r)
            flag = " ⚠️ out-of-range" if pb <= 14.8 else ""
            pb_data.append({
                'Region': REGIONS[r].split('—')[0].strip(),
                'Pb [psia]': f"{pb:.0f}{flag}",
            })
        st.dataframe(pd.DataFrame(pb_data), use_container_width=True,
                     hide_index=True)

        st.markdown("**Selected region Pb**")
        st.metric(
            REGIONS[region].split('—')[0].strip(),
            f"{Pb_corr:.1f} psia"
        )

    st.divider()
    st.subheader("Regional Comparison")
    fig_reg, _ = P.plot_regional_comparison(API, GOR, T, gas_SG)
    st.pyplot(fig_reg, use_container_width=True)

    st.divider()
    st.subheader("PVT Data Table")
    df_tbl = pd.DataFrame({
        'P [psia]'   : tbl['P'].round(1),
        'Rs [scf/STB]': tbl['Rs'].round(2),
        'Bo [RB/STB]' : tbl['Bo'].round(4),
        'μo [cp]'     : tbl['mu_o'].round(4),
    })
    st.dataframe(df_tbl, use_container_width=True, height=300)
    st.download_button(
        "⬇️ Download PVT table as CSV",
        df_tbl.to_csv(index=False),
        file_name=f"pvt_table_API{API}_GOR{GOR}.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — EOS
# ════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.subheader("Peng-Robinson EOS — Z-factor and Density")

    st.markdown(
        '<div class="assumption-box">'
        '<b>Limitations:</b> Uses bulk API/GOR/gas_SG to generate two '
        'pseudo-components (gas C1-C6 via Sutton 1985; oil C7+ via '
        'Katz-Firoozabadi 1978). C7+ critical pressure is adjusted so '
        'b = 2.0 ft³/lbmol, following volume-translated PR practice. '
        'Z-factors reflect the overall liquid-phase mixture above P_b. '
        'Full two-phase flash requires real compositional data '
        '(see Composition tab). '
        'Bubble point uses Wilson (1969) K-values — qualitative only.'
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
    fig_cmp, _ = P.plot_correlations_vs_eos(API, GOR, T, gas_SG,
                                              mixture=pseudo)
    st.pyplot(fig_cmp, use_container_width=True)

    st.divider()
    st.subheader("EOS Data Table")
    tbl_eos = eos_table(pseudo, T_F=T)
    df_eos  = pd.DataFrame({
        'P [psia]'      : tbl_eos['P'].round(1),
        'Z [-]'         : tbl_eos['Z'].round(4),
        'Density [lb/ft³]': tbl_eos['rho'].round(3),
        'V_mol [ft³/lbmol]': tbl_eos['V_mol'].round(4),
    })
    st.dataframe(df_eos, use_container_width=True, height=300)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — 3D SURFACES
# ════════════════════════════════════════════════════════════════════════════

with tabs[2]:
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
            fig_3d = P.plot_bo_surface(
                API, GOR, gas_SG,
                T_range=T_range_3d, P_range=P_range_3d, region=region
            )
        elif surface_choice.startswith("μ"):
            fig_3d = P.plot_viscosity_surface(
                API, GOR, gas_SG,
                T_range=T_range_3d, P_range=P_range_3d, region=region
            )
        elif surface_choice.startswith("Z"):
            mix_fn = lambda T_F: pseudo_components(API, GOR, gas_SG)
            fig_3d = P.plot_z_surface(
                mix_fn, T_range=T_range_3d, P_range=P_range_3d,
                label=f'API={API}°'
            )
        else:
            mix_fn = lambda T_F: pseudo_components(API, GOR, gas_SG)
            fig_3d = P.plot_pvt_3d(
                mix_fn, T_range=T_range_3d, P_range=P_range_3d,
                label=f'API={API}°'
            )

    st.plotly_chart(fig_3d, use_container_width=True)

    st.markdown(
        '<div class="assumption-box">'
        'Bo and μ surfaces use empirical correlations (selected region). '
        'Z and V surfaces use PR EOS with pseudo-components. '
        'Out-of-range temperatures for a given correlation are silently '
        'excluded (NaN); this may cause gaps in the surface.'
        '</div>',
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — PHASE / Pxy
# ════════════════════════════════════════════════════════════════════════════

with tabs[3]:
    col_env, col_pxy = st.columns(2)

    with col_env:
        st.subheader("PT Phase Envelope")
        st.markdown(
            '<div class="assumption-box">'
            'Wilson (1969) K-values give a qualitative first-order envelope. '
            'Dew curve may be incomplete for heavy pseudo-components. '
            'Quantitative accuracy requires PR fugacity flash '
            '(future extension).'
            '</div>',
            unsafe_allow_html=True,
        )
        T_env_min = st.slider("T range min [°F]", 50, 150, 80,  key='envmin')
        T_env_max = st.slider("T range max [°F]", 200, 500, 380, key='envmax')

        with st.spinner("Building phase envelope…"):
            fig_env, _ = P.plot_phase_envelope(
                pseudo,
                T_range=(float(T_env_min), float(T_env_max)),
                label=f'API={API}°, GOR={GOR}'
            )
        st.pyplot(fig_env, use_container_width=True)

    with col_pxy:
        st.subheader("Binary Pxy Diagram")
        st.markdown(
            '<div class="assumption-box">'
            'Wilson K-values: qualitative phase boundary shape. '
            'Only pure components from the CSV are available. '
            'C7+ is not available for Pxy (requires characterization).'
            '</div>',
            unsafe_allow_html=True,
        )
        available_comps = [c for c in PURE_COMPONENTS.index]
        comp_A = st.selectbox("Component A", available_comps,
                               index=available_comps.index('C1'), key='pxyA')
        comp_B = st.selectbox("Component B", available_comps,
                               index=available_comps.index('C3'), key='pxyB')
        T_pxy  = st.slider("Temperature [°F]", 50, 300, 100, key='tpxy')
        P_max_pxy = st.slider("P search max [psia]", 500, 5000, 2000,
                               key='pmaxpxy')

        if comp_A == comp_B:
            st.warning("Select two different components.")
        else:
            with st.spinner("Computing Pxy…"):
                fig_pxy, _ = P.plot_pxy(comp_A, comp_B,
                                         T_F=float(T_pxy),
                                         P_range=(14.7, float(P_max_pxy)))
            st.pyplot(fig_pxy, use_container_width=True)

            st.markdown(
                '<div class="source-box">'
                f'Tc, Pc, ω from: '
                f'{PURE_COMPONENTS.loc[comp_A, "source_Tc_Pc"]} | '
                f'{PURE_COMPONENTS.loc[comp_B, "source_Tc_Pc"]}'
                '</div>',
                unsafe_allow_html=True,
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — COMPOSITION
# ════════════════════════════════════════════════════════════════════════════

with tabs[4]:
    st.subheader("Real Fluid Composition Input")
    st.markdown(
        "Enter mole fractions from a lab PVT compositional report. "
        "C7+ is always a lumped fraction and requires measured MW and SG."
    )

    st.markdown(
        '<div class="source-box">'
        'Pure component Tc, Pc, ω loaded from '
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
                "Must sum to 100% before running EOS."
            )

    if abs(total_sum - 1.0) <= 0.01 and st.button("▶ Run EOS with this composition"):
        # Build composition dict (exclude zero entries)
        comp_dict = {k: v for k, v in comp_inputs.items() if v > 1e-6}
        if c7_frac > 1e-6:
            comp_dict['C7+'] = c7_frac

        # Normalize
        total = sum(comp_dict.values())
        comp_dict = {k: v / total for k, v in comp_dict.items()}

        c7_props = characterize_c7plus(MW_c7, SG_c7)
        mixture  = build_mixture(comp_dict, c7plus_props=c7_props)

        st.success("Mixture built from real composition.")

        # Display component table
        mix_df = pd.DataFrame({
            'Component': mixture['names'],
            'z [-]'    : np.round(mixture['z'], 5),
            'Tc [°R]'  : np.round(mixture['Tc'], 1),
            'Pc [psia]': np.round(mixture['Pc'], 1),
            'ω [-]'    : np.round(mixture['omega'], 4),
            'MW'       : np.round(mixture['MW'], 3),
        })
        st.dataframe(mix_df, use_container_width=True, hide_index=True)

        T_comp = st.session_state.get('T_comp', T)

        Pb_real = pr_bubble_point(mixture, T)
        st.metric("EOS Bubble Point (Wilson)", f"{Pb_real:.1f} psia")

        fig_eos_real, _ = P.plot_eos_curves(
            mixture, T_F=T,
            label='Real composition'
        )
        st.pyplot(fig_eos_real, use_container_width=True)

        fig_env_real, _ = P.plot_phase_envelope(
            mixture, label='Real composition'
        )
        st.pyplot(fig_env_real, use_container_width=True)

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
# TAB 6 — ABOUT
# ════════════════════════════════════════════════════════════════════════════

with tabs[5]:
    st.subheader("Methodology and References")

    st.markdown("""
### Empirical Correlations

| Correlation | Region | Source |
|---|---|---|
| Standing (1947) | Western USA / California | *Drill. & Prod. Prac.*, API |
| Al-Marhoun (1988) | Middle East / Saudi Arabia | *JPT* Vol.40 No.5, SPE-13718-PA |
| Glaso (1980) | North Sea | *JPT*, SPE-8016 |
| Vasquez & Beggs (1980) | Global / Latin America | *JPT*, SPE-6719 |
| Petrosky & Farshad (1993) | Gulf of Mexico | SPE-26644 |

Viscosity uses **Beggs & Robinson (1975)** dead-oil and saturated-oil correlations
for all regions, with Vasquez & Beggs (1980) undersaturated correction.
Undersaturated oil compressibility uses a fixed **c_o = 5×10⁻⁶ psi⁻¹** as
a representative value for light-to-medium crudes; the full Vasquez & Beggs
correlation would tie this to Rs, T, and API.

### Equation of State

**Peng-Robinson (1976)** EOS with van der Waals mixing rules and zero binary
interaction parameters (k_ij = 0, ideal mixing assumption). This is acceptable
for hydrocarbon-hydrocarbon pairs but introduces error for CO₂/H₂S-hydrocarbon
mixtures; calibrated k_ij values would be required for those systems.

Phase equilibrium uses **Wilson (1969) K-values** as a first-order approximation.
Full PR fugacity-based flash requires ≥6 components with calibrated k_ij values.

### Pure Component Data

| Property | Source |
|---|---|
| Tc, Pc (C1–C6) | Ahmed (2019) Table 1-1 (GPA Engineering Data Book) |
| Tc, Pc, ω (CO₂, N₂, H₂S) | Peng & Robinson (1976) Table 1 |
| ω (all hydrocarbons) | Peng & Robinson (1976) Table 1 |
| MW | IUPAC standard atomic weights |

### C7+ Characterization

Critical temperature and acentric factor from **Katz & Firoozabadi (1978)**
table (JPT, November 1978, pp. 1649-1655), interpolated by molecular weight.
Critical pressure is adjusted so the PR co-volume parameter b = 2.0 ft³/lbmol,
following volume-translated PR EOS practice for heavy pseudo-components
(see Ahmed 2019, Ch. 2).

### Known Limitations

- Two pseudo-components (gas + oil) yield extreme Tc/Pc asymmetry that
  prevents stable Rachford-Rice flash. Full flash requires 6–12 components.
- Wilson K-values give qualitative phase boundaries only.
- Fixed c_o for undersaturated compressibility.
- k_ij = 0 (ideal mixing); sour gas or CO₂-rich systems need calibrated k_ij.
- Petrosky & Farshad is development-range limited; results outside that range
  are flagged.

### References

1. Ahmed, T. (2019). *Reservoir Engineering Handbook*, 5th ed. Elsevier.
2. Al-Marhoun, M.A. (1988). PVT Correlations for Middle East Crude Oils.
   *JPT*, 40(5), 650–666. SPE-13718-PA.
3. Beggs, H.D. & Robinson, J.R. (1975). Estimating the Viscosity of Crude Oil
   Systems. *JPT*, 27(9), 1140–1141.
4. Glaso, O. (1980). Generalized Pressure-Volume-Temperature Correlations.
   *JPT*, 32(5), 785–795. SPE-8016.
5. Katz, D.L. & Firoozabadi, A. (1978). Predicting Phase Behavior of
   Condensate/Crude-Oil Systems Using Methane Interaction Coefficients.
   *JPT*, 30(11), 1649–1655.
6. Peng, D.Y. & Robinson, D.B. (1976). A New Two-Constant Equation of State.
   *Ind. Eng. Chem. Fundam.*, 15(1), 59–64.
7. Petrosky, G.E. & Farshad, F.F. (1993). Pressure-Volume-Temperature
   Correlations for Gulf of Mexico Crude Oils. SPE-26644.
8. Standing, M.B. (1947). A Pressure-Volume-Temperature Correlation for
   Mixtures of California Oils and Gases. *Drill. & Prod. Prac.*, API.
9. Sutton, R.P. (1985). Compressibility Factors for High-Molecular-Weight
   Reservoir Gases. SPE-14265.
10. Vasquez, M. & Beggs, H.D. (1980). Correlations for Fluid Physical Property
    Prediction. *JPT*, 32(6), 968–970. SPE-6719.
11. Wilson, G.M. (1969). A Modified Redlich-Kwong EOS, Application to General
    Physical Data Calculations. AIChE 65th National Meeting, Paper 15C.
""")
