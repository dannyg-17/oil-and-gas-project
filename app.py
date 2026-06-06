"""
app.py
------
Streamlit interactive frontend for the PVT Analyzer.

Run with:
    streamlit run app.py

Tabs
----
  1. Oil PVT       — Rs, Bo, viscosity, compressibility curves + PVT table
  2. Gas PVT       — Gas Z-factor, Bg, μg, ρg from gas correlations
  3. Phase Behavior — PT phase envelope + Binary Pxy diagram
  4. About          — methodology, sources, references
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib
matplotlib.use('Agg')

# Project modules
from correlations import REGIONS, pvt_table, bubble_point, check_range, solution_gor
from correlations import formation_volume_factor, viscosity, oil_compressibility
from eos import (
    pseudo_components, build_mixture, characterize_c7plus,
    pr_bubble_point, pr_bubble_point_fugacity, pr_phase_envelope,
    pxy_diagram_fugacity, PURE_COMPONENTS,
)
from gas_pvt import gas_pvt_table
import plotting as _P

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PVT Analyzer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    /* ── Global font ── */
    html, body, [class*="css"], .stApp, .block-container,
    .stMarkdown, .stText, .stMetric, .stDataFrame,
    .stSelectbox, .stSlider, .stRadio, .stCheckbox,
    .stNumberInput, .stExpander, .stDownloadButton,
    h1, h2, h3, h4, h5, h6, p, div, span, label, button {
        font-family: 'Times New Roman', Times, serif !important;
        line-height: 2.0 !important;
    }

    .block-container { padding-top: 1.5rem; }

    /* ── Tab labels ── */
    .stTabs [data-baseweb="tab"] {
        font-size: 1.15rem !important;
        font-weight: 700 !important;
        padding: 0.75rem 1.8rem !important;
        font-family: 'Times New Roman', Times, serif !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    /* ── Assumption / source boxes ── */
    .assumption-box {
        background: #FFF8E1; border-left: 4px solid #FFC107;
        padding: 0.6rem 1rem; border-radius: 4px; margin: 0.5rem 0;
        font-size: 0.85rem; color: #5D4037;
        font-family: 'Times New Roman', Times, serif !important;
        line-height: 1.8 !important;
    }
    .source-box {
        background: #E8F5E9; border-left: 4px solid #4CAF50;
        padding: 0.4rem 0.8rem; border-radius: 4px;
        font-size: 0.78rem; color: #1B5E20;
        font-family: 'Times New Roman', Times, serif !important;
        line-height: 1.8 !important;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# REGION LABELS
# ---------------------------------------------------------------------------

_REGION_DISPLAY = {
    "western_usa"   : "Standing (1947) — Western USA",
    "middle_east"   : "Al-Marhoun (1988) — Middle East",
    "north_sea"     : "Glaso (1980) — North Sea",
    "global"        : "Vasquez & Beggs (1980) — Global",
    "gulf_of_mexico": "Petrosky & Farshad (1993) — Gulf of Mexico",
}

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
def compute_gas_pvt(gas_SG, T, y_CO2, y_H2S, fluid_type):
    return gas_pvt_table(
        gas_SG=gas_SG, T_F=T,
        y_CO2=y_CO2, y_H2S=y_H2S,
        fluid_type=fluid_type,
        z_method='hall_yarborough',
        P_min=14.7, P_max=10000.0, n_points=200,
    )


@st.cache_data(show_spinner=False)
def compute_pvt_table_grid(API, GOR, T, gas_SG, region, T_sp, P_sp):
    """Fixed 50-psia step PVT table for the Oil PVT tab."""
    tbl    = pvt_table(API, GOR, T, gas_SG, region=region, T_sp=T_sp, P_sp=P_sp)
    Pb_val = tbl['Pb']
    P_max_tbl = int(np.ceil(max(Pb_val * 2.5, 6000) / 50) * 50)
    P_arr  = np.arange(50, P_max_tbl + 1, 50)
    Rs_arr = np.array([float(solution_gor(np.array([p]), API, T, gas_SG, Pb_val, GOR, region=region)[0])
                       for p in P_arr])
    Bo_arr = np.array([float(formation_volume_factor(np.array([p]), API, T, gas_SG, Pb_val, GOR, region=region)[0])
                       for p in P_arr])
    mu_arr = np.array([float(viscosity(np.array([p]), API, T, gas_SG, Pb_val, GOR, region=region)[0])
                       for p in P_arr])
    co_arr = np.array([float(oil_compressibility(np.array([p]), API, T, gas_SG, Pb_val, GOR)[0])
                       for p in P_arr])
    return Pb_val, P_arr, Rs_arr, Bo_arr, mu_arr, co_arr


@st.cache_data(show_spinner=False)
def compute_phase_envelope(comp_tuple, T_range=(50.0, 400.0)):
    """Phase envelope — cache on a hashable key."""
    # comp_tuple: tuple of (name, z, Tc, Pc, omega, MW) for each component
    names  = [r[0] for r in comp_tuple]
    z      = np.array([r[1] for r in comp_tuple])
    Tc     = np.array([r[2] for r in comp_tuple])
    Pc     = np.array([r[3] for r in comp_tuple])
    omega  = np.array([r[4] for r in comp_tuple])
    MW     = np.array([r[5] for r in comp_tuple])
    mixture = {'names': names, 'z': z, 'Tc': Tc, 'Pc': Pc, 'omega': omega, 'MW': MW}
    return pr_phase_envelope(mixture, T_range=T_range, n_T=60)


@st.cache_data(show_spinner=False)
def compute_pxy(comp_A, comp_B, T_F, P_lo, P_hi, n_x):
    return pxy_diagram_fugacity(comp_A, comp_B, T_F,
                                P_range=(P_lo, P_hi), n_x=n_x)


# ---------------------------------------------------------------------------
# TOP-OF-PAGE INPUTS
# ---------------------------------------------------------------------------

st.title("PVT Analyzer")
st.caption(
    "Empirical correlations (Standing, Al-Marhoun, Glaso, Vasquez & Beggs, "
    "Petrosky & Farshad) + Peng-Robinson EOS with real or pseudo-compositions. "
    "Gas PVT: Hall-Yarborough (1974) Z-factor, Lee-Gonzalez-Eakin (1966) viscosity."
)

# ── A. Region selector ──────────────────────────────────────────────────────
region = st.selectbox(
    "Select Regional Correlation:",
    list(_REGION_DISPLAY.keys()),
    index=3,
    format_func=lambda r: _REGION_DISPLAY[r],
)

# ── B. Fluid inputs row ─────────────────────────────────────────────────────
f_col1, f_col2, f_col3, f_col4 = st.columns(4)
with f_col1:
    API = st.number_input("API Gravity [°API]", min_value=10.0, max_value=70.0,
                          value=35.0, step=0.1,
                          help="Stock-tank oil API gravity.")
with f_col2:
    GOR = st.number_input("Surface GOR [scf/STB]", min_value=10.0, max_value=5000.0,
                          value=500.0, step=1.0,
                          help="Gas-oil ratio at surface (stock-tank) conditions.")
with f_col3:
    T = st.number_input("Reservoir Temperature [°F]", min_value=60.0, max_value=400.0,
                        value=180.0, step=0.5)
with f_col4:
    gas_SG = st.number_input("Gas Specific Gravity [-]", min_value=0.50, max_value=1.50,
                             value=0.65, step=0.001,
                             help="Gas SG relative to air = 1.0.")

# ── C. Composition expander ─────────────────────────────────────────────────
_COMP_DESCRIPTIONS = {
    'C1'  : 'Methane: primary natural gas component',
    'C2'  : 'Ethane: light gas, common in associated gas',
    'C3'  : 'Propane: liquefies under moderate pressure',
    'nC4' : 'n-Butane: NGLs fraction',
    'iC4' : 'iso-Butane: branched C4 isomer',
    'nC5' : 'n-Pentane: lightest liquid alkane at surface',
    'iC5' : 'iso-Pentane: branched C5 isomer',
    'nC6' : 'n-Hexane: lower end of gasoline fraction',
    'CO2' : 'Carbon dioxide: non-hydrocarbon contaminant',
    'N2'  : 'Nitrogen: inert non-hydrocarbon',
    'H2S' : 'Hydrogen sulfide: toxic acid gas',
    'C7+' : 'Heptanes-plus: the oil fraction (all components heavier than C6)',
}

_COMP_ORDER = ['C1', 'C2', 'C3', 'nC4', 'iC4', 'nC5', 'iC5', 'nC6',
               'CO2', 'N2', 'H2S', 'C7+']

# Preset fluid templates (mol%)
_PRESETS = {
    'None (enter manually)': None,
    'Typical GOM Black Oil': {
        'C1': 45.0, 'C2': 5.0, 'C3': 5.0, 'nC4': 3.0, 'iC4': 1.0,
        'nC5': 1.0, 'iC5': 0.5, 'nC6': 1.0, 'CO2': 0.5, 'N2': 0.0,
        'H2S': 0.0, 'C7+': 38.0, 'MW_c7': 215.0, 'SG_c7': 0.87,
    },
    'Typical North Sea Volatile Oil': {
        'C1': 55.0, 'C2': 8.0, 'C3': 6.0, 'nC4': 3.0, 'iC4': 1.5,
        'nC5': 1.5, 'iC5': 0.5, 'nC6': 1.0, 'CO2': 1.5, 'N2': 0.5,
        'H2S': 0.0, 'C7+': 21.5, 'MW_c7': 195.0, 'SG_c7': 0.84,
    },
    'Typical Middle East Crude': {
        'C1': 35.0, 'C2': 4.0, 'C3': 4.0, 'nC4': 2.5, 'iC4': 0.5,
        'nC5': 1.5, 'iC5': 0.5, 'nC6': 1.5, 'CO2': 2.0, 'N2': 0.5,
        'H2S': 1.0, 'C7+': 47.0, 'MW_c7': 230.0, 'SG_c7': 0.89,
    },
    'Lean Dry Gas': {
        'C1': 85.0, 'C2': 8.0, 'C3': 3.0, 'nC4': 1.0, 'iC4': 0.5,
        'nC5': 0.3, 'iC5': 0.2, 'nC6': 0.2, 'CO2': 1.0, 'N2': 0.8,
        'H2S': 0.0, 'C7+': 0.0, 'MW_c7': 100.0, 'SG_c7': 0.74,
    },
}

with st.expander("Enter Fluid Composition (Mole Fractions)"):

    # ── Preset selector ──────────────────────────────────────────────────────
    preset_choice = st.selectbox(
        "Load a preset composition:",
        list(_PRESETS.keys()),
        key='preset_choice',
    )
    preset = _PRESETS[preset_choice]

    # ── CSV upload ───────────────────────────────────────────────────────────
    st.markdown("**Or upload a CSV file** (two columns: `component`, `mole_fraction` or `mol_pct`)")
    uploaded_file = st.file_uploader(
        "Upload composition CSV",
        type=["csv"],
        key='comp_csv',
        label_visibility='collapsed',
    )

    # Parse uploaded CSV if provided
    _csv_values = {}
    _csv_mw_c7 = None
    _csv_sg_c7 = None
    if uploaded_file is not None:
        try:
            import io
            _df_upload = pd.read_csv(io.StringIO(uploaded_file.read().decode('utf-8')))
            _df_upload.columns = [c.strip().lower() for c in _df_upload.columns]
            val_col = 'mole_fraction' if 'mole_fraction' in _df_upload.columns else \
                      'mol_pct'       if 'mol_pct'       in _df_upload.columns else \
                      _df_upload.columns[1]
            comp_col = _df_upload.columns[0]
            for _, row in _df_upload.iterrows():
                comp = str(row[comp_col]).strip()
                val  = float(row[val_col])
                # Normalise to mol%
                if val <= 1.0:
                    val *= 100.0
                if comp.lower() in ('mw_c7', 'mw c7', 'c7+ mw', 'c7_mw'):
                    _csv_mw_c7 = val
                elif comp.lower() in ('sg_c7', 'sg c7', 'c7+ sg', 'c7_sg'):
                    _csv_sg_c7 = val
                elif comp in _COMP_ORDER:
                    _csv_values[comp] = val
            st.success(f"Loaded {len(_csv_values)} components from CSV.")
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")

    # ── Manual inputs ────────────────────────────────────────────────────────
    st.markdown("**Component mole fractions [mol%]:**")
    col_a, col_b, col_c = st.columns(3)
    comp_inputs_pct = {}

    non_c7_comps = [c for c in _COMP_ORDER if c != 'C7+']
    _col_assignments = [
        (col_a, non_c7_comps[0:4]),
        (col_b, non_c7_comps[4:8]),
        (col_c, non_c7_comps[8:11]),
    ]

    for col_obj, comp_list in _col_assignments:
        with col_obj:
            for comp in comp_list:
                # Priority: CSV upload > preset > zero default
                if comp in _csv_values:
                    default_val = _csv_values[comp]
                elif preset and comp in preset:
                    default_val = float(preset[comp])
                else:
                    default_val = 0.0
                st.markdown(f"**{comp}** — *{_COMP_DESCRIPTIONS[comp]}*")
                comp_inputs_pct[comp] = st.number_input(
                    f"{comp} [mol%]",
                    min_value=0.0, max_value=100.0,
                    value=default_val,
                    step=0.01,
                    key=f'comp_{comp}',
                    label_visibility='collapsed',
                )

    with col_c:
        c7_default = _csv_values.get('C7+', preset['C7+'] if preset else 0.0)
        st.markdown(f"**C7+** — *{_COMP_DESCRIPTIONS['C7+']}*")
        comp_inputs_pct['C7+'] = st.number_input(
            "C7+ [mol%]", min_value=0.0, max_value=100.0,
            value=float(c7_default), step=0.01,
            key='comp_C7plus', label_visibility='collapsed',
        )
        mw_default = _csv_mw_c7 if _csv_mw_c7 else (preset.get('MW_c7', 215.0) if preset else 215.0)
        sg_default = _csv_sg_c7 if _csv_sg_c7 else (preset.get('SG_c7', 0.87)  if preset else 0.87)
        mw_col, sg_col = st.columns(2)
        with mw_col:
            MW_c7 = st.number_input("C7+ MW [lb/lbmol]", 90.0, 600.0,
                                    float(mw_default), 1.0, key='c7_MW')
        with sg_col:
            SG_c7 = st.number_input("C7+ SG [-]", 0.60, 1.20,
                                    float(sg_default), 0.001, key='c7_SG')

    total_pct = sum(comp_inputs_pct.values())
    if abs(total_pct - 100.0) < 1.0:
        st.success(f"Mole fraction sum: {total_pct:.2f}% — composition valid.")
    else:
        st.warning(
            f"Mole fractions sum to {total_pct:.2f}%. Must sum to 100% (±1%) "
            "to use real composition for EOS calculations."
        )

    st.caption(
        "CSV format: two columns — `component` and `mole_fraction` (0–1) or `mol_pct` (0–100). "
        "Accepted component names: C1, C2, C3, nC4, iC4, nC5, iC5, nC6, CO2, N2, H2S, C7+. "
        "Optionally include rows named MW_c7 and SG_c7 for C7+ characterization."
    )

# Determine if composition is valid
_comp_valid = abs(total_pct - 100.0) < 1.0

# Build mixture if composition is valid
if _comp_valid:
    _comp_dict = {k: v / 100.0 for k, v in comp_inputs_pct.items() if v > 1e-6}
    _total_frac = sum(_comp_dict.values())
    _comp_dict_norm = {k: v / _total_frac for k, v in _comp_dict.items()}
    _c7_props = characterize_c7plus(MW_c7, SG_c7)
    _mixture  = build_mixture(_comp_dict_norm, c7plus_props=_c7_props)
    # Hashable tuple for caching
    _mix_tuple = tuple(
        (_mixture['names'][i],
         float(_mixture['z'][i]),
         float(_mixture['Tc'][i]),
         float(_mixture['Pc'][i]),
         float(_mixture['omega'][i]),
         float(_mixture['MW'][i]))
        for i in range(len(_mixture['names']))
    )
else:
    _mixture  = None
    _mix_tuple = None

# ── Shared PVT computation ─────────────────────────────────────────────────
# Vasquez & Beggs separator correction — used internally when global region
_T_sp = 60.0
_P_sp = 114.7

tbl     = compute_pvt(API, GOR, T, gas_SG, region, _T_sp, _P_sp)
pseudo  = compute_pseudo(API, GOR, gas_SG)
Pb_corr = tbl['Pb']
Pb_eos  = pr_bubble_point(pseudo, T)

# EOS bubble point from real composition if available
if _comp_valid and _mixture is not None:
    _Pb_fug, _conv_fug, _wb_fug = pr_bubble_point_fugacity(_mixture, T)
    Pb_eos_display = f"{_Pb_fug:.0f} psia" if _conv_fug else "Not converged"
else:
    Pb_eos_display = f"{Pb_eos:.0f} psia" if not np.isnan(Pb_eos) else "—"

# Range warnings below region selector
range_warns = check_range(GOR, T, API, region)
if range_warns:
    for w in range_warns:
        st.warning(w)

# PVT table warnings
if tbl['warnings']:
    for w in tbl['warnings']:
        st.warning(w)

# ── D. Key metrics row ──────────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Bubble Point (Correlation)", f"{Pb_corr:.0f} psia")
m2.metric("Bubble Point (PR EOS)", Pb_eos_display)
m3.metric("Reservoir Temperature", f"{T} °F")
m4.metric("Gas Gravity", f"{gas_SG:.3f}")

st.divider()


# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------

tabs = st.tabs(["Oil PVT", "Gas PVT", "Phase Behavior", "About"])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — OIL PVT
# ════════════════════════════════════════════════════════════════════════════

with tabs[0]:
    st.subheader("Oil PVT Properties")

    region_label = _REGION_DISPLAY[region]

    # Retrieve plotly PVT curves
    fig_rs, fig_bo, fig_visc = _P.plot_pvt_curves_plotly(
        API, GOR, T, gas_SG, region=region, T_sp=_T_sp, P_sp=_P_sp
    )
    st.plotly_chart(fig_rs,   use_container_width=True)
    st.plotly_chart(fig_bo,   use_container_width=True)
    st.plotly_chart(fig_visc, use_container_width=True)

    # Oil Compressibility co — above Pb only, using plotly directly
    Pb_val, P_arr, Rs_arr, Bo_arr, mu_arr, co_arr = compute_pvt_table_grid(
        API, GOR, T, gas_SG, region, _T_sp, _P_sp
    )
    co_mask = ~np.isnan(co_arr)
    if co_mask.any():
        fig_co = go.Figure()
        fig_co.add_trace(go.Scatter(
            x=P_arr[co_mask], y=co_arr[co_mask],
            mode='lines',
            line=dict(color='#6A1B9A', width=2),
            name='co [psi⁻¹]',
            hovertemplate='P = %{x:,.0f} psia<br>co = %{y:.3e} psi⁻¹<extra></extra>',
        ))
        fig_co.add_vline(
            x=float(Pb_val),
            line_dash='dash', line_color='red', line_width=1.5,
            annotation_text=f'Pb = {Pb_val:.0f} psia',
            annotation_position='top left',
            annotation_font_color='red',
        )
        fig_co.update_layout(
            title=dict(
                text=f'Oil Compressibility co vs Pressure — {region_label}  '
                     f'(API={API}°, GOR={GOR} scf/STB, T={T}°F)',
                font=dict(size=13),
            ),
            xaxis=dict(title='Pressure [psia]', tickformat=',.0f'),
            yaxis=dict(title='co [psi⁻¹]'),
            height=450,
            margin=dict(l=60, r=30, t=60, b=50),
            hovermode='x unified',
        )
        st.plotly_chart(fig_co, use_container_width=True)

    st.divider()

    # PVT Data Table
    st.subheader("PVT Data Table")

    df_tbl = pd.DataFrame({
        'P [psia]'                     : P_arr,
        'T [°F]'                       : T,
        'Rs [scf/STB]'                 : np.round(Rs_arr, 2),
        'Bo [RB/STB]'                  : np.round(Bo_arr, 4),
        'μo [cp]'                      : np.round(mu_arr, 4),
        'co [psi⁻¹]'                   : co_arr,
    })

    df_display = df_tbl.copy()
    df_display['co [psi⁻¹]'] = df_tbl['co [psi⁻¹]'].apply(
        lambda v: f"{v:.3e}" if not np.isnan(v) else "< Pb"
    )

    st.dataframe(df_display, use_container_width=True, height=350, hide_index=True)

    # Download CSV
    csv_tbl = df_tbl.copy()
    csv_tbl['co [psi⁻¹]'] = csv_tbl['co [psi⁻¹]'].fillna('')
    st.download_button(
        "Download PVT table as CSV",
        csv_tbl.to_csv(index=False),
        file_name=f"pvt_table_API{API}_GOR{GOR}_T{T}F.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — GAS PVT
# ════════════════════════════════════════════════════════════════════════════

with tabs[1]:
    st.subheader("Gas PVT Properties")

    # Gas inputs inline — 3 columns, no fluid-type selector
    g_c1, g_c2, g_c3 = st.columns(3)
    with g_c1:
        gas_SG_g = st.number_input("Gas SG [-]", 0.50, 1.50, 0.65, 0.001, key='g_sg')
    with g_c2:
        y_CO2    = st.number_input("CO2 mole fraction [-]", 0.0, 0.50, 0.0, 0.001, key='g_co2')
    with g_c3:
        y_H2S    = st.number_input("H2S mole fraction [-]", 0.0, 0.50, 0.0, 0.001, key='g_h2s')

    with st.spinner("Computing gas PVT..."):
        gas_tbl = compute_gas_pvt(gas_SG_g, T, y_CO2, y_H2S, 'dry_gas')

    for w in gas_tbl['warnings']:
        st.warning(w)

    # Pseudocritical metrics
    Tpc_disp = gas_tbl['Tpc']
    Ppc_disp = gas_tbl['Ppc']
    Tpr_disp = (T + 459.67) / Tpc_disp

    pm1, pm2, pm3 = st.columns(3)
    pm1.metric("Tpc", f"{Tpc_disp:.1f} °R  ({Tpc_disp - 459.67:.1f} °F)")
    pm2.metric("Ppc", f"{Ppc_disp:.1f} psia")
    pm3.metric("Tpr at reservoir T", f"{Tpr_disp:.4f}")

    if Tpr_disp < 1.05:
        st.error(
            f"Tpr = {Tpr_disp:.4f} < 1.05. Hall-Yarborough correlation is "
            "undefined below Tpr = 1.05 (Hall & Yarborough 1974). "
            "Increase temperature or reduce gas SG."
        )

    if y_CO2 + y_H2S > 0.0:
        st.info(
            f"Wichert-Aziz (1972) acid-gas correction applied "
            f"(A = {y_CO2 + y_H2S:.3f})."
        )

    # Plotly Gas PVT charts
    valid_mask = ~np.isnan(gas_tbl['Z'])
    P_g  = gas_tbl['P'][valid_mask]
    Z_g  = gas_tbl['Z'][valid_mask]
    Bg_g = gas_tbl['Bg_ft3'][valid_mask]
    mu_g = gas_tbl['mu_g'][valid_mask]
    rho_g = gas_tbl['rho_g'][valid_mask]

    def _gas_fig(x, y, y_label, color, title, hover_fmt='.5f'):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode='lines',
            line=dict(color=color, width=2),
            name=y_label,
            hovertemplate=f'P = %{{x:,.0f}} psia<br>{y_label} = %{{y:{hover_fmt}}}<extra></extra>',
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=13)),
            xaxis=dict(title='Pressure [psia]', tickformat=',.0f'),
            yaxis=dict(title=y_label),
            height=450,
            margin=dict(l=60, r=30, t=60, b=50),
            hovermode='x unified',
        )
        return fig

    st.plotly_chart(
        _gas_fig(P_g, Z_g, 'Z-factor [-]', '#1565C0',
                 f'Z-Factor vs Pressure — Gas SG={gas_SG_g:.3f}, T={T}°F'),
        use_container_width=True,
    )
    st.plotly_chart(
        _gas_fig(P_g, Bg_g, 'Bg [ft³/scf]', '#2E7D32',
                 f'Gas FVF (Bg) vs Pressure — Gas SG={gas_SG_g:.3f}, T={T}°F',
                 hover_fmt='.4e'),
        use_container_width=True,
    )
    st.plotly_chart(
        _gas_fig(P_g, mu_g, 'μg [cp]', '#C62828',
                 f'Gas Viscosity vs Pressure — Gas SG={gas_SG_g:.3f}, T={T}°F'),
        use_container_width=True,
    )
    st.plotly_chart(
        _gas_fig(P_g, rho_g, 'ρg [lb/ft³]', '#6A1B9A',
                 f'Gas Density vs Pressure — Gas SG={gas_SG_g:.3f}, T={T}°F'),
        use_container_width=True,
    )

    st.divider()
    st.subheader("Gas PVT Data Table")

    # Resample to 50-psia steps for a clean display table
    from scipy.interpolate import interp1d
    P_raw = gas_tbl['P'][valid_mask]
    P_step = np.arange(50, int(P_raw.max()) + 1, 50)
    def _interp(arr):
        return interp1d(P_raw, arr, bounds_error=False, fill_value=np.nan)(P_step)
    Z_step   = _interp(gas_tbl['Z'][valid_mask])
    Bg_step  = _interp(gas_tbl['Bg_ft3'][valid_mask])
    Bgb_step = _interp(gas_tbl['Bg_bbl'][valid_mask])
    rho_step = _interp(gas_tbl['rho_g'][valid_mask])
    mu_step  = _interp(gas_tbl['mu_g'][valid_mask])
    cg_step  = _interp(gas_tbl['cg'][valid_mask])

    df_gas = pd.DataFrame({
        'P [psia]'     : P_step,
        'Z [-]'        : np.round(Z_step, 5),
        'Bg [ft³/scf]' : [f"{v:.4e}" if not np.isnan(v) else "" for v in Bg_step],
        'Bg [bbl/scf]' : [f"{v:.4e}" if not np.isnan(v) else "" for v in Bgb_step],
        'ρg [lb/ft³]'  : np.round(rho_step, 4),
        'μg [cp]'      : np.round(mu_step, 6),
        'cg [psi⁻¹]'   : [f"{v:.4e}" if not np.isnan(v) else "" for v in cg_step],
    })

    st.dataframe(df_gas, use_container_width=True, height=300, hide_index=True)

    csv_gas = pd.DataFrame({
        'P_psia'         : P_step,
        'Z'              : Z_step,
        'Bg_ft3_per_scf' : Bg_step,
        'Bg_bbl_per_scf' : Bgb_step,
        'rho_g_lb_ft3'   : rho_step,
        'mu_g_cp'        : mu_step,
        'cg_psi_inv'     : cg_step,
    })
    st.download_button(
        "Download gas PVT table as CSV",
        csv_gas.to_csv(index=False),
        file_name=f"gas_pvt_SG{gas_SG_g:.3f}_T{T}F.csv",
        mime="text/csv",
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — PHASE BEHAVIOR
# ════════════════════════════════════════════════════════════════════════════

with tabs[2]:
    st.subheader("Phase Behavior")

    ph_left, ph_right = st.columns(2)

    # ── Left: PT Phase Envelope ─────────────────────────────────────────────
    with ph_left:
        st.markdown("#### PT Phase Envelope")

        # Determine composition to use
        if _comp_valid and _mixture is not None:
            _env_mixture    = _mixture
            _env_mix_tuple  = _mix_tuple
            _comp_label     = "Real composition (from mole fraction input)"
        else:
            _env_mixture    = pseudo
            _pseudo_tuple   = tuple(
                (pseudo['names'][i],
                 float(pseudo['z'][i]),
                 float(pseudo['Tc'][i]),
                 float(pseudo['Pc'][i]),
                 float(pseudo['omega'][i]),
                 float(pseudo['MW'][i]))
                for i in range(len(pseudo['names']))
            )
            _env_mix_tuple  = _pseudo_tuple
            _comp_label     = f"Pseudo-components (API={API}°, GOR={GOR} scf/STB, gas SG={gas_SG:.3f})"

        st.caption(f"Using: {_comp_label}")

        with st.spinner("Building phase envelope..."):
            env = compute_phase_envelope(_env_mix_tuple, T_range=(50.0, 400.0))

        T_env   = env['T']
        bub_mask = ~np.isnan(env['P_bubble'])
        dew_mask = ~np.isnan(env['P_dew'])
        Tc_mix_F = env['Tc_mix'] - 459.67

        fig_env = go.Figure()
        if bub_mask.any():
            fig_env.add_trace(go.Scatter(
                x=T_env[bub_mask], y=env['P_bubble'][bub_mask],
                mode='lines+markers',
                name='Bubble curve',
                line=dict(color='#1565C0', width=2),
                marker=dict(size=3),
                hovertemplate='T = %{x:.1f} °F<br>P_bubble = %{y:,.0f} psia<extra></extra>',
            ))
        if dew_mask.any():
            fig_env.add_trace(go.Scatter(
                x=T_env[dew_mask], y=env['P_dew'][dew_mask],
                mode='lines+markers',
                name='Dew curve',
                line=dict(color='#C62828', width=2, dash='dash'),
                marker=dict(size=3),
                hovertemplate='T = %{x:.1f} °F<br>P_dew = %{y:,.0f} psia<extra></extra>',
            ))
        fig_env.add_vline(
            x=float(Tc_mix_F),
            line_dash='dot', line_color='gray', line_width=1.2,
            annotation_text=f"Tc mix (Kay's) = {Tc_mix_F:.1f} °F",
            annotation_position='top right',
            annotation_font_color='gray',
            annotation_font_size=10,
        )
        fig_env.update_layout(
            title=dict(
                text='PT Phase Envelope (PR EOS, fugacity-based)',
                font=dict(size=13),
            ),
            xaxis=dict(title='Temperature [°F]'),
            yaxis=dict(title='Pressure [psia]', tickformat=',.0f'),
            height=450,
            margin=dict(l=60, r=30, t=60, b=50),
            hovermode='closest',
            legend=dict(x=0.02, y=0.98),
        )
        st.plotly_chart(fig_env, use_container_width=True)

        n_bub = int(bub_mask.sum())
        n_dew = int(dew_mask.sum())
        n_T   = len(T_env)
        st.caption(
            f"Bubble: {n_bub}/{n_T} converged  |  Dew: {n_dew}/{n_T} converged  "
            "|  T range: 50–400 °F (fixed)"
        )

    # ── Right: Binary Pxy Diagram ───────────────────────────────────────────
    with ph_right:
        st.markdown("#### Binary Pxy Diagram")

        available_components = [c for c in PURE_COMPONENTS.index
                                 if c not in ('C7+', 'C7PLUS')]

        pxy_c1, pxy_c2 = st.columns(2)
        with pxy_c1:
            comp_A_sel = st.selectbox(
                "Component A",
                available_components,
                index=available_components.index('C1') if 'C1' in available_components else 0,
                key='pxy_compA',
            )
        with pxy_c2:
            _b_options = [c for c in available_components if c != comp_A_sel]
            comp_B_sel = st.selectbox(
                "Component B",
                _b_options,
                index=0,
                key='pxy_compB',
            )

        T_pxy = st.number_input(
            "Temperature [°F]", -200.0, 400.0, 100.0, 5.0,
            key='pxy_T',
            help="Temperature for the Pxy diagram.",
        )

        if comp_A_sel == comp_B_sel:
            st.warning("Component A and Component B must be different.")
        else:
            with st.spinner(f"Computing Pxy: {comp_A_sel}/{comp_B_sel} at T={T_pxy:.0f}°F..."):
                pxy_result = compute_pxy(
                    comp_A_sel, comp_B_sel, float(T_pxy),
                    14.7, 5000.0, 50,
                )

            fig_pxy = _P.plot_pxy_plotly(pxy_result)
            fig_pxy.update_layout(height=450)
            st.plotly_chart(fig_pxy, use_container_width=True)

            bub_conv = int(np.sum(~np.isnan(pxy_result['P_bubble'])))
            dew_conv = int(np.sum(~np.isnan(pxy_result['P_dew'])))
            st.caption(
                f"Bubble: {bub_conv}/50 converged  |  "
                f"Dew: {dew_conv}/50 converged  |  "
                "P range: 14.7–5000 psia (fixed)."
            )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ════════════════════════════════════════════════════════════════════════════

with tabs[3]:
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

Defined only above bubble point (P >= Pb). Below Pb, co is shown as "< Pb".

---

### Gas PVT Correlations

#### Pseudocritical Properties
- **Dry gas**: Sutton (1985) SPE-14265, Eqs. 3-4
  - Tpc = 169.2 + 349.5·γg − 74.0·γg²  [°R]
  - Ppc = 756.8 − 131.0·γg − 3.6·γg²   [psia]
- **Gas condensate**: Standing (1977) 9th ed. SPE, p. 204
- **Acid-gas correction**: Wichert & Aziz (1972) *Hydrocarbon Processing*, May 1972, pp. 119-122

#### Z-Factor Method
- **Hall-Yarborough (1974)** SPE-3350-PA — the only method used
  - Newton-Raphson on Starling-Carnahan EOS; valid Tpr >= 1.05
  - Convergence: |Δy/y| < 1×10⁻¹⁰; max 200 iterations
  - Returns NaN with warning if Tpr < 1.05 (correlation undefined)
  - DPR (Dranchuk-Purvis-Robinson 1974) is used only internally as a cross-check

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
Algorithm: Whitson & Brule (2000) SPE Monograph Vol.20, Section 3.3.

#### Pxy Diagram
Binary Pxy diagram via PR EOS fugacity iteration at fixed T.
Bubble and dew curves computed by solving fugacity equality at each feed composition.
Source: Peng & Robinson (1976); Michelsen & Mollerup (2007) Thermodynamic
Models: Fundamentals & Computational Aspects, Ch. 5.

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
12. Michelsen, M.L. & Mollerup, J.M. (2007). *Thermodynamic Models: Fundamentals & Computational Aspects*, 2nd ed.
13. Peng, D.Y. & Robinson, D.B. (1976). A New Two-Constant Equation of State. *Ind. Eng. Chem. Fundam.* 15(1):59–64.
14. Petrosky, G.E. & Farshad, F.F. (1993). PVT Correlations for Gulf of Mexico Crude Oils. SPE-26644.
15. Rachford, H.H. & Rice, J.D. (1952). Procedure for Use of Electronic Digital Computers in Calculating Flash Vaporization. *Trans. AIME* 195:327–328.
16. Standing, M.B. (1947). A Pressure-Volume-Temperature Correlation for Mixtures of California Oils and Gases. *Drill. & Prod. Prac.*, API.
17. Standing, M.B. (1977). *Volumetric and Phase Behavior of Oil Field Hydrocarbon Systems*, 9th ed. SPE.
18. Sutton, R.P. (1985). Compressibility Factors for High-Molecular-Weight Reservoir Gases. SPE-14265.
19. Vasquez, M. & Beggs, H.D. (1980). Correlations for Fluid Physical Property Prediction. *JPT* 32(6):968–970. SPE-6719.
20. Whitson, C.H. & Brule, M.R. (2000). *Phase Behavior of Petroleum Reservoir Fluids*. SPE Monograph Vol. 20.
21. Wichert, E. & Aziz, K. (1972). Calculate Z's for Sour Gases. *Hydrocarbon Processing*, May 1972, pp. 119–122.
22. Wilson, G.M. (1969). A Modified Redlich-Kwong EOS. AIChE 65th National Meeting, Paper 15C.
""")
