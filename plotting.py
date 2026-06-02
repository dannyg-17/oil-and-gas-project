"""
plotting.py
-----------
All visualization functions for the PVT project.

2D static figures  — matplotlib  (Spyder-friendly, publication quality)
3D interactive     — plotly      (rotatable surfaces, embeds in Streamlit)

Functions
---------
  plot_pvt_curves()          Rs, Bo, viscosity vs pressure (one region)
  plot_regional_comparison() Pb bar chart + curve overlays (all regions)
  plot_eos_curves()          Z-factor and density vs pressure
  plot_phase_envelope()      PT phase envelope from EOS
  plot_pxy()                 Pxy diagram for binary pair
  plot_3d_surface()          Generic Plotly 3D surface (P-T-Z grid)
  plot_bo_surface()          Bo(P, T) surface
  plot_viscosity_surface()   mu(P, T) surface
  plot_z_surface()           Z-factor(P, T) surface
  plot_pvt_3d()              Full P-V-T molar volume surface
  plot_correlations_vs_eos() Side-by-side Pb comparison

All matplotlib functions return (fig, axes).
All plotly functions return a plotly Figure object.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend for Spyder + Streamlit
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Project modules
from correlations import pvt_table, bubble_point, REGIONS
from eos import (pseudo_components, build_mixture, characterize_c7plus,
                 eos_table, phase_envelope_wilson, pxy_diagram, pr_bubble_point)

# ---------------------------------------------------------------------------
# STYLE CONSTANTS
# ---------------------------------------------------------------------------

_REGION_COLORS = {
    'western_usa'   : '#2196F3',
    'middle_east'   : '#FF9800',
    'north_sea'     : '#4CAF50',
    'global'        : '#9C27B0',
    'gulf_of_mexico': '#F44336',
}

_REGION_LABELS = {
    'western_usa'   : 'Standing (1947)\nW. USA',
    'middle_east'   : 'Al-Marhoun (1988)\nMiddle East',
    'north_sea'     : 'Glaso (1980)\nNorth Sea',
    'global'        : 'Vasquez & Beggs (1980)\nGlobal',
    'gulf_of_mexico': 'Petrosky & Farshad (1993)\nGulf of Mexico',
}

_MPL_STYLE = {
    'figure.facecolor' : 'white',
    'axes.facecolor'   : '#FAFAFA',
    'axes.grid'        : True,
    'grid.alpha'       : 0.4,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 10,
}


def _apply_style():
    plt.rcParams.update(_MPL_STYLE)



def _add_pb_line(ax, Pb, color='red', label=True):
    """Add vertical dashed line at bubble point."""
    ax.axvline(Pb, color=color, linestyle='--', linewidth=1.2, alpha=0.7,
               label=f'$P_b$ = {Pb:.0f} psia' if label else None)


# ---------------------------------------------------------------------------
# 1. PVT CURVES — Rs, Bo, Viscosity vs Pressure
# ---------------------------------------------------------------------------

def plot_pvt_curves(API, GOR, T, gas_SG, region='global',
                    T_sp=60.0, P_sp=114.7):
    """
    Three-panel plot: Rs, Bo, and viscosity vs pressure.

    Parameters
    ----------
    API, GOR, T, gas_SG : fluid inputs
    region  : str — correlation region key
    T_sp    : float — separator temperature [°F] (for V&B correction)
    P_sp    : float — separator pressure [psia]  (default = no correction)

    Returns
    -------
    fig, axes : matplotlib Figure and axes array [3]
    """
    _apply_style()
    tbl = pvt_table(API, GOR, T, gas_SG, region=region,
                    T_sp=T_sp, P_sp=P_sp)
    P   = tbl['P']
    Pb  = tbl['Pb']
    region_label = REGIONS[region].split('—')[0].strip()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f'PVT Properties — {region_label}\n'
        f'API={API}°, GOR={GOR} scf/STB, T={T}°F, $\\gamma_g$={gas_SG}',
        fontsize=11, y=1.02
    )

    props = [
        ('Rs',   tbl['Rs'],   '#2196F3', 'Solution GOR $R_s$ [scf/STB]'),
        ('Bo',   tbl['Bo'],   '#4CAF50', 'Oil FVF $B_o$ [RB/STB]'),
        ('mu_o', tbl['mu_o'], '#FF5722', 'Viscosity $\\mu_o$ [cp]'),
    ]

    for ax, (key, vals, color, ylabel) in zip(axes, props):
        ax.plot(P, vals, color=color, linewidth=2)
        _add_pb_line(ax, Pb)
        ax.set_xlabel('Pressure [psia]')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f'{x:,.0f}'))

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 2. REGIONAL COMPARISON
# ---------------------------------------------------------------------------

def plot_regional_comparison(API, GOR, T, gas_SG):
    """
    Two-panel figure:
      Left:  bar chart of Pb across all five regional correlations
      Right: Rs vs pressure overlay for all valid regions

    Petrosky & Farshad is included but flagged if Pb ≤ 14.7 psia
    (outside development range).

    Returns
    -------
    fig, axes : matplotlib Figure and axes [2]
    """
    _apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'Regional Correlation Comparison\n'
        f'API={API}°, GOR={GOR} scf/STB, T={T}°F, $\\gamma_g$={gas_SG}',
        fontsize=11, y=1.02
    )

    # --- Bubble point bar chart
    ax_bar = axes[0]
    regions = list(REGIONS.keys())
    pbs     = []
    flags   = []
    for r in regions:
        pb = bubble_point(API, GOR, T, gas_SG, region=r)
        pbs.append(pb)
        flags.append(pb <= 14.8)   # clipped to atmospheric — out of range

    colors = [
        '#BDBDBD' if flag else _REGION_COLORS[r]
        for r, flag in zip(regions, flags)
    ]
    bars = ax_bar.bar(range(len(regions)), pbs, color=colors, width=0.6,
                      edgecolor='white', linewidth=0.8)
    ax_bar.set_xticks(range(len(regions)))
    ax_bar.set_xticklabels(
        [_REGION_LABELS[r].split('\n')[0] for r in regions],
        rotation=30, ha='right', fontsize=8
    )
    ax_bar.set_ylabel('Bubble Point Pressure $P_b$ [psia]')
    ax_bar.set_title('Bubble Point — Regional Comparison')

    for bar, pb, flag in zip(bars, pbs, flags):
        label = f'{pb:.0f}' + (' *' if flag else '')
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(pbs) * 0.01,
                    label, ha='center', va='bottom', fontsize=8)

    if any(flags):
        ax_bar.text(0.02, 0.97,
                    '* clipped to 14.7 psia\n  (out of development range)',
                    transform=ax_bar.transAxes, fontsize=7,
                    va='top', color='#757575', style='italic')

    # --- Rs overlay
    ax_rs = axes[1]
    for r in regions:
        pb = bubble_point(API, GOR, T, gas_SG, region=r)
        if pb <= 14.8:
            continue
        tbl = pvt_table(API, GOR, T, gas_SG, region=r)
        label = _REGION_LABELS[r].replace('\n', ' ')
        ax_rs.plot(tbl['P'], tbl['Rs'],
                   color=_REGION_COLORS[r], linewidth=1.8, label=label)

    ax_rs.set_xlabel('Pressure [psia]')
    ax_rs.set_ylabel('Solution GOR $R_s$ [scf/STB]')
    ax_rs.set_title('$R_s$ vs Pressure — Regional Overlay')
    ax_rs.legend(fontsize=7, loc='upper left')
    ax_rs.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{x:,.0f}'))

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 3. EOS CURVES — Z-factor and Density
# ---------------------------------------------------------------------------

def plot_eos_curves(mixture, T_F, label='Fluid'):
    """
    Two-panel plot: Z-factor and density vs pressure from PR EOS.

    Parameters
    ----------
    mixture : dict — from eos.build_mixture() or eos.pseudo_components()
    T_F     : float — temperature [°F]
    label   : str   — title descriptor

    Returns
    -------
    fig, axes : matplotlib Figure and axes [2]
    """
    _apply_style()
    tbl = eos_table(mixture, T_F)
    P   = tbl['P']
    Pb  = tbl['Pb_eos']

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        f'PR EOS Properties — {label}  (T={T_F}°F)\n'
        f'[Wilson K-value bubble point: {Pb:.0f} psia]',
        fontsize=11, y=1.02
    )

    ax_Z, ax_rho = axes

    ax_Z.plot(P, tbl['Z'], color='#1565C0', linewidth=2)
    if not np.isnan(Pb):
        _add_pb_line(ax_Z, Pb)
    ax_Z.set_xlabel('Pressure [psia]')
    ax_Z.set_ylabel('Compressibility Factor Z [-]')
    ax_Z.set_title('Z-Factor vs Pressure')
    ax_Z.legend(fontsize=8)

    ax_rho.plot(P, tbl['rho'], color='#2E7D32', linewidth=2)
    if not np.isnan(Pb):
        _add_pb_line(ax_rho, Pb)
    ax_rho.set_xlabel('Pressure [psia]')
    ax_rho.set_ylabel('Density [lb/ft³]')
    ax_rho.set_title('Mixture Density vs Pressure')
    ax_rho.legend(fontsize=8)

    for ax in axes:
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f'{x:,.0f}'))

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 4. PT PHASE ENVELOPE
# ---------------------------------------------------------------------------

def plot_phase_envelope(mixture, T_range=(50.0, 400.0), label='Fluid'):
    """
    PT phase envelope (bubble and dew curves) from Wilson K-values.

    Includes annotation noting this is a first-order approximation.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    _apply_style()
    env = phase_envelope_wilson(mixture, T_range=T_range, n_T=80)
    T   = env['T']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f'PT Phase Envelope — {label}\n'
                 '(Wilson K-value approximation)', fontsize=11)

    bub_mask = ~np.isnan(env['P_bubble'])
    dew_mask = ~np.isnan(env['P_dew'])

    if bub_mask.any():
        ax.plot(T[bub_mask], env['P_bubble'][bub_mask],
                'b-o', markersize=3, linewidth=2, label='Bubble point curve')
    if dew_mask.any():
        ax.plot(T[dew_mask], env['P_dew'][dew_mask],
                'r--s', markersize=3, linewidth=2, label='Dew point curve')

    # Mark cricondentherm and cricondenbar if computable
    Tc_mix_F = env['Tc_mix'] - 459.67
    Pc_mix   = env['Pc_mix']
    ax.axvline(Tc_mix_F, color='gray', linestyle=':', alpha=0.6,
               label=f"Tc mix (Kay's) = {Tc_mix_F:.1f}°F")

    ax.set_xlabel('Temperature [°F]')
    ax.set_ylabel('Pressure [psia]')
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{x:,.0f}'))

    ax.text(0.02, 0.02,
            'Note: Wilson K-values give qualitative envelope shape.\n'
            'Quantitative accuracy requires PR fugacity flash.',
            transform=ax.transAxes, fontsize=7,
            va='bottom', color='#757575', style='italic')

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 5. PXY DIAGRAM
# ---------------------------------------------------------------------------

def plot_pxy(comp_A, comp_B, T_F, P_range=(14.7, 3000.0)):
    """
    Pxy diagram for a binary mixture at fixed temperature.

    X-axis: mole fraction of component A
    Y-axis: pressure [psia]
    Shows bubble and dew curves.

    Wilson K-value approximation — noted on figure.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    _apply_style()
    pxy = pxy_diagram(comp_A, comp_B, T_F, P_range=P_range)
    x   = pxy['x_A']

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(f'Pxy Diagram: {comp_A} – {comp_B}  at  T={T_F}°F\n'
                 '(Wilson K-value approximation)', fontsize=11)

    bub_m = ~np.isnan(pxy['P_bubble'])
    dew_m = ~np.isnan(pxy['P_dew'])

    if bub_m.any():
        ax.plot(x[bub_m], pxy['P_bubble'][bub_m],
                'b-', linewidth=2, label='Bubble point')
    if dew_m.any():
        ax.plot(x[dew_m], pxy['P_dew'][dew_m],
                'r--', linewidth=2, label='Dew point')

    ax.set_xlabel(f'Mole fraction {comp_A}  $x_{{{comp_A}}}$')
    ax.set_ylabel('Pressure [psia]')
    ax.legend(fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{x:,.0f}'))

    ax.text(0.02, 0.97,
            'Wilson K-values: qualitative shape only.',
            transform=ax.transAxes, fontsize=7,
            va='top', color='#757575', style='italic')

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 6. CORRELATIONS vs EOS — Pb COMPARISON
# ---------------------------------------------------------------------------

def plot_correlations_vs_eos(API, GOR, T, gas_SG, mixture=None):
    """
    Bar chart comparing Pb from all empirical correlations against
    EOS Wilson K-value estimate.

    Parameters
    ----------
    mixture : dict or None — if None, pseudo_components() is used

    Returns
    -------
    fig, ax
    """
    _apply_style()

    if mixture is None:
        mixture = pseudo_components(API, GOR, gas_SG)
    Pb_eos = pr_bubble_point(mixture, T)

    labels, pbs, colors = [], [], []
    for r in REGIONS:
        pb = bubble_point(API, GOR, T, gas_SG, region=r)
        labels.append(_REGION_LABELS[r].replace('\n', '\n'))
        pbs.append(pb)
        colors.append(_REGION_COLORS[r] if pb > 14.8 else '#BDBDBD')

    labels.append('PR EOS\n(Wilson K)')
    pbs.append(Pb_eos if not np.isnan(Pb_eos) else 0)
    colors.append('#607D8B')

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(len(labels)), pbs, color=colors, width=0.6,
                  edgecolor='white', linewidth=0.8)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Bubble Point Pressure $P_b$ [psia]')
    ax.set_title(
        f'Correlations vs EOS — Bubble Point\n'
        f'API={API}°, GOR={GOR} scf/STB, T={T}°F, $\\gamma_g$={gas_SG}'
    )

    for bar, pb in zip(bars, pbs):
        if pb > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(pbs) * 0.01,
                    f'{pb:.0f}', ha='center', va='bottom', fontsize=8)

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 7–9. 3D INTERACTIVE SURFACES (PLOTLY)
# ---------------------------------------------------------------------------

def _build_pt_grid(mixture_fn, T_range, P_range, n_T, n_P, property_fn):
    """
    Build P-T grid and evaluate a scalar property at each point.

    mixture_fn : callable(T_F) → mixture dict  (allows T-dependent props)
    property_fn: callable(P, T_F, mixture) → float
    Returns P_arr, T_arr, Z_grid (n_T × n_P)
    """
    T_arr = np.linspace(T_range[0], T_range[1], n_T)
    P_arr = np.linspace(P_range[0], P_range[1], n_P)
    Z_grid = np.full((n_T, n_P), np.nan)

    for i, T_F in enumerate(T_arr):
        mix = mixture_fn(T_F)
        for j, P in enumerate(P_arr):
            try:
                Z_grid[i, j] = property_fn(P, T_F, mix)
            except Exception:
                pass

    return P_arr, T_arr, Z_grid


def plot_3d_surface(P_arr, T_arr, Z_grid, title, z_label,
                    colorscale='Viridis', opacity=0.9):
    """
    Generic interactive 3D surface plot using Plotly.

    Parameters
    ----------
    P_arr    : array [n_P] — pressure axis [psia]
    T_arr    : array [n_T] — temperature axis [°F]
    Z_grid   : array [n_T, n_P] — property values
    title    : str
    z_label  : str — Z-axis label
    colorscale : str — Plotly colorscale name
    opacity  : float

    Returns
    -------
    plotly Figure
    """
    P_mesh, T_mesh = np.meshgrid(P_arr, T_arr)

    fig = go.Figure(data=[go.Surface(
        x=P_mesh, y=T_mesh, z=Z_grid,
        colorscale=colorscale,
        opacity=opacity,
        colorbar=dict(title=z_label, thickness=18),
        hovertemplate=(
            'P = %{x:.0f} psia<br>'
            'T = %{y:.1f} °F<br>'
            + z_label + ' = %{z:.4f}<extra></extra>'
        ),
    )])

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=14)),
        scene=dict(
            xaxis_title='Pressure [psia]',
            yaxis_title='Temperature [°F]',
            zaxis_title=z_label,
            camera=dict(eye=dict(x=1.5, y=-1.8, z=0.8)),
        ),
        margin=dict(l=0, r=0, b=0, t=60),
        height=600,
    )
    return fig


def plot_bo_surface(API, GOR, gas_SG,
                    T_range=(100.0, 300.0), P_range=(50.0, 6000.0),
                    region='global', n_T=30, n_P=40):
    """
    Bo(P, T) surface from empirical correlations.
    Temperature is swept; at each T, Pb is recomputed so Bo is always
    physically consistent.
    """
    from correlations import formation_volume_factor, bubble_point, solution_gor

    T_arr  = np.linspace(T_range[0], T_range[1], n_T)
    P_arr  = np.linspace(P_range[0], P_range[1], n_P)
    Bo_grid = np.full((n_T, n_P), np.nan)

    for i, T in enumerate(T_arr):
        try:
            Pb = bubble_point(API, GOR, T, gas_SG, region=region)
            if Pb <= 14.8:
                continue
            for j, P in enumerate(P_arr):
                Bo_grid[i, j] = float(formation_volume_factor(
                    P, API, T, gas_SG, Pb, GOR, region=region
                )[0])
        except Exception:
            pass

    region_label = REGIONS[region].split('—')[0].strip()
    return plot_3d_surface(
        P_arr, T_arr, Bo_grid,
        title=f'Oil FVF $B_o$(P,T) — {region_label}',
        z_label='Bo [RB/STB]',
        colorscale='Blues',
    )


def plot_viscosity_surface(API, GOR, gas_SG,
                           T_range=(100.0, 300.0), P_range=(50.0, 6000.0),
                           region='global', n_T=30, n_P=40):
    """
    Viscosity(P, T) surface from Beggs & Robinson / Vasquez & Beggs.
    """
    from correlations import viscosity, bubble_point

    T_arr   = np.linspace(T_range[0], T_range[1], n_T)
    P_arr   = np.linspace(P_range[0], P_range[1], n_P)
    mu_grid = np.full((n_T, n_P), np.nan)

    for i, T in enumerate(T_arr):
        try:
            Pb = bubble_point(API, GOR, T, gas_SG, region=region)
            if Pb <= 14.8:
                continue
            for j, P in enumerate(P_arr):
                mu_grid[i, j] = float(viscosity(
                    P, API, T, gas_SG, Pb, GOR, region=region
                )[0])
        except Exception:
            pass

    region_label = REGIONS[region].split('—')[0].strip()
    return plot_3d_surface(
        P_arr, T_arr, mu_grid,
        title=f'Oil Viscosity $\\mu_o$(P,T) — {region_label}',
        z_label='μo [cp]',
        colorscale='Reds',
    )


def plot_z_surface(mixture_fn, T_range=(100.0, 300.0),
                   P_range=(50.0, 6000.0), n_T=25, n_P=35, label='Fluid'):
    """
    Z-factor(P, T) surface from PR EOS.

    Parameters
    ----------
    mixture_fn : callable(T_F) → mixture dict
                 Use lambda T: pseudo_components(API, GOR, gas_SG)
                 or lambda T: your_fixed_mixture  for real compositions.
    """
    from eos import pr_density

    def z_fn(P, T_F, mix):
        _, Z, _ = pr_density(P, T_F, mix)
        return Z

    P_arr, T_arr, Z_grid = _build_pt_grid(
        mixture_fn, T_range, P_range, n_T, n_P, z_fn
    )
    return plot_3d_surface(
        P_arr, T_arr, Z_grid,
        title=f'Z-Factor(P,T) — PR EOS — {label}',
        z_label='Z [-]',
        colorscale='Plasma',
    )


def plot_pvt_3d(mixture_fn, T_range=(100.0, 300.0),
                P_range=(50.0, 6000.0), n_T=25, n_P=35, label='Fluid'):
    """
    Full P-V-T surface: molar volume V [ft³/lbmol] vs P and T.
    This is the EOS equivalent of the classic thermodynamics PVT surface.
    """
    from eos import pr_density

    def v_fn(P, T_F, mix):
        _, _, V = pr_density(P, T_F, mix)
        return V

    P_arr, T_arr, V_grid = _build_pt_grid(
        mixture_fn, T_range, P_range, n_T, n_P, v_fn
    )
    return plot_3d_surface(
        P_arr, T_arr, V_grid,
        title=f'P-V-T Surface (Molar Volume) — {label}',
        z_label='V [ft³/lbmol]',
        colorscale='Turbo',
    )


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.makedirs("test_plots", exist_ok=True)

    API, GOR, T, gas_SG = 35, 500, 180.0, 0.65
    print("plotting.py — running self-test...")

    # 1. PVT curves
    fig, _ = plot_pvt_curves(API, GOR, T, gas_SG, region='western_usa')
    fig.savefig("test_plots/pvt_curves.png", dpi=120, bbox_inches='tight')
    print("  [1] pvt_curves.png saved")
    plt.close(fig)

    # 2. Regional comparison
    fig, _ = plot_regional_comparison(API, GOR, T, gas_SG)
    fig.savefig("test_plots/regional_comparison.png", dpi=120, bbox_inches='tight')
    print("  [2] regional_comparison.png saved")
    plt.close(fig)

    # 3. EOS curves
    pseudo = pseudo_components(API, GOR, gas_SG)
    fig, _ = plot_eos_curves(pseudo, T_F=T, label='35°API Black Oil (pseudo)')
    fig.savefig("test_plots/eos_curves.png", dpi=120, bbox_inches='tight')
    print("  [3] eos_curves.png saved")
    plt.close(fig)

    # 4. Phase envelope
    fig, _ = plot_phase_envelope(pseudo, label='35°API Black Oil (pseudo)')
    fig.savefig("test_plots/phase_envelope.png", dpi=120, bbox_inches='tight')
    print("  [4] phase_envelope.png saved")
    plt.close(fig)

    # 5. Pxy
    fig, _ = plot_pxy('C1', 'C3', T_F=100.0, P_range=(14.7, 2500.0))
    fig.savefig("test_plots/pxy_C1_C3.png", dpi=120, bbox_inches='tight')
    print("  [5] pxy_C1_C3.png saved")
    plt.close(fig)

    # 6. Correlations vs EOS
    fig, _ = plot_correlations_vs_eos(API, GOR, T, gas_SG, mixture=pseudo)
    fig.savefig("test_plots/corr_vs_eos.png", dpi=120, bbox_inches='tight')
    print("  [6] corr_vs_eos.png saved")
    plt.close(fig)

    # 7. 3D Bo surface
    fig_3d = plot_bo_surface(API, GOR, gas_SG, T_range=(100, 280),
                              P_range=(100, 5000), region='western_usa')
    fig_3d.write_html("test_plots/bo_surface.html")
    print("  [7] bo_surface.html saved")

    # 8. 3D Viscosity surface
    fig_3d = plot_viscosity_surface(API, GOR, gas_SG, T_range=(100, 280),
                                     P_range=(100, 5000))
    fig_3d.write_html("test_plots/viscosity_surface.html")
    print("  [8] viscosity_surface.html saved")

    # 9. 3D Z-factor surface
    mix_fn = lambda T_F: pseudo_components(API, GOR, gas_SG)
    fig_3d = plot_z_surface(mix_fn, T_range=(100, 280), P_range=(100, 5000),
                             label='35°API Black Oil')
    fig_3d.write_html("test_plots/z_surface.html")
    print("  [9] z_surface.html saved")

    # 10. P-V-T surface
    fig_3d = plot_pvt_3d(mix_fn, T_range=(100, 280), P_range=(100, 5000),
                          label='35°API Black Oil')
    fig_3d.write_html("test_plots/pvt_3d.html")
    print("  [10] pvt_3d.html saved")

    print("\nAll plots generated successfully.")
