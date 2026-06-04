"""
plotting.py
-----------
All visualization functions for the PVT project.

2D static figures  — matplotlib  (publication quality)
3D interactive     — plotly      (rotatable surfaces for Streamlit)

Functions
---------
  plot_pvt_curves()          Rs, Bo, viscosity vs pressure (one region)
  plot_regional_comparison() Pb bar chart + curve overlays (all regions)
  plot_eos_curves()          Z-factor and density vs pressure (PR EOS)
  plot_phase_envelope()      PT phase envelope (PR fugacity bubble/dew)
  plot_gas_pvt_curves()      Z, Bg, μg, ρg vs pressure for gas systems
  plot_flash_result()        Beta, x, y from PR flash at given P, T
  plot_correlations_vs_eos() Bubble point comparison (correlations vs EOS)
  plot_3d_surface()          Generic Plotly 3D surface (P-T grid)
  plot_bo_surface()          Bo(P, T) surface
  plot_viscosity_surface()   μo(P, T) surface
  plot_z_surface()           Z-factor(P, T) surface
  plot_pvt_3d()              P-V-T molar volume surface

Grid computation functions (for @st.cache_data decoration in app.py)
----------------------------------------------------------------------
  _compute_bo_grid()         Bo grid computation
  _compute_visc_grid()       Viscosity grid computation
  _compute_z_grid()          Z-factor grid computation
  _compute_pvt_grid()        Molar volume grid computation

All matplotlib functions return (fig, axes).
All plotly functions return a plotly Figure object.

Removed
-------
  plot_pxy()  — was Wilson K-value binary diagram (qualitative, removed)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import plotly.graph_objects as go

# Project modules
from correlations import pvt_table, bubble_point, REGIONS
from eos import (
    pseudo_components, build_mixture, characterize_c7plus,
    eos_table, pr_bubble_point, pr_phase_envelope
)

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
    Four-panel plot: Rs, Bo, viscosity, and co vs pressure.

    Returns
    -------
    fig, axes : matplotlib Figure and axes array [4]
    """
    _apply_style()
    tbl = pvt_table(API, GOR, T, gas_SG, region=region,
                    T_sp=T_sp, P_sp=P_sp)
    P   = tbl['P']
    Pb  = tbl['Pb']
    region_label = REGIONS[region].split('—')[0].strip()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(
        f'Oil PVT Properties — {region_label}\n'
        f'API={API}°, GOR={GOR} scf/STB, T={T}°F, $\\gamma_g$={gas_SG}',
        fontsize=11, y=1.02
    )

    props = [
        ('Rs',   tbl['Rs'],   '#2196F3', 'Solution GOR $R_s$ [scf/STB]'),
        ('Bo',   tbl['Bo'],   '#4CAF50', 'Oil FVF $B_o$ [RB/STB]'),
        ('mu_o', tbl['mu_o'], '#FF5722', 'Viscosity $\\mu_o$ [cp]'),
        ('co',   tbl['co'],   '#9C27B0', 'Oil Compressibility $c_o$ [psi$^{-1}$]'),
    ]

    for ax, (key, vals, color, ylabel) in zip(axes, props):
        # co is NaN below Pb — plot only valid values
        valid = ~np.isnan(vals)
        if valid.any():
            ax.plot(P[valid], vals[valid], color=color, linewidth=2)
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

    ax_bar = axes[0]
    regions = list(REGIONS.keys())
    pbs, flags = [], []
    for r in regions:
        pb = bubble_point(API, GOR, T, gas_SG, region=r)
        pbs.append(pb)
        flags.append(pb <= 14.8)

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
# 4. PT PHASE ENVELOPE (PR FUGACITY)
# ---------------------------------------------------------------------------

def plot_phase_envelope(mixture, T_range=(50.0, 400.0), label='Fluid'):
    """
    PT phase envelope using PR EOS fugacity-based bubble and dew points.

    Source: pr_phase_envelope() in eos.py, which uses
    pr_bubble_point_fugacity() and pr_dew_point_fugacity().

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    _apply_style()
    env = pr_phase_envelope(mixture, T_range=T_range, n_T=60)
    T   = env['T']

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(f'PT Phase Envelope — {label}\n'
                 '(Peng-Robinson EOS, fugacity-based bubble and dew points)',
                 fontsize=11)

    bub_mask = ~np.isnan(env['P_bubble'])
    dew_mask = ~np.isnan(env['P_dew'])

    if bub_mask.any():
        ax.plot(T[bub_mask], env['P_bubble'][bub_mask],
                'b-o', markersize=3, linewidth=2, label='Bubble point curve')
    if dew_mask.any():
        ax.plot(T[dew_mask], env['P_dew'][dew_mask],
                'r--s', markersize=3, linewidth=2, label='Dew point curve')

    # Kay's rule mixture Tc as approximate reference
    Tc_mix_F = env['Tc_mix'] - 459.67
    ax.axvline(Tc_mix_F, color='gray', linestyle=':', alpha=0.6,
               label=f"Tc mix (Kay's) = {Tc_mix_F:.1f}°F")

    ax.set_xlabel('Temperature [°F]')
    ax.set_ylabel('Pressure [psia]')
    ax.legend(fontsize=8)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f'{x:,.0f}'))

    # Convergence info
    n_bub = int(bub_mask.sum())
    n_dew = int(dew_mask.sum())
    n_T   = len(T)
    ax.text(0.02, 0.02,
            f'Bubble: {n_bub}/{n_T} converged  |  Dew: {n_dew}/{n_T} converged\n'
            'Source: Peng & Robinson (1976); fugacity equality iteration.',
            transform=ax.transAxes, fontsize=7,
            va='bottom', color='#555555', style='italic')

    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# 5. GAS PVT CURVES
# ---------------------------------------------------------------------------

def plot_gas_pvt_curves(gas_tbl):
    """
    Four-panel plot of gas PVT properties vs pressure:
      Z-factor, Bg [ft³/scf], μg [cp], ρg [lb/ft³]

    Parameters
    ----------
    gas_tbl : dict — output of gas_pvt.gas_pvt_table()

    Returns
    -------
    fig, axes : matplotlib Figure and axes [4]
    """
    _apply_style()
    P      = gas_tbl['P']
    Z      = gas_tbl['Z']
    Bg     = gas_tbl['Bg_ft3']
    mu_g   = gas_tbl['mu_g']
    rho_g  = gas_tbl['rho_g']
    T_F    = gas_tbl['T_F']
    sg     = gas_tbl['gas_SG']
    method = gas_tbl['z_method']
    ft     = gas_tbl['fluid_type']
    Tpc    = gas_tbl['Tpc']
    Ppc    = gas_tbl['Ppc']

    method_label = 'Hall-Yarborough (1974)' if method == 'hall_yarborough' else 'DPR (1974)'
    ft_label     = 'Dry Gas (Sutton 1985)' if ft == 'dry_gas' else 'Gas Condensate (Standing 1977)'

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(
        f'Gas PVT Properties — {ft_label}\n'
        f'$\\gamma_g$={sg:.3f}, T={T_F}°F, '
        f'Tpc={Tpc:.1f}°R, Ppc={Ppc:.1f} psia, Z-method: {method_label}',
        fontsize=10, y=1.02
    )

    panels = [
        (Z,     '#1565C0', 'Z-factor [-]',          'Z-Factor vs Pressure'),
        (Bg,    '#2E7D32', 'Bg [ft³/scf]',           'Gas FVF vs Pressure'),
        (mu_g,  '#E65100', 'μg [cp]',                'Gas Viscosity vs Pressure'),
        (rho_g, '#6A1B9A', 'ρg [lb/ft³]',            'Gas Density vs Pressure'),
    ]

    for ax, (vals, color, ylabel, title) in zip(axes, panels):
        valid = ~np.isnan(vals)
        if valid.any():
            ax.plot(P[valid], vals[valid], color=color, linewidth=2)
        ax.set_xlabel('Pressure [psia]')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f'{x:,.0f}'))

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 6. FLASH RESULT
# ---------------------------------------------------------------------------

def plot_flash_result(flash_result, mixture, P, T_F):
    """
    Display PR flash result: vapor fraction β and phase compositions x, y.

    Parameters
    ----------
    flash_result : dict — output of eos.pr_flash()
    mixture      : dict — from build_mixture() or pseudo_components()
    P            : float [psia]
    T_F          : float [°F]

    Returns
    -------
    fig, axes : matplotlib Figure and axes [2]
    """
    _apply_style()
    names = mixture['names']
    beta  = flash_result['beta']
    phase = flash_result['phase']
    x     = flash_result['x']
    y     = flash_result['y']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f'PR EOS Flash Result at P={P:.0f} psia, T={T_F:.0f}°F\n'
        f'Phase: {phase}  |  β (vapor fraction) = {beta:.4f}\n'
        f'Source: Rachford-Rice flash, PR fugacity coefficients — '
        f'Peng & Robinson (1976)',
        fontsize=10
    )

    n = len(names)
    x_pos = np.arange(n)
    width = 0.35

    ax_comp = axes[0]
    if x is not None and not np.all(np.isnan(x)):
        ax_comp.bar(x_pos - width/2, x, width, label='Liquid (x)', color='#1565C0', alpha=0.8)
    if y is not None and not np.all(np.isnan(y)):
        ax_comp.bar(x_pos + width/2, y, width, label='Vapor (y)',  color='#E65100', alpha=0.8)
    ax_comp.plot(x_pos, mixture['z'], 'ko', markersize=5, label='Feed (z)')
    ax_comp.set_xticks(x_pos)
    ax_comp.set_xticklabels(names, rotation=30, ha='right', fontsize=8)
    ax_comp.set_ylabel('Mole Fraction')
    ax_comp.set_title('Phase Compositions')
    ax_comp.legend(fontsize=8)

    ax_beta = axes[1]
    ax_beta.axis('off')
    info_text = (
        f"Flash Summary\n"
        f"{'─'*30}\n"
        f"Phase:           {phase}\n"
        f"Vapor fraction β: {beta:.5f}\n"
        f"Converged:        {flash_result['converged']}\n"
        f"Iterations:       {flash_result['iterations']}\n"
        f"\nZ_L = {flash_result['Z_L'] if flash_result['Z_L'] is not None else 'N/A'}\n"
        f"Z_V = {flash_result['Z_V'] if flash_result['Z_V'] is not None else 'N/A'}\n"
    )
    if flash_result['warnings']:
        info_text += "\nWarnings:\n" + "\n".join(f"  {w}" for w in flash_result['warnings'])
    ax_beta.text(0.05, 0.95, info_text,
                 transform=ax_beta.transAxes,
                 fontsize=9, va='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.8))

    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# 7. CORRELATIONS vs EOS — Pb COMPARISON
# ---------------------------------------------------------------------------

def plot_correlations_vs_eos(API, GOR, T, gas_SG, mixture=None):
    """
    Bar chart comparing Pb from all empirical correlations against
    EOS Wilson K-value estimate (internal helper, for pseudo-components).

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

    labels.append('PR EOS\n(Wilson K,\ninternal)')
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
# GRID COMPUTATION FUNCTIONS (extractable for @st.cache_data)
# ---------------------------------------------------------------------------

def _compute_bo_grid(API, GOR, gas_SG, T_range, P_range,
                     region='global', n_T=30, n_P=40):
    """
    Compute Bo(P, T) grid for 3D surface.

    Suitable for decoration with @st.cache_data in app.py.
    Returns P_arr, T_arr, Bo_grid.
    """
    from correlations import formation_volume_factor, bubble_point as bp

    T_arr   = np.linspace(T_range[0], T_range[1], n_T)
    P_arr   = np.linspace(P_range[0], P_range[1], n_P)
    Bo_grid = np.full((n_T, n_P), np.nan)

    for i, T in enumerate(T_arr):
        try:
            Pb = bp(API, GOR, T, gas_SG, region=region)
            if Pb <= 14.8:
                continue
            for j, P in enumerate(P_arr):
                Bo_grid[i, j] = float(
                    formation_volume_factor(P, API, T, gas_SG, Pb, GOR, region=region)[0]
                )
        except Exception:
            pass

    return P_arr, T_arr, Bo_grid


def _compute_visc_grid(API, GOR, gas_SG, T_range, P_range,
                       region='global', n_T=30, n_P=40):
    """
    Compute μo(P, T) grid for 3D surface.

    Suitable for decoration with @st.cache_data in app.py.
    Returns P_arr, T_arr, mu_grid.
    """
    from correlations import viscosity as vis, bubble_point as bp

    T_arr   = np.linspace(T_range[0], T_range[1], n_T)
    P_arr   = np.linspace(P_range[0], P_range[1], n_P)
    mu_grid = np.full((n_T, n_P), np.nan)

    for i, T in enumerate(T_arr):
        try:
            Pb = bp(API, GOR, T, gas_SG, region=region)
            if Pb <= 14.8:
                continue
            for j, P in enumerate(P_arr):
                mu_grid[i, j] = float(vis(P, API, T, gas_SG, Pb, GOR, region=region)[0])
        except Exception:
            pass

    return P_arr, T_arr, mu_grid


def _compute_z_grid(API, GOR, gas_SG, T_range, P_range, n_T=25, n_P=35):
    """
    Compute Z(P, T) grid for 3D surface using PR EOS.

    Suitable for decoration with @st.cache_data in app.py.
    Returns P_arr, T_arr, Z_grid.
    """
    from eos import pr_density as prd

    T_arr  = np.linspace(T_range[0], T_range[1], n_T)
    P_arr  = np.linspace(P_range[0], P_range[1], n_P)
    Z_grid = np.full((n_T, n_P), np.nan)

    for i, T_F in enumerate(T_arr):
        mix = pseudo_components(API, GOR, gas_SG)
        for j, P in enumerate(P_arr):
            try:
                _, Z, _ = prd(P, T_F, mix)
                Z_grid[i, j] = Z
            except Exception:
                pass

    return P_arr, T_arr, Z_grid


def _compute_pvt_grid(API, GOR, gas_SG, T_range, P_range, n_T=25, n_P=35):
    """
    Compute V_mol(P, T) grid for 3D surface using PR EOS.

    Suitable for decoration with @st.cache_data in app.py.
    Returns P_arr, T_arr, V_grid.
    """
    from eos import pr_density as prd

    T_arr  = np.linspace(T_range[0], T_range[1], n_T)
    P_arr  = np.linspace(P_range[0], P_range[1], n_P)
    V_grid = np.full((n_T, n_P), np.nan)

    for i, T_F in enumerate(T_arr):
        mix = pseudo_components(API, GOR, gas_SG)
        for j, P in enumerate(P_arr):
            try:
                _, _, V = prd(P, T_F, mix)
                V_grid[i, j] = V
            except Exception:
                pass

    return P_arr, T_arr, V_grid


# ---------------------------------------------------------------------------
# 3D SURFACE PLOTS (PLOTLY)
# ---------------------------------------------------------------------------

def plot_3d_surface(P_arr, T_arr, Z_grid, title, z_label,
                    colorscale='Viridis', opacity=0.9):
    """
    Generic interactive 3D surface plot using Plotly.

    Parameters
    ----------
    P_arr    : array [n_P]
    T_arr    : array [n_T]
    Z_grid   : array [n_T, n_P]
    title    : str
    z_label  : str
    colorscale : str
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
    Bo(P, T) interactive 3D surface from empirical correlations.
    Uses _compute_bo_grid() for the data (can be cached separately in app.py).
    """
    P_arr, T_arr, Bo_grid = _compute_bo_grid(
        API, GOR, gas_SG, T_range, P_range, region, n_T, n_P
    )
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
    Viscosity(P, T) interactive 3D surface.
    Uses _compute_visc_grid() for the data (can be cached separately in app.py).
    """
    P_arr, T_arr, mu_grid = _compute_visc_grid(
        API, GOR, gas_SG, T_range, P_range, region, n_T, n_P
    )
    region_label = REGIONS[region].split('—')[0].strip()
    return plot_3d_surface(
        P_arr, T_arr, mu_grid,
        title=f'Oil Viscosity $\\mu_o$(P,T) — {region_label}',
        z_label='μo [cp]',
        colorscale='Reds',
    )


def plot_z_surface(API=35, GOR=500, gas_SG=0.65,
                   T_range=(100.0, 300.0), P_range=(50.0, 6000.0),
                   n_T=25, n_P=35, label='Fluid'):
    """
    Z-factor(P, T) interactive 3D surface from PR EOS.
    Uses _compute_z_grid() for the data (can be cached separately in app.py).
    """
    P_arr, T_arr, Z_grid = _compute_z_grid(
        API, GOR, gas_SG, T_range, P_range, n_T, n_P
    )
    return plot_3d_surface(
        P_arr, T_arr, Z_grid,
        title=f'Z-Factor(P,T) — PR EOS — {label}',
        z_label='Z [-]',
        colorscale='Plasma',
    )


def plot_pvt_3d(API=35, GOR=500, gas_SG=0.65,
                T_range=(100.0, 300.0), P_range=(50.0, 6000.0),
                n_T=25, n_P=35, label='Fluid'):
    """
    P-V-T surface: molar volume [ft³/lbmol] vs P and T from PR EOS.
    Uses _compute_pvt_grid() for the data (can be cached separately in app.py).
    """
    P_arr, T_arr, V_grid = _compute_pvt_grid(
        API, GOR, gas_SG, T_range, P_range, n_T, n_P
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

    # 1. PVT curves (now 4 panels)
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

    # 4. Phase envelope (fugacity-based)
    fig, _ = plot_phase_envelope(pseudo, T_range=(80, 300), label='35°API Black Oil (pseudo)')
    fig.savefig("test_plots/phase_envelope.png", dpi=120, bbox_inches='tight')
    print("  [4] phase_envelope.png saved")
    plt.close(fig)

    # 5. Gas PVT curves
    from gas_pvt import gas_pvt_table
    gtbl = gas_pvt_table(gas_SG=0.65, T_F=200.0, fluid_type='dry_gas',
                         z_method='hall_yarborough', P_min=100.0, P_max=8000.0, n_points=80)
    fig, _ = plot_gas_pvt_curves(gtbl)
    fig.savefig("test_plots/gas_pvt_curves.png", dpi=120, bbox_inches='tight')
    print("  [5] gas_pvt_curves.png saved")
    plt.close(fig)

    # 6. Flash result
    from eos import build_mixture, characterize_c7plus, pr_flash
    comp = {'C1':0.45,'C2':0.05,'C3':0.05,'nC4':0.03,'nC5':0.01,'nC6':0.01,'C7+':0.40}
    c7   = characterize_c7plus(215.0, 0.87)
    mix  = build_mixture(comp, c7plus_props=c7)
    fr   = pr_flash(mix, P=1500.0, T_F=180.0)
    fig, _ = plot_flash_result(fr, mix, P=1500.0, T_F=180.0)
    fig.savefig("test_plots/flash_result.png", dpi=120, bbox_inches='tight')
    print("  [6] flash_result.png saved")
    plt.close(fig)

    # 7. Correlations vs EOS
    fig, _ = plot_correlations_vs_eos(API, GOR, T, gas_SG, mixture=pseudo)
    fig.savefig("test_plots/corr_vs_eos.png", dpi=120, bbox_inches='tight')
    print("  [7] corr_vs_eos.png saved")
    plt.close(fig)

    # 8. 3D Bo surface
    fig_3d = plot_bo_surface(API, GOR, gas_SG, T_range=(100, 280),
                              P_range=(100, 5000), region='western_usa')
    fig_3d.write_html("test_plots/bo_surface.html")
    print("  [8] bo_surface.html saved")

    # 9. 3D Viscosity surface
    fig_3d = plot_viscosity_surface(API, GOR, gas_SG, T_range=(100, 280),
                                     P_range=(100, 5000))
    fig_3d.write_html("test_plots/viscosity_surface.html")
    print("  [9] viscosity_surface.html saved")

    # 10. 3D Z-factor surface
    fig_3d = plot_z_surface(API, GOR, gas_SG, T_range=(100, 280),
                             P_range=(100, 5000), label='35°API Black Oil')
    fig_3d.write_html("test_plots/z_surface.html")
    print("  [10] z_surface.html saved")

    # 11. P-V-T surface
    fig_3d = plot_pvt_3d(API, GOR, gas_SG, T_range=(100, 280),
                          P_range=(100, 5000), label='35°API Black Oil')
    fig_3d.write_html("test_plots/pvt_3d.html")
    print("  [11] pvt_3d.html saved")

    print("\nAll plots generated successfully.")
