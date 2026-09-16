# -*- coding: utf-8 -*-
"""
Example: Series configuration with single U-tube boreholes, heat_flux mode.

Runs a 9-borehole field in series mode (3 groups of 3 boreholes) for one
year, in heat_flux mode with a constant heat load and a constant supply
temperature: constant extraction for the first half of the year, plant off
for the second half. Plots extracted heat and outlet fluid temperature.
"""

import matplotlib.pyplot as plt
import numpy as np

from carm import (
    BoreholeGeometry,
    BoreholeMesh,
    BoreholeThermalProperties,
    SingleUtube,
)
from carm import EnvironmentalProperties, EnvironmentalTimeSeries
from carm import FieldInput
from carm import Fluid
from carm import GroundGeometry, GroundMesh
from carm import PhysicalModel
from carm import Simulation


def _synthetic_weather(n_steps: int, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Synthetic external air temperature / solar irradiance profile (seasonal +
    daily cycle), used here in place of measured data. For real data, build
    ``T_ext``/``SolarRad`` from your own source (e.g. ``pandas.read_excel``)
    and pass them to ``EnvironmentalTimeSeries.from_array``.
    """
    hours = np.arange(n_steps) * dt / 3600.0
    T_ext = 12.5 - 11.0 * np.cos(2 * np.pi * hours / 8760.0) + 3.0 * np.sin(2 * np.pi * hours / 24.0)
    daylight = np.clip(np.sin(2 * np.pi * (hours % 24.0 - 6.0) / 24.0), 0.0, None)
    seasonal = 0.5 - 0.5 * np.cos(2 * np.pi * hours / 8760.0)
    SolarRad = 900.0 * daylight * seasonal
    return T_ext, SolarRad


def main():

    # -------------------------------------------------------------------------
    # Input parameters
    # -------------------------------------------------------------------------

    n_bhes = 9
    x_min, y_min = -2.5, -2.5
    x_max, y_max = 12.5, 12.5

    stratification = [(1.8, 947.37, 1900, 111)]
    n_mesh = 20
    m_mesh = 40
    Tg = 13
    L = 100
    m_mesh_sup = 4
    m_mesh_inf = 40
    L_sup = 1
    L_inf = 10

    k_w = 0.568709114496803
    rho_w = 1000.1435933169
    cp_w = 4207.40834247225
    ni_w = 1.49626063208248e-6

    Dpi = 0.026
    Lbore = 100
    D0 = 0.15
    Rp0 = 0.25
    RppB = 0.72
    n_pipes = 2
    pipe_thick = 0.003
    pipe_spacing = 0.0823
    cp_0 = 1460
    rho_0 = 1655
    k0 = 1.8

    absorptance = 0.7
    eps = 0.95
    At = 10
    tau = 0
    tau_y = 365 * 24 * 3600
    tau_shift = 210 * 24 * 3600
    R_ext = 0.04

    Tm = 13

    dt = 3600
    n_steps = 8760  # one year

    # series groups: each group defines an ordered chain of boreholes
    groups = {
        "group_0": [0, 1, 2],
        "group_1": [3, 4, 5],
        "group_2": [6, 7, 8],
    }
    n_groups = len(groups)

    # -------------------------------------------------------------------------
    # Heat flux mode: constant extraction load for the first half of the
    # year, plant off for the second half. Same load and supply temperature
    # for every group.
    # -------------------------------------------------------------------------

    Q_load = 5000  # W, constant heat extracted from the ground (heating case)
    T_supply_value = 45  # °C, constant supply temperature to the building

    Q_buildings = np.zeros(n_steps, dtype=np.float64)
    Q_buildings[: n_steps // 2] = Q_load  # extraction: first half of the year on

    T_supply = np.full(n_steps, T_supply_value, dtype=np.float64)

    mw_tot = np.full((n_groups, n_steps), 0.1657, dtype=np.float64)
    mw_tot[:, n_steps // 2 :] = 0.0

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------

    myfield = FieldInput(n_bhes=n_bhes, xmin=x_min, ymin=y_min, xmax=x_max, ymax=y_max, rb = D0 / 2.0, layout = "irregular")
    grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 3), np.linspace(0, 10, 3))
    myfield.from_array(grid_x.ravel(), grid_y.ravel())

    fluid = Fluid(k_w=k_w, rho_w=rho_w, cp_w=cp_w, ni_w=ni_w)

    bore_geom = BoreholeGeometry(Lbore=Lbore, D0=D0)
    bore_mesh = BoreholeMesh(m_mesh=m_mesh)
    bore_th_props = BoreholeThermalProperties(cp_0=cp_0, rho_0=rho_0, k0=k0)
    props_b = SingleUtube(
        geom=bore_geom,
        mesh=bore_mesh,
        thermalprops=bore_th_props,
        fluid=fluid,
        Rp0=Rp0,
        RppB=RppB,
        pipe_spacing=pipe_spacing,
        pipe_thick=pipe_thick,
        Dpi=Dpi,
        n_pipes=n_pipes,
    )

    ground_geom = GroundGeometry(D0=D0, L=L, L_sup=L_sup, L_inf=L_inf, rn=None)
    ground_mesh = GroundMesh(
        n_mesh=n_mesh,
        m_mesh=m_mesh,
        m_mesh_sup=m_mesh_sup,
        m_mesh_inf=m_mesh_inf,
    )

    T_ext, SolarRad = _synthetic_weather(n_steps, dt)
    env_input = EnvironmentalTimeSeries.from_array(Tm=Tm, T_ext=T_ext, SolarRad=SolarRad)
    env_props = EnvironmentalProperties(
        R_ext=R_ext,
        absorptance=absorptance,
        eps=eps,
        At=At,
        tau=tau,
        tau_y=tau_y,
        tau_shift=tau_shift,
    )

    model = PhysicalModel(
        ground_geom=ground_geom,
        ground_mesh=ground_mesh,
        borehole=props_b,
        fluid=fluid,
        Tg=Tg,
        stratification=stratification,
        fieldinput=myfield,
    )

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    simulation = Simulation(
        model=model, envinput=env_input, timesteps=dt, n_steps=n_steps,
        envprops=env_props, mw_tot=mw_tot, Tf1=None,
        heat_flux=True, Q_buildings=Q_buildings, T_supply=T_supply,
        groups=groups,
    )
    T_history = simulation.run(series=True)

    # -------------------------------------------------------------------------
    # Post-processing indices
    # -------------------------------------------------------------------------

    # T_history shape: (n_steps + 1, n_bhes, n_dof)
    nsup = m_mesh_sup + 1
    nground = n_mesh * m_mesh
    reference_borehole = 0

    time = np.arange(dt, dt * (n_steps + 1), dt, dtype=np.float64)

    Tfout = T_history[1:, reference_borehole, nsup + nground + (props_b.n_equations - 1)]
    q_extracted = simulation.q_nbhes[:, reference_borehole]

    # -------------------------------------------------------------------------
    # Plot: field layout
    # -------------------------------------------------------------------------

    model.field.plot_field(show_ids=True, show_graph=True)

    # -------------------------------------------------------------------------
    # Plot: extracted heat and outlet fluid temperature
    # -------------------------------------------------------------------------

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)

    ax1.plot(time / 3600, q_extracted, color="tab:red")
    ax1.set_ylabel("$q$ [W]")
    ax1.set_title(f"Extracted heat - Borehole {reference_borehole}")
    ax1.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.4)

    ax2.plot(time / 3600, Tfout, color="tab:blue")
    ax2.set_xlabel("Time [h]")
    ax2.set_ylabel(r"$T_{f,out}$ [°C]")
    ax2.set_title(f"Outlet fluid temperature - Borehole {reference_borehole}")
    ax2.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.4)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()