# -*- coding: utf-8 -*-
"""
Example: Single borehole with helical pipe configuration.
"""

import numpy as np

from carm import (
    BoreholeGeometry,
    BoreholeMesh,
    BoreholeThermalProperties,
    Helical,
)
from carm import EnvironmentalProperties, EnvironmentalTimeSeries
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

    stratification = [(1.8, 947.37, 1900, 23)]
    n_mesh = 20
    m_mesh = 40
    Tg = 13
    L = 12
    m_mesh_sup = 4
    m_mesh_inf = 40
    L_sup = 1
    L_inf = 10
    rn = 10

    k_w = 0.568709114496803
    rho_w = 1000.1435933169
    cp_w = 4207.40834247225
    ni_w = 1.49626063208248e-6

    Dpi1 = Dpi2 = 0.0204
    Lbore = 12
    D0 = 0.5
    pipe_thick = 0.0023
    cp_0 = 1460
    rho_0 = 1655
    k0 = 1.8
    k_pipe = 0.38
    rih = 0.045
    N = 8
    supply_and_return = "1_2"
    P = 1.5
    Lp2tot = N * np.sqrt(P**2 + np. pi * (rih + Dpi1 / 2.0 + pipe_thick)**2)

    absorptance = 0.7
    eps = 0.95
    At = 10
    tau = 0
    tau_y = 365 * 24 * 3600
    tau_shift = 210 * 24 * 3600
    R_ext = 0.04

    Tm = 13

    dt = 3600
    n_steps = 276

    Tf1 = np.full((1, n_steps), 2, dtype=np.float64)
    mw_tot = np.full((1, n_steps), 0.1657, dtype=np.float64)

    # -------------------------------------------------------------------------
    # Build model
    # -------------------------------------------------------------------------

    fluid = Fluid(k_w=k_w, rho_w=rho_w, cp_w=cp_w, ni_w=ni_w)

    bore_geom = BoreholeGeometry(Lbore=Lbore, D0=D0)
    bore_mesh = BoreholeMesh(m_mesh=m_mesh)
    bore_th_props = BoreholeThermalProperties(cp_0=cp_0, rho_0=rho_0, k0=k0)
    props_b = Helical(
        geom=bore_geom,
        mesh=bore_mesh,
        thermalprops=bore_th_props,
        fluid=fluid,
        Dpi1=Dpi1,
        Dpi2=Dpi2,
        P=P,
        Lp2tot=Lp2tot,
        supply_and_return=supply_and_return,
        rih=rih,
        pipe_thick=pipe_thick,
        N=N,
        k_pipe=k_pipe,
    )

    ground_geom = GroundGeometry(rn=rn, D0=D0, L=L, L_sup=L_sup, L_inf=L_inf)
    ground_mesh = GroundMesh(
        n_mesh=n_mesh, m_mesh=m_mesh, m_mesh_sup=m_mesh_sup, m_mesh_inf=m_mesh_inf
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
    )

    # -------------------------------------------------------------------------
    # Simulation
    # -------------------------------------------------------------------------

    simulation = Simulation(
        model=model, envinput=env_input, timesteps=dt, n_steps=n_steps,
        envprops=env_props, mw_tot=mw_tot, Tf1=Tf1,
    )
    T_history = simulation.run()

    # -------------------------------------------------------------------------
    # Post-processing indices
    # -------------------------------------------------------------------------

    import matplotlib.pyplot as plt

    nsup = m_mesh_sup + 1
    nground = n_mesh * m_mesh
    nb = n_mesh * props_b.n_equations
    ninf = m_mesh_inf
    reference_borehole = 0

    time = np.arange(dt, dt * (n_steps + 1), dt, dtype=np.float64,)

    Tfout = T_history[1:, reference_borehole, nsup + nground + (props_b.n_equations - 1)]
    Tf1 = Tf1.reshape(-1)

    slice_down = [nsup + nground +  (props_b.n_equations * j) + (props_b.n_equations - 2) for j in range(m_mesh)]
    slice_up = [nsup + nground + (props_b.n_equations * j) + (props_b.n_equations - 1) for j in range(m_mesh)]

    slice_shell = [nsup + nground + props_b.id_shell + (props_b.n_equations * j) for j in range(m_mesh)]

    radius = np.linspace(D0/2, rn, n_mesh)

    steps = [10, 150, 276]
    r0 = model.ground[0].r0
    rn = model.ground[0].rn
    dz = model.ground[0].dz
    
    depth = np.arange(-L_sup, -L_sup - dz * m_mesh, -dz)
    # -------------------------------------------------------------------------
    # Plot: outlet fluid temperature over time
    # -------------------------------------------------------------------------

    Tfout = T_history[1:, reference_borehole, nsup + nground + (props_b.n_equations - 1)]

    fig, ax = plt.subplots(figsize = (4,3))
    ax.plot(time / 3600, Tfout, color = "tab:red", label = r"$T_{f,out}$")
    ax.plot(time / 3600, Tf1, color = "tab:blue", label = r"$T_{f,in}$")

    ax.set_xlabel("Time [h]")
    ax.set_ylabel("Temperature [°C]")
    ax.legend(fontsize = 7)
    ax.set_title(f"Outlet fluid temperature Helical configuration")

    ax.grid(True, which = 'major', linestyle = '--', linewidth = 0.5, alpha = 0.4)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Plot: shell temperature vertical profile
    # -------------------------------------------------------------------------

    Tshell_10  = T_history[11,  reference_borehole, slice_shell]
    Tshell_150 = T_history[151, reference_borehole, slice_shell]
    Tshell_276 = T_history[276, reference_borehole, slice_shell]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(Tshell_10,  depth, label='Step = 10')
    ax.plot(Tshell_150, depth, label='Step = 150')
    ax.plot(Tshell_276, depth, label='Step = 276')
    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel("Depth [m]")
    ax.set_xlim(2, 15)
    ax.legend(fontsize=7)
    ax.set_title(f"Shell temperature for Borehole {reference_borehole}")
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Plot: fluid (down/up) temperature vertical profile
    # -------------------------------------------------------------------------

    T_down_10= T_history[10, reference_borehole, slice_down]
    T_down_150= T_history[150, reference_borehole, slice_down]
    T_down_276= T_history[276, reference_borehole, slice_down]
    T_up_10= T_history[10, reference_borehole, slice_up]
    T_up_150= T_history[150, reference_borehole, slice_up]
    T_up_276= T_history[276, reference_borehole, slice_up]

    fig, ax = plt.subplots(figsize = (4, 3))

    ax.plot(T_down_10, depth, color = "tab:blue", linestyle = "-", label = "Step = 10")
    ax.plot(T_up_10, depth, color = "tab:blue", linestyle = "-")
    ax.plot(T_down_150, depth, color = "tab:orange", linestyle = "-", label = "Step = 150")
    ax.plot(T_up_150, depth, color = "tab:orange", linestyle = "-")
    ax.plot(T_down_276, depth, color = "tab:green", linestyle = "-", label = "Step = 276")
    ax.plot(T_up_276, depth, color = "tab:green", linestyle = "-")
    ax.set_xlabel("Temperature [°C]")
    ax.set_ylabel("Depth [m]")
    ax.set_xlim(1, 4)
    ax.legend(fontsize = 7)
    ax.set_title(f"Fluid temperature for Borehole {reference_borehole}")

    ax.grid(True, which = 'major', linestyle = '--', linewidth = 0.5, alpha = 0.4)

    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Plot: ground temperature heatmap
    # -------------------------------------------------------------------------

    nsup = m_mesh_sup + 1
    steps = [10, 150, 276]
    r0 = model.ground[0].r0
    rn = model.ground[0].rn
    dz = model.ground[0].dz

    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    ax1, ax2, ax3 = axes

    T_all = np.array([T_history[steps, reference_borehole, nsup : nsup + nground ]]).reshape(len(steps), m_mesh, n_mesh)

    vmin, vmax = T_all.min(), T_all.max()

    x_ticks = np.round(np.linspace(r0, rn, 5), decimals = 2)

    depth = np.arange(-L_sup, -L_sup - dz * m_mesh, -dz)
    radius = np.linspace(D0/2, rn, n_mesh)
    R, D = np.meshgrid(radius, depth)

    for i, (t, ax) in enumerate(zip(steps, axes)):

        pc = ax.pcolormesh(R, D, T_all[i], cmap='RdYlGn_r', shading='gouraud', vmin=vmin, vmax=vmax)
        ax.set_xlabel("Radius [m]")
        ax.set_ylabel("Depth [m]")
        fig.colorbar(pc, ax=ax, label="Temperature [°C]")
        ax.set_title(f"Step {t}")
        ax.set_xlim((r0, rn))
        ax.set_xticks(x_ticks)

    fig.suptitle(f"Ground temperature - BHE {reference_borehole}")
    plt.tight_layout()
    plt.show()

    # -------------------------------------------------------------------------
    # Plot: Boundary condition temperature
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(12, 3))
    ax1, ax2, ax3 = axes

    T_all = np.array([T_history[steps, reference_borehole, nsup : nsup + nground ]]).reshape(len(steps), m_mesh, n_mesh)

    vmin, vmax = T_all.min(), T_all.max()

    x_ticks = np.round(np.linspace(r0, rn, 5), decimals = 2)

    R, D = np.meshgrid(radius, depth)

    for i, (t, ax) in enumerate(zip(steps, axes)):

        pc = ax.pcolormesh(R, D, T_all[i], cmap='RdYlGn_r', shading='gouraud', vmin=vmin, vmax=vmax)
        contours = ax.contour(R, D, T_all[i], levels=6, colors='black', linewidths=0.5, alpha=0.6)
        ax.clabel(contours, inline=True, fontsize=7, fmt='%.1f', inline_spacing=5)
        ax.set_xlabel("Radius [m]")
        ax.set_ylabel("Depth [m]")
        fig.colorbar(pc, ax=ax, label="Temperature [°C]")
        ax.set_title(f"Step {t}")
        ax.set_xlim((r0, rn))
        ax.set_xticks(x_ticks)

    fig.suptitle(f"Ground temperature - BHE {reference_borehole}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()