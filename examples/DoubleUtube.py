# -*- coding: utf-8 -*-
"""
Example: Single borehole with double U-tube configuration.
"""

import numpy as np

from carm import (
    BoreholeGeometry,
    BoreholeMesh,
    BoreholeThermalProperties,
    DoubleUtube,
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

    Dpi = 0.026
    Lbore = 12
    D0 = 0.15
    Rp0 = 0.25
    RppB = 0.72
    RppA = 0.55
    n_pipes = 4
    pipe_thick = 0.003
    pipe_spacing = 0.0823
    cp_0 = 1460
    rho_0 = 1655
    k0 = 1.8
    connection = "P"  # "P" or "p" for parallel, "S" or "s" for series

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
    props_b = DoubleUtube(
        geom=bore_geom,
        mesh=bore_mesh,
        thermalprops=bore_th_props,
        fluid=fluid,
        Rp0=Rp0,
        RppB=RppB,
        RppA=RppA,
        pipe_spacing=pipe_spacing,
        pipe_thick=pipe_thick,
        Dpi=Dpi,
        n_pipes=n_pipes,
        connection=connection,
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


if __name__ == "__main__":
    main()