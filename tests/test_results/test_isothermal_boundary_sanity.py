# -*- coding: utf-8 -*-
"""
Results test: isothermal boundary sanity check.

Every boundary the solver couples to is pinned at the same 13 C as the
undisturbed ground temperature, and nothing is left to drive a gradient:

  - above-surface convection uses ``T_ext = 13`` with ``absorptance = 0``
    and ``eps = 0``, so neither solar nor sky-radiation exchange forces
    the surface node (see ``build_rhs_ground`` in
    ``carm/matrix/assembly.py``, where both terms are scaled by
    ``absorptance``/``eps``);
  - the below-ground and outer-radial far-field boundaries come from the
    Kusuda-Achenbach profile (``carm/initial_conditions/kusuda.py``),
    which collapses to the constant ``Tm`` when the seasonal amplitude
    ``At = 0``; ``Tm = 13`` matches ``Tg``;
  - the fluid inlet ``Tf1`` is held at 13 too, so the BHE loop carries no
    forcing either.

With every coupling at the same temperature there is zero net heat flux
anywhere, so the field starts at trivial steady state and must stay there:
no node anywhere in the model (surface, upper/middle/lower ground, or the
borehole itself) may drift off 13 C for the whole simulated span. Any
drift means something is generating or absorbing energy that shouldn't be
there.

``Simulation.run()`` can write a ``results/*.npz`` file when
``save_results=True`` (``Simulation._save_results``); ``monkeypatch.
chdir(tmp_path)`` keeps that possible side effect out of the repository
working directory.
"""
import numpy as np
import pytest

from carm import (
    GroundGeometry,
    GroundMesh,
    BoreholeGeometry,
    BoreholeMesh,
    BoreholeThermalProperties,
    SingleUtube,
    Fluid,
    PhysicalModel,
    EnvironmentalProperties,
    EnvironmentalTimeSeries,
    Simulation,
)

N_STEPS = 48  # 2 days, hourly steps
T_ISO = 13.0


@pytest.fixture
def fluid():
    return Fluid(
        k_w=0.568709114496803,
        rho_w=1000.1435933169,
        cp_w=4207.40834247225,
        ni_w=1.49626063208248e-6,
    )


@pytest.fixture
def model(fluid):
    ground_mesh = GroundMesh(n_mesh=4, m_mesh=6, m_mesh_sup=2, m_mesh_inf=2)
    stratification = [(1.8, 947.37, 1900.0, 26.0)]
    borehole = SingleUtube(
        geom=BoreholeGeometry(Lbore=20.0, D0=0.15),
        mesh=BoreholeMesh(m_mesh=6),
        thermalprops=BoreholeThermalProperties(cp_0=1460.0, rho_0=1655.0, k0=1.8),
        fluid=fluid,
        pipe_thick=0.003,
        pipe_spacing=0.0823,
        Dpi=0.026,
        n_pipes=2,
        Rp0=0.25,
        RppB=0.72,
    )
    return PhysicalModel(
        ground_geom=GroundGeometry(rn=5.0, D0=0.15, L=20.0, L_sup=1.0, L_inf=5.0),
        ground_mesh=ground_mesh,
        borehole=borehole,
        fluid=fluid,
        Tg=T_ISO,
        stratification=stratification,
    )


@pytest.fixture
def env_props():
    # absorptance = eps = 0: no solar/sky radiation forcing at the surface;
    # At = 0: the Kusuda-Achenbach profile collapses to the constant Tm,
    # which fixes both the below-ground and outer-radial far-field
    # boundaries at 13 C.
    return EnvironmentalProperties(
        R_ext=0.04, absorptance=0.0, eps=0.0, At=0.0,
        tau=3600.0, tau_y=31_536_000.0, tau_shift=1_296_000.0,
    )


@pytest.fixture
def env_series():
    T_ext = np.full(N_STEPS, T_ISO, dtype=np.float64)
    solar_rad = np.zeros(N_STEPS, dtype=np.float64)
    return EnvironmentalTimeSeries.from_array(T_ISO, T_ext, solar_rad)


def test_isothermal_boundaries_hold_every_node_at_ground_temperature(
    model, env_props, env_series, fluid, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), T_ISO, dtype=np.float64)

    sim = Simulation(
        model=model, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
    )
    T_history = sim.run()

    assert np.allclose(T_history, T_ISO, atol=1e-6)
