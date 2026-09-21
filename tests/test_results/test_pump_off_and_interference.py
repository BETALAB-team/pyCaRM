# -*- coding: utf-8 -*-
"""
Results test: mass-flow-off ("pump off") steps, multi-borehole thermal
interference (FLS) through ``_run_parallel``, and soil-moisture property
updates through ``_run_series``.

Each of these existed only as a code path with no end-to-end test before
this file:

- ``mw_tot == 0`` at a mid-run step (the "idx_null" branches in both
  ``_run_parallel`` and ``_run_series``, and the matching ``mw_tot_j == 0``
  branch in the per-BHE-type matrix assembly functions) — a real
  operating condition (an intermittently-running pump), not an edge case.
- ``Simulation.fls`` is only ever exercised at construction (``sim.fls is
  not None``) or by testing ``FiniteLineSolution`` directly against
  pygfunction (test_fls_vs_pygfunction.py); its actual use inside
  ``_run_parallel``'s Picard loop (``self.fls._compute_delta_t(...)``,
  boundary-condition penalty from a neighboring borehole) was untested.
- Soil-moisture-driven property updates (``_props_calculation``) are
  exercised end-to-end in parallel mode by
  examples/Helical_variable_properties.py (see test_examples_smoke.py),
  but ``_run_series`` has its own, separate implementation of the same
  update and no series example uses water_input.
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
    FieldInput,
    PhysicalModel,
    EnvironmentalProperties,
    EnvironmentalTimeSeries,
    Simulation,
)

N_STEPS = 24
TG = 13.0


@pytest.fixture
def fluid():
    return Fluid(
        k_w=0.568709114496803,
        rho_w=1000.1435933169,
        cp_w=4207.40834247225,
        ni_w=1.49626063208248e-6,
    )


@pytest.fixture
def ground_mesh():
    return GroundMesh(n_mesh=4, m_mesh=6, m_mesh_sup=2, m_mesh_inf=2)


@pytest.fixture
def stratification():
    return [(1.8, 947.37, 1900.0, 26.0)]


@pytest.fixture
def borehole(fluid):
    return SingleUtube(
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


@pytest.fixture
def irrigation_borehole(fluid):
    return SingleUtube(
        geom=BoreholeGeometry(Lbore=20.0, D0=0.15, D_irrigation=0.02, perf_fraction=0.3),
        mesh=BoreholeMesh(m_mesh=6),
        thermalprops=BoreholeThermalProperties(
            cp_0=1460.0, rho_0=1655.0, k0=1.8, soil_type="sand"
        ),
        fluid=fluid,
        pipe_thick=0.003,
        pipe_spacing=0.0823,
        Dpi=0.026,
        n_pipes=2,
        Rp0=0.25,
        RppB=0.72,
    )


@pytest.fixture
def env_props():
    return EnvironmentalProperties(
        R_ext=0.04, absorptance=0.6, eps=0.9, At=10.0,
        tau=3600.0, tau_y=31_536_000.0, tau_shift=1_296_000.0,
    )


@pytest.fixture
def env_series():
    T_ext = np.full(N_STEPS, 10.0, dtype=np.float64)
    solar_rad = np.full(N_STEPS, 200.0, dtype=np.float64)
    return EnvironmentalTimeSeries.from_array(13.0, T_ext, solar_rad)


@pytest.fixture
def two_bhe_irregular_model(ground_mesh, stratification, borehole, fluid):
    fi = FieldInput(
        n_bhes=2, xmin=-5.0, ymin=-5.0, xmax=15.0, ymax=5.0, rb=0.075, layout="irregular"
    )
    fi.from_array(np.array([0.0, 5.0]), np.array([0.0, 0.0]))
    return PhysicalModel(
        ground_geom=GroundGeometry(rn=None, D0=0.15, L=20.0, L_sup=1.0, L_inf=5.0),
        ground_mesh=ground_mesh,
        borehole=borehole,
        fluid=fluid,
        Tg=TG,
        stratification=stratification,
        fieldinput=fi,
    )


@pytest.fixture
def two_bhe_irrigation_model(ground_mesh, stratification, irrigation_borehole, fluid):
    fi = FieldInput(
        n_bhes=2, xmin=-5.0, ymin=-5.0, xmax=15.0, ymax=5.0, rb=0.075, layout="irregular"
    )
    fi.from_array(np.array([0.0, 5.0]), np.array([0.0, 0.0]))
    return PhysicalModel(
        ground_geom=GroundGeometry(rn=None, D0=0.15, L=20.0, L_sup=1.0, L_inf=5.0),
        ground_mesh=ground_mesh,
        borehole=irrigation_borehole,
        fluid=fluid,
        Tg=TG,
        stratification=stratification,
        fieldinput=fi,
    )


# ============================================================
# Pump-off steps (mw_tot == 0 mid-run)
# ============================================================

def test_parallel_pump_off_step_has_zero_flux(
    two_bhe_irregular_model, env_props, env_series, tmp_path, monkeypatch
):
    """Borehole 0 is switched off (mw_tot == 0) for one step while borehole
    1 keeps running; the off borehole must show exactly zero fluid heat
    flux at that step (no flow -> no heat transfer), and the still-running
    borehole must be unaffected."""
    monkeypatch.chdir(tmp_path)

    off_step = N_STEPS // 2
    mw_tot = np.full((2, N_STEPS), 0.2, dtype=np.float64)
    mw_tot[0, off_step] = 0.0
    Tf1 = np.full((2, N_STEPS), TG - 5.0, dtype=np.float64)

    sim = Simulation(
        model=two_bhe_irregular_model, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
    )
    sim.run(parallel=True)

    assert sim.q_nbhes[off_step, 0] == 0.0
    assert sim.q_nbhes[off_step, 1] < 0.0


def test_series_pump_off_step_has_zero_flux(
    two_bhe_irregular_model, env_props, env_series, tmp_path, monkeypatch
):
    """The single series group is switched off (mw_tot == 0) for one step;
    both boreholes in the chain share the group's flow, so both must show
    exactly zero fluid heat flux at that step."""
    monkeypatch.chdir(tmp_path)

    off_step = N_STEPS // 2
    groups = {0: [0, 1]}
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    mw_tot[0, off_step] = 0.0
    Tf1 = np.full((1, N_STEPS), TG - 5.0, dtype=np.float64)

    sim = Simulation(
        model=two_bhe_irregular_model, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1, groups=groups,
    )
    sim.run(series=True)

    assert sim.q_nbhes[off_step, 0] == 0.0
    assert sim.q_nbhes[off_step, 1] == 0.0


# ============================================================
# Multi-borehole thermal interference (FLS) through _run_parallel
# ============================================================

def test_parallel_run_with_thermal_interference_extracts_heat(
    two_bhe_irregular_model, env_props, env_series, tmp_path, monkeypatch
):
    """Two independent (parallel), irregularly-laid-out boreholes build an
    FLS thermal-interference model at construction time
    (test_solver.py::test_fls_built_for_irregular_multi_borehole); this
    test is the only one in the suite that actually runs the Picard loop
    that consumes it (``self.fls._compute_delta_t`` perturbing the
    boundary condition each iteration), rather than just instantiating it
    or testing FiniteLineSolution in isolation."""
    monkeypatch.chdir(tmp_path)

    mw_tot = np.full((2, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((2, N_STEPS), TG - 5.0, dtype=np.float64)

    sim = Simulation(
        model=two_bhe_irregular_model, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
    )
    assert sim.fls is not None

    sim.run(parallel=True)

    assert np.all(np.isfinite(sim.T_history))
    assert np.all(sim.q_nbhes[1:, 0] < 0.0)
    assert np.all(sim.q_nbhes[1:, 1] < 0.0)


# ============================================================
# Soil moisture / water_input through _run_series
# ============================================================

def test_series_water_input_updates_borehole_properties(
    two_bhe_irrigation_model, env_props, tmp_path, monkeypatch
):
    """A wet pulse partway through the run must be picked up by
    ``_run_series``'s own soil-moisture update (mirroring the
    already-covered ``_run_parallel`` path exercised by
    examples/Helical_variable_properties.py), producing finite non-zero
    property histories, and ``save_results=True`` must write them to the
    .npz (the water_input branch of ``Simulation._save_results``)."""
    monkeypatch.chdir(tmp_path)

    T_ext = np.full(N_STEPS, 10.0, dtype=np.float64)
    solar_rad = np.full(N_STEPS, 200.0, dtype=np.float64)
    water_input = np.zeros(N_STEPS, dtype=np.float64)
    water_input[N_STEPS // 2 :] = 5e-5
    env_series_wet = EnvironmentalTimeSeries.from_array(13.0, T_ext, solar_rad, water_input)

    groups = {0: [0, 1]}
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), TG - 5.0, dtype=np.float64)

    sim = Simulation(
        model=two_bhe_irrigation_model, envprops=env_props, envinput=env_series_wet,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1, groups=groups,
    )
    sim.run(series=True, save_results=True)

    assert np.all(np.isfinite(sim.k_borehole_history))
    assert np.any(sim.k_borehole_history[N_STEPS // 2 :] != 0.0)

    saved = list((tmp_path / "results").glob("*.npz"))
    assert len(saved) == 1
    with np.load(saved[0]) as data:
        assert "water_content_borehole" in data
        assert "k_borehole" in data
