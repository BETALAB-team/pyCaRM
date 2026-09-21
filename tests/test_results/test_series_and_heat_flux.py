# -*- coding: utf-8 -*-
"""
Results test: series connection (_run_series) and heat_flux_mode (COP/EER)
mode, executed end-to-end.

Both code paths existed only as construction/validation checks before this
file (tests/test_solver.py never calls Simulation.run()) and are exercised
by the real example scripts only at full scale (examples/SingleUtube_
multi_series.py, examples/SingleUtube_multi_series_heat_flux.py both use
9 boreholes over 8760 steps — too slow to run on every test pass, see
tests/test_examples_smoke.py). This file runs the same code paths at a
small, fast scale and checks physical direction/sanity, in the same style
as test_energy_flow_direction.py.

``Simulation.run()`` unconditionally writes a ``results/*.npz`` file
(``Simulation._save_results``); ``monkeypatch.chdir(tmp_path)`` keeps that
side effect out of the repository working directory.
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
    HeatFluxMode,
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


# ============================================================
# Series mode (_run_series)
# ============================================================

@pytest.fixture
def two_bhe_series_model(ground_mesh, stratification, borehole, fluid):
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


def test_series_extracts_heat_at_both_boreholes(
    two_bhe_series_model, env_props, env_series, tmp_path, monkeypatch
):
    """A cold Tf1 (extraction) feeding a 2-borehole series group must extract
    heat at both boreholes (q_nbhes < 0 throughout, sign convention per
    soil_moisture.py: 'Negative in heat extraction')."""
    monkeypatch.chdir(tmp_path)

    groups = {0: [0, 1]}
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), TG - 5.0, dtype=np.float64)

    sim = Simulation(
        model=two_bhe_series_model, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1, groups=groups,
    )
    sim.run(series=True)

    assert np.all(sim.q_nbhes[1:, 0] < 0.0)
    assert np.all(sim.q_nbhes[1:, 1] < 0.0)


def test_series_downstream_borehole_extracts_less(
    two_bhe_series_model, env_props, env_series, tmp_path, monkeypatch
):
    """In a series chain, the downstream borehole (index 1) receives fluid
    already warmed by the upstream one (index 0); with a smaller temperature
    gap to the ground it must extract less heat at every step."""
    monkeypatch.chdir(tmp_path)

    groups = {0: [0, 1]}
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), TG - 5.0, dtype=np.float64)

    sim = Simulation(
        model=two_bhe_series_model, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1, groups=groups,
    )
    sim.run(series=True)

    assert np.all(np.abs(sim.q_nbhes[1:, 1]) < np.abs(sim.q_nbhes[1:, 0]))


def test_series_heat_flux_rhs_old_state_frozen_across_picard_iterations(
    two_bhe_series_model, env_props, env_series, tmp_path, monkeypatch
):
    """Regression test for an operation-order bug in ``_run_series``: the
    'old' temperatures fed into ``build_global_rhs`` (the implicit
    capacitance term, ``-(C/dt) * T_old``) must stay frozen at the previous
    *timestep*'s converged value for every Picard iteration within a step,
    exactly as ``_run_parallel`` does by passing ``use_old=True`` to
    ``model._get_temperatures``. ``_run_series`` called it without
    ``use_old=True``, so from the second Picard iteration onward (only
    reachable when heat_flux_mode forces the while-loop to iterate) it fed
    back the *previous iteration's own solve* instead — collapsing the
    transient term at convergence. Same class of bug as the Tf1_groups fix
    (a value read before/instead of being pinned to its correct point in
    the step)."""
    monkeypatch.chdir(tmp_path)

    import carm.simulation.solver as solver_mod
    from carm.state import State

    groups = {0: [0, 1]}
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Q_buildings = np.full(N_STEPS, 3000.0, dtype=np.float64)
    T_supply = np.full(N_STEPS, 45.0, dtype=np.float64)
    heat_flux_mode = HeatFluxMode(Q_buildings=Q_buildings, T_supply=T_supply)

    sim = Simulation(
        model=two_bhe_series_model, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=None, groups=groups,
        heat_flux_mode=heat_flux_mode,
    )

    step_counter = {"step": -1}
    real_save_old = State.save_old

    def tracking_save_old(self):
        step_counter["step"] += 1
        real_save_old(self)

    monkeypatch.setattr(State, "save_old", tracking_save_old)

    seen: dict = {}
    max_iterations_for_one_borehole = {"count": 0}
    real_build_global_rhs = solver_mod.build_global_rhs

    def spy_build_global_rhs(*args, **kwargs):
        gr_p, T_old_ground, T_old_borehole = args[1], args[4], args[5]
        key = (step_counter["step"], id(gr_p))
        entry = (np.array(T_old_ground), np.array(T_old_borehole))
        if key in seen:
            prev = seen[key]
            np.testing.assert_array_equal(
                prev[0], entry[0],
                err_msg="T_ground_old drifted across Picard iterations within one step",
            )
            np.testing.assert_array_equal(
                prev[1], entry[1],
                err_msg="T_borehole_old drifted across Picard iterations within one step",
            )
            max_iterations_for_one_borehole["count"] += 1
        else:
            seen[key] = entry
        return real_build_global_rhs(*args, **kwargs)

    monkeypatch.setattr(solver_mod, "build_global_rhs", spy_build_global_rhs)

    sim.run(series=True)

    # The invariant above is only exercised when a step's Picard loop runs
    # more than once; assert that actually happened, so this test cannot
    # pass vacuously.
    assert max_iterations_for_one_borehole["count"] > 0


# ============================================================
# heat_flux_mode (COP/EER)
# ============================================================

@pytest.fixture
def single_bhe_model(ground_mesh, stratification, borehole, fluid):
    return PhysicalModel(
        ground_geom=GroundGeometry(rn=5.0, D0=0.15, L=20.0, L_sup=1.0, L_inf=5.0),
        ground_mesh=ground_mesh,
        borehole=borehole,
        fluid=fluid,
        Tg=TG,
        stratification=stratification,
    )


def test_heat_flux_extraction_gives_negative_q_ground(
    single_bhe_model, env_props, env_series, tmp_path, monkeypatch
):
    """A constant, positive Q_buildings (heating case) must draw heat from
    the ground (Q_ground < 0) with a plausible COP (> 1)."""
    monkeypatch.chdir(tmp_path)

    Q_buildings = np.full(N_STEPS, 1000.0, dtype=np.float64)
    T_supply = np.full(N_STEPS, 45.0, dtype=np.float64)
    heat_flux_mode = HeatFluxMode(Q_buildings=Q_buildings, T_supply=T_supply)
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)

    sim = Simulation(
        model=single_bhe_model, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=None,
        heat_flux_mode=heat_flux_mode,
    )
    sim.run()

    # step 0 is skipped by design (see solver.py Simulation._run_parallel)
    assert np.all(np.isfinite(sim.COP[1:]))
    assert np.all(sim.COP[1:] > 1.0)
    assert np.all(sim.Q_ground[1:] < 0.0)
    assert np.all(sim.q_nbhes[1:, 0] < 0.0)


def test_heat_flux_mode_independent_curves_produce_different_cop_and_eer(
    single_bhe_model, env_props, env_series, tmp_path, monkeypatch
):
    """cop_curve and eer_curve are configured independently on HeatFluxMode:
    distinct constant curves must produce COP/EER equal to those constants,
    not the shared default polynomial (and not each other's value)."""
    monkeypatch.chdir(tmp_path)

    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)

    heating_mode = HeatFluxMode(
        Q_buildings=np.full(N_STEPS, 1000.0, dtype=np.float64),
        T_supply=np.full(N_STEPS, 45.0, dtype=np.float64),
        cop_curve=lambda dT: 3.0,
        eer_curve=lambda dT: 4.0,
    )
    sim_heating = Simulation(
        model=single_bhe_model, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=None,
        heat_flux_mode=heating_mode,
    )
    sim_heating.run()
    assert np.allclose(sim_heating.COP[1:], 3.0)
    assert np.all(np.isnan(sim_heating.EER))

    cooling_mode = HeatFluxMode(
        Q_buildings=np.full(N_STEPS, -1000.0, dtype=np.float64),
        T_supply=np.full(N_STEPS, 7.0, dtype=np.float64),
        cop_curve=lambda dT: 3.0,
        eer_curve=lambda dT: 4.0,
    )
    sim_cooling = Simulation(
        model=single_bhe_model, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=None,
        heat_flux_mode=cooling_mode,
    )
    sim_cooling.run()
    assert np.allclose(sim_cooling.EER[1:], 4.0)
    assert np.all(np.isnan(sim_cooling.COP))


def test_heat_flux_off_gives_zero_load(
    single_bhe_model, env_props, env_series, tmp_path, monkeypatch
):
    """Q_buildings == 0 must skip the COP/Q_ground update for that step,
    leaving it at the NaN fill value."""
    monkeypatch.chdir(tmp_path)

    Q_buildings = np.zeros(N_STEPS, dtype=np.float64)
    T_supply = np.full(N_STEPS, 45.0, dtype=np.float64)
    heat_flux_mode = HeatFluxMode(Q_buildings=Q_buildings, T_supply=T_supply)
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)

    sim = Simulation(
        model=single_bhe_model, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=None,
        heat_flux_mode=heat_flux_mode,
    )
    sim.run()

    assert np.all(np.isnan(sim.COP))
    assert np.all(np.isnan(sim.Q_ground))


def test_heat_flux_rejection_gives_positive_q_ground(
    single_bhe_model, env_props, env_series, tmp_path, monkeypatch
):
    """A constant, negative Q_buildings (cooling case) must reject heat into
    the ground (Q_ground > 0, sign convention per solver.py) with a
    plausible EER (> 1). This is the EER branch's only coverage: every
    other heat_flux test in this file uses a positive (heating) load."""
    monkeypatch.chdir(tmp_path)

    Q_buildings = np.full(N_STEPS, -1000.0, dtype=np.float64)
    T_supply = np.full(N_STEPS, 7.0, dtype=np.float64)
    heat_flux_mode = HeatFluxMode(Q_buildings=Q_buildings, T_supply=T_supply)
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)

    sim = Simulation(
        model=single_bhe_model, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=None,
        heat_flux_mode=heat_flux_mode,
    )
    sim.run()

    assert np.all(np.isfinite(sim.EER[1:]))
    assert np.all(sim.EER[1:] > 1.0)
    assert np.all(sim.Q_ground[1:] > 0.0)
    assert np.all(sim.q_nbhes[1:, 0] > 0.0)


def test_series_heat_flux_rejection_gives_positive_q_ground(
    two_bhe_series_model, env_props, env_series, tmp_path, monkeypatch
):
    """Same EER (cooling) branch as
    test_heat_flux_rejection_gives_positive_q_ground, but for _run_series:
    the series Picard loop has its own, separate 'elif Q_buildings < 0'
    block. Also exercises save_results=True end-to-end for a series +
    heat_flux run (writes COP/EER/Q_ground/Q_buildings/abs_err into the
    saved .npz, a branch no other test in the suite reaches for series)."""
    monkeypatch.chdir(tmp_path)

    groups = {0: [0, 1]}
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Q_buildings = np.full(N_STEPS, -1000.0, dtype=np.float64)
    T_supply = np.full(N_STEPS, 7.0, dtype=np.float64)
    heat_flux_mode = HeatFluxMode(Q_buildings=Q_buildings, T_supply=T_supply)

    sim = Simulation(
        model=two_bhe_series_model, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=None, groups=groups,
        heat_flux_mode=heat_flux_mode,
    )
    sim.run(series=True, save_results=True)

    assert np.all(np.isfinite(sim.EER[1:]))
    assert np.all(sim.EER[1:] > 1.0)
    assert np.all(sim.Q_ground[1:] > 0.0)
    assert np.all(sim.q_nbhes[1:, 0] > 0.0)
    assert np.all(sim.q_nbhes[1:, 1] > 0.0)

    saved = list((tmp_path / "results").glob("*.npz"))
    assert len(saved) == 1
    with np.load(saved[0]) as data:
        assert "COP" in data
        assert "EER" in data
