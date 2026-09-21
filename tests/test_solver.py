# -*- coding: utf-8 -*-
"""
Tests for the simulation orchestrator module.

Covers: Simulation.__post_init__ — input validation (heat_flux/Tf1
combinations, mw_tot/Tf1 shape checks, environmental time series length,
water_input/soil-moisture prerequisites, fls_mode validation), and the
derived state it builds at construction time (adiabatic flag, FLS
instantiation, Kusuda-Achenbach profile shapes, boundary condition shape).

These are unit tests on construction, plus ``Simulation.run()``'s own
dispatch validation (parallel/series flag combinations, series-group
checks) — those guard clauses raise before any time-stepping happens, so
they stay as cheap as a construction-only test. The full time-stepping
loop itself is deliberately out of scope here and belongs to the
"results" test suite (energy-balance / regression checks). Mesh and step
counts are kept small on purpose to keep construction fast.
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
from carm.properties import SoilMoisture

N_STEPS = 5


# ============================================================
# FIXTURES — building blocks
# ============================================================

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
    return [(1.8, 947.37, 1900.0, 26.0)]  # L + L_sup + L_inf = 20 + 1 + 5


@pytest.fixture
def single_utube(fluid):
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
def single_utube_irrigation(fluid):
    """Same BHE, but with irrigation and soil_type enabled for the water_input tests."""
    return SingleUtube(
        geom=BoreholeGeometry(
            Lbore=20.0, D0=0.15, D_irrigation=0.02, perf_fraction=0.3
        ),
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


def _model_single(ground_mesh, stratification, borehole, fluid, rn=5.0):
    return PhysicalModel(
        ground_geom=GroundGeometry(rn=rn, D0=0.15, L=20.0, L_sup=1.0, L_inf=5.0),
        ground_mesh=ground_mesh,
        borehole=borehole,
        fluid=fluid,
        Tg=13.0,
        stratification=stratification,
    )


def _model_multi(ground_mesh, stratification, borehole, fluid, layout):
    fi = FieldInput(
        n_bhes=2, xmin=-5.0, ymin=-5.0, xmax=15.0, ymax=5.0, rb=0.075, layout=layout
    )
    fi.from_array(np.array([0.0, 5.0]), np.array([0.0, 0.0]))
    return PhysicalModel(
        ground_geom=GroundGeometry(rn=None, D0=0.15, L=20.0, L_sup=1.0, L_inf=5.0),
        ground_mesh=ground_mesh,
        borehole=borehole,
        fluid=fluid,
        Tg=13.0,
        stratification=stratification,
        fieldinput=fi,
    )


@pytest.fixture
def model_single(ground_mesh, stratification, single_utube, fluid):
    return _model_single(ground_mesh, stratification, single_utube, fluid)


@pytest.fixture
def model_multi_regular(ground_mesh, stratification, single_utube, fluid):
    return _model_multi(ground_mesh, stratification, single_utube, fluid, "regular")


@pytest.fixture
def model_multi_irregular(ground_mesh, stratification, single_utube, fluid):
    return _model_multi(ground_mesh, stratification, single_utube, fluid, "irregular")


# ============================================================
# Simulation — heat_flux / Tf1 validation (single borehole)
# ============================================================

def test_heat_flux_true_requires_q_and_supply(model_single, env_props, env_series):
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    with pytest.raises(ValueError):
        Simulation(
            model=model_single, envprops=env_props, envinput=env_series,
            timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot,
            heat_flux=True,
        )


def test_heat_flux_true_rejects_tf1(model_single, env_props, env_series):
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Q_buildings = np.full(N_STEPS, 1000.0, dtype=np.float64)
    T_supply = np.full(N_STEPS, 45.0, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), 2.0, dtype=np.float64)
    with pytest.raises(ValueError):
        Simulation(
            model=model_single, envprops=env_props, envinput=env_series,
            timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot,
            heat_flux=True, Q_buildings=Q_buildings, T_supply=T_supply, Tf1=Tf1,
        )


def test_heat_flux_true_valid_construction(model_single, env_props, env_series):
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Q_buildings = np.full(N_STEPS, 1000.0, dtype=np.float64)
    T_supply = np.full(N_STEPS, 45.0, dtype=np.float64)
    sim = Simulation(
        model=model_single, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot,
        heat_flux=True, Q_buildings=Q_buildings, T_supply=T_supply,
    )
    assert sim.heat_flux is True
    assert sim.Tf1 is None


def test_heat_flux_false_rejects_q_and_supply(model_single, env_props, env_series):
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), 2.0, dtype=np.float64)
    Q_buildings = np.full(N_STEPS, 1000.0, dtype=np.float64)
    with pytest.raises(ValueError):
        Simulation(
            model=model_single, envprops=env_props, envinput=env_series,
            timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot,
            heat_flux=False, Tf1=Tf1, Q_buildings=Q_buildings,
        )


def test_heat_flux_false_requires_tf1(model_single, env_props, env_series):
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    with pytest.raises(ValueError):
        Simulation(
            model=model_single, envprops=env_props, envinput=env_series,
            timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot,
            heat_flux=False, Tf1=None,
        )


# ============================================================
# Simulation — mw_tot / Tf1 shape validation
# ============================================================

def test_single_borehole_mw_tot_wrong_n_steps(model_single, env_props, env_series):
    mw_tot = np.full((1, N_STEPS + 1), 0.2, dtype=np.float64)  # wrong length
    Tf1 = np.full((1, N_STEPS), 2.0, dtype=np.float64)
    with pytest.raises(ValueError):
        Simulation(
            model=model_single, envprops=env_props, envinput=env_series,
            timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
        )


def test_single_borehole_tf1_wrong_n_steps(model_single, env_props, env_series):
    """In single-borehole mode only the number of columns (n_steps) is validated."""
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS + 1), 2.0, dtype=np.float64)  # wrong length
    with pytest.raises(ValueError):
        Simulation(
            model=model_single, envprops=env_props, envinput=env_series,
            timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
        )


def test_multi_borehole_mw_tot_must_match_n_bhes(model_multi_regular, env_props, env_series):
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)  # should be (2, N_STEPS)
    Tf1 = np.full((2, N_STEPS), 2.0, dtype=np.float64)
    with pytest.raises(ValueError):
        Simulation(
            model=model_multi_regular, envprops=env_props, envinput=env_series,
            timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
        )


def test_multi_borehole_valid_shapes(model_multi_regular, env_props, env_series):
    mw_tot = np.full((2, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((2, N_STEPS), 2.0, dtype=np.float64)
    sim = Simulation(
        model=model_multi_regular, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
    )
    assert sim.mw_tot.shape == (2, N_STEPS)


def test_multi_borehole_groups_mw_tot_must_match_n_groups(
    model_multi_irregular, env_props, env_series
):
    groups = {0: [0, 1]}  # 1 group, not 2 independent boreholes
    mw_tot = np.full((2, N_STEPS), 0.2, dtype=np.float64)  # should be (1, N_STEPS)
    Tf1 = np.full((1, N_STEPS), 2.0, dtype=np.float64)
    with pytest.raises(ValueError):
        Simulation(
            model=model_multi_irregular, envprops=env_props, envinput=env_series,
            timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1, groups=groups,
        )


def test_multi_borehole_groups_valid_shapes(model_multi_irregular, env_props, env_series):
    groups = {0: [0, 1]}
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), 2.0, dtype=np.float64)
    sim = Simulation(
        model=model_multi_irregular, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1, groups=groups,
    )
    assert sim.groups == groups


# ============================================================
# Simulation — environmental series validation
# ============================================================

def test_envinput_shorter_than_n_steps_raises(model_single, env_props):
    T_ext = np.full(N_STEPS - 1, 10.0, dtype=np.float64)
    solar_rad = np.full(N_STEPS - 1, 200.0, dtype=np.float64)
    env_series_short = EnvironmentalTimeSeries.from_array(13.0, T_ext, solar_rad)

    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), 2.0, dtype=np.float64)
    with pytest.raises(ValueError):
        Simulation(
            model=model_single, envprops=env_props, envinput=env_series_short,
            timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
        )


# ============================================================
# Simulation — water_input / soil moisture
# ============================================================

def test_water_input_requires_irrigation_geometry(
    ground_mesh, stratification, single_utube, fluid, env_props
):
    """water_input set but borehole without D_irrigation/perf_fraction must fail."""
    model = _model_single(ground_mesh, stratification, single_utube, fluid)
    T_ext = np.full(N_STEPS, 10.0, dtype=np.float64)
    solar_rad = np.full(N_STEPS, 200.0, dtype=np.float64)
    water_input = np.full(N_STEPS, 1e-7, dtype=np.float64)
    env_series_wet = EnvironmentalTimeSeries.from_array(13.0, T_ext, solar_rad, water_input)

    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), 2.0, dtype=np.float64)
    with pytest.raises(ValueError):
        Simulation(
            model=model, envprops=env_props, envinput=env_series_wet,
            timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
        )


def test_water_input_builds_soil_moisture_and_histories(
    ground_mesh, stratification, single_utube_irrigation, fluid, env_props
):
    model = _model_single(ground_mesh, stratification, single_utube_irrigation, fluid)
    T_ext = np.full(N_STEPS, 10.0, dtype=np.float64)
    solar_rad = np.full(N_STEPS, 200.0, dtype=np.float64)
    water_input = np.full(N_STEPS, 1e-7, dtype=np.float64)
    env_series_wet = EnvironmentalTimeSeries.from_array(13.0, T_ext, solar_rad, water_input)

    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), 2.0, dtype=np.float64)
    sim = Simulation(
        model=model, envprops=env_props, envinput=env_series_wet,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
    )
    assert isinstance(sim.bh_p_varprops, SoilMoisture)
    assert sim.k_borehole_history.shape == (N_STEPS,)
    assert sim.cp_borehole_history.shape == (N_STEPS,)
    assert sim.rho_borehole_history.shape == (N_STEPS,)
    np.testing.assert_array_equal(sim.k_borehole_history, np.zeros(N_STEPS))


def test_water_input_shorter_than_n_steps_raises(
    ground_mesh, stratification, single_utube_irrigation, fluid, env_props
):
    model = _model_single(ground_mesh, stratification, single_utube_irrigation, fluid)
    T_ext = np.full(N_STEPS, 10.0, dtype=np.float64)
    solar_rad = np.full(N_STEPS, 200.0, dtype=np.float64)
    water_input_short = np.full(N_STEPS - 1, 1e-7, dtype=np.float64)
    env_series_wet = EnvironmentalTimeSeries.from_array(
        13.0, T_ext, solar_rad, water_input_short
    )

    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), 2.0, dtype=np.float64)
    with pytest.raises(ValueError):
        Simulation(
            model=model, envprops=env_props, envinput=env_series_wet,
            timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
        )


# ============================================================
# Simulation — adiabatic flag
# ============================================================

def test_adiabatic_false_for_single_borehole(model_single, env_props, env_series):
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), 2.0, dtype=np.float64)
    sim = Simulation(
        model=model_single, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
    )
    assert sim.adiabatic is False


def test_adiabatic_true_for_regular_layout(model_multi_regular, env_props, env_series):
    mw_tot = np.full((2, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((2, N_STEPS), 2.0, dtype=np.float64)
    sim = Simulation(
        model=model_multi_regular, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
    )
    assert sim.adiabatic is True


def test_adiabatic_false_for_irregular_layout(model_multi_irregular, env_props, env_series):
    mw_tot = np.full((2, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((2, N_STEPS), 2.0, dtype=np.float64)
    sim = Simulation(
        model=model_multi_irregular, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
    )
    assert sim.adiabatic is False


# ============================================================
# Simulation — fls_mode and FLS instantiation
# ============================================================

def test_invalid_fls_mode_raises(model_single, env_props, env_series):
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), 2.0, dtype=np.float64)
    with pytest.raises(ValueError):
        Simulation(
            model=model_single, envprops=env_props, envinput=env_series,
            timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
            fls_mode="invalid_mode",
        )


def test_fls_not_built_for_single_borehole(model_single, env_props, env_series):
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), 2.0, dtype=np.float64)
    sim = Simulation(
        model=model_single, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
    )
    assert sim.fls is None


def test_fls_not_built_for_regular_layout(model_multi_regular, env_props, env_series):
    mw_tot = np.full((2, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((2, N_STEPS), 2.0, dtype=np.float64)
    sim = Simulation(
        model=model_multi_regular, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
    )
    assert sim.fls is None


def test_fls_built_for_irregular_multi_borehole(model_multi_irregular, env_props, env_series):
    mw_tot = np.full((2, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((2, N_STEPS), 2.0, dtype=np.float64)
    sim = Simulation(
        model=model_multi_irregular, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
    )
    assert sim.fls is not None
    assert sim.fls.response_matrix.shape == (N_STEPS + 1, 2, 2)


# ============================================================
# Simulation — Kusuda-Achenbach profiles and boundary condition
# ============================================================

def test_kusuda_profile_shapes(model_single, env_props, env_series, ground_mesh):
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), 2.0, dtype=np.float64)
    sim = Simulation(
        model=model_single, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
    )
    assert sim.T_sup_kusuda.shape == (N_STEPS, ground_mesh.m_mesh_sup + 1)
    assert sim.T_inf_kusuda.shape == (N_STEPS, ground_mesh.m_mesh_inf)
    n_equations = model_single.borehole.n_equations
    expected_middle_cols = ground_mesh.m_mesh * (ground_mesh.n_mesh + n_equations)
    assert sim.T_middle_kusuda.shape == (N_STEPS, expected_middle_cols)


def test_boundary_condition_shape(model_single, env_props, env_series, ground_mesh):
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), 2.0, dtype=np.float64)
    sim = Simulation(
        model=model_single, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
    )
    assert sim.T_bc.shape == (N_STEPS, 1, ground_mesh.m_mesh)


# ============================================================
# Simulation — heat_flux mode mw_tot/Q_buildings/T_supply shape validation
#
# Mirrors the "mw_tot / Tf1 shape validation" section above, but for the
# heat_flux=True branch of __post_init__ (a separate set of shape checks
# that never shared coverage with the Tf1-based ones).
# ============================================================

def test_heat_flux_true_single_borehole_wrong_n_steps(model_single, env_props, env_series):
    Q_buildings = np.full(N_STEPS + 1, 1000.0, dtype=np.float64)  # wrong length
    T_supply = np.full(N_STEPS, 45.0, dtype=np.float64)
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    with pytest.raises(ValueError):
        Simulation(
            model=model_single, envprops=env_props, envinput=env_series,
            timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=None,
            heat_flux=True, Q_buildings=Q_buildings, T_supply=T_supply,
        )


def test_heat_flux_true_multi_borehole_groups_shape_mismatch(
    model_multi_irregular, env_props, env_series
):
    groups = {0: [0, 1]}  # 1 group, so mw_tot must have 1 row, not 2
    Q_buildings = np.full(N_STEPS, 1000.0, dtype=np.float64)
    T_supply = np.full(N_STEPS, 45.0, dtype=np.float64)
    mw_tot = np.full((2, N_STEPS), 0.2, dtype=np.float64)
    with pytest.raises(ValueError):
        Simulation(
            model=model_multi_irregular, envprops=env_props, envinput=env_series,
            timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=None, groups=groups,
            heat_flux=True, Q_buildings=Q_buildings, T_supply=T_supply,
        )


def test_heat_flux_true_multi_borehole_shape_mismatch(model_multi_regular, env_props, env_series):
    Q_buildings = np.full(N_STEPS, 1000.0, dtype=np.float64)
    T_supply = np.full(N_STEPS, 45.0, dtype=np.float64)
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)  # should be (2, N_STEPS)
    with pytest.raises(ValueError):
        Simulation(
            model=model_multi_regular, envprops=env_props, envinput=env_series,
            timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=None,
            heat_flux=True, Q_buildings=Q_buildings, T_supply=T_supply,
        )


# ============================================================
# Simulation.run() — parallel/series dispatch validation
#
# These raise before any time-stepping happens (the checks sit at the top
# of run(), ahead of the dispatch to _run_parallel/_run_series), so they
# stay cheap despite exercising run() rather than just __post_init__.
# ============================================================

def test_run_series_without_groups_raises(model_multi_irregular, env_props, env_series):
    mw_tot = np.full((2, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((2, N_STEPS), 2.0, dtype=np.float64)
    sim = Simulation(
        model=model_multi_irregular, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
    )
    with pytest.raises(ValueError):
        sim.run(series=True)


def test_run_series_disconnected_group_raises(
    ground_mesh, stratification, single_utube, fluid, env_props, env_series
):
    """Borehole 1 sits geometrically between 0 and 2, so their Voronoi cells
    never touch; grouping them in series must be rejected."""
    fi = FieldInput(
        n_bhes=3, xmin=-5.0, ymin=-5.0, xmax=15.0, ymax=5.0, rb=0.075, layout="irregular"
    )
    fi.from_array(np.array([0.0, 5.0, 10.0]), np.array([0.0, 0.0, 0.0]))
    model = PhysicalModel(
        ground_geom=GroundGeometry(rn=None, D0=0.15, L=20.0, L_sup=1.0, L_inf=5.0),
        ground_mesh=ground_mesh, borehole=single_utube, fluid=fluid, Tg=13.0,
        stratification=stratification, fieldinput=fi,
    )
    groups = {0: [0, 2]}
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), 2.0, dtype=np.float64)
    sim = Simulation(
        model=model, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1, groups=groups,
    )
    with pytest.raises(ValueError):
        sim.run(series=True)


def test_run_multi_borehole_no_flags_raises(model_multi_irregular, env_props, env_series):
    mw_tot = np.full((2, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((2, N_STEPS), 2.0, dtype=np.float64)
    sim = Simulation(
        model=model_multi_irregular, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
    )
    with pytest.raises(ValueError):
        sim.run()


def test_run_single_borehole_both_flags_raises(model_single, env_props, env_series):
    mw_tot = np.full((1, N_STEPS), 0.2, dtype=np.float64)
    Tf1 = np.full((1, N_STEPS), 2.0, dtype=np.float64)
    sim = Simulation(
        model=model_single, envprops=env_props, envinput=env_series,
        timesteps=3600.0, n_steps=N_STEPS, mw_tot=mw_tot, Tf1=Tf1,
    )
    with pytest.raises(ValueError):
        sim.run(parallel=True, series=True)
