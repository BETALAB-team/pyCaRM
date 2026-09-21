# -*- coding: utf-8 -*-
"""
Tests for borehole properties module.

Covers: BoreholeGeometry, BoreholeMesh, BoreholeThermalProperties,
        SingleUtube, DoubleUtube, Coaxial, Helical.

Reference values are computed analytically from the example scripts
(SingleUtube.py, DoubleUtube.py, Coaxial.py, Helical.py).
"""
import numpy as np
import pytest

from carm import (
    BoreholeGeometry,
    BoreholeMesh,
    BoreholeThermalProperties,
    SingleUtube,
    DoubleUtube,
    Coaxial,
    Helical,
)
from carm import Fluid


# ============================================================
# FIXTURES — parametri reali dagli script di esempio
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
def thermal_props():
    return BoreholeThermalProperties(cp_0=1460.0, rho_0=1655.0, k0=1.8)


@pytest.fixture
def single_utube(fluid, thermal_props):
    return SingleUtube(
        geom=BoreholeGeometry(Lbore=100.0, D0=0.15),
        mesh=BoreholeMesh(m_mesh=40),
        thermalprops=thermal_props,
        fluid=fluid,
        pipe_thick=0.003,
        pipe_spacing=0.0823,
        Dpi=0.026,
        n_pipes=2,
        Rp0=0.25,
        RppB=0.72,
    )


@pytest.fixture
def double_utube(fluid, thermal_props):
    return DoubleUtube(
        geom=BoreholeGeometry(Lbore=12.0, D0=0.15),
        mesh=BoreholeMesh(m_mesh=40),
        thermalprops=thermal_props,
        fluid=fluid,
        pipe_thick=0.003,
        pipe_spacing=0.0823,
        Dpi=0.026,
        n_pipes=4,
        Rp0=0.25,
        RppB=0.72,
        RppA=0.55,
        connection="P",
    )


@pytest.fixture
def coaxial(fluid, thermal_props):
    return Coaxial(
        geom=BoreholeGeometry(Lbore=12.0, D0=0.15),
        mesh=BoreholeMesh(m_mesh=40),
        thermalprops=thermal_props,
        fluid=fluid,
        Dp1i=0.032,
        Dp2i=0.110,
        pipe1_thick=0.003,
        pipe2_thick=0.006,
        k_pipe1=0.38,
        k_pipe2=0.38,
        supply_and_return="1_2",
    )


@pytest.fixture
def helical(fluid, thermal_props):
    return Helical(
        geom=BoreholeGeometry(Lbore=5.1816, D0=0.590),
        mesh=BoreholeMesh(m_mesh=40),
        thermalprops=thermal_props,
        fluid=fluid,
        Dpi1=0.01694,
        Dpi2=0.01694,
        rih=0.269 - 0.01694/2 - 0.0021,
        pipe_thick=0.0021,
        N=34,
        P=0.1524,
        supply_and_return="1_2",
        Lp2tot=34 * np.pi * (0.01694 + 2*0.0021),
        k_pipe=0.47,
    )



# ============================================================
# BoreholeGeometry
# ============================================================

def test_geometry_stores_values():
    geom = BoreholeGeometry(Lbore=100.0, D0=0.15)
    assert geom.Lbore == 100.0
    assert geom.D0 == 0.15


def test_geometry_r0():
    geom = BoreholeGeometry(Lbore=100.0, D0=0.15)
    assert geom.r0 == pytest.approx(0.075)


def test_geometry_invalid_lbore():
    with pytest.raises(ValueError):
        BoreholeGeometry(Lbore=-1.0, D0=0.15)


def test_geometry_zero_lbore():
    with pytest.raises(ValueError):
        BoreholeGeometry(Lbore=0.0, D0=0.15)


def test_geometry_invalid_d0():
    with pytest.raises(ValueError):
        BoreholeGeometry(Lbore=100.0, D0=0.0)


def test_geometry_valid_irrigation():
    geom = BoreholeGeometry(Lbore=100.0, D0=0.15, D_irrigation=0.02, perf_fraction=0.3)
    assert geom.D_irrigation == 0.02
    assert geom.perf_fraction == 0.3


def test_geometry_irrigation_requires_both_fields():
    """D_irrigation and perf_fraction must be both None or both set."""
    with pytest.raises(ValueError):
        BoreholeGeometry(Lbore=100.0, D0=0.15, D_irrigation=0.02, perf_fraction=None)
    with pytest.raises(ValueError):
        BoreholeGeometry(Lbore=100.0, D0=0.15, D_irrigation=None, perf_fraction=0.3)


def test_geometry_invalid_irrigation_diameter():
    with pytest.raises(ValueError):
        BoreholeGeometry(Lbore=100.0, D0=0.15, D_irrigation=0.0, perf_fraction=0.3)
    with pytest.raises(ValueError):
        # must not exceed the borehole diameter
        BoreholeGeometry(Lbore=100.0, D0=0.15, D_irrigation=0.2, perf_fraction=0.3)


def test_geometry_invalid_perf_fraction():
    with pytest.raises(ValueError):
        BoreholeGeometry(Lbore=100.0, D0=0.15, D_irrigation=0.02, perf_fraction=-0.1)
    with pytest.raises(ValueError):
        BoreholeGeometry(Lbore=100.0, D0=0.15, D_irrigation=0.02, perf_fraction=1.0)


# ============================================================
# BoreholeMesh
# ============================================================

def test_mesh_stores_values():
    mesh = BoreholeMesh(m_mesh=20)
    assert mesh.m_mesh == 20


def test_mesh_invalid_zero():
    with pytest.raises(ValueError):
        BoreholeMesh(m_mesh=0)


def test_mesh_invalid_negative():
    with pytest.raises(ValueError):
        BoreholeMesh(m_mesh=-5)


# ============================================================
# BoreholeThermalProperties
# ============================================================

def test_thermal_props_stores_values():
    tp = BoreholeThermalProperties(cp_0=1460.0, rho_0=1655.0, k0=1.8)
    assert tp.cp_0 == 1460.0
    assert tp.rho_0 == 1655.0
    assert tp.k0 == 1.8


def test_thermal_props_invalid_cp():
    with pytest.raises(ValueError):
        BoreholeThermalProperties(cp_0=0.0, rho_0=1655.0, k0=1.8)


def test_thermal_props_invalid_rho():
    with pytest.raises(ValueError):
        BoreholeThermalProperties(cp_0=1460.0, rho_0=0.0, k0=1.8)


def test_thermal_props_invalid_k0():
    with pytest.raises(ValueError):
        BoreholeThermalProperties(cp_0=1460.0, rho_0=1655.0, k0=0.0)


def test_thermal_props_requires_stratification_or_equivalent():
    """Neither stratification nor a full set of equivalent properties given."""
    with pytest.raises(ValueError):
        BoreholeThermalProperties()
    with pytest.raises(ValueError):
        BoreholeThermalProperties(cp_0=1460.0, rho_0=1655.0)  # k0 missing


def test_thermal_props_rejects_both_stratification_and_equivalent():
    with pytest.raises(ValueError):
        BoreholeThermalProperties(
            cp_0=1460.0, rho_0=1655.0, k0=1.8,
            stratification=[(1.8, 1460.0, 1655.0, 100.0)],
        )


def test_thermal_props_invalid_stratification_layer():
    """Every (k, cp, rho, thickness) entry must be strictly positive."""
    with pytest.raises(ValueError):
        BoreholeThermalProperties(stratification=[(1.8, 1460.0, 1655.0, 0.0)])
    with pytest.raises(ValueError):
        BoreholeThermalProperties(stratification=[(-1.8, 1460.0, 1655.0, 100.0)])


def test_thermal_props_valid_stratification():
    tp = BoreholeThermalProperties(stratification=[(1.8, 1460.0, 1655.0, 100.0)])
    assert tp.cp_0 is None
    assert tp.stratification == [(1.8, 1460.0, 1655.0, 100.0)]


def test_thermal_props_normalizes_soil_type():
    tp = BoreholeThermalProperties(cp_0=1460.0, rho_0=1655.0, k0=1.8, soil_type="  Sand ")
    assert tp.soil_type == "sand"


def test_thermal_props_invalid_soil_type():
    with pytest.raises(ValueError):
        BoreholeThermalProperties(cp_0=1460.0, rho_0=1655.0, k0=1.8, soil_type="granite")


# ============================================================
# BoreholeProperties._variable_properties — grout stratification
# ============================================================

def test_stratified_properties_mixed_cell_is_length_weighted(fluid):
    """A layer boundary that falls inside a mesh cell must yield a
    thickness-weighted average of the two layers for that cell (not just
    whichever layer the cell midpoint falls in)."""
    # Lbore=100, m_mesh=40 -> dz=2.5; cell 20 spans z=[50.0, 52.5].
    # Layer split at 51.25 puts exactly 1.25 m of each layer in that cell.
    thermalprops = BoreholeThermalProperties(
        stratification=[
            (1.5, 1400.0, 1600.0, 51.25),
            (2.0, 1500.0, 1700.0, 48.75),
        ]
    )
    bh = SingleUtube(
        geom=BoreholeGeometry(Lbore=100.0, D0=0.15),
        mesh=BoreholeMesh(m_mesh=40),
        thermalprops=thermalprops,
        fluid=fluid,
        pipe_thick=0.003,
        pipe_spacing=0.0823,
        Dpi=0.026,
        n_pipes=2,
        Rp0=0.25,
        RppB=0.72,
    )

    assert bh.k0.shape == (40, 1)
    # Cells fully inside the first layer (0..19).
    np.testing.assert_allclose(bh.k0[:20, 0], 1.5)
    np.testing.assert_allclose(bh.cp_0[:20, 0], 1400.0)
    np.testing.assert_allclose(bh.rho_0[:20, 0], 1600.0)
    # Cell 20 (index 20) is split 50/50 between the two layers.
    assert bh.k0[20, 0] == pytest.approx(1.75)
    assert bh.cp_0[20, 0] == pytest.approx(1450.0)
    assert bh.rho_0[20, 0] == pytest.approx(1650.0)
    # Cells fully inside the second layer.
    np.testing.assert_allclose(bh.k0[21:, 0], 2.0)


def test_stratified_properties_length_mismatch_raises(fluid):
    thermalprops = BoreholeThermalProperties(
        stratification=[(1.5, 1400.0, 1600.0, 40.0)]  # != Lbore=100.0
    )
    with pytest.raises(ValueError):
        SingleUtube(
            geom=BoreholeGeometry(Lbore=100.0, D0=0.15),
            mesh=BoreholeMesh(m_mesh=40),
            thermalprops=thermalprops,
            fluid=fluid,
            pipe_thick=0.003,
            pipe_spacing=0.0823,
            Dpi=0.026,
            n_pipes=2,
            Rp0=0.25,
            RppB=0.72,
        )


# ============================================================
# SingleUtube — valori numerici
# ============================================================

def test_single_utube_dz(single_utube):
    assert single_utube.dz == pytest.approx(100.0 / 40)


def test_single_utube_n_equations(single_utube):
    assert single_utube.n_equations == 6


def test_single_utube_rp0_dz(single_utube):
    # Rp0 / dz = 0.25 / 2.5
    assert single_utube.Rp0_dz == pytest.approx(0.1, rel=1e-6)


def test_single_utube_rppb_dz(single_utube):
    # RppB / dz = 0.72 / 2.5
    assert single_utube.RppB_dz == pytest.approx(0.288, rel=1e-6)


def test_single_utube_s_core(single_utube):
    assert single_utube.S_core == pytest.approx(0.0045820946, rel=1e-5)


def test_single_utube_s_shell(single_utube):
    assert single_utube.S_shell == pytest.approx(0.0114808687, rel=1e-5)


def test_single_utube_c_fluid(single_utube):
    # C_fluid = rho_w * cp_w * dz * pi * Dpi^2 / 4
    expected = 1000.1435933169 * 4207.40834247225 * 2.5 * np.pi * 0.026**2 / 4
    assert single_utube.C_fluid == pytest.approx(expected, rel=1e-6)


def test_single_utube_c_shell(single_utube):
    # C_shell = rho_0 * cp_0 * dz * S_shell
    expected = 1655.0 * 1460.0 * 2.5 * 0.0114808687
    assert single_utube.C_shell[0, 0] == pytest.approx(expected, rel=1e-4)


def test_single_utube_c_core(single_utube):
    expected = 1655.0 * 1460.0 * 2.5 * 0.0045820946
    assert single_utube.C_core[0, 0] == pytest.approx(expected, rel=1e-4)


def test_single_utube_r_axial_shell(single_utube):
    # R_axial_shell = dz / (k0 * S_shell)
    expected = 2.5 / (1.8 * 0.0114808687)
    assert single_utube.R_axial_shell[0, 0] == pytest.approx(expected, rel=1e-5)


def test_single_utube_r_axial_core(single_utube):
    expected = 2.5 / (1.8 * 0.0045820946)
    assert single_utube.R_axial_core[0, 0] == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize(
    "mw_tot, expected_regime",
    [
        (0.0306, "laminar"),      # Re ~= 1000
        (0.1528, "transition"),   # Re ~= 5000 (the flow rate every other test uses)
        (0.6112, "turbulent"),    # Re ~= 20000
    ],
)
def test_single_utube_alpha_calculation_across_flow_regimes(
    single_utube, mw_tot, expected_regime
):
    """The convective coefficient uses a different Nusselt correlation per
    Reynolds-number regime; every other test in this file only exercises
    the default fixture's flow rate (mw_tot=0.2 elsewhere), which falls in
    the transition regime. This checks all three branches directly."""
    Re = (4 * mw_tot) / (
        np.pi * single_utube.Dpi * single_utube.fluid.rho_w * single_utube.fluid.ni_w
    )
    if expected_regime == "laminar":
        assert Re <= 2000
    elif expected_regime == "transition":
        assert 2000 < Re <= 10000
    else:
        assert Re > 10000

    alpha_w = single_utube._alpha_calculation(mw_tot=mw_tot)
    assert np.isfinite(alpha_w)
    assert alpha_w > 0.0


# ============================================================
# DoubleUtube — valori numerici
# ============================================================

def test_double_utube_dz(double_utube):
    assert double_utube.dz == pytest.approx(12.0 / 40)


def test_double_utube_n_equations(double_utube):
    assert double_utube.n_equations == 10


def test_double_utube_rp0_dz(double_utube):
    assert double_utube.Rp0_dz == pytest.approx(0.25 / (12.0/40), rel=1e-6)


def test_double_utube_rppb_dz(double_utube):
    assert double_utube.RppB_dz == pytest.approx(0.72 / (12.0/40), rel=1e-6)


def test_double_utube_rppa_dz(double_utube):
    assert double_utube.RppA_dz == pytest.approx(0.55 / (12.0/40), rel=1e-6)


def test_double_utube_s_core(double_utube):
    assert double_utube.S_core == pytest.approx(0.0038444596, rel=1e-5)


def test_double_utube_s_shell(double_utube):
    assert double_utube.S_shell == pytest.approx(0.0106100082, rel=1e-5)


def test_double_utube_c_fluid(double_utube):
    expected = 1000.1435933169 * 4207.40834247225 * (12.0/40) * np.pi * 0.026**2 / 4
    assert double_utube.C_fluid == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "mw_tot, expected_regime",
    [
        (0.0611, "laminar"),      # Re ~= 1000 (per branch, mw halved by "P" connection)
        (0.3056, "transition"),   # Re ~= 5000
        (1.2224, "turbulent"),    # Re ~= 20000
    ],
)
def test_double_utube_alpha_calculation_across_flow_regimes(
    double_utube, mw_tot, expected_regime
):
    """Same three-regime Nusselt correlation as SingleUtube, but
    DoubleUtube has its own separate implementation (with the extra
    parallel/series mass-flow split) — not shared code, so needs its own
    coverage. The default fixture flow rate elsewhere only reaches the
    transition regime."""
    mw = mw_tot / 2 if double_utube.connection in ("P", "p") else mw_tot
    Re = (4 * mw) / (
        np.pi * double_utube.Dpi * double_utube.fluid.rho_w * double_utube.fluid.ni_w
    )
    if expected_regime == "laminar":
        assert Re <= 2000
    elif expected_regime == "transition":
        assert 2000 < Re <= 10000
    else:
        assert Re > 10000

    alpha_w = double_utube._alpha_calculation(mw_tot=mw_tot)
    assert np.isfinite(alpha_w)
    assert alpha_w > 0.0


def test_double_utube_invalid_connection():
    with pytest.raises(ValueError):
        DoubleUtube(
            geom=BoreholeGeometry(Lbore=12.0, D0=0.15),
            mesh=BoreholeMesh(m_mesh=40),
            thermalprops=BoreholeThermalProperties(cp_0=1460.0, rho_0=1655.0, k0=1.8),
            fluid=Fluid(k_w=0.57, rho_w=1000.0, cp_w=4200.0, ni_w=1.5e-6),
            pipe_thick=0.003, pipe_spacing=0.0823, Dpi=0.026,
            n_pipes=4, Rp0=0.25, RppB=0.72, RppA=0.55,
            connection="X",  # invalido
        )


# ============================================================
# Coaxial — valori numerici
# ============================================================

def test_coaxial_dz(coaxial):
    assert coaxial.dz == pytest.approx(12.0 / 40)


def test_coaxial_n_equations(coaxial):
    assert coaxial.n_equations == 5


def test_coaxial_geometry(coaxial):
    assert coaxial.r1i == pytest.approx(0.016, rel=1e-6)
    assert coaxial.r1o == pytest.approx(0.019, rel=1e-6)
    assert coaxial.r2i == pytest.approx(0.055, rel=1e-6)
    assert coaxial.r2o == pytest.approx(0.061, rel=1e-6)


def test_coaxial_s_shell(coaxial):
    assert coaxial.S_shell == pytest.approx(0.0059815924, rel=1e-5)


def test_coaxial_c_fluid1(coaxial):
    expected = 1000.1435933169 * 4207.40834247225 * (12.0/40) * np.pi * 0.016**2
    assert coaxial.C_fluid1 == pytest.approx(expected, rel=1e-6)


def test_coaxial_c_fluid2(coaxial):
    expected = 1000.1435933169 * 4207.40834247225 * (12.0/40) * np.pi * (0.055**2 - 0.019**2)
    assert coaxial.C_fluid2 == pytest.approx(expected, rel=1e-6)


def test_coaxial_r_cond1(coaxial):
    expected = 1 / (2 * np.pi * (12.0/40) * 0.38) * np.log(0.019 / 0.016)
    assert coaxial.R_cond1 == pytest.approx(expected, rel=1e-6)


def test_coaxial_r_cond2(coaxial):
    expected = 1 / (2 * np.pi * (12.0/40) * 0.38) * np.log(0.061 / 0.055)
    assert coaxial.R_cond2 == pytest.approx(expected, rel=1e-6)


def test_coaxial_invalid_supply_return():
    with pytest.raises(ValueError):
        Coaxial(
            geom=BoreholeGeometry(Lbore=12.0, D0=0.15),
            mesh=BoreholeMesh(m_mesh=40),
            thermalprops=BoreholeThermalProperties(cp_0=1460.0, rho_0=1655.0, k0=1.8),
            fluid=Fluid(k_w=0.57, rho_w=1000.0, cp_w=4200.0, ni_w=1.5e-6),
            Dp1i=0.032, Dp2i=0.110,
            pipe1_thick=0.003, pipe2_thick=0.006,
            k_pipe1=0.38,
            k_pipe2=0.38,
            supply_and_return="3_4",  # invalido
        )


# ============================================================
# Helical — valori numerici
# ============================================================

def test_helical_dz(helical):
    assert helical.dz == pytest.approx(5.1816 / 40)


def test_helical_n_equations(helical):
    assert helical.n_equations == 7


def test_helical_F(helical):
    Lp2tot = 34 * np.pi * (0.01694 + 2 * 0.0021)
    Ds = Lp2tot / (34 * np.pi)
    expected = 34 / 5.1816 * np.pi * Ds
    assert helical.F == pytest.approx(expected, rel=1e-6)


def test_helical_geometry(helical):
    r1o = 0.01694 / 2 + 0.0021
    assert helical.r1i == pytest.approx(r1o - 0.0021, rel=1e-6)
    assert helical.r2i == pytest.approx(r1o - 0.0021, rel=1e-6)


def test_helical_s_shell(helical):
    # S_shell = π * ((D0/2)² - rshell_middle²)
    rih = 0.269 - 0.01694 / 2 - 0.0021
    roh = rih + 0.01694 + 2 * 0.0021
    rshell_middle = np.sqrt((roh**2 + (0.590 / 2.0) ** 2) / 2.0)
    expected = np.pi * ((0.590 / 2) ** 2 - rshell_middle**2)
    assert helical.S_shell == pytest.approx(expected, rel=1e-5)


def test_helical_s_shell_middle(helical):
    rih = 0.269 - 0.01694 / 2 - 0.0021
    roh = rih + 0.01694 + 2 * 0.0021
    rshell_middle = np.sqrt((roh**2 + (0.590 / 2.0) ** 2) / 2.0)
    expected = np.pi * (rshell_middle**2 - roh**2)
    assert helical.S_shell_middle == pytest.approx(expected, rel=1e-5)


def test_helical_s_core(helical):
    # S_core = π * (rih² - r1o²)
    rih = 0.269 - 0.01694 / 2 - 0.0021
    r1o = 0.01694 / 2 + 0.0021
    expected = np.pi * (rih**2 - r1o**2)
    assert helical.S_core == pytest.approx(expected, rel=1e-5)


def test_helical_c_fluid1(helical):
    dz = 5.1816 / 40
    r1i = 0.01694 / 2  # r1o - pipe_thick = (0.01694/2 + 0.0021) - 0.0021
    expected = 1000.1435933169 * 4207.40834247225 * np.pi * dz * r1i**2
    assert helical.C_fluid1 == pytest.approx(expected, rel=1e-6)


def test_helical_c_fluid2(helical):
    dz = 5.1816 / 40
    r2i = 0.01694 / 2
    Lp2tot = 34 * np.pi * (0.01694 + 2 * 0.0021)
    Ds = Lp2tot / (34 * np.pi)
    F = 34 / 5.1816 * np.pi * Ds
    expected = 1000.1435933169 * 4207.40834247225 * np.pi * dz * F * r2i**2
    assert helical.C_fluid2 == pytest.approx(expected, rel=1e-6)


def test_helical_invalid_rih():
    with pytest.raises(ValueError):
        Helical(
            geom=BoreholeGeometry(Lbore=5.1816, D0=0.590),
            mesh=BoreholeMesh(m_mesh=40),
            thermalprops=BoreholeThermalProperties(cp_0=1460.0, rho_0=1655.0, k0=1.8),
            fluid=Fluid(k_w=0.57, rho_w=1000.0, cp_w=4200.0, ni_w=1.5e-6),
            Dpi1=0.01694,
            Dpi2=0.01694,
            rih=0.001,  # troppo piccolo
            pipe_thick=0.0021,
            N=34,
            P=0.1524,
            supply_and_return="1_2",
            Lp2tot=34 * np.pi * 0.02114,
            k_pipe=0.47,
        )


def test_helical_invalid_supply_return():
    with pytest.raises(ValueError):
        Helical(
            geom=BoreholeGeometry(Lbore=5.1816, D0=0.590),
            mesh=BoreholeMesh(m_mesh=40),
            thermalprops=BoreholeThermalProperties(cp_0=1460.0, rho_0=1655.0, k0=1.8),
            fluid=Fluid(k_w=0.57, rho_w=1000.0, cp_w=4200.0, ni_w=1.5e-6),
            Dpi1=0.01694,
            Dpi2=0.01694,
            rih=0.260,
            pipe_thick=0.0021,
            N=34,
            P=0.1524,
            supply_and_return="3_4",  # invalido
            Lp2tot=34 * np.pi * 0.02114,
            k_pipe=0.47,
        )