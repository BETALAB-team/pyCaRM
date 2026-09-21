# -*- coding: utf-8 -*-
"""
Tests for ground properties module.

Covers: GroundGeometry, GroundMesh, GroundProperties.

Reference values are computed analytically from the SingleUtube example
parameters (SingleUtube.py): monostrato k=1.8, cp=947.37, rho=1900,
L=100, L_sup=1, L_inf=10, n_mesh=20, m_mesh=40, f=1.2, rn=10, D0=0.15.
"""
import numpy as np
import pytest

from carm import GroundGeometry, GroundMesh
from carm.properties import GroundProperties


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def simple_geom():
    return GroundGeometry(D0=0.15, L=100.0, L_sup=1.0, L_inf=10.0, rn=10.0)


@pytest.fixture
def simple_mesh():
    return GroundMesh(n_mesh=20, m_mesh=40, m_mesh_sup=4, m_mesh_inf=40)


@pytest.fixture
def ground(simple_geom, simple_mesh):
    # monostrato, spessore totale = L_sup + L + L_inf = 111 m
    strat = [(1.8, 947.37, 1900.0, 111.0)]
    return GroundProperties(
        geom=simple_geom,
        mesh=simple_mesh,
        Tg=13.0,
        stratification=strat,
    )


# ============================================================
# GroundGeometry
# ============================================================

def test_ground_geometry_stores_values():
    geom = GroundGeometry(D0=0.15, L=100.0, L_sup=1.0, L_inf=10.0, rn=10.0)
    assert geom.D0 == 0.15
    assert geom.L == 100.0
    assert geom.L_sup == 1.0
    assert geom.L_inf == 10.0
    assert geom.rn == 10.0


def test_ground_geometry_r0():
    geom = GroundGeometry(D0=0.15, L=100.0, L_sup=1.0, L_inf=10.0, rn=10.0)
    assert geom.r0 == pytest.approx(0.075)


def test_ground_geometry_invalid_d0():
    with pytest.raises(ValueError):
        GroundGeometry(D0=0.0, L=100.0, L_sup=1.0, L_inf=10.0, rn=10.0)


def test_ground_geometry_rn_smaller_than_r0():
    with pytest.raises(ValueError):
        GroundGeometry(D0=0.15, L=100.0, L_sup=1.0, L_inf=10.0, rn=0.05)


def test_ground_geometry_invalid_L():
    with pytest.raises(ValueError):
        GroundGeometry(D0=0.15, L=0.0, L_sup=1.0, L_inf=10.0, rn=10.0)


def test_ground_geometry_invalid_L_sup():
    with pytest.raises(ValueError):
        GroundGeometry(D0=0.15, L=100.0, L_sup=0.0, L_inf=10.0, rn=10.0)


def test_ground_geometry_invalid_L_inf():
    with pytest.raises(ValueError):
        GroundGeometry(D0=0.15, L=100.0, L_sup=1.0, L_inf=0.0, rn=10.0)


# ============================================================
# GroundMesh
# ============================================================

def test_ground_mesh_stores_values():
    mesh = GroundMesh(n_mesh=20, m_mesh=40, m_mesh_sup=4, m_mesh_inf=40)
    assert mesh.n_mesh == 20
    assert mesh.m_mesh == 40
    assert mesh.m_mesh_sup == 4
    assert mesh.m_mesh_inf == 40
    assert mesh.f == pytest.approx(1.2)


def test_ground_mesh_invalid_n_mesh():
    with pytest.raises(ValueError):
        GroundMesh(n_mesh=1, m_mesh=40, m_mesh_sup=4, m_mesh_inf=40)


def test_ground_mesh_invalid_m_mesh():
    with pytest.raises(ValueError):
        GroundMesh(n_mesh=20, m_mesh=0, m_mesh_sup=4, m_mesh_inf=40)
        

# ============================================================
# GroundProperties — shapes
# ============================================================

def test_ground_shapes(ground):
    assert ground.radius.shape == (1, 21)   # (1, n_mesh + 1)
    assert ground.rm.shape == (1, 22)        # (1, n_mesh + 2)
    assert ground.R_ground.shape == (40, 21) # (m_mesh, n_mesh + 1)
    assert ground.C_ground.shape == (40, 20)
    assert ground.R_axial.shape == (40, 20)
    assert ground.R_sup.shape == (4,)        # (m_mesh_sup,)
    assert ground.C_sup.shape == (4,)
    assert ground.R_inf.shape == (40,)       # (m_mesh_inf,)
    assert ground.C_inf.shape == (40,)


# ============================================================
# GroundProperties — discretizzazione
# ============================================================

def test_ground_dz(ground):
    assert ground.dz == pytest.approx(100.0 / 40)


def test_ground_dz_sup(ground):
    assert ground.dz_sup == pytest.approx(1.0 / 4)


def test_ground_dz_inf(ground):
    assert ground.dz_inf == pytest.approx(10.0 / 40)


def test_ground_radius_first(ground):
    # il primo raggio è sempre r0 = D0/2
    assert ground.radius[0, 0] == pytest.approx(0.075)


def test_ground_radius_last(ground):
    # l'ultimo raggio è sempre rn
    assert ground.radius[0, -1] == pytest.approx(10.0)


def test_ground_rm_first(ground):
    # rm[0] = r0 per convenzione
    assert ground.rm[0, 0] == pytest.approx(0.075)


def test_ground_rm_last(ground):
    # rm[-1] = rn per convenzione
    assert ground.rm[0, -1] == pytest.approx(10.0)


# ============================================================
# GroundProperties — resistenze e capacità radiali
# ============================================================

def test_ground_R_ground_formula(ground):
    # R = 1/(2π·k·dz) · log(rm[i+1]/rm[i])
    # prima corona, primo strato
    k = 1.8
    dz = 100.0 / 40
    expected = 1 / (2 * np.pi * k * dz) * np.log(ground.rm[0, 1] / ground.rm[0, 0])
    assert ground.R_ground[0, 0] == pytest.approx(expected, rel=1e-6)


def test_ground_R_ground_value(ground):
    assert ground.R_ground[0, 0] == pytest.approx(0.01190099185, rel=1e-5)


def test_ground_R_ground_last_value(ground):
    assert ground.R_ground[0, -1] == pytest.approx(0.0029871496, rel=1e-5)


def test_ground_C_ground_formula(ground):
    # C = ρ·cp·dz·π·(r_out² - r_in²)
    cp = 947.37
    rho = 1900.0
    dz = 100.0 / 40
    area = np.pi * (ground.radius[0, 1]**2 - ground.radius[0, 0]**2)
    expected = rho * cp * dz * area
    assert ground.C_ground[0, 0] == pytest.approx(expected, rel=1e-6)


def test_ground_C_ground_value(ground):
    assert ground.C_ground[0, 0] == pytest.approx(152694.380044, rel=1e-5)


def test_ground_C_ground_last_value(ground):
    assert ground.C_ground[0, -1] == pytest.approx(439448822.423360, rel=1e-5)


# ============================================================
# GroundProperties — resistenze assiali
# ============================================================

def test_ground_R_axial_formula(ground):
    # R_axial = dz / (k · π · (r_out² - r_in²))
    k = 1.8
    dz = 100.0 / 40
    area = np.pi * (ground.radius[0, 1]**2 - ground.radius[0, 0]**2)
    expected = dz / (k * area)
    assert ground.R_axial[0, 0] == pytest.approx(expected, rel=1e-6)


def test_ground_R_axial_value(ground):
    assert ground.R_axial[0, 0] == pytest.approx(40.931503, rel=1e-5)


# ============================================================
# GroundProperties — sup e inf
# ============================================================

def test_ground_R_sup_formula(ground):
    # R_sup = dz_sup / (k · π · rn²)
    k = 1.8
    dz_sup = 1.0 / 4
    Area = np.pi * 10.0**2
    expected = dz_sup / (k * Area)
    assert ground.R_sup[0] == pytest.approx(expected, rel=1e-6)


def test_ground_R_sup_value(ground):
    assert ground.R_sup[0] == pytest.approx(0.0004420971, rel=1e-5)


def test_ground_C_sup_formula(ground):
    # C_sup = ρ·cp·dz_sup·π·rn²
    cp = 947.37
    rho = 1900.0
    dz_sup = 1.0 / 4
    Area = np.pi * 10.0**2
    expected = rho * cp * dz_sup * Area
    assert ground.C_sup[0] == pytest.approx(expected, rel=1e-6)


def test_ground_C_sup_value(ground):
    assert ground.C_sup[0] == pytest.approx(141371905.030990, rel=1e-5)


def test_ground_R_inf_value(ground):
    # monostrato omogeneo: R_inf uguale a R_sup (stesso dz_inf=dz_sup=0.25)
    assert ground.R_inf[0] == pytest.approx(0.0004420971, rel=1e-5)


def test_ground_C_inf_value(ground):
    assert ground.C_inf[0] == pytest.approx(141371905.030990, rel=1e-5)


# ============================================================
# GroundProperties — proprietà medie
# ============================================================

def test_ground_k_mean(ground):
    # monostrato uniforme: k_mean == k
    assert ground.k_mean == pytest.approx(1.8, rel=1e-6)


def test_ground_cp_mean(ground):
    assert ground.cp_mean == pytest.approx(947.37, rel=1e-6)


def test_ground_rho_mean(ground):
    assert ground.rho_mean == pytest.approx(1900.0, rel=1e-6)


# ============================================================
# GroundProperties — stratification mismatch
# ============================================================

def test_ground_stratification_length_mismatch(simple_geom, simple_mesh):
    # spessore totale sbagliato: 111 m richiesti, 100 forniti
    with pytest.raises(ValueError):
        GroundProperties(
            geom=simple_geom,
            mesh=simple_mesh,
            Tg=13.0,
            stratification=[(1.8, 947.37, 1900.0, 100.0)],
        )


def test_ground_stratification_multistrato(simple_geom, simple_mesh):
    # due strati che sommano esattamente a 111 m
    strat = [
        (1.8, 947.37, 1900.0, 55.5),
        (2.0, 1000.0, 2000.0, 55.5),
    ]
    gr = GroundProperties(
        geom=simple_geom,
        mesh=simple_mesh,
        Tg=13.0,
        stratification=strat,
    )

    expected_k_mean = (1.8 * 54.5 + 2.0 * 45.5) / 100.0

    assert gr.k_mean == pytest.approx(expected_k_mean, rel=1e-5)