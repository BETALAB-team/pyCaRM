# -*- coding: utf-8 -*-
"""
Results test: FLS equation vs pygfunction (external reference).

Checks that the FLS equation h(d, t) implemented in
``FiniteLineSolution._integrand`` is the same as pygfunction's
``finite_line_source`` (Cimmino, 2018), real part only and with the image
part. Only the equation is checked, at given distances and times: the
``fls_mode`` options ("sqrt", "continuous") are different formulations of
which distances enter h, so their response matrices are not expected to
match a single pygfunction call.
"""
import numpy as np
import pytest
import pygfunction as gt
from scipy.integrate import quad_vec

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
)
from carm.thermal_interference import FiniteLineSolution

H = 20.0
D0 = 0.15
D = 1.0  # buried depth of the BHE top, ground_geom.L_sup
k = 1.8
rho = 1900.0
cp = 947.37
alpha = k / (rho * cp)

distances = [0.1, 1.0, 5.0, 20.0]
# 1 hour to 10 years: the image goes from round-off to dominant
times = np.logspace(np.log10(3600.0), np.log10(10 * 365 * 86400.0), 12)


@pytest.fixture
def two_borehole_model():
    ground_geom = GroundGeometry(D0=D0, L=H, L_sup=D, L_inf=5.0, rn=None)
    ground_mesh = GroundMesh(n_mesh=4, m_mesh=6, m_mesh_sup=2, m_mesh_inf=2)
    fluid = Fluid(k_w=0.568, rho_w=1000.0, cp_w=4207.0, ni_w=1.496e-6)

    borehole = SingleUtube(
        geom=BoreholeGeometry(Lbore=H, D0=D0),
        mesh=BoreholeMesh(m_mesh=6),
        thermalprops=BoreholeThermalProperties(cp_0=1460.0, rho_0=1655.0, k0=1.8),
        fluid=fluid,
        pipe_thick=0.003,
        pipe_spacing=0.05,
        Dpi=0.032,
        n_pipes=2,
        Rp0=0.1,
        RppB=0.1,
    )

    fi = FieldInput(n_bhes=2, xmin=-2.5, ymin=-2.5, xmax=7.5, ymax=2.5, rb=D0 / 2)
    fi.from_array(np.array([0.0, 5.0]), np.array([0.0, 0.0]))

    stratification = [(k, cp, rho, H + D + 5.0)]

    return PhysicalModel(
        ground_geom=ground_geom,
        ground_mesh=ground_mesh,
        borehole=borehole,
        fluid=fluid,
        Tg=13.0,
        stratification=stratification,
        fieldinput=fi,
    )


def _pygfunction_h(d, t, image):
    """h(d, t) from pygfunction for two boreholes ``d`` meters apart."""
    b1 = gt.boreholes.Borehole(H=H, D=D, r_b=1e-6, x=0.0, y=0.0)
    b2 = gt.boreholes.Borehole(H=H, D=D, r_b=1e-6, x=d, y=0.0)
    return gt.heat_transfer.finite_line_source(
        t, alpha, b1, b2, reaSource=True, imgSource=image
    )


@pytest.mark.parametrize("image", [False, True])
def test_fls_equation_matches_pygfunction(two_borehole_model, image):
    fls = FiniteLineSolution(
        physicalmodel=two_borehole_model, n_steps=1,
        time_hist=np.array([3600.0]), fls_image=image,
    )
    for d in distances:
        func = fls._integrand(np.array(d))
        for t in times:
            h_carm = quad_vec(func, 1.0 / np.sqrt(4.0 * alpha * t), np.inf)[0] / H
            h_pyg = _pygfunction_h(d, t, image)
            # abs: at the diffusion front (h < 1e-10) the two quadratures
            # differ in relative terms; h reaches ~5 over this grid
            assert h_carm == pytest.approx(h_pyg, rel=1e-9, abs=1e-12), (d, t)
