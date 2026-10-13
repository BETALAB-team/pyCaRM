# -*- coding: utf-8 -*-
"""
Tests for the Finite Line Source (FLS) module.

Covers: FiniteLineSolution (constructor, response_matrix shape and
        physical sanity properties, response increments and the rebuilt
        response_matrix, _compute_delta_t against the original einsum,
        the image source, the on-disk increments cache).

These are unit/property-based checks (shape, monotonicity, reciprocity,
self- vs cross-response) that do not require an external reference.
A numeric comparison against pygfunction (as ground truth) belongs to the
"results" test suite, not covered here.
"""
import json

import numpy as np
import pytest
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


# ============================================================
# FIXTURES
# ============================================================

H = 20.0
D0 = 0.15
n_steps = 10
time_hist = np.arange(3600, 3600 * (n_steps + 1), 3600, dtype=np.float64)


@pytest.fixture
def fluid():
    return Fluid(k_w=0.568, rho_w=1000.0, cp_w=4207.0, ni_w=1.496e-6)


def _two_borehole_model(
    fluid, x1=5.0, xmax=7.5, k=1.8, cp=947.37, rho=1900.0, L=H, D0=D0, L_sup=1.0
):
    ground_geom = GroundGeometry(D0=D0, L=L, L_sup=L_sup, L_inf=5.0, rn=None)
    ground_mesh = GroundMesh(n_mesh=4, m_mesh=6, m_mesh_sup=2, m_mesh_inf=2)

    borehole = SingleUtube(
        geom=BoreholeGeometry(Lbore=L, D0=D0),
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

    fi = FieldInput(n_bhes=2, xmin=-2.5, ymin=-2.5, xmax=xmax, ymax=2.5, rb=D0 / 2)
    fi.from_array(np.array([0.0, x1]), np.array([0.0, 0.0]))

    stratification = [(k, cp, rho, L + L_sup + 5.0)]

    return PhysicalModel(
        ground_geom=ground_geom,
        ground_mesh=ground_mesh,
        borehole=borehole,
        fluid=fluid,
        Tg=13.0,
        stratification=stratification,
        fieldinput=fi,
    )


@pytest.fixture
def two_borehole_model(fluid):
    return _two_borehole_model(fluid)


@pytest.fixture
def fls(two_borehole_model):
    return FiniteLineSolution(
        physicalmodel=two_borehole_model,
        n_steps=n_steps,
        time_hist=time_hist,
    )


# ============================================================
# FiniteLineSolution — response_matrix, shape and physics
# ============================================================

def test_response_matrix_shape(fls):
    assert fls.response_matrix.shape == (n_steps + 1, 2, 2)


def test_response_matrix_zero_at_t0(fls):
    np.testing.assert_array_equal(fls.response_matrix[0], np.zeros((2, 2)))


def test_response_matrix_monotonic_in_time(fls):
    """The G thermal response must grow (or stay constant) over time."""
    diffs = np.diff(fls.response_matrix, axis=0)
    assert np.all(diffs >= -1e-9)


def test_response_matrix_self_response_positive(fls):
    assert np.all(fls.response_matrix[1:, 0, 0] > 0)
    assert np.all(fls.response_matrix[1:, 1, 1] > 0)


def test_response_matrix_self_greater_than_cross(fls):
    """A borehole 'sees' itself more than it sees a neighbor."""
    assert np.all(fls.response_matrix[1:, 0, 0] > fls.response_matrix[1:, 0, 1])
    assert np.all(fls.response_matrix[1:, 1, 1] > fls.response_matrix[1:, 1, 0])


def test_response_matrix_reciprocity(fls):
    """Identical boreholes and a symmetric field: cross response is reciprocal and self-response is equal."""
    np.testing.assert_allclose(
        fls.response_matrix[:, 0, 1], fls.response_matrix[:, 1, 0], rtol=1e-10
    )
    np.testing.assert_allclose(
        fls.response_matrix[:, 0, 0], fls.response_matrix[:, 1, 1], rtol=1e-10
    )


def test_continuous_mode_shape(two_borehole_model):
    fls_cont = FiniteLineSolution(
        physicalmodel=two_borehole_model,
        n_steps=n_steps,
        time_hist=time_hist,
        fls_mode="continuous",
    )
    assert fls_cont.response_matrix.shape == (n_steps + 1, 2, 2)
    assert np.all(fls_cont.response_matrix[1:, 0, 0] > 0)


@pytest.mark.parametrize("src, rcv", [(0, 1), (1, 0)])
def test_sqrt_mode_uses_receiver_req(fluid, src, rcv):
    """Cells of different size: the sqrt penalty at the receiver is h at
    sqrt(d^2 + req_rcv^2), the receiver's own req, not the source's."""
    model = _two_borehole_model(fluid, xmax=20.0)
    req = [model.field.field_dict[j]["req"] for j in range(2)]
    assert req[1] > 1.5 * req[0]

    t = 30 * 86400.0  # long enough for the heat to reach the neighbour
    fls = FiniteLineSolution(
        physicalmodel=model, n_steps=1, time_hist=np.array([t]), fls_mode="sqrt"
    )
    q = np.zeros((1, 2))
    q[0, src] = 1.0
    k = model.ground[0].k_mean
    dT = fls._compute_delta_t(q, step=1)[rcv, 0] * 2.0 * np.pi * k * H

    alpha = k / (model.ground[0].rho_mean * model.ground[0].cp_mean)
    a = 1.0 / np.sqrt(4.0 * alpha * t)

    def h(dist):
        return quad_vec(fls._integrand(np.array(dist)), a, np.inf)[0] / H

    d = 5.0
    assert dT == pytest.approx(h(np.hypot(d, req[rcv])), rel=1e-9)
    assert dT != pytest.approx(h(np.hypot(d, req[src])), rel=1e-3)


# ============================================================
# FiniteLineSolution — _compute_delta_t
# ============================================================

def test_compute_delta_t_zero_at_step0(fls, two_borehole_model):
    q_nbhes = np.zeros((n_steps, 2), dtype=np.float64)
    dT = fls._compute_delta_t(q_nbhes=q_nbhes, step=0)
    expected_shape = (2, two_borehole_model.ground_mesh.m_mesh)
    assert dT.shape == expected_shape
    np.testing.assert_array_equal(dT, np.zeros(expected_shape))


def test_compute_delta_t_positive_load_gives_positive_penalty(fls):
    """A positive thermal load (injection) must give a positive thermal penalty."""
    q_nbhes = np.full((n_steps, 2), 1000.0, dtype=np.float64)
    dT = fls._compute_delta_t(q_nbhes=q_nbhes, step=3)
    assert np.all(dT > 0)


# ============================================================
# FiniteLineSolution — response increments
# ============================================================

@pytest.fixture(params=["sqrt", "continuous"])
def fls_any_mode(request, two_borehole_model):
    return FiniteLineSolution(
        physicalmodel=two_borehole_model,
        n_steps=n_steps,
        time_hist=time_hist,
        fls_mode=request.param,
        fls_image=True,
    )


def _einsum_delta_t(fls, q_nbhes, step):
    """Reference: the original convolution over the full response matrix."""
    R = fls.response_matrix
    const = 1.0 / (
        2.0 * np.pi
        * fls.physicalmodel.ground[0].k_mean
        * fls.physicalmodel.ground_geom.L
    )
    return np.einsum("ti,tij -> j", q_nbhes[0:step], R[step:0:-1] - R[step - 1 :: -1]) * const


def test_response_increments_shape(fls_any_mode):
    assert fls_any_mode._response_increments.shape == (n_steps, 2, 2)


def test_response_matrix_rebuilt_from_increments(fls_any_mode):
    """R[0] = 0 and R[k] is the running sum of the first k increments."""
    R = fls_any_mode.response_matrix
    dR = fls_any_mode._response_increments
    np.testing.assert_array_equal(R[0], np.zeros((2, 2)))
    np.testing.assert_array_equal(R[1:], np.cumsum(dR, axis=0))


@pytest.mark.parametrize("step", [1, 2, 5, n_steps])
def test_compute_delta_t_matches_einsum(fls_any_mode, step):
    """The increment-based convolution equals the original einsum over R."""
    q_nbhes = np.random.default_rng(step).normal(0.0, 1000.0, (n_steps, 2))
    dT = fls_any_mode._compute_delta_t(q_nbhes=q_nbhes, step=step)
    expected = _einsum_delta_t(fls_any_mode, q_nbhes, step)
    np.testing.assert_allclose(dT, np.broadcast_to(expected[:, None], dT.shape), rtol=1e-12)


# ============================================================
# FiniteLineSolution — image source (isothermal ground surface)
# ============================================================

def test_fls_image_is_on_by_default(two_borehole_model):
    fls = FiniteLineSolution(
        physicalmodel=two_borehole_model, n_steps=n_steps, time_hist=time_hist
    )
    assert fls.fls_image is True


@pytest.mark.parametrize("mode", ["sqrt", "continuous"])
def test_fls_image_lowers_response(two_borehole_model, mode):
    """The image sink lowers the response at every step, strictly once its
    effect leaves round-off. Monthly steps over 10 months: the surface is
    D = 1 m above the BHE top, so the image is not negligible from month 1."""
    grid = _grid(n_steps, dt=30 * 86400.0)
    R = {
        img: FiniteLineSolution(
            physicalmodel=two_borehole_model, n_steps=n_steps, time_hist=grid,
            fls_mode=mode, fls_image=img,
        ).response_matrix[1:]
        for img in (True, False)
    }
    assert np.all(R[True] <= R[False])
    assert np.all(R[True][1:] < R[False][1:])
    # the relative drop grows with time
    drop = 1.0 - R[True][:, 0, 1] / R[False][:, 0, 1]
    assert np.all(np.diff(drop) > 0)


# ============================================================
# FiniteLineSolution — response increments cache
# ============================================================

def _grid(n, dt=3600.0):
    return np.arange(dt, dt * (n + 1), dt, dtype=np.float64)


def _forbid_compute(monkeypatch):
    """Make any FLS integration fail, so that only a cache hit can succeed."""
    def fail(self, out):
        raise AssertionError("increments were computed, expected a cache hit")
    monkeypatch.setattr(FiniteLineSolution, "_fill_increments_sqrt", fail)
    monkeypatch.setattr(FiniteLineSolution, "_fill_increments_continuous", fail)


def test_cache_none_writes_nothing(two_borehole_model, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    FiniteLineSolution(physicalmodel=two_borehole_model, n_steps=n_steps, time_hist=time_hist)
    assert list(tmp_path.iterdir()) == []


def test_cache_miss_writes_npy_and_sidecar(two_borehole_model, tmp_path):
    fls = FiniteLineSolution(
        physicalmodel=two_borehole_model, n_steps=n_steps, time_hist=time_hist,
        fls_mode="continuous", cache_dir=tmp_path,
    )
    (npy,) = tmp_path.glob("*.npy")
    (sidecar,) = tmp_path.glob("*.json")
    assert sidecar.stem == npy.stem
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([npy.name, sidecar.name])
    assert np.load(npy).shape == (n_steps, 2, 2)
    meta = json.loads(sidecar.read_text())
    assert meta["fls_mode"] == "continuous"
    assert meta["dt"] == 3600.0
    assert meta["fls_image"] is True
    assert meta["D"] == 1.0
    assert meta["format_version"] == 3
    assert isinstance(fls._response_increments, np.memmap)


@pytest.mark.parametrize("mode", ["sqrt", "continuous"])
def test_cache_hit_equals_uncached(two_borehole_model, tmp_path, monkeypatch, mode):
    uncached = FiniteLineSolution(
        physicalmodel=two_borehole_model, n_steps=n_steps, time_hist=time_hist, fls_mode=mode,
    )
    FiniteLineSolution(
        physicalmodel=two_borehole_model, n_steps=n_steps, time_hist=time_hist,
        fls_mode=mode, cache_dir=tmp_path,
    )
    _forbid_compute(monkeypatch)
    hit = FiniteLineSolution(
        physicalmodel=two_borehole_model, n_steps=n_steps, time_hist=time_hist,
        fls_mode=mode, cache_dir=tmp_path,
    )
    assert isinstance(hit._response_increments, np.memmap)
    np.testing.assert_array_equal(hit._response_increments, uncached._response_increments)


def test_cache_shorter_run_served_by_slicing(two_borehole_model, tmp_path, monkeypatch):
    FiniteLineSolution(
        physicalmodel=two_borehole_model, n_steps=n_steps, time_hist=time_hist, cache_dir=tmp_path,
    )
    short = FiniteLineSolution(physicalmodel=two_borehole_model, n_steps=4, time_hist=_grid(4))
    _forbid_compute(monkeypatch)
    sliced = FiniteLineSolution(
        physicalmodel=two_borehole_model, n_steps=4, time_hist=_grid(4), cache_dir=tmp_path,
    )
    assert sliced._response_increments.shape == (4, 2, 2)
    np.testing.assert_array_equal(sliced._response_increments, short._response_increments)


def test_cache_too_short_is_recomputed_and_replaced(two_borehole_model, tmp_path):
    FiniteLineSolution(
        physicalmodel=two_borehole_model, n_steps=4, time_hist=_grid(4), cache_dir=tmp_path,
    )
    longer = FiniteLineSolution(
        physicalmodel=two_borehole_model, n_steps=n_steps, time_hist=time_hist, cache_dir=tmp_path,
    )
    (npy,) = tmp_path.glob("*.npy")
    assert np.load(npy).shape == (n_steps, 2, 2)
    assert longer._response_increments.shape == (n_steps, 2, 2)


@pytest.mark.parametrize(
    "model_change, fls_change",
    [
        ({"x1": 6.0}, {}),          # one coordinate (and with it req)
        ({"xmax": 9.0}, {}),        # req only: wider Voronoi cell, same coordinates
        ({"k": 2.0}, {}),
        ({"cp": 1000.0}, {}),
        ({"rho": 2000.0}, {}),
        ({"L": 25.0}, {}),
        ({"D0": 0.2}, {}),          # rb
        ({"L_sup": 2.0}, {}),       # D, the buried depth of the image
        ({}, {"dt": 1800.0}),
        ({}, {"fls_mode": "continuous"}),
        ({}, {"n_theta": 16}),
        ({}, {"fls_image": False}),
    ],
)
def test_cache_key_change_misses(fluid, tmp_path, monkeypatch, model_change, fls_change):
    def build(
        model_kwargs, dt=3600.0, fls_mode="sqrt", n_theta=FiniteLineSolution.N_THETA,
        fls_image=True,
    ):
        monkeypatch.setattr(FiniteLineSolution, "N_THETA", n_theta)
        FiniteLineSolution(
            physicalmodel=_two_borehole_model(fluid, **model_kwargs),
            n_steps=n_steps,
            time_hist=_grid(n_steps, dt),
            fls_mode=fls_mode,
            fls_image=fls_image,
            cache_dir=tmp_path,
        )
    build({})
    build(model_change, **fls_change)
    assert len(list(tmp_path.glob("*.npy"))) == 2


def test_cache_tmp_file_removed_on_failure(two_borehole_model, tmp_path, monkeypatch):
    _forbid_compute(monkeypatch)
    with pytest.raises(AssertionError):
        FiniteLineSolution(
            physicalmodel=two_borehole_model,
            n_steps=n_steps,
            time_hist=time_hist,
            cache_dir=tmp_path,
        )
    assert list(tmp_path.iterdir()) == []


def test_cache_requires_uniform_time_grid(two_borehole_model, tmp_path):
    uneven = np.cumsum(np.linspace(3600.0, 7200.0, n_steps))
    with pytest.raises(ValueError, match="uniform"):
        FiniteLineSolution(
            physicalmodel=two_borehole_model, n_steps=n_steps, time_hist=uneven, cache_dir=tmp_path,
        )
