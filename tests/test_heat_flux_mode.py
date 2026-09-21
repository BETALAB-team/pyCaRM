# -*- coding: utf-8 -*-
"""
Tests for HeatFluxMode: construction, validation, and default curves.

Covers only what HeatFluxMode can know standalone (Q_buildings/T_supply
length match, curve callability). n_steps/n_bhes/groups cross-checks live
in Simulation.__post_init__ (see tests/test_solver.py).
"""
import numpy as np
import pytest

from carm import HeatFluxMode

N_STEPS = 5


def test_valid_construction_uses_default_curves():
    Q_buildings = np.full(N_STEPS, 1000.0, dtype=np.float64)
    T_supply = np.full(N_STEPS, 45.0, dtype=np.float64)

    mode = HeatFluxMode(Q_buildings=Q_buildings, T_supply=T_supply)

    assert mode.cop_curve(10.0) == mode.eer_curve(10.0)
    assert mode.cop_curve(10.0) == pytest.approx(10.29 - 0.21 * 10.0 + 0.0012 * 10.0**2.0)


def test_mismatched_lengths_raises():
    Q_buildings = np.full(N_STEPS + 1, 1000.0, dtype=np.float64)
    T_supply = np.full(N_STEPS, 45.0, dtype=np.float64)

    with pytest.raises(ValueError):
        HeatFluxMode(Q_buildings=Q_buildings, T_supply=T_supply)


def test_non_callable_cop_curve_raises():
    Q_buildings = np.full(N_STEPS, 1000.0, dtype=np.float64)
    T_supply = np.full(N_STEPS, 45.0, dtype=np.float64)

    with pytest.raises(ValueError):
        HeatFluxMode(Q_buildings=Q_buildings, T_supply=T_supply, cop_curve=1.0)


def test_non_callable_eer_curve_raises():
    Q_buildings = np.full(N_STEPS, 1000.0, dtype=np.float64)
    T_supply = np.full(N_STEPS, 45.0, dtype=np.float64)

    with pytest.raises(ValueError):
        HeatFluxMode(Q_buildings=Q_buildings, T_supply=T_supply, eer_curve=1.0)


def test_independent_cop_and_eer_curves():
    Q_buildings = np.full(N_STEPS, 1000.0, dtype=np.float64)
    T_supply = np.full(N_STEPS, 45.0, dtype=np.float64)

    mode = HeatFluxMode(
        Q_buildings=Q_buildings,
        T_supply=T_supply,
        cop_curve=lambda dT: 5.0,
        eer_curve=lambda dT: 3.0,
    )

    assert mode.cop_curve(10.0) == 5.0
    assert mode.eer_curve(10.0) == 3.0


def test_is_frozen():
    Q_buildings = np.full(N_STEPS, 1000.0, dtype=np.float64)
    T_supply = np.full(N_STEPS, 45.0, dtype=np.float64)
    mode = HeatFluxMode(Q_buildings=Q_buildings, T_supply=T_supply)

    with pytest.raises(AttributeError):
        mode.Q_buildings = Q_buildings


def test_equality_and_hash_use_identity_not_array_value():
    """Array fields make value-based __eq__/__hash__ unsound (elementwise
    array comparison has no unambiguous truth value, and arrays are
    unhashable); HeatFluxMode must fall back to identity-based comparison
    instead of raising."""
    Q_buildings = np.full(N_STEPS, 1000.0, dtype=np.float64)
    T_supply = np.full(N_STEPS, 45.0, dtype=np.float64)
    a = HeatFluxMode(Q_buildings=Q_buildings, T_supply=T_supply)
    b = HeatFluxMode(Q_buildings=Q_buildings.copy(), T_supply=T_supply.copy())

    assert a == a
    assert a != b
    assert hash(a) == hash(a)
