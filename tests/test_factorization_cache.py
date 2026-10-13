# -*- coding: utf-8 -*-
"""
Tests for FactorizationCache: build-on-miss, keying by rn and clearing on a
property change.

Uses a fake builder that counts calls, so the cache's rules are tested
through ``get`` without assembling or solving any physics.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import identity

from carm.simulation.factorization_cache import FactorizationCache

N_NODES = 4


class CountingBuilder:
    def __init__(self, fill: float = 1.0):
        self.calls = []
        self.fill = fill

    def __call__(self, model, gr_p, env, timesteps, mw, adiabatic):
        self.calls.append((gr_p, mw, adiabatic))
        return (identity(N_NODES, format="csc") * self.fill).tocsc()


def _fake_model(n_bhes: int = 2, rn=None):
    borehole = SimpleNamespace(
        k0=np.full((3, 1), 2.0),
        cp_0=np.full((3, 1), 1000.0),
        rho_0=np.full((3, 1), 1800.0),
    )
    rn = rn if rn is not None else [1.0 + j for j in range(n_bhes)]
    return SimpleNamespace(borehole=borehole, ground=[SimpleNamespace(rn=r) for r in rn])


def _cache(builder, model=None):
    return FactorizationCache(
        model if model is not None else _fake_model(),
        env=object(),
        timesteps=object(),
        builder=builder,
    )


def test_same_key_twice_builds_once_and_returns_same_objects():
    builder = CountingBuilder()
    cache = _cache(builder)

    A1, lu1 = cache.get(0, 0.5, False)
    A2, lu2 = cache.get(0, 0.5, False)

    assert len(builder.calls) == 1
    assert A1 is A2
    assert lu1 is lu2


def test_flow_that_comes_back_reuses_its_factorization():
    builder = CountingBuilder()
    cache = _cache(builder)

    first = cache.get(0, 0.5, False)
    cache.get(0, 0.0, False)
    again = cache.get(0, 0.5, False)

    assert len(builder.calls) == 2
    assert again[1] is first[1]


def test_boreholes_with_different_rn_and_same_flow_are_separate_entries():
    builder = CountingBuilder()
    model = _fake_model(n_bhes=2, rn=[1.72, 1.9])
    cache = _cache(builder, model)

    _, lu0 = cache.get(0, 0.5, False)
    _, lu1 = cache.get(1, 0.5, False)

    assert len(builder.calls) == 2
    assert lu0 is not lu1
    assert builder.calls[0][0] is model.ground[0]
    assert builder.calls[1][0] is model.ground[1]


def test_boreholes_with_same_rn_and_flow_share_one_entry():
    builder = CountingBuilder()
    cache = _cache(builder, _fake_model(n_bhes=2, rn=[1.72, 1.72]))

    A0, lu0 = cache.get(0, 0.5, False)
    A1, lu1 = cache.get(1, 0.5, False)

    assert len(builder.calls) == 1
    assert A1 is A0
    assert lu1 is lu0


def test_adiabatic_flag_is_part_of_the_key():
    builder = CountingBuilder()
    cache = _cache(builder)

    _, lu_false = cache.get(0, 0.5, False)
    _, lu_true = cache.get(0, 0.5, True)

    assert len(builder.calls) == 2
    assert lu_false is not lu_true
    assert [call[2] for call in builder.calls] == [False, True]


def test_property_change_rebuilds_and_drops_old_property_entries():
    builder = CountingBuilder()
    model = _fake_model(n_bhes=1)
    cache = _cache(builder, model)
    _, lu_old = cache.get(0, 0.5, False)

    model.borehole.k0[:, 0] = 2.5  # in place, as _update_properties does
    _, lu_new = cache.get(0, 0.5, False)

    assert len(builder.calls) == 2
    assert lu_new is not lu_old

    model.borehole.k0[:, 0] = 2.0  # back to the old bytes
    _, lu_back = cache.get(0, 0.5, False)

    assert len(builder.calls) == 3
    assert lu_back is not lu_old


def test_property_change_rebuilds_once_for_boreholes_sharing_rn():
    builder = CountingBuilder()
    model = _fake_model(n_bhes=2, rn=[1.72, 1.72])
    cache = _cache(builder, model)
    cache.get(0, 0.5, False)
    cache.get(1, 0.5, False)

    model.borehole.k0[:, 0] = 2.5  # in place, as _update_properties does
    _, lu0 = cache.get(0, 0.5, False)
    _, lu1 = cache.get(1, 0.5, False)

    assert len(builder.calls) == 2
    assert lu1 is lu0


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_matrix_raises_at_build_time(bad):
    builder = CountingBuilder(fill=bad)
    cache = _cache(builder)

    with pytest.raises(ValueError):
        cache.get(0, 0.5, False)
