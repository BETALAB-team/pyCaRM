# -*- coding: utf-8 -*-
"""
Tests for environmental conditions module.

Covers: EnvironmentalProperties, EnvironmentalTimeSeries (from_array),
        ExternalEnvironment.
"""
import numpy as np
import pytest

from carm import (
    EnvironmentalProperties,
    EnvironmentalTimeSeries,
)
from carm.external_environment import ExternalEnvironment


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def env_props():
    return EnvironmentalProperties(
        R_ext=0.04,
        absorptance=0.6,
        eps=0.9,
        At=10.0,
        tau=3600.0,
        tau_y=31_536_000.0,
        tau_shift=1_296_000.0,
    )


@pytest.fixture
def t_ext():
    return np.linspace(-5.0, 25.0, 8760)


@pytest.fixture
def solar_rad():
    return np.abs(np.sin(np.linspace(0, 2 * np.pi, 8760))) * 600.0


@pytest.fixture
def env_series(t_ext, solar_rad):
    return EnvironmentalTimeSeries.from_array(12.0, t_ext, solar_rad)


@pytest.fixture
def ext_env(env_props, env_series):
    return ExternalEnvironment(envprops=env_props, envinput=env_series)


# ============================================================
# EnvironmentalProperties — storage
# ============================================================

def test_env_props_stores_values(env_props):
    assert env_props.R_ext == 0.04
    assert env_props.absorptance == 0.6
    assert env_props.eps == 0.9
    assert env_props.At == 10.0
    assert env_props.tau == 3600.0
    assert env_props.tau_y == 31_536_000.0
    assert env_props.tau_shift == 1_296_000.0


# ============================================================
# EnvironmentalProperties — validazione
# ============================================================

def test_env_props_invalid_r_ext():
    with pytest.raises(ValueError):
        EnvironmentalProperties(
            R_ext=-0.01, absorptance=0.6, eps=0.9,
            At=10.0, tau=3600.0, tau_y=31_536_000.0, tau_shift=1_296_000.0,
        )


def test_env_props_invalid_absorptance():
    with pytest.raises(ValueError):
        EnvironmentalProperties(
            R_ext=0.04, absorptance=-0.1, eps=0.9,
            At=10.0, tau=3600.0, tau_y=31_536_000.0, tau_shift=1_296_000.0,
        )


def test_env_props_invalid_eps():
    with pytest.raises(ValueError):
        EnvironmentalProperties(
            R_ext=0.04, absorptance=0.6, eps=-0.5,
            At=10.0, tau=3600.0, tau_y=31_536_000.0, tau_shift=1_296_000.0,
        )


def test_env_props_invalid_at():
    with pytest.raises(ValueError):
        EnvironmentalProperties(
            R_ext=0.04, absorptance=0.6, eps=0.9,
            At=-1.0, tau=3600.0, tau_y=31_536_000.0, tau_shift=1_296_000.0,
        )


def test_env_props_invalid_tau():
    with pytest.raises(ValueError):
        EnvironmentalProperties(
            R_ext=0.04, absorptance=0.6, eps=0.9,
            At=10.0, tau=-1.0, tau_y=31_536_000.0, tau_shift=1_296_000.0,
        )


def test_env_props_invalid_tau_y():
    with pytest.raises(ValueError):
        EnvironmentalProperties(
            R_ext=0.04, absorptance=0.6, eps=0.9,
            At=10.0, tau=3600.0, tau_y=-1.0, tau_shift=1_296_000.0,
        )


def test_env_props_invalid_tau_shift():
    with pytest.raises(ValueError):
        EnvironmentalProperties(
            R_ext=0.04, absorptance=0.6, eps=0.9,
            At=10.0, tau=3600.0, tau_y=31_536_000.0, tau_shift=-1.0,
        )


def test_env_props_is_frozen(env_props):
    """frozen=True: qualsiasi assignment deve sollevare un'eccezione."""
    with pytest.raises(Exception):
        env_props.R_ext = 99.0


# ============================================================
# EnvironmentalTimeSeries — from_array
# ============================================================

def test_env_series_stores_tm(env_series):
    assert env_series.Tm == 12.0


def test_env_series_stores_t_ext(env_series, t_ext):
    np.testing.assert_array_equal(env_series.T_ext, t_ext)


def test_env_series_stores_solar_rad(env_series, solar_rad):
    np.testing.assert_array_equal(env_series.SolarRad, solar_rad)


def test_env_series_invalid_t_ext_nan(solar_rad):
    t_bad = np.full(8760, np.nan)
    with pytest.raises(ValueError):
        EnvironmentalTimeSeries.from_array(12.0, t_bad, solar_rad)


def test_env_series_invalid_t_ext_inf(solar_rad):
    t_bad = np.full(8760, np.inf)
    with pytest.raises(ValueError):
        EnvironmentalTimeSeries.from_array(12.0, t_bad, solar_rad)


def test_env_series_invalid_solar_rad_nan(t_ext):
    rad_bad = np.full(8760, np.nan)
    with pytest.raises(ValueError):
        EnvironmentalTimeSeries.from_array(12.0, t_ext, rad_bad)


def test_env_series_invalid_solar_rad_inf(t_ext):
    rad_bad = np.full(8760, np.inf)
    with pytest.raises(ValueError):
        EnvironmentalTimeSeries.from_array(12.0, t_ext, rad_bad)


def test_env_series_single_nan_in_middle(t_ext):
    """Un solo NaN in posizione arbitraria deve far fallire la validazione."""
    rad_bad = np.abs(np.sin(np.linspace(0, 2 * np.pi, 8760))) * 600.0
    rad_bad[4380] = np.nan
    with pytest.raises(ValueError):
        EnvironmentalTimeSeries.from_array(12.0, t_ext, rad_bad)


# ============================================================
# ExternalEnvironment — T_sky
# ============================================================

def test_ext_env_t_sky_shape(ext_env, t_ext):
    assert ext_env.T_sky.shape == t_ext.shape


def test_ext_env_t_sky_formula(ext_env, t_ext):
    """T_sky = 0.0552 * (T_ext + 273.15)^1.5"""
    expected = 0.0552 * (t_ext + 273.15) ** 1.5
    np.testing.assert_array_almost_equal(ext_env.T_sky, expected)


def test_ext_env_t_sky_first_value(ext_env, t_ext):
    expected_first = 0.0552 * (t_ext[0] + 273.15) ** 1.5
    assert ext_env.T_sky[0] == pytest.approx(expected_first, rel=1e-6)


def test_ext_env_t_sky_last_value(ext_env, t_ext):
    expected_last = 0.0552 * (t_ext[-1] + 273.15) ** 1.5
    assert ext_env.T_sky[-1] == pytest.approx(expected_last, rel=1e-6)


def test_ext_env_stores_boltzmann(ext_env):
    assert ext_env.boltzmann == pytest.approx(5.670374419e-8, rel=1e-9)


def test_ext_env_stores_envprops(ext_env, env_props):
    assert ext_env.envprops is env_props


def test_ext_env_stores_envinput(ext_env, env_series):
    assert ext_env.envinput is env_series