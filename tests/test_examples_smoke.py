# -*- coding: utf-8 -*-
"""
Smoke tests for the example scripts in examples/.

These scripts are the first thing a new user runs after installing CaRM
(see README.md "Examples" section), but nothing else in the test suite
ever calls them — a renamed/reshaped public API (as happened with
Coaxial's k_pipe1/k_pipe2, see tests/test_properties/test_borehole.py)
could break them silently.

Single-borehole examples and the 9-borehole parallel example run in a
few seconds each and are executed here. The two full-year (n_steps=8760),
9-borehole series examples are only import-checked for speed (still
catches renamed/removed public API used at import time); the same
_run_series/heat_flux_mode code paths are exercised end-to-end at a
reduced scale in tests/test_results/test_series_and_heat_flux.py.

matplotlib.use("Agg") avoids blocking on plt.show()/opening windows.
Simulation.run() unconditionally writes results/*.npz
(Simulation._save_results); monkeypatch.chdir(tmp_path) keeps that side
effect out of the repository working directory.
"""
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"

# Single-borehole + one modest multi-borehole example: a few seconds each.
FAST_EXAMPLES = [
    "SingleUtube",
    "DoubleUtube",
    "Coaxial",
    "Helical",
    "Helical_variable_properties",
    "SingleUtube_multi_parallel",
    "Excel_input",
]

# 9 boreholes x 8760 steps: import-only, too slow to run on every test pass.
IMPORT_ONLY_EXAMPLES = [
    "SingleUtube_multi_series",
    "SingleUtube_multi_series_heat_flux",
]


def _load_example(name):
    path = EXAMPLES_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("name", FAST_EXAMPLES)
def test_example_runs(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    module = _load_example(name)
    module.main()


@pytest.mark.parametrize("name", IMPORT_ONLY_EXAMPLES)
def test_example_imports(name):
    module = _load_example(name)
    assert hasattr(module, "main")
