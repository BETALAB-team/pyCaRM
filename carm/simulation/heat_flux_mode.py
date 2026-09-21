# -*- coding: utf-8 -*-
"""
Heat-flux-mode configuration for the BHE simulation.

Groups the building-side load, supply temperature, and heat pump
performance curves needed to drive ``Simulation`` from a building thermal
load instead of a prescribed inlet fluid temperature.
"""

from dataclasses import dataclass

from numpy.typing import NDArray
from typing import Callable

import numpy as np


def _default_performance_curve(dT: float) -> float:
    """Ruhnau et al. (2019) quadratic GSHP performance regression."""
    return 10.29 - 0.21 * dT + 0.0012 * dT**2.0


@dataclass(frozen=True, slots=True, eq=False)
class HeatFluxMode:
    """
    Building-side load and heat pump performance configuration.

    ``eq=False`` keeps the default identity-based ``__eq__``/``__hash__``:
    the array fields make the auto-generated dataclass ``__eq__`` raise
    (elementwise array comparison has no unambiguous truth value) and
    ``__hash__`` raise (arrays are unhashable).

    Attributes
    ----------
    Q_buildings : NDArray[np.float64]
        Building thermal load time series, shape (n_steps,) [W].
        Positive values are a heating load (COP branch), negative values
        a cooling load (EER branch), zero skips the heat pump update for
        that step.
    T_supply : NDArray[np.float64]
        Supply temperature time series to the building, shape (n_steps,) [°C].
    cop_curve : Callable[[float], float]
        Heat pump COP as a function of the temperature lift
        ``T_supply - T_f,out`` [-]. Defaults to the shared default
        performance curve.
    eer_curve : Callable[[float], float]
        Heat pump EER as a function of the temperature lift
        ``T_f,out - T_supply`` [-]. Defaults to the shared default
        performance curve.
    """

    Q_buildings: NDArray[np.float64]
    T_supply: NDArray[np.float64]
    cop_curve: Callable[[float], float] = _default_performance_curve
    eer_curve: Callable[[float], float] = _default_performance_curve

    def __post_init__(self):
        if len(self.Q_buildings) != len(self.T_supply):
            raise ValueError("Q_buildings and T_supply must have the same length")
        if not callable(self.cop_curve):
            raise ValueError("cop_curve must be callable")
        if not callable(self.eer_curve):
            raise ValueError("eer_curve must be callable")
