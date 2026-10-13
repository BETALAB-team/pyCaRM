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
    Building load and heat pump settings for heat flux mode.

    Attributes
    ----------
    Q_buildings : NDArray[np.float64]
        Building thermal load per time step, shape (n_steps,) [W].
        Positive for heating, negative for cooling, zero when the heat
        pump is off.
    T_supply : NDArray[np.float64]
        Supply temperature to the building per time step, shape (n_steps,) [°C].
    cop_curve : Callable[[float], float]
        Heat pump COP as a function of the temperature lift
        ``T_supply - T_f,out`` [-]. Defaults to Ruhnau et al. (2019).
    eer_curve : Callable[[float], float]
        Heat pump EER as a function of the temperature lift
        ``T_f,out - T_supply`` [-]. Defaults to Ruhnau et al. (2019).
    """
    # eq=False: the array fields would make the generated __eq__/__hash__ raise.

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
