# -*- coding: utf-8 -*-
"""
Environmental conditions module.

Defines the thermal and radiative properties of the external environment
used as boundary conditions in the borehole heat exchanger simulation.
"""

from dataclasses import dataclass, field

from typing import Self
from numpy.typing import NDArray

import numpy as np


@dataclass(frozen=True, slots=True)
class EnvironmentalProperties:
    """
    Physical and thermal properties of the external environment.

    Groups all site-specific parameters that characterize the surface
    boundary condition: radiative exchange coefficients, surface optical
    properties, and the seasonal air temperature signal.

    Attributes
    ----------
    R_ext : float
        External thermal resistance [W / (K m²)].
    absorptance : float
        Surface absorptance for solar radiation [-].
    eps : float
        Surface emittance for longwave radiation [-].
    At : float
        Annual amplitude of monthly average air temperature [K].
    tau : float
        Current simulation time [s].
    tau_y : float
        Duration of one year (315,536,000 s) [s].
    tau_shift : float
        Time offset to account for the date of minimum surface temperature [s].
    """

    R_ext: float  # W / (K * m2)
    absorptance: float  # -, surface absorptance
    eps: float  # -, surface emittance
    At: float  # annual amplitude of monthly average air temperature
    tau: float  # s, time
    tau_y: float  # 315,536,000 s, time of 1 year
    tau_shift: float  # s, time to account for the date of minimum surface temperature

    def __post_init__(self):
        if self.R_ext < 0:
            raise ValueError("R_ext value must be >= 0")
        if self.absorptance < 0:
            raise ValueError("absorptance value must be >= 0")
        if self.eps < 0:
            raise ValueError("eps value must be >= 0")
        if self.At < 0:
            raise ValueError("At value must be >= 0")
        if self.tau < 0:
            raise ValueError("tau value must be >= 0")
        if self.tau_y < 0:
            raise ValueError("tau_y value must be > 0")
        if self.tau_shift < 0:
            raise ValueError("tau_shift value must be > 0")


@dataclass
class EnvironmentalTimeSeries:
    """
    Time series of external environmental inputs.

    Stores the external air temperature and solar irradiance arrays
    used to drive the surface boundary condition over the simulation period.
    Instances should be created via the class methods to ensure input validation.

    Attributes
    ----------
    Tm : float
        Mean annual air temperature [°C].
    T_ext : NDArray[np.float64]
        External air temperature time series [°C].
    SolarRad : NDArray[np.float64]
        Solar irradiance time series [W/m²].
    water_input : NDArray[np.float64] | None = None
        Water input series by irrigation or
        metherological phenomena [m3 / s].
    """

    Tm: float
    T_ext: NDArray[np.float64]
    SolarRad: NDArray[np.float64]
    water_input: NDArray[np.float64] | None = None

    @classmethod
    def from_array(
        cls,
        Tm: float,
        T_ext: NDArray[np.float64],
        SolarRad: NDArray[np.float64],
        water_input: NDArray[np.float64] | None = None,
    ) -> Self:
        """
        Construct an instance from NumPy arrays.

        All values must be finite (no NaN or Inf).

        Parameters
        ----------
        Tm : float
            Mean annual air temperature [°C].
        T_ext : NDArray[np.float64]
            External air temperature time series [°C].
        SolarRad : NDArray[np.float64]
            Solar irradiance time series [W/m²].
        water_input : NDArray[np.float64] | None = None
            Water input series by irrigation or
            metherological phenomena [m3 / s].

        Returns
        -------
        EnvironmentalTimeSeries
            A validated instance populated from the arrays.

        Raises
        ------
        ValueError
            If ``T_ext``, ``SolarRad``, or ``water_input`` contain non-finite values.

        Examples
        --------
        >>> import numpy as np
        >>> T = np.linspace(-5, 25, 8760)
        >>> rad = np.abs(np.sin(np.linspace(0, 2 * np.pi, 8760))) * 600
        >>> env = EnvironmentalTimeSeries.from_array(12.0, T, rad)
        """
        arrays = {"T_ext": T_ext, "SolarRad": SolarRad}
        if water_input is not None:
            arrays["water_input"] = water_input

        invalid = [name for name, arr in arrays.items() if not np.all(np.isfinite(arr))]
        if invalid:
            raise ValueError(f"{', '.join(invalid)} may contain invalid values")

        return cls(Tm, T_ext, SolarRad, water_input)


@dataclass
class ExternalEnvironment:
    """
    Assembled external environment for simulation boundary conditions.

    Combines physical properties and time-series inputs, and derives
    the sky radiation temperature from the external air temperature
    at construction time.

    Attributes
    ----------
    envprops : EnvironmentalProperties
        Static physical and radiative properties of the site.
    envinput : EnvironmentalTimeSeries
        Time-varying environmental inputs (temperature and solar radiation).
    T_sky : NDArray[np.float64]
        Effective sky temperature time series, derived from ``T_ext`` [K].
    boltzmann : float
        Stefan-Boltzmann constant (5.670374419 × 10⁻⁸) [W / (m² K⁴)].

    Examples
    --------
    >>> env = ExternalEnvironment(envprops=props, envinput=series)
    >>> env.T_sky.shape
    (8760,)
    """

    envprops: EnvironmentalProperties
    envinput: EnvironmentalTimeSeries
    T_sky: NDArray[np.float64] = field(init=False)

    boltzmann: float = field(default=5.670374419e-8, init=False)

    def __post_init__(self) -> None:
        # T_sky calculation
        self._sky_temperature_calc()

    def _sky_temperature_calc(self) -> None:
        self.T_sky = 0.0552 * (self.envinput.T_ext + 273.15) ** 1.5
