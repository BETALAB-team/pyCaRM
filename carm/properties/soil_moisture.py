"""
Soil moisture module.

Empirical equations are here implemented to account for water content
in porous media. This module provides insights on borehole
properties variability against water content, tracking a per-timestep
water volume balance (irrigation input, drainage loss, evaporation loss)
around the borehole.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class SoilMoisture:
    """
    Soil moisture and thermophysical properties module.

    Computes borehole thermophysical properties (thermal conductivity,
    specific heat capacity, density) as a function of volumetric water content,
    using Chung-Horton (1987) for thermal conductivity, de Vries (1963) for
    volumetric heat capacity, and a mass-weighted average for density.

    At each timestep, the water volume stored around the borehole is updated
    from a water balance: irrigation input increases it, while gravity
    drainage (Brooks-Corey, 1964, unsaturated hydraulic conductivity) and
    evaporation (driven by the exchanged thermal power) deplete it. The
    resulting volume is converted back to a volumetric water content and fed
    into the property correlations above.

    Attributes
    ----------
    w_rho : float
        Density of water [kg/m³].
    w_latent : float
        Latent heat of vaporization of water [J/kg].
    SOIL_PARAMS : dict
        Tabulated parameters for sand, loam, and clay soil types.
        Each entry contains: b1, b2, b3 (Chung-Horton), theta_s, theta_r
        (Rawls et al.), xs (solid volume fraction), x0 (organic matter
        fraction), lambda (Brooks-Corey pore-size distribution index),
        Ks (saturated hydraulic conductivity [m/s]).
    water_input : NDArray
        Water flux time series [m/s].
    rho_dry : float
        Dry soil density [kg/m³].
    b1_loc : float
        Chung-Horton parameter b1 for the selected soil type [W/(m K)].
    b2_loc : float
        Chung-Horton parameter b2 for the selected soil type [W/(m K)].
    b3_loc : float
        Chung-Horton parameter b3 for the selected soil type [W/(m K)].
    theta_s_loc : float
        Saturated volumetric water content for the selected soil type [-].
    theta_r_loc : float
        Residual volumetric water content for the selected soil type [-].
    xs_loc : float
        Solid volume fraction for the selected soil type [-].
    x0_loc : float
        Organic matter volume fraction for the selected soil type [-].
    lambda_loc : float
        Brooks-Corey pore-size distribution index for the selected soil
        type [-]. Governs how steeply unsaturated hydraulic conductivity
        (and thus drainage loss) drops as water content decreases.
    Ks_loc : float
        Saturated hydraulic conductivity for the selected soil type [m/s].
    Wvol_prev : float | None
        Water volume at the previous timestep [m³]. ``None`` before the
        first call to :meth:`_properties_calculation`, when it is
        initialized to the residual water volume (``theta_r_loc * V``).
    Wvol_r : float | None
        Water volume at the current timestep [m³], clipped between the
        residual (``theta_r_loc * V``) and saturated (``theta_s_loc * V``)
        volumes. ``None`` until the first call to
        :meth:`_properties_calculation`.
    Wvol_loss : float
        Water volume lost by gravity drainage at the current timestep [m³],
        computed from the Brooks-Corey unsaturated hydraulic conductivity.
    Wvol_evap : float
        Water volume lost by evaporation at the current timestep [m³].
    W_content : float
        Volumetric water content at the current timestep [-].
    """

    #loss_factor = 0.1 waiting fo further analyses
    w_rho = 1000.0
    w_latent = 2250000.0
    SOIL_PARAMS = {
        "sand": {
            "b1": 0.228,
            "b2": -2.406,
            "b3": 4.909,
            "theta_s": 0.417,
            "theta_r": 0.0295, #0.020
            "xs": 1 - 0.417 - 0.012,
            "x0": 0.012,
            "lambda": 0.694,
            "Ks": 5.83 * 10**-5,
        },
        "loam": {
            "b1": 0.243,
            "b2": 0.393,
            "b3": 1.534,
            "theta_s": 0.434,
            "theta_r": 0.027,
            "xs": 1 - 0.434 - 0.018,
            "x0": 0.018,
            "lambda": 0.252,
            "Ks": 3.67 * 10**-6,
        },
        "clay": {
            "b1": -0.197,
            "b2": -0.962,
            "b3": 2.521,
            "theta_s": 0.385,
            "theta_r": 0.090,
            "xs": 1 - 0.385 - 0.024,
            "x0": 0.024,
            "lambda": 0.165,
            "Ks": 1.67 * 10**-7,
        },
    }

    def __init__(
        self,
        *,
        water_input: NDArray,
        rho_dry: float,
        soil_type: str,
    ) -> None:

        self.water_input = water_input
        self.rho_dry = rho_dry

        self.Wvol_prev = None
        self.Wvol_r = None
        self.Wvol_loss = 0.0
        self.Wvol_evap = 0.0

        if soil_type not in list(self.SOIL_PARAMS.keys()):
            raise ValueError(
                "Soil  type must match one between Clay, Loam, and Sand soil types."
            )

        self.b1_loc = self.SOIL_PARAMS[soil_type]["b1"]
        self.b2_loc = self.SOIL_PARAMS[soil_type]["b2"]
        self.b3_loc = self.SOIL_PARAMS[soil_type]["b3"]
        self.theta_s_loc = self.SOIL_PARAMS[soil_type]["theta_s"]
        self.theta_r_loc = self.SOIL_PARAMS[soil_type]["theta_r"]
        self.xs_loc = self.SOIL_PARAMS[soil_type]["xs"]
        self.x0_loc = self.SOIL_PARAMS[soil_type]["x0"]
        self.lambda_loc = self.SOIL_PARAMS[soil_type]["lambda"]
        self.Ks_loc = self.SOIL_PARAMS[soil_type]["Ks"]

    def _properties_calculation(
        self,
        step: int,
        timesteps: int,
        V: float,
        L: float,
        A_irr: float,
        q: float,
    ) -> Tuple[float, float, float]:
        """
        Compute thermophysical properties from volumetric water content at a given timestep.

        Updates the water volume balance and computes thermal conductivity via
        Chung-Horton (1987), volumetric heat capacity via de Vries (1963), and
        density via mass-weighted average.

        The water balance itself combines three terms over the timestep:
        irrigation input (``water_input * A_irr * timesteps``), gravity
        drainage loss (``Wvol_loss``, from the Brooks-Corey (1964) unsaturated
        hydraulic conductivity at the current water content), and evaporation
        loss (``Wvol_evap``, from the exchanged thermal power ``q``). The
        result is clipped between the residual and saturated water volumes
        before being converted back to a volumetric water content.

        Parameters
        ----------
        step : int
            Current simulation timestep index.
        timesteps : float
            Duration of each timestep [s].
        V : float
            Reference soil volume [m³].
        L : float
            Borehole length [m], used as the drainage path length in the
            Brooks-Corey unsaturated hydraulic conductivity term.
        A_irr : float
            Irrigation pipe area. Surface area over which water input is applied [m²].
        q : float
            Thermal power exchanged by the system [W]. Negative in heat extraction,
            positive in heat injection. However, it is correlated only with water
            injection. Hence, the cooling case is the only one supported.

        Returns
        -------
        k : float
            Thermal conductivity [W/(m K)].
        cp : float
            Volumetric heat capacity [J/(kg K)].
        rho : float
            Density [kg/m³].
        """

        f_k = lambda wr: self.b1_loc + self.b2_loc * wr + self.b3_loc * wr**0.5
        f_cp = (
            lambda wr: 1.92 * 10**6 * self.xs_loc + 2.51 * 10**6 * self.x0_loc + 4.18 * 10**6 * wr
        )
        f_rho = lambda wr: wr * self.w_rho + self.rho_dry
        # Brooks-Corey (1964) unsaturated hydraulic conductivity, driving gravity drainage.
        f_hydr_k = lambda theta: self.Ks_loc * (
            (theta - self.theta_r_loc) / (self.theta_s_loc - self.theta_r_loc)
        ) ** (3 + 2 / self.lambda_loc)

        if self.Wvol_prev is None:
            self.Wvol_prev = self.theta_r_loc * V
            self.Wvol_r = self.theta_r_loc * V

        self.Wvol_loss = f_hydr_k(self.Wvol_r / V) * (V / L) * timesteps
        self.Wvol_evap = max(((q * timesteps) / self.w_latent) / self.w_rho, 0)

        self.Wvol_r = np.clip(
            self.Wvol_prev
            + self.water_input[step] * A_irr * timesteps
            - self.Wvol_loss
            - self.Wvol_evap,
            self.theta_r_loc * V,
            self.theta_s_loc * V,
        )

        assert self.Wvol_r >= 0

        self.Wvol_prev = self.Wvol_r

        self.W_content = self.Wvol_r / V

        self.rho = f_rho(self.W_content)
        self.k = f_k(self.W_content)
        self.cp = f_cp(self.W_content) / self.rho

        return self.k, self.cp, self.rho
