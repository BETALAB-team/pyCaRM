# -*- coding: utf-8 -*-
"""
Finite Line Source (FLS) module.

Computes the far-field thermal response of the borehole field using the
Finite Line Source analytical model. Handles both single and multi-borehole
configurations via a precomputed response matrix and a superposition scheme.
"""
from carm import PhysicalModel

from numpy.typing import NDArray

import numpy as np
from scipy.special import erf
from scipy.integrate import quad_vec


class FiniteLineSolution:
    """
    Finite Line Source thermal response for a borehole field.

    Precomputes the FLS response matrix at construction time and uses it
    at each time step to evaluate the borehole wall temperature perturbation
    via load superposition (convolution over the heat load history).

    Attributes
    ----------
    physicalmodel : PhysicalModel
        Full physical model of the BHE system.
    n_steps : int
        Total number of simulation time steps.
    time_hist : NDArray
        Array of cumulative simulation times [s], shape (n_steps,).
    response_matrix : NDArray
        Precomputed FLS response matrix of shape (n_steps + 1, n_bhes, n_bhes).
        Entry ``[t, i, j]`` is the unit thermal response at borehole ``j``
        due to borehole ``i`` up to time step ``t``.
    The numerical implementation of the integral follows the approach used in
    pygfunction (Cimmino, 2018).
    """
    def __init__(
        self,
        *,
        physicalmodel: PhysicalModel,
        n_steps: int,
        time_hist: NDArray,
        fls_mode: str = "sqrt"
    ) -> None:

        self.physicalmodel = physicalmodel
        self.n_steps = n_steps
        self.time_hist = time_hist

        if fls_mode == "sqrt":
            self._response_matrix: NDArray = self._response_matrix_sqrt()
        else:
            self._response_matrix: NDArray = self._response_matrix_continuous()

    @property
    def response_matrix(self) -> NDArray:
        return self._response_matrix

    def _compute_delta_t(self, q_nbhes: NDArray, step: int) -> NDArray:
        # ground conductivity
        k_mean = self.physicalmodel.ground[0].k_mean

        # ground perforation
        L = self.physicalmodel.ground_geom.L

        # ground mesh
        m_mesh = self.physicalmodel.ground_mesh.m_mesh

        # n_bhes
        n_bhes = self.physicalmodel.fieldinput.n_bhes

        dT = np.zeros((n_bhes, m_mesh), dtype=np.float64)
        if step == 0:
            return dT

        dT_j = np.zeros(n_bhes, dtype=np.float64)
        const = 1.0 / (2.0 * np.pi * k_mean * L)

        assert q_nbhes[0:step].shape == (step, n_bhes), "q_nbhes shape must be 2D"

        dT_j = (
            np.einsum(
                "ti,tij -> j",
                q_nbhes[0:step],
                (
                    self._response_matrix[step:0:-1]
                    - self._response_matrix[step - 1 :: -1]
                ),
            )[:, None]
            * const
        )
        dT[:] = dT_j

        return dT

    def _response_matrix_sqrt(self) -> NDArray:
        # ground properties
        k_mean = self.physicalmodel.ground[0].k_mean
        cp_mean = self.physicalmodel.ground[0].cp_mean
        rho_mean = self.physicalmodel.ground[0].rho_mean
        alpha_mean = k_mean / (rho_mean * cp_mean)

        # ground perforation
        L = self.physicalmodel.ground_geom.L

        # nbhes
        n_bhes = self.physicalmodel.fieldinput.n_bhes

        # distance matrix squared
        d = self.physicalmodel.field._distance_matrix

        response_matrix = np.zeros((self.n_steps + 1, n_bhes, n_bhes), dtype=np.float64)

        a = 1 / np.sqrt(4 * alpha_mean * self.time_hist)
        b = np.concatenate(([np.inf], a[:-1]))

        def func(s):
            return s**-2.0 * np.exp(-(d**2.0) * s**2.0) * (
                (s * L) * erf(s * L)
                - (1.0 / np.sqrt(np.pi)) * (1.0 - np.exp(-((s * L) ** 2.0)))
            )

        arrays = [1 / L * quad_vec(func, a_i, b_i)[0] for a_i, b_i in zip(a, b)]
        arrays_stacked = np.stack(arrays, axis=0)

        response_matrix[1:] = np.cumsum(arrays_stacked, axis=0)

        return response_matrix

    def _response_matrix_continuous(self) -> NDArray:
        # ground properties
        k_mean = self.physicalmodel.ground[0].k_mean
        cp_mean = self.physicalmodel.ground[0].cp_mean
        rho_mean = self.physicalmodel.ground[0].rho_mean
        alpha_mean = k_mean / (rho_mean * cp_mean)

        # ground perforation
        L = self.physicalmodel.ground_geom.L

        # nbhes
        n_bhes = self.physicalmodel.fieldinput.n_bhes

        #field parameters
        coords = np.array(self.physicalmodel.fieldinput._borehole_coordinates) #shape (n, 2)
        req = np.array([self.physicalmodel.field.field_dict[j]["req"] for j in range(n_bhes)])
        rb = self.physicalmodel.borehole.D0 / 2.0
        n_theta = 8

        response_matrix = np.zeros((self.n_steps + 1, n_bhes, n_bhes), dtype=np.float64)

        a = 1 / np.sqrt(4 * alpha_mean * self.time_hist)
        b = np.concatenate(([np.inf], a[:-1]))

        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

        x = coords[:, 0]
        y = coords[:, 1]

        dx = x[None, :, None] - x[:, None, None] + req[None, :, None] * np.cos(theta)[None, None, :]
        dy = y[None, :, None] - y[:, None, None] + req[None, :, None] * np.sin(theta)[None, None, :]
        d = np.maximum(np.sqrt(dx**2 + dy**2), rb)

        def func(s):
            return s**-2.0 * np.exp(-(d**2.0) * s**2.0) * (
                (s * L) * erf(s * L)
                - (1.0 / np.sqrt(np.pi)) * (1.0 - np.exp(-((s * L) ** 2.0)))
            )

        arrays = [1 / L * np.mean(quad_vec(func, a_i, b_i)[0], axis=-1)
                for a_i, b_i in zip(a, b)]

        response_matrix[1:] = np.cumsum(np.stack(arrays, axis=0), axis=0)

        return response_matrix
