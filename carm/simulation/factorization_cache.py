# -*- coding: utf-8 -*-
"""
Factorization cache for the BHE simulation.

Per-run, in-memory store of the boreholes' system matrices and their ``splu``
factorizations. A matrix is built and factorized only for a key not seen
before, so a mass flow that returns to an earlier value reuses its
factorization, and boreholes with the same ``rn`` share one.
"""

from typing import Callable

import numpy as np
from scipy.sparse.linalg import splu

from ..matrix import build_global_matrix


class FactorizationCache:
    """
    Factorized system matrices keyed by ``(rn, mw, adiabatic)``, where ``rn``
    is ``model.ground[j].rn``.

    Parameters
    ----------
    model : PhysicalModel
        Provides ``model.ground[j]`` for the builder.
    env : ExternalEnvironment
        Passed through to the builder.
    timesteps : float
        Passed through to the builder.
    builder : Callable
        Called as ``builder(model, model.ground[j], env, timesteps, mw,
        adiabatic)``. Defaults to ``build_global_matrix``.
    """

    def __init__(
        self,
        model,
        env,
        timesteps,
        builder: Callable = build_global_matrix,
    ) -> None:
        self._model = model
        self._env = env
        self._timesteps = timesteps
        self._builder = builder
        self._entries: dict[int, dict] = {}
        self._fingerprints: dict[int, bytes] = {}

    def _fingerprint(self) -> bytes:
        bh = self._model.borehole
        return bh.k0.tobytes() + bh.cp_0.tobytes() + bh.rho_0.tobytes()

    def get(self, j: int, mw: float, adiabatic: bool):
        """Return ``(A, lu)`` for borehole ``j``, building it on a miss.

        If the borehole's material properties changed since the entries of
        ``j``'s ``rn`` were built, those entries are dropped first: property
        values never come back bit-for-bit, so they could not be found again.
        """
        # Boreholes differ only by their ground's rn, so equal rn means equal
        # ground and the matrix can be shared. When ground properties vary per
        # borehole, the key must also include the ground property arrays.
        rn = self._model.ground[j].rn
        fingerprint = self._fingerprint()
        if self._fingerprints.get(rn) != fingerprint:
            self._entries[rn] = {}
            self._fingerprints[rn] = fingerprint

        entries = self._entries[rn]
        key = (mw, adiabatic)
        if key not in entries:
            A = self._builder(
                self._model, self._model.ground[j], self._env, self._timesteps,
                mw, adiabatic,
            )
            if not np.all(np.isfinite(A.data)):
                raise ValueError("A contains NaN or Inf")
            entries[key] = (A, splu(A))
        return entries[key]
