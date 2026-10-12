# -*- coding: utf-8 -*-
"""
Finite Line Source (FLS) module.

Computes the far-field thermal response of the borehole field using the
Finite Line Source analytical model. Handles both single and multi-borehole
configurations via precomputed response increments and a superposition scheme.

The FLS uses the real part (the borehole as a line source in an infinite
medium) and, by default, the image part (a mirrored sink above the ground
surface). The image keeps the FLS perturbation at zero on the surface, which
then stays at the undisturbed (Kusuda) temperature.
"""
from carm import PhysicalModel

from numpy.typing import NDArray
from pathlib import Path
from typing import Callable

import contextlib
import hashlib
import json
import os
import traceback
import numpy as np
from scipy.special import erf
from scipy.integrate import quad_vec


class FiniteLineSolution:
    """
    Finite Line Source thermal response for a borehole field.

    Precomputes the FLS response increments at construction time and uses
    them at each time step to evaluate the borehole wall temperature
    perturbation via load superposition (convolution over the heat load
    history). Only the increments are kept in memory; the cumulative response
    matrix is rebuilt on demand.

    Parameters
    ----------
    physicalmodel : PhysicalModel
        Full physical model of the BHE system.
    n_steps : int
        Total number of simulation time steps.
    time_hist : NDArray
        Array of cumulative simulation times [s], shape (n_steps,).
    fls_mode : str
        ``'sqrt'`` or ``'continuous'``, see :doc:`fls_methods`.
    fls_image : bool
        ``True`` (default) adds the image part, a sink mirrored above the
        ground surface, so the FLS perturbation is zero at the surface. The BHE
        top is buried at depth ``D = ground_geom.L_sup``. ``False`` uses the
        real part only (a borehole in an infinite medium).
    cache_dir : str, Path or None
        Folder of the on-disk increments cache. ``None`` (default) computes
        the increments in memory and writes nothing. When set, the increments
        are looked up in the folder by a key built from the inputs they depend
        on (see Notes), served memory-mapped from a hit, or computed, written
        and then memory-mapped on a miss. The time grid must be uniform
        (``time_hist = dt, 2 dt, ...``).

    Attributes
    ----------
    physicalmodel : PhysicalModel
        Full physical model of the BHE system.
    n_steps : int
        Total number of simulation time steps.
    time_hist : NDArray
        Array of cumulative simulation times [s], shape (n_steps,).
    response_matrix : NDArray
        FLS response matrix of shape (n_steps + 1, n_bhes, n_bhes), rebuilt
        from the increments on each access (zeros row + cumulative sum).
        Entry ``[t, i, j]`` is the unit thermal response at borehole ``j``
        due to borehole ``i`` up to time step ``t``. Not used by the solver.

    Notes
    -----
    With ``Ierf(x) = x erf(x) - (1 - exp(-x^2)) / sqrt(pi)``, borehole length
    ``H = L``, buried depth ``D = L_sup`` and ``s`` from ``1 / sqrt(4 alpha t)``
    to infinity, the response is the integral of::

        exp(-d^2 s^2) / (2 H s^2) * (real + image)
        real  = 2 Ierf(H s)
        image = 2 Ierf((2 D + H) s) - Ierf(2 D s) - Ierf((2 D + 2 H) s)

    where ``image`` (always <= 0) is dropped when ``fls_image=False``. ``d``
    is ``Field._distance_matrix`` in ``'sqrt'`` mode and the receiver wall
    points in ``'continuous'`` mode.

    The increments ``dR[k] = R[k + 1] - R[k]``, shape (n_steps, n_bhes, n_bhes),
    are the FLS integrals over each time interval. The penalty at ``step`` is
    ``sum_t q[t] @ (R[step - t] - R[step - 1 - t])``, i.e. ``q[t]`` paired with
    ``dR[step - 1 - t]``, evaluated as one matrix-vector product.

    **Increments cache.** The cache file is ``fls_<key>.npy`` (float64,
    shape (rows, n_bhes, n_bhes)) in ``cache_dir``, next to a ``.json``
    sidecar listing the key inputs. The key is a hash of everything the
    increments depend on: ``fls_mode``, ``fls_image``, the time step ``dt``,
    the ground means k, rho, cp, the borehole length L, the buried depth
    ``D = L_sup``, the borehole coordinates, the equivalent radius ``req`` of
    every borehole, the borehole radius ``rb``, ``n_theta`` and a cache-format
    version. Flows, temperatures, weather and
    the number of steps are not part of the key.

    Because the time grid is ``k * dt``, the increments of a shorter run are
    the first rows of a longer run's increments: a cached file with at least
    ``n_steps`` rows serves the run by slicing. A shorter file is recomputed
    and replaced. Files are written atomically (temporary file, then rename)
    and opened with ``np.load(..., mmap_mode="r")``, so parallel runs on the
    same field share one copy through the OS page cache. On a miss the
    increments are filled straight into the temporary file, so they are never
    held in RAM as a whole.

    The numerical implementation of the integral, real and image parts,
    follows the approach used in pygfunction (Cimmino, 2018).
    """
    CACHE_FORMAT_VERSION = 2
    N_THETA = 8  # receiver wall points of the continuous mode

    def __init__(
        self,
        *,
        physicalmodel: PhysicalModel,
        n_steps: int,
        time_hist: NDArray,
        fls_mode: str = "sqrt",
        fls_image: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:

        self.physicalmodel = physicalmodel
        self.n_steps = n_steps
        self.time_hist = time_hist
        self.fls_mode = fls_mode
        self.fls_image = fls_image

        n_bhes = self.physicalmodel.fieldinput.n_bhes
        if cache_dir is None:
            self._response_increments: NDArray = np.empty(
                (n_steps, n_bhes, n_bhes), dtype=np.float64
            )
            self._fill_increments(self._response_increments)
        else:
            self._response_increments: NDArray = self._cached_increments(Path(cache_dir))

    @property
    def response_matrix(self) -> NDArray:
        dR = self._response_increments
        response_matrix = np.zeros((dR.shape[0] + 1, *dR.shape[1:]), dtype=np.float64)
        np.cumsum(dR, axis=0, out=response_matrix[1:])
        return response_matrix

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

        const = 1.0 / (2.0 * np.pi * k_mean * L)

        assert q_nbhes[0:step].shape == (step, n_bhes), "q_nbhes shape must be 2D"

        # q[t] pairs with dR[step - 1 - t]: reverse q and flatten (t, i) on both sides
        dT_j = (
            q_nbhes[step - 1 :: -1].reshape(-1)
            @ self._response_increments[:step].reshape(-1, n_bhes)
        )[:, None] * const
        dT[:] = dT_j

        return dT

    def _fill_increments(self, out: NDArray) -> None:
        if self.fls_mode == "sqrt":
            self._fill_increments_sqrt(out)
        else:
            self._fill_increments_continuous(out)

    def _cache_key_inputs(self) -> dict:
        """Everything the increments depend on, as JSON-serialisable values."""
        dt = float(self.time_hist[0])
        if not np.allclose(np.diff(self.time_hist), dt, rtol=1e-9, atol=0.0):
            raise ValueError(
                "The FLS increments cache needs a uniform time grid (time_hist = dt, 2 dt, ...)"
            )
        ground = self.physicalmodel.ground[0]
        n_bhes = self.physicalmodel.fieldinput.n_bhes
        return {
            "format_version": self.CACHE_FORMAT_VERSION,
            "fls_mode": self.fls_mode,
            "fls_image": self.fls_image,
            "dt": dt,
            "k": float(ground.k_mean),
            "rho": float(ground.rho_mean),
            "cp": float(ground.cp_mean),
            "L": float(self.physicalmodel.ground_geom.L),
            "D": float(self.physicalmodel.ground_geom.L_sup),
            "coordinates": [
                [float(x), float(y)]
                for x, y in self.physicalmodel.fieldinput._borehole_coordinates
            ],
            "req": [
                float(self.physicalmodel.field.field_dict[j]["req"]) for j in range(n_bhes)
            ],
            "rb": float(self.physicalmodel.borehole.D0 / 2.0),
            "n_theta": self.N_THETA,
        }

    def _cached_increments(self, cache_dir: Path) -> NDArray:
        inputs = self._cache_key_inputs()
        key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]
        path = cache_dir / f"fls_{key}.npy"

        if path.exists():
            cached = np.load(path, mmap_mode="r")
            if cached.shape[0] >= self.n_steps:
                return cached[: self.n_steps]
            del cached  # too short: release the map before replacing the file

        # miss: fill a temporary .npy on disk, then rename it into place
        cache_dir.mkdir(parents=True, exist_ok=True)
        n_bhes = self.physicalmodel.fieldinput.n_bhes
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp")
        out = None
        try:
            out = np.lib.format.open_memmap(
                tmp, mode="w+", dtype=np.float64, shape=(self.n_steps, n_bhes, n_bhes)
            )
            self._fill_increments(out)
            out.flush()
            out = None  # close the map before the rename
            try:
                os.replace(tmp, path)
            except PermissionError as err:
                raise PermissionError(
                    f"Cannot replace the cached FLS increments {path}: the file is open "
                    "(memory-mapped) in this or another process. Close those simulations, "
                    "or build the longest run first so that shorter runs only read it."
                ) from err
        except BaseException as err:
            # close the map (also referenced by the failed frames) so the
            # temporary file can be removed, on Windows too
            out = None
            traceback.clear_frames(err.__traceback__)
            with contextlib.suppress(OSError):
                tmp.unlink()
            raise

        sidecar = {"key": key, "n_steps": self.n_steps, **inputs}
        tmp_json = path.with_name(f"{path.stem}.{os.getpid()}.json.tmp")
        tmp_json.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        os.replace(tmp_json, path.with_suffix(".json"))

        return np.load(path, mmap_mode="r")

    def _integrand(self, d: NDArray) -> Callable[[float], NDArray]:
        """FLS integrand in ``s`` for the distances ``d``, real part plus the
        image part when ``fls_image`` (see Notes). ``1 / L`` is applied by the
        caller."""
        L = self.physicalmodel.ground_geom.L
        D = self.physicalmodel.ground_geom.L_sup

        def ierf(x):
            return x * erf(x) - (1.0 / np.sqrt(np.pi)) * (1.0 - np.exp(-(x**2.0)))

        def func(s):
            parts = ierf(s * L)
            if self.fls_image:
                parts = parts + (
                    ierf((2.0 * D + L) * s)
                    - 0.5 * (ierf(2.0 * D * s) + ierf((2.0 * D + 2.0 * L) * s))
                )
            return s**-2.0 * np.exp(-(d**2.0) * s**2.0) * parts

        return func

    def _fill_increments_sqrt(self, out: NDArray) -> None:
        # ground properties
        k_mean = self.physicalmodel.ground[0].k_mean
        cp_mean = self.physicalmodel.ground[0].cp_mean
        rho_mean = self.physicalmodel.ground[0].rho_mean
        alpha_mean = k_mean / (rho_mean * cp_mean)

        # ground perforation
        L = self.physicalmodel.ground_geom.L

        # distance matrix squared
        d = self.physicalmodel.field._distance_matrix

        a = 1 / np.sqrt(4 * alpha_mean * self.time_hist)
        b = np.concatenate(([np.inf], a[:-1]))

        func = self._integrand(d)

        for k, (a_k, b_k) in enumerate(zip(a, b)):
            out[k] = 1 / L * quad_vec(func, a_k, b_k)[0]

    def _fill_increments_continuous(self, out: NDArray) -> None:
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
        n_theta = self.N_THETA

        a = 1 / np.sqrt(4 * alpha_mean * self.time_hist)
        b = np.concatenate(([np.inf], a[:-1]))

        theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)

        x = coords[:, 0]
        y = coords[:, 1]

        dx = x[None, :, None] - x[:, None, None] + req[None, :, None] * np.cos(theta)[None, None, :]
        dy = y[None, :, None] - y[:, None, None] + req[None, :, None] * np.sin(theta)[None, None, :]
        d = np.maximum(np.sqrt(dx**2 + dy**2), rb)

        func = self._integrand(d)

        for k, (a_k, b_k) in enumerate(zip(a, b)):
            out[k] = 1 / L * np.mean(quad_vec(func, a_k, b_k)[0], axis=-1)
