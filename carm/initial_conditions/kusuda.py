# -*- coding: utf-8 -*-
from numpy.typing import NDArray
import numpy as np


def kusuda_achenbach(ground, borehole, envprops, Tm, timesteps, n_steps) -> tuple[NDArray, NDArray, NDArray]:
    At = envprops.At
    tau = envprops.tau
    tau_y = envprops.tau_y
    tau_shift = envprops.tau_shift
    tau_array = np.arange(tau, tau + n_steps * timesteps, timesteps)

    dz_tot = (
        [0]
        + [ground.dz_sup] * (ground.m_mesh_sup)
        + [ground.dz] * ground.m_mesh
        + [ground.dz_inf] * ground.m_mesh_inf
    )
    T_kusuda = np.zeros((n_steps,
        1 + ground.m_mesh_sup + ground.m_mesh + ground.m_mesh_inf), dtype=np.float64
    )
    z = 0.0

    a_mean = ground.k_mean / (ground.cp_mean * ground.rho_mean)

    for j in range(1 + ground.m_mesh_sup + ground.m_mesh + ground.m_mesh_inf):
        z = z + dz_tot[j] / 2

        T_kusuda[:, j] = Tm - At * np.exp(-z * np.sqrt(np.pi / (tau_y * a_mean))) * np.cos(
            (2.0 * np.pi / tau_y)
            * (tau_array - tau_shift - z / 2.0 * np.sqrt(tau_y / (np.pi * a_mean)))
        )

        z = z + dz_tot[j] / 2.0

    T_sup = T_kusuda[:, : ground.m_mesh_sup + 1]
    T_inf = T_kusuda[:, ground.m_mesh_sup + 1 + ground.m_mesh :]

    T_middle_loc = T_kusuda[:, 
        ground.m_mesh_sup + 1 : ground.m_mesh_sup + 1 + ground.m_mesh
    ]

    T_middle = np.concatenate(
        (
            np.repeat(T_middle_loc, ground.n_mesh, axis = 1),
            np.repeat(T_middle_loc, borehole.n_equations, axis = 1),
        ), axis = 1
    )

    return T_sup, T_middle, T_inf
