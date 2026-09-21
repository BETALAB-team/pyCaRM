# -*- coding: utf-8 -*-
"""
Single U-tube BHE matrix assembly module.

Builds the local coefficient matrix and right-hand side vector for the
single U-tube BHE configuration, used internally by the global assembler.
"""
import numpy as np
from scipy.sparse import coo_matrix


def build_coefficient_matrix_bhe_single_u_tube(model, gr_p, timesteps, mw_tot_j):

    bh_p = model.borehole
    n = bh_p.n_equations
    m = bh_p.m_mesh
    fluid = model.fluid
    id_core = bh_p.id_core
    id_shell = bh_p.id_shell
    R_axial_core = bh_p.R_axial_core
    R_axial_shell = bh_p.R_axial_shell
    R_ground = gr_p.R_ground

    if mw_tot_j == 0:
        R_fluid = bh_p.R_pipes
    else:
        R_fluid = bh_p._borehole_resistances(mw_tot_j)

    rows = []
    columns = []
    data = []

    for j in range(m):
        # ----------------------------------------------------------------------
        # HEAT EXCHANGER GROUP OF EQUATIONS
        # ----------------------------------------------------------------------
        rows.extend([j * n + id_shell] * 3)
        columns.append(j * n + id_shell)
        columns.append(j * n + 2)
        columns.append(j * n + 3)
        data.append(
            -1 / bh_p.Rp0_dz
            - 1 / bh_p.Rp0_dz
            - bh_p.C_shell[j, 0] / timesteps
            - 1 / R_ground[j, 0]
        )  # shell
        data.append(1 / (bh_p.Rp0_dz))
        data.append(1 / (bh_p.Rp0_dz))

        rows.extend([j * n + id_core] * 3)
        columns.append(j * n + id_core)
        columns.append(j * n + 2)
        columns.append(j * n + 3)
        data.append(
            -1 / (0.5 * bh_p.RppB_dz)
            - 1 / (0.5 * bh_p.RppB_dz)
            - bh_p.C_core[j, 0] / timesteps
        )  # core
        data.append(1 / (0.5 * bh_p.RppB_dz))
        data.append(1 / (0.5 * bh_p.RppB_dz))

        rows.extend([j * n + 2] * 4)
        columns.append(j * n + 0)
        columns.append(j * n + 1)
        columns.append(j * n + 2)
        columns.append(j * n + bh_p.id_inlet)
        data.append(1 / (bh_p.Rp0_dz))  # pipe 1
        data.append(1 / (0.5 * bh_p.RppB_dz))
        data.append(-1 / bh_p.Rp0_dz - 1 / (0.5 * bh_p.RppB_dz) - 1 / R_fluid)
        data.append(1 / R_fluid)

        rows.extend([j * n + 3] * 4)
        columns.append(j * n + 0)
        columns.append(j * n + 1)
        columns.append(j * n + 3)
        columns.append(j * n + bh_p.id_outlet)
        data.append(1 / bh_p.Rp0_dz)  # pipe 2
        data.append(1 / (0.5 * bh_p.RppB_dz))
        data.append(-1 / bh_p.Rp0_dz - 1 / (0.5 * bh_p.RppB_dz) - 1 / R_fluid)
        data.append(1 / R_fluid)

        rows.extend([j * n + bh_p.id_inlet] * 2)
        columns.append(j * n + 2)
        columns.append(j * n + bh_p.id_inlet)
        data.append(1 / (R_fluid))  # fluid 1
        data.append(-mw_tot_j * fluid.cp_w - 1 / (R_fluid) - bh_p.C_fluid / (timesteps))

        rows.extend([j * n + bh_p.id_outlet] * 2)
        columns.append(j * n + 3)
        columns.append(j * n + bh_p.id_outlet)
        data.append(1 / R_fluid)  # fluid 2
        data.append(-mw_tot_j * fluid.cp_w - 1 / R_fluid - bh_p.C_fluid / timesteps)

        # ----------------------------------------------------------------------
        # AXIAL COUPLING TERMS FOR SHELL AND CORE
        # ----------------------------------------------------------------------

        t = j * n + id_shell

        if id_core is not None:
            k = j * n + id_core
            rows.append(k)
            columns.append(k)

            if j == 0:
                data.append(
                    -1 / (0.5 * R_axial_core[j, 0] + 0.5 * R_axial_core[j + 1, 0])
                )  # core manca termine precedente

                rows.append(k)
                columns.append(k + n)
                data.append(
                    1 / (0.5 * R_axial_core[j + 1, 0] + 0.5 * R_axial_core[j, 0])
                )  # diag

            elif j == (m - 1):
                data.append(
                    -1 / (0.5 * R_axial_core[j, 0] + 0.5 * R_axial_core[j - 1, 0])
                )  # core manca termine successivo

                rows.append(k)
                columns.append(k - n)
                data.append(
                    1 / (0.5 * R_axial_core[j - 1, 0] + 0.5 * R_axial_core[j, 0])
                )  # diag

            else:
                data.append(
                    -1 / (0.5 * R_axial_core[j, 0] + 0.5 * R_axial_core[j - 1, 0])
                    - 1 / (0.5 * R_axial_core[j, 0] + 0.5 * R_axial_core[j + 1, 0])
                )  # core

                rows.append(k)
                columns.append(k - n)
                data.append(
                    1 / (0.5 * R_axial_core[j - 1, 0] + 0.5 * R_axial_core[j, 0])
                )  # diag

                rows.append(k)
                columns.append(k + n)
                data.append(
                    1 / (0.5 * R_axial_core[j + 1, 0] + 0.5 * R_axial_core[j, 0])
                )  # diag

        # per shell
        rows.append(t)
        columns.append(t)

        if j == 0:
            data.append(
                -1 / (0.5 * R_axial_shell[j, 0] + 0.5 * R_axial_shell[j + 1, 0])
            )  # shell manca termine prec.

            rows.append(t)
            columns.append(t + n)
            data.append(
                1 / (0.5 * R_axial_shell[j + 1, 0] + 0.5 * R_axial_shell[j, 0])
            )  # diag

        elif j == (m - 1):
            data.append(
                -1 / (0.5 * R_axial_shell[j, 0] + 0.5 * R_axial_shell[j - 1, 0])
            )  # shell manca termine succ.

            rows.append(t)
            columns.append(t - n)
            data.append(
                1 / (0.5 * R_axial_shell[j - 1, 0] + 0.5 * R_axial_shell[j, 0])
            )  # diag

        else:
            data.append(
                -1 / (0.5 * R_axial_shell[j, 0] + 0.5 * R_axial_shell[j - 1, 0])
                - 1 / (0.5 * R_axial_shell[j, 0] + 0.5 * R_axial_shell[j + 1, 0])
            )  # shell

            rows.append(t)
            columns.append(t - n)
            data.append(
                1 / (0.5 * R_axial_shell[j - 1, 0] + 0.5 * R_axial_shell[j, 0])
            )  # diag

            rows.append(t)
            columns.append(t + n)
            data.append(
                1 / (0.5 * R_axial_shell[j + 1, 0] + 0.5 * R_axial_shell[j, 0])
            )  # diag

    N = n * m

    # aggiungo termini di portata
    for j in range(m):
        if j != 0:
            rows.append(j * n + bh_p.id_inlet)
            columns.append((j - 1) * n + bh_p.id_inlet)
            data.append(mw_tot_j * fluid.cp_w)

        if j == bh_p.m_mesh - 1:
            rows.append(j * n + bh_p.id_outlet)
            columns.append(j * n + bh_p.id_inlet)
            data.append(mw_tot_j * fluid.cp_w)

        else:
            rows.append(j * n + bh_p.id_outlet)
            columns.append((j + 1) * n + bh_p.id_outlet)
            data.append(mw_tot_j * fluid.cp_w)

    A_bhe_single = coo_matrix(
        (data, (rows, columns)), shape=(N, N), dtype=np.float64
    )
    A_bhe_single = A_bhe_single.tocsc()

    return A_bhe_single


def build_rhs_single_u_tube(model, gr_p, timesteps, T_old_borehole, mw_tot_j, Tf1_j):

    bh_p = model.borehole
    fluid = model.fluid
    n = bh_p.n_equations

    b_bhe_single = np.zeros((n * bh_p.m_mesh))

    for j in range(bh_p.m_mesh):

        b_bhe_single[j * n + bh_p.id_shell] = (
            -bh_p.C_shell[j, 0] * T_old_borehole[j * n + bh_p.id_shell] / (timesteps)
        )
        b_bhe_single[j * n + bh_p.id_core] = (
            -bh_p.C_core[j, 0] * T_old_borehole[j * n + bh_p.id_core] / (timesteps)
        )

        b_bhe_single[j * n + bh_p.id_inlet] = (
            -bh_p.C_fluid * T_old_borehole[j * n + bh_p.id_inlet] / (timesteps)
        )  # fluid 1
        b_bhe_single[j * n + bh_p.id_outlet] = (
            -bh_p.C_fluid * T_old_borehole[j * n + bh_p.id_outlet] / (timesteps)
        )  # fluid 2

    if mw_tot_j != 0:
        b_bhe_single[bh_p.id_inlet] -= mw_tot_j * fluid.cp_w * Tf1_j

    return b_bhe_single
