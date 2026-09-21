# -*- coding: utf-8 -*-
"""
Physical model module.

Assembles the complete physical description of the borehole heat exchanger
system: ground geometry, mesh, borehole type, fluid properties, and field
layout. Handles both single-borehole and multi-borehole configurations.
"""
from dataclasses import dataclass, field
from typing import Sequence

from .properties import GroundProperties, GroundGeometry, GroundMesh
from .properties import SingleUtube, DoubleUtube, Helical, Coaxial
from .fluid import Fluid
from .field_layout import FieldInput, Field


@dataclass
class PhysicalModel:
    """
    Container for the full physical model of the BHE system.

    Combines ground, borehole, fluid, and field layout into a single object.
    On construction, validates the configuration and builds one
    ``GroundProperties`` instance per borehole (with Voronoi-derived ``r_eq``
    in the multi-borehole case).

    Attributes
    ----------
    ground_geom : GroundGeometry
        Geometric parameters of the borehole and surrounding ground.
    ground_mesh : GroundMesh
        Discretization settings for the ground domain.
    borehole : SingleUtube or DoubleUtube or Helical or Coaxial
        Borehole heat exchanger type and its geometric parameters.
    fluid : Fluid
        Thermophysical properties of the heat carrier fluid.
    Tg : float
        Undisturbed ground temperature [°C].
    stratification : Sequence[tuple[float, float, float, float]]
        Ground layer stratification as a sequence of
        ``(z_top, z_bot, k, rho_cp)`` tuples.
    ground : list[GroundProperties]
        One ``GroundProperties`` instance per borehole, populated at
        construction time.
    fieldinput : FieldInput or None
        Field layout object. If ``None`` or ``n_bhes == 1``, single-borehole
        mode is used.
    field : Field
        Voronoi field decomposition. Present only in multi-borehole mode.
    """
    ground_geom: GroundGeometry
    ground_mesh: GroundMesh
    borehole: SingleUtube | DoubleUtube | Helical | Coaxial
    fluid: Fluid

    Tg: float
    stratification: Sequence[tuple[float, float, float, float]]

    ground: list = field(default_factory=list, init=False)
    fieldinput: FieldInput | None = None

    def __post_init__(self) -> None:
        rn = self.ground_geom.rn

        if self.fieldinput is None or self.fieldinput.n_bhes == 1:

            if rn is None:
                raise ValueError("Single borehole: rn must be given as input")

            gr_p = GroundProperties(
                geom=self.ground_geom,
                mesh=self.ground_mesh,
                Tg=self.Tg,
                stratification=self.stratification,
            )

            self.ground.append(gr_p)

        else:
            if rn is not None:
                raise ValueError("Multi-borehole: rn must not be defined")

            self.field = Field(fieldinput=self.fieldinput)

            for j in range(self.fieldinput.n_bhes):
                rn_j = self.field.field_dict[j]["req"]

                geom_j = GroundGeometry(
                    rn=rn_j,
                    D0=self.ground_geom.D0,
                    L=self.ground_geom.L,
                    L_sup=self.ground_geom.L_sup,
                    L_inf=self.ground_geom.L_inf,
                )

                gr_p = GroundProperties(
                    geom=geom_j,
                    mesh=self.ground_mesh,
                    Tg=self.Tg,
                    stratification=self.stratification,
                )

                self.ground.append(gr_p)

    def _get_temperatures(self, state, j, use_old = False):

        ngs = self.ground[0].m_mesh_sup + 1
        ng = self.ground[0].n_mesh * self.ground[0].m_mesh
        nb = self.borehole.n_equations * self.borehole.m_mesh

        T = state.T_state if not use_old else state.T_old

        T_borehole = T[j, (ngs + ng) : (ngs + ng + nb)]
        T_ground = T[j, ngs : (ngs + ng)]
        T_ground_sup = T[j, 1:(ngs)]
        T_ground_inf = T[j, (ngs + ng + nb) :]
        Ts = T[j, 0]

        return T_borehole, T_ground, T_ground_sup, T_ground_inf, Ts
