# -*- coding: utf-8 -*-
"""
Borehole heat exchanger types module.

Defines the geometry, mesh, thermal properties, and fluid dynamics for
all supported BHE configurations: single U-tube, double U-tube, coaxial,
and helical pipes. The class hierarchy is:

    BoreholeProperties
    ├── Utube
    │   ├── SingleUtube
    │   └── DoubleUtube
    ├── Coaxial
    └── Helical
"""
from dataclasses import dataclass
from typing import Sequence
from numpy.typing import NDArray

import numpy as np

from ..fluid import Fluid


@dataclass(frozen=True, slots=True)
class BoreholeGeometry:
    """
    Geometric parameters of the borehole.
    D_irrigation and perf_fraction parameters must be set as None if
        grout variable properties are not taken into account.
    
    Note: the irrigation system is feasible only with shallow helical
    heat exchangers.

    Attributes
    ----------
    Lbore : float
        Active borehole length [m].
    D0 : float
        Borehole diameter [m].
    D_irrigation: None | float = None
        Irrigation pipe diameter [m].
    perf_fraction: None | float = None
        Irrigation pipe perforation fraction [-].

    """

    Lbore: float  # m
    D0: float  # m
    D_irrigation: None | float = None
    perf_fraction: None | float = None

    def __post_init__(self) -> None:
        if self.Lbore <= 0:
            raise ValueError("Lbore must be > 0")
        if self.D0 <= 0:
            raise ValueError("D0 must be > 0")
        if (self.D_irrigation is None) != (self.perf_fraction is None):
            raise ValueError("D_irrigation and perf_fraction must be both None or float values")
        if self.D_irrigation is not None:
            if (self.D_irrigation <= 0.0) | (self.D_irrigation > self.D0):
                raise ValueError("Invalid irrigation pipe diameter")
            elif (self.perf_fraction < 0.0) or (self.perf_fraction >= 1.0):
                raise ValueError("Invalid perforation percentage")
        

    @property
    def r0(self) -> float:
        return self.D0 / 2.0


@dataclass(frozen=True, slots=True)
class BoreholeMesh:
    """
    Axial discretization of the borehole.

    Attributes
    ----------
    m_mesh : int
        Number of axial mesh elements along the borehole.
    """

    m_mesh: int

    def __post_init__(self) -> None:
        if self.m_mesh <= 0:
            raise ValueError("Invalid axial mesh, m_mesh must be > 0")


@dataclass
class BoreholeThermalProperties:
    """
    Thermal properties of the borehole filling material (grout).
    The user must define whether he wants to define equivalent properties
    or grout stratification.

    Attributes
    ----------
    cp_0 : float | None
        Specific heat capacity [J / (kg K)].
    rho_0 : float | None
        Density [kg/m³].
    k0 : float | None
        Thermal conductivity [W / (m K)].
    stratification : Sequence[tuple[float, float, float, float]] | None
        Grout layering as a sequence of ``(k, cp, rho, thickness)`` tuples.
        The sum of layer thicknesses must equal the borehole discretized length.
        Stratification is set as None by default.
    soil_type: str
        Soil type string. This is set as None by default. If accounting for
        time variable properties, it must be set as 'sand', 'loam', or 'clay' 
        and the correct properties must be given as input.
    """
    cp_0: float | None = None  # J/kgK
    rho_0: float | None = None  # kg/m3
    k0: float | None = None  # W/mK
    stratification: Sequence[tuple[float, float, float, float]] | None = None
    soil_type: str | None = None

    def __post_init__(self) -> None:
        if self.stratification is None and (
            self.cp_0 is None or self.rho_0 is None or self.k0 is None
        ):
            raise ValueError(
                "Define grout stratification or single equivalent properties"
            )

        if self.stratification is not None and (
            self.cp_0 is not None or self.rho_0 is not None or self.k0 is not None
        ):
            raise ValueError(
                "Choose to define only stratification or single equivalent properties."
            )

        if (
            self.cp_0 is not None and self.rho_0 is not None and self.k0 is not None
        ) and ((self.cp_0 <= 0) or (self.rho_0 <= 0) or (self.k0 <= 0)):
            raise ValueError(
                "Invalid borehole properties, cp_0, rho_0, and k0 must be > 0"
            )
        
        if self.stratification is not None and not all(min(layer) > 0 for layer in self.stratification):
            raise ValueError("Invalid borehole properties, k0, cp_0, rho_0, and thicnkess must be > 0.")
        
        if self.soil_type is not None:
            self.soil_type = self.soil_type.strip().lower()
            if self.soil_type not in {"sand", "loam", "clay"}:
                raise ValueError("If soil_type is not None it must be set as 'sand', 'loam', or 'clay'.")



class BoreholeProperties:
    """
    Base class for all BHE configurations.

    Assembles geometry, mesh, thermal properties, and fluid into a single
    object. Computes derived quantities (``dz``, mesh-shaped property arrays)
    shared by all BHE types.

    Attributes
    ----------
    geom : BoreholeGeometry
        Borehole geometric parameters.
    mesh : BoreholeMesh
        Axial discretization settings.
    thermalprops : BoreholeThermalProperties
        Thermal properties of the grout.
    fluid : Fluid
        Thermophysical properties of the heat carrier fluid.
    Lbore : float
        Active borehole length [m].
    D0 : float
        Borehole diameter [m].
    m_mesh : int
        Number of axial mesh elements.
    dz : float
        Axial mesh element size [m].
    cp_0 : NDArray
        Specific heat capacity array, shape (m_mesh, 1) [J / (kg K)].
    rho_0 : NDArray
        Density array, shape (m_mesh, 1) [kg/m³].
    k0 : NDArray
        Thermal conductivity array, shape (m_mesh, 1) [W / (m K)].
    """

    def __init__(
        self,
        *,
        geom: BoreholeGeometry,
        mesh: BoreholeMesh,
        thermalprops: BoreholeThermalProperties,
        fluid: Fluid,
    ) -> None:

        self.configuration = {
            1: "single U-tube",
            2: "double U-tube",
            3: "coaxial pipes",
            4: "helical pipes",
        }
        self.geom = geom
        self.mesh = mesh
        self.thermalprops = thermalprops
        self.fluid = fluid

        # alias
        self.Lbore = self.geom.Lbore
        self.D0 = self.geom.D0
        self.soil_type = self.thermalprops.soil_type
        self.D_irrigation = self.geom.D_irrigation
        self.perf_fraction = self.geom.perf_fraction

        self.m_mesh = self.mesh.m_mesh

        # calculation
        self.dz = self.Lbore / self.m_mesh

        # properties shaping
        if self.thermalprops.stratification is not None:
            self.k0, self.cp_0, self.rho_0 = self._variable_properties()
        else:
            k0 = self.thermalprops.k0
            cp_0 = self.thermalprops.cp_0
            rho_0 = self.thermalprops.rho_0
            
            self.cp_0 = np.full((self.m_mesh, 1), cp_0, dtype=np.float64)
            self.rho_0 = np.full(
                (self.m_mesh, 1), rho_0, dtype=np.float64
            )
            self.k0 = np.full((self.m_mesh, 1), k0, dtype=np.float64)

    def _variable_properties(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dz_tot = np.full(self.m_mesh, self.dz, dtype=np.float64)

        tol = 1e-6
        if abs((sum(Ls for _, _, _, Ls in self.thermalprops.stratification) - np.sum(dz_tot))) > tol:
            raise ValueError(
                "Length of stratifications must match the total length of discretization"
            )

        n_cells = self.m_mesh

        k0 = np.zeros(n_cells, dtype=np.float64)
        cp_0 = np.zeros(n_cells, dtype=np.float64)
        rho_0 = np.zeros(n_cells, dtype=np.float64)

        i = 0
        j = 0
        dz_res = dz_tot[i]
        k_m, cp_m, rho_m, L_res = self.thermalprops.stratification[j]

        k_acc = cp_acc = rho_acc = 0.0

        while i < n_cells and j < len(self.thermalprops.stratification):
            delta = min(dz_res, L_res)

            k_acc = k_acc + k_m * delta
            cp_acc = cp_acc + cp_m * delta
            rho_acc = rho_acc + rho_m * delta

            dz_res = dz_res - delta
            L_res = L_res - delta

            if dz_res <= tol:
                k0[i] = k_acc / dz_tot[i]
                cp_0[i] = cp_acc / dz_tot[i]
                rho_0[i] = rho_acc / dz_tot[i]

                i += 1
                if i >= n_cells:
                    break

                dz_res = dz_tot[i]
                k_acc = cp_acc = rho_acc = 0

            if L_res <= tol:
                j += 1
                if j >= len(self.thermalprops.stratification):
                    break

                k_m, cp_m, rho_m, L_res = self.thermalprops.stratification[j]

        if np.any(k0 <= 0) or np.any(rho_0 <= 0) or np.any(cp_0 <= 0):
            bad = [
                (idx, k0[idx], cp_0[idx], rho_0[idx])
                for idx in range(n_cells)
                if k0[idx] <= 0 or cp_0[idx] <= 0 or rho_0[idx] <= 0
            ]
            raise ValueError(
                f"Unfilled/invalid cell properties at indices (idx,k,cp,rho): {bad[:10]} ..."
            )

        k0 = np.asarray(k0)[:, None]
        cp_0 = np.asarray(cp_0)[:, None]
        rho_0 = np.asarray(rho_0)[:, None]

        return k0, cp_0, rho_0
      

class Utube(BoreholeProperties):
    """
    Base class for U-tube BHE configurations (single and double).

    Extends ``BoreholeProperties`` with pipe geometry, grout cross-sectional
    areas (shell and core), thermal capacitances, and axial resistances.

    Attributes
    ----------
    pipe_thick : float
        Pipe wall thickness [m].
    pipe_spacing : float
        Centre-to-centre spacing between pipes [m].
    Dpi : float
        Inner pipe diameter [m].
    n_pipes : int
        Number of pipes (2 for single U-tube, 4 for double U-tube).
    S_shell : float
        Cross-sectional area of the grout shell region [m²].
    S_core : float
        Cross-sectional area of the grout core region [m²].
    C_shell : NDArray
        Thermal capacitance of the shell, shape (m_mesh, 1) [J/K].
    C_core : NDArray
        Thermal capacitance of the core, shape (m_mesh, 1) [J/K].
    C_fluid : float
        Thermal capacitance of the fluid per pipe element [J/K].
    R_axial_shell : NDArray
        Axial thermal resistance of the shell, shape (m_mesh, 1) [K/W].
    R_axial_core : NDArray
        Axial thermal resistance of the core, shape (m_mesh, 1) [K/W].
    """

    def __init__(
        self,
        *,
        geom: BoreholeGeometry,
        mesh: BoreholeMesh,
        thermalprops: BoreholeThermalProperties,
        fluid: Fluid,
        pipe_thick: float,
        pipe_spacing: float,
        Dpi: float,
        n_pipes: int,
    ) -> None:

        super().__init__(
            geom=geom,
            mesh=mesh,
            thermalprops=thermalprops,
            fluid=fluid,
        )

        if n_pipes not in (2, 4):
            raise ValueError("Invalid number of pipes, n_pipes must be = 2 | 4")

        self.pipe_thick = pipe_thick
        self.pipe_spacing = pipe_spacing
        self.Dpi = Dpi
        self.n_pipes = n_pipes
        self.id_shell = 0
        self.id_core = 1

        # calculation
        # areas chell and core
        self._core_shell_area_u_tube()

        # shell
        self.C_shell = (
            self.rho_0 * self.cp_0 * self.dz * self.S_shell
        )  # shape (m_mesh, 1)

        # core
        self.C_core = (
            self.rho_0 * self.cp_0 * self.dz * self.S_core
        )  # shape (m_mesh, 1)

        # axial resistance shell and core
        self._bhe_res_axial()

        # fluid
        self._fluid_capacitance()

        # pipes
        self.R_pipes = (
            1
            / (2 * np.pi * self.dz * self.fluid.k_w)
            * np.log(self.Dpi / (self.Dpi / 100))
        )

    def _update_properties(self, k0: float, cp_0: float, rho_0: float) -> None:
        self.k0[:, 0] = k0
        self.cp_0[:, 0] = cp_0
        self.rho_0[:, 0] = rho_0

        self.C_shell = (
            self.rho_0 * self.cp_0 * self.dz * self.S_shell
        )
        self.C_core = (
            self.rho_0 * self.cp_0 * self.dz * self.S_core
        )
        self._bhe_res_axial()



    def _core_shell_area_u_tube(self) -> None:
        x = (self.Dpi / 2 + self.pipe_thick) / self.pipe_spacing

        if x > 0 and x <= 1:
            angleA = np.acos(x)
            angleB = np.pi - 2 * angleA
        else:
            raise ValueError(
                "Spacing between pipes must be >= (pipe diameter/2 + pipe thickness)"
            )

        self.S_core = (np.pi * (self.pipe_spacing**2) / 4) - self.n_pipes * (
            (self.pipe_spacing**2) / 8 * (2 * angleB - np.sin(2 * angleB))
            + ((self.Dpi + 2 * self.pipe_thick) ** 2 / 8)
            * (np.pi - angleB - np.sin(angleB))
        )

        self.S_shell = (
            (np.pi * self.D0**2 / 4)
            - self.S_core
            - self.n_pipes * np.pi * (self.Dpi + 2 * self.pipe_thick) ** 2 / 4
        )

    def _fluid_capacitance(self) -> None:
        self.C_fluid = (
            self.fluid.rho_w * self.fluid.cp_w * self.dz * np.pi * (self.Dpi) ** 2 / 4
        )

    def _bhe_res_axial(self) -> None:
        self.R_axial_core = self.dz / (self.k0 * self.S_core)  # shape (m_mesh, 1)
        self.R_axial_shell = self.dz / (self.k0 * self.S_shell)  # shape (m_mesh, 1)


class SingleUtube(Utube):
    """
    Single U-tube BHE configuration (2 pipes).

    Extends ``Utube`` with pipe-to-grout and grout-to-ground resistances
    specific to the single U-tube layout.

    Attributes
    ----------
    Rp0 : float
        Pipe-to-grout thermal resistance [K m / W].
    RppB : float
        Grout-to-ground thermal resistance [K m / W].
    n_equations : int
        Number of nodal equations in the discretized system (6).
    Rp0_dz : float
        ``Rp0`` normalized by ``dz`` [K/W].
    RppB_dz : float
        ``RppB`` normalized by ``dz`` [K/W].
    """

    def __init__(
        self,
        *,
        geom: BoreholeGeometry,
        mesh: BoreholeMesh,
        thermalprops: BoreholeThermalProperties,
        fluid: Fluid,
        pipe_thick: float,
        pipe_spacing: float,
        Dpi: float,
        n_pipes: int,
        Rp0: float,
        RppB: float,
    ) -> None:

        super().__init__(
            geom=geom,
            mesh=mesh,
            thermalprops=thermalprops,
            fluid=fluid,
            pipe_thick=pipe_thick,
            pipe_spacing=pipe_spacing,
            Dpi=Dpi,
            n_pipes=n_pipes,
        )

        if self.n_pipes != 2:
            raise ValueError("Single Utube configuration requires 2 pipes")

        self.Rp0 = Rp0
        self.RppB = RppB
        self.n_equations = 6
        self.id_inlet = self.n_equations - 2
        self.id_outlet = self.n_equations - 1

        # calculation
        self.Rp0_dz = self.Rp0 / self.dz
        self.RppB_dz = self.RppB / self.dz


    def crossing_time_calculation(self, mw_tot: NDArray) -> NDArray:
        """
        Compute the fluid transit time through the U-tube.

        Parameters
        ----------
        mw_tot : NDArray
            Total mass flow rate [kg/s].

        Returns
        -------
        NDArray
            Time for the fluid to travel the full U-tube length (2 × Lbore) [s].

        Examples
        --------
        >>> mw_tot = np.full(n_steps, 2)
        >>> t = bhe.crossing_time_calculation(mw_tot=mw_tot)
        """
        mw = mw_tot
        v_w = mw / (self.fluid.rho_w * np.pi * ((self.Dpi) ** 2 / 4))
        crossing_time = 2 * self.Lbore / v_w

        return crossing_time

    def _alpha_calculation(self, mw_tot: float) -> float:
        mw = mw_tot
        Re = (4 * mw) / (np.pi * (self.Dpi) * self.fluid.rho_w * self.fluid.ni_w)
        Pr = (self.fluid.cp_w * self.fluid.rho_w * self.fluid.ni_w) / self.fluid.k_w

        if Re <= 2000:
            Nu = 1.61 * (2000 * Pr * (self.Dpi) / self.Lbore) ** (1 / 3)

        elif Re > 2000 and Re <= 10000:
            Nu = (
                0.116
                * (Re ** (2 / 3) - 125)
                * (Pr ** (1 / 3))
                * (1 + ((self.Dpi) / self.Lbore) ** (2 / 3))
            )

        else:
            Nu = 0.023 * (Re**0.8) * (Pr ** (1 / 3))

        alpha_w = (self.fluid.k_w * Nu) / (self.Dpi)

        return alpha_w

    def _borehole_resistances(self, mw_tot: float) -> float:
        alpha_w = self._alpha_calculation(mw_tot=mw_tot)
        R_conv = 1 / (np.pi * (self.Dpi) * alpha_w * self.dz)

        return R_conv


class DoubleUtube(Utube):
    """
    Double U-tube BHE configuration (4 pipes).

    Extends ``Utube`` with an additional pipe-to-pipe resistance and
    support for series (S) or parallel (P) pipe connection.

    Attributes
    ----------
    connection : str
        Pipe connection mode: ``'S'`` for series, ``'P'`` for parallel.
    Rp0 : float
        Pipe-to-grout thermal resistance [K m / W].
    RppB : float
        Grout-to-ground thermal resistance [K m / W].
    RppA : float
        Pipe-to-pipe thermal resistance [K m / W].
    n_equations : int
        Number of nodal equations in the discretized system (10).
    Rp0_dz : float
        ``Rp0`` normalized by ``dz`` [K/W].
    RppB_dz : float
        ``RppB`` normalized by ``dz`` [K/W].
    RppA_dz : float
        ``RppA`` normalized by ``dz`` [K/W].
    """

    def __init__(
        self,
        *,
        geom: BoreholeGeometry,
        mesh: BoreholeMesh,
        thermalprops: BoreholeThermalProperties,
        fluid: Fluid,
        pipe_thick: float,
        pipe_spacing: float,
        Dpi: float,
        n_pipes: int,
        Rp0: float,
        RppB: float,
        RppA: float,
        connection: str,
    ) -> None:

        super().__init__(
            geom=geom,
            mesh=mesh,
            thermalprops=thermalprops,
            fluid=fluid,
            pipe_thick=pipe_thick,
            pipe_spacing=pipe_spacing,
            Dpi=Dpi,
            n_pipes=n_pipes,
        )

        if self.n_pipes != 4:
            raise ValueError("Double Utubes configuration requires 4 pipes")

        if connection not in ("S", "s", "P", "p"):
            raise ValueError(
                "Connection value must be S or s for series connection, P or p for parallel connection"
            )

        self.connection = connection
        self.Rp0 = Rp0
        self.RppB = RppB
        self.RppA = RppA
        self.n_equations = 10
        self.id_inlet = self.n_equations - 2
        self.id_outlet = self.n_equations - 1

        # calculation
        self.Rp0_dz = self.Rp0 / self.dz
        self.RppB_dz = self.RppB / self.dz
        self.RppA_dz = self.RppA / self.dz

    def crossing_time_calculation(self, mw_tot: NDArray) -> NDArray:
        """
        Compute the fluid transit time through the double U-tube.

        Accounts for series (full flow in each pipe) vs. parallel
        (half flow in each pipe) connection.

        Parameters
        ----------
        mw_tot : NDArray
            Total mass flow rate [kg/s].

        Returns
        -------
        NDArray
            Fluid transit time [s].
        """

        if (self.connection == "S") or (self.connection == "s"):
            mw = mw_tot
            v_w = mw / (self.fluid.rho_w * np.pi * ((self.Dpi) ** 2 / 4))
            crossing_time = 4 * self.Lbore / v_w
        else:
            mw = mw_tot / 2
            v_w = mw / (self.fluid.rho_w * np.pi * ((self.Dpi) ** 2 / 4))
            crossing_time = 2 * self.Lbore / v_w

        return crossing_time

    def _alpha_calculation(self, mw_tot: float) -> float:

        if self.connection in ("P", "p"):
            mw = mw_tot / 2
        else:
            mw = mw_tot

        Re = (4 * mw) / (np.pi * (self.Dpi) * self.fluid.rho_w * self.fluid.ni_w)
        Pr = (self.fluid.cp_w * self.fluid.rho_w * self.fluid.ni_w) / self.fluid.k_w

        if Re <= 2000:
            Nu = 1.61 * (2000 * Pr * (self.Dpi) / self.Lbore) ** (1 / 3)

        elif Re > 2000 and Re <= 10000:
            Nu = (
                0.116
                * (Re ** (2 / 3) - 125)
                * (Pr ** (1 / 3))
                * (1 + ((self.Dpi) / self.Lbore) ** (2 / 3))
            )

        else:
            Nu = 0.023 * (Re**0.8) * (Pr ** (1 / 3))

        alpha_w = (self.fluid.k_w * Nu) / (self.Dpi)

        return alpha_w

    def _borehole_resistances(self, mw_tot: float) -> float:

        alpha_w = self._alpha_calculation(mw_tot=mw_tot)
        R_conv = 1 / (np.pi * (self.Dpi) * alpha_w * self.dz)

        return R_conv


class Coaxial(BoreholeProperties):
    """
    Coaxial pipe BHE configuration.

    Two concentric pipes: inner pipe (1) and annular outer pipe (2).
    Flow direction is set by ``supply_and_return``.

    Attributes
    ----------
    Dp1i : float
        Inner diameter of pipe 1 (inner pipe) [m].
    Dp2i : float
        Inner diameter of pipe 2 (outer annulus) [m].
    pipe1_thick : float
        Wall thickness of pipe 1 [m].
    pipe2_thick : float
        Wall thickness of pipe 2 [m].
    k_pipe1 : float
        Thermal conductivity of the pipe 1 material [W / (m K)].
    k_pipe2 : float
        Thermal conductivity of the pipe 2 material [W / (m K)].
    supply_and_return : str
        Flow direction: ``'1_2'`` (supply in pipe 1) or ``'2_1'`` (supply in pipe 2).
    n_equations : int
        Number of nodal equations in the discretized system (5).
    De : float
        Hydraulic diameter of the annular region [m].
    S_shell : float
        Cross-sectional area of the grout annulus [m²].
    R_cond1 : float
        Conductive resistance of pipe 1 wall [K/W].
    R_cond2 : float
        Conductive resistance of pipe 2 wall [K/W].
    R_shell : float
        Conductive resistance of the grout annulus [K/W].
    R_pipes1 : float
        Conductive resistance of stationary fluid in pipe 1, used when mw=0 [K/W].
    R_pipes2 : float
        Conductive resistance of stationary fluid in the annulus, used when mw=0 [K/W].
    R_axial_shell : float
        Axial conductive resistance of the grout shell [K/W].
    C_shell : float
        Thermal capacitance of the grout shell [J/K].
    C_fluid1 : float
        Thermal capacitance of the fluid in pipe 1 [J/K].
    C_fluid2 : float
        Thermal capacitance of the fluid in the annulus [J/K].
    """

    def __init__(
        self,
        *,
        geom: BoreholeGeometry,
        mesh: BoreholeMesh,
        thermalprops: BoreholeThermalProperties,
        fluid: Fluid,
        Dp1i: float,
        Dp2i: float,
        pipe1_thick: float,
        pipe2_thick: float,
        k_pipe1: float,
        k_pipe2: float,
        supply_and_return: str,
    ) -> None:

        super().__init__(
            geom=geom,
            mesh=mesh,
            thermalprops=thermalprops,
            fluid=fluid,
        )

        allowed_supply_and_return = {"1_2", "2_1"}

        if supply_and_return not in allowed_supply_and_return:
            raise ValueError("supply and return must be 1_2 or 2_1 type")
        if (pipe1_thick <= 0) or (pipe2_thick <= 0):
            raise ValueError(
                "Invalid pipe thicknesses, check pipe2_thick and pipe2_thick values"
            )
        if (pipe1_thick >= Dp1i / 2) or (pipe2_thick >= Dp2i / 2):
            raise ValueError(
                "pipe1_thick must be >= Dp1i/2 and pipe2_thick must be >= Dp2i/2"
            )
        if (Dp1i <= 0) or (Dp2i <= 0) or (Dp2i <= Dp1i * 2 * pipe1_thick):
            raise ValueError(
                "Invalid internal diameters, check Dp1i and Dp2i values. Dp2i >= Dp1i+2*pipe1_thick"
            )
        if k_pipe1 <= 0:
            raise ValueError("k_pipe1 must be > 0")
        if k_pipe2 <= 0:
            raise ValueError("k_pipe2 must be > 0")

        self.supply_and_return = supply_and_return
        self.k_pipe1 = k_pipe1
        self.k_pipe2 = k_pipe2
        self.Dp1i = Dp1i
        self.Dp2i = Dp2i
        self.pipe1_thick = pipe1_thick
        self.pipe2_thick = pipe2_thick
        self.n_equations = 5
        self.id_shell = 0
        self.id_core = None

        if self.supply_and_return == "1_2":
            self.id_inlet = self.n_equations - 2
            self.id_outlet = self.n_equations - 1
        else:
            self.id_inlet = self.n_equations - 1
            self.id_outlet = self.n_equations - 2

        # calculation
        # geometry
        self.Dp1 = Dp1i + 2 * pipe1_thick
        self.Dp2 = Dp2i + 2 * pipe2_thick
        self.r1o = self.Dp1 / 2
        self.r1i = self.r1o - self.pipe1_thick
        self.r2o = self.Dp2 / 2
        self.r2i = self.r2o - self.pipe2_thick

        # shell
        self.S_shell = np.pi * ((self.D0 / 2) ** 2 - self.r2o**2)
        self.R_cond1 = (
            1 / (2 * np.pi * self.dz * self.k_pipe1) * np.log(self.r1o / self.r1i)
        )
        self.R_cond2 = (
            1 / (2 * np.pi * self.dz * self.k_pipe2) * np.log(self.r2o / self.r2i)
        )
        self.R_shell = (
            1 / (2 * np.pi * self.dz * self.k0) * np.log((self.D0 / 2) / self.r2o)
        )  # shape (m_mesh, 1)
        self.C_shell = (
            self.rho_0 * self.cp_0 * self.dz * self.S_shell
        )  # shape (m_mesh, 1)
        self._bhe_res_axial()

        # fluid resistances when mw = 0
        self.R_pipes1 = (
            1
            / (2.0 * np.pi * self.dz * fluid.k_w)
            * np.log(self.r1i / (self.r1i / 100))
        )
        self.R_pipes2 = (
            1 / (2.0 * np.pi * self.dz * fluid.k_w) * np.log(self.r2i / self.r1o)
        )

        # fluid
        self.C_fluid1, self.C_fluid2 = self._fluid_capacitance()

    def _update_properties(self, k0: float, cp_0: float, rho_0: float) -> None:
        self.k0[:, 0] = k0
        self.cp_0[:, 0] = cp_0
        self.rho_0[:, 0] = rho_0

        self.R_shell = (
            1 / (2 * np.pi * self.dz * self.k0) * np.log((self.D0 / 2) / self.r2o)
        )
        self.C_shell = (
            self.rho_0 * self.cp_0 * self.dz * self.S_shell
        )
        self._bhe_res_axial()



    def _fluid_capacitance(self) -> tuple[float, float]:

        C_fluid1 = self.fluid.rho_w * self.fluid.cp_w * self.dz * np.pi * self.r1i**2
        C_fluid2 = (
            self.fluid.rho_w
            * self.fluid.cp_w
            * self.dz
            * np.pi
            * (self.r2i**2 - self.r1o**2)
        )

        return C_fluid1, C_fluid2

    def _alpha_calculation(self, mw_tot: float) -> tuple[float, float]:
        mw = mw_tot
        # for pipe 1
        Re1 = (4 * mw) / (np.pi * (self.r1i * 2) * self.fluid.rho_w * self.fluid.ni_w)
        Pr1 = (self.fluid.cp_w * self.fluid.rho_w * self.fluid.ni_w) / self.fluid.k_w

        if Re1 <= 2000:
            Nu1 = 1.61 * (2000 * Pr1 * self.r1i * 2 / self.Lbore) ** (1 / 3)

        elif Re1 > 2000 and Re1 <= 10000:
            Nu1 = (
                0.116
                * (Re1 ** (2 / 3) - 125)
                * (Pr1 ** (1 / 3))
                * (1 + (self.r1i * 2 / self.Lbore) ** (2 / 3))
            )

        else:
            Nu1 = 0.023 * (Re1**0.8) * (Pr1 ** (1 / 3))

        alpha_w1 = (self.fluid.k_w * Nu1) / (self.r1i * 2)

        # for pipe 2
        self.De = (self.Dp2i) - (self.Dp1i + 2 * self.pipe1_thick)
        Re2 = (4 * mw) / (np.pi * self.De * self.fluid.rho_w * self.fluid.ni_w)
        Pr2 = (self.fluid.cp_w * self.fluid.rho_w * self.fluid.ni_w) / self.fluid.k_w

        if Re2 <= 2000:
            Nu2 = 1.61 * (2000 * Pr2 * self.De / self.Lbore) ** (1 / 3)

        elif Re2 > 2000 and Re2 <= 10000:
            Nu2 = (
                0.116
                * (Re2 ** (2 / 3) - 125)
                * (Pr2 ** (1 / 3))
                * (1 + (self.De / self.Lbore) ** (2 / 3))
            )

        else:
            Nu2 = 0.023 * (Re2**0.8) * (Pr2 ** (1 / 3))

        alpha_w2 = (self.fluid.k_w * Nu2) / self.De

        return alpha_w1, alpha_w2

    def _borehole_resistances(self, mw_tot: float) -> tuple[float, float, float]:

        alpha_w1, alpha_w2 = self._alpha_calculation(mw_tot=mw_tot)

        R_conv1 = 1 / (2 * np.pi * self.r1i * self.dz * alpha_w1)
        R_conv2_1 = 1 / (2 * np.pi * self.r1o * self.dz * alpha_w2)
        R_conv2_2 = 1 / (2 * np.pi * self.r2i * self.dz * alpha_w2)

        return R_conv1, R_conv2_1, R_conv2_2

    def crossing_time_calculation(self, mw_tot: NDArray) -> NDArray:
        """
        Compute fluid transit times for both flow paths in the coaxial BHE.

        Parameters
        ----------
        mw_tot : NDArray
            Total mass flow rate [kg/s].

        Returns
        -------
        crossing_time : NDArray
            Transit time through pipes [s].
        """
        mw = mw_tot
        v_w1 = mw / (self.fluid.rho_w * np.pi * (self.Dp1i**2 / 4))
        crossing_time = self.Lbore / v_w1

        v_w2 = mw / (self.fluid.rho_w * np.pi * (self.De**2 / 4))
        crossing_time += self.Lbore / v_w2

        return crossing_time

    def _bhe_res_axial(self) -> None:
        self.R_axial_shell = self.dz / (self.k0 * self.S_shell)


class Helical(BoreholeProperties):
    """
    Helical pipe BHE configuration.

    A helical coil wound inside the borehole. The geometry is parameterized
    by the helix radius, pipe diameter, and number of turns.

    Attributes
    ----------
    Dpi1 : float
        Inner pipe 1 diameter (straight tube) [m].
    Dpi2 : float
        Inner pipe 2 diameter (helical tube) [m].
    rih : float
        Inner helix radius (centre of pipe to borehole axis) [m].
    pipe_thick : float
        Pipe wall thickness [m].
    N : int
        Number of helix turns.
    P : float
        Helix pitch [m].
    supply_and_return : str
        Flow direction: ``'1_2'`` (supply in pipe 1) or ``'2_1'`` (supply in pipe 2).
    Lp2tot: float
        Total length helical pipe [m].
    k_pipe : float
        Thermal conductivity of the pipe material [W / (m K)].
    n_equations : int
        Number of nodal equations in the discretized system (6).
    F : float
        Turn density (turns per metre) [1/m].
    S_shell : float
        Cross-sectional area of the outer grout annulus [m²].
    S_core : float
        Cross-sectional area of the inner grout core [m²].
    C_shell : NDArray
        Thermal capacitance of the shell, shape (m_mesh, 1) [J/K].
    C_shell_middle : NDArray
        Thermal capacitance of the node between pipe 2 and shell, shape (m_mesh, 1) [J/K]
    C_core : NDArray
        Thermal capacitance of the core, shape (m_mesh, 1) [J/K].
    C_fluid1 : float
        Thermal capacitance of the supply fluid [J/K].
    C_fluid2 : float
        Thermal capacitance of the return fluid [J/K].
    """

    def __init__(
        self,
        *,
        geom: BoreholeGeometry,
        mesh: BoreholeMesh,
        thermalprops: BoreholeThermalProperties,
        fluid: Fluid,
        Dpi1: float,
        Dpi2: float,
        rih: float,
        pipe_thick: float,
        N: int,
        P: float,
        supply_and_return: str,
        Lp2tot: float,
        k_pipe: float,
    ) -> None:

        super().__init__(
            geom=geom,
            mesh=mesh,
            thermalprops=thermalprops,
            fluid=fluid,
        )

        allowed_supply_and_return = {"1_2", "2_1"}

        if supply_and_return not in allowed_supply_and_return:
            raise ValueError("supply and return must be 1_2 or 2_1 type")
        if Dpi1 <= 0:
            raise ValueError("Dp must be > 0")
        if pipe_thick <= 0 or pipe_thick >= (Dpi1 / 2):
            raise ValueError("Invalid pipe thickness")
        if rih <= (Dpi1 / 2):
            raise ValueError("Invalid helical geometry: rih must be > r1o")
        if (rih + Dpi2 + 2 * pipe_thick) >= self.D0 / 2:
            raise ValueError("Invalid helical geometry: roh must be < D0/2")
        if Dpi2 <= 0:
            raise ValueError("Dp must be > 0")
        if pipe_thick <= 0 or pipe_thick >= (Dpi2 / 2):
            raise ValueError("Invalid pipe thickness")
        if abs(N - self.Lbore / P) > 1e-6:
            raise ValueError(r"N, Lbore, and P must be consistent: N = Lbore / P")
        if Lp2tot <= 0.0:
            raise ValueError(r"Invalid Lp2tot: Lp2tot must be > 0")

        self.Dpi1 = Dpi1
        self.Dpi2 = Dpi2
        self.k_pipe = k_pipe
        self.N = N
        self.P = P
        self.supply_and_return = supply_and_return
        self.Lp2tot = Lp2tot
        self.pipe_thick = pipe_thick
        self.rih = rih
        self.n_equations = 7
        self.id_shell = 0
        self.id_core = 1
        self.id_shell_middle = 2
        if self.supply_and_return == "1_2":
            self.id_inlet = self.n_equations - 2
            self.id_outlet = self.n_equations - 1
        else:
            self.id_inlet = self.n_equations - 1
            self.id_outlet = self.n_equations - 2

        # calculation
        # geometry
        self.Ds = self.Lp2tot / (self.N * np.pi)
        self.F = self.N / self.Lbore * np.pi * self.Ds
        self.Lp2 = self.F * self.dz
        self.r1o = self.Dpi1 / 2 + self.pipe_thick
        self.r1i = self.r1o - self.pipe_thick
        self.r2o = self.Dpi2 / 2 + self.pipe_thick
        self.r2i = self.r2o - self.pipe_thick
        self.roh = self.rih + Dpi2 + 2 * pipe_thick
        self.rcore = np.sqrt((self.r1o**2 + self.rih**2) / 2.0)
        self.rshell_middle = np.sqrt((self.roh**2 + (self.D0 / 2.0) ** 2) / 2.0)
        self.capshellcoeff = (
            ((self.roh * 2 + self.D0) / 2) ** 2 - (self.roh * 2) ** 2
        ) / (self.D0**2 - (self.roh * 2) ** 2)

        # generic shell
        self.Cgeneric = (
            self.rho_0
            * self.cp_0
            * self.dz
            * np.pi
            / 4.0
            * (self.D0**2 - (self.roh * 2) ** 2)
        )
        # shell middle
        self.S_shell_middle = np.pi * (self.rshell_middle**2 - self.roh**2)
        self.R_shell_middle = (
            1
            / (2 * np.pi * self.dz * self.k0)
            * np.log(self.rshell_middle / self.roh)
            * self.k_pipe
            / self.k0
            * (2 * np.pi * self.roh * self.dz)
            / (2 * (self.Dpi2 / 2.0 + self.pipe_thick) * self.Lp2)
        )  # shape (m_mesh, 1)
        self.C_shell_middle = self.Cgeneric * self.capshellcoeff  # shape (m_mesh, 1)

        # shell
        self.S_shell = np.pi * ((self.D0 / 2) ** 2 - self.rshell_middle**2)
        self.R_shell = (
            1
            / (2 * np.pi * self.dz * self.k0)
            * np.log((self.D0 / 2) / self.rshell_middle)
        )  # shape (m_mesh, 1)
        self.C_shell = self.Cgeneric * (1 - self.capshellcoeff)  # shape (m_mesh, 1)

        # core
        self.S_core = np.pi * (self.rih**2 - self.r1o**2)
        self.R_core1 = (
            1 / (2 * np.pi * self.dz * self.k0) * np.log(self.rcore / self.r1o)
        )  # shape (m_mesh, 1)
        self.R_core2 = (
            1
            / (2 * np.pi * self.dz * self.k0)
            * np.log(self.rih / self.rcore)
            * self.k_pipe
            / self.k0
            * (2 * np.pi * self.rih * self.dz)
            / (2.0 * (self.Dpi2 / 2 + self.pipe_thick) * self.Lp2)
        )  # shape (m_mesh, 1)

        self.R_core_shell_middle = (
            1
            / (2 * np.pi * self.P * self.k0)
            * np.log(self.rshell_middle / self.rcore)
            * 1
            / (1 + self.dz / self.P)
        )  # shape (m_mesh, 1)
        self.R_core_shell = (
            1
            / (2 * np.pi * self.P * self.k0)
            * np.log((self.D0 / 2.0) / self.rcore)
            * 1
            / (1 + self.dz / self.P)
        )
        self.C_core = (
            self.rho_0 * self.cp_0 * self.dz * self.S_core
        )  # shape (m_mesh, 1)

        # axial resistances core and shell
        self._bhe_res_axial()

        # pipe resistances
        self.R_cond1 = (
            1 / (2 * np.pi * self.dz * self.k_pipe) * np.log(self.r1o / self.r1i)
        )
        self.R_cond2 = (
            1
            / (2 * np.pi * self.F * self.dz * self.k_pipe)
            * np.log(self.r2o / self.r2i)
        )

        # fluid resistances when mw = 0
        self.R_pipes1 = (
            1
            / (2.0 * np.pi * self.dz * self.fluid.k_w)
            * np.log(self.r1i / (self.r1i / 100))
        )
        self.R_pipes2 = (
            1
            / (2.0 * np.pi * self.F * self.dz * self.fluid.k_w)
            * np.log(self.r2i / (self.r2i / 100))
        )

        # fluid
        self.C_fluid1, self.C_fluid2 = self._fluid_capacitance()

    def _update_properties(self, k0: float, cp_0: float, rho_0: float) -> None:
        self.k0[:, 0] = k0
        self.cp_0[:, 0] = cp_0
        self.rho_0[:, 0] = rho_0

        self.Cgeneric = (
            self.rho_0
            * self.cp_0
            * self.dz
            * np.pi
            / 4.0
            * (self.D0**2 - (self.roh * 2) ** 2)
        )
        self.R_shell_middle = (
            1
            / (2 * np.pi * self.dz * self.k0)
            * np.log(self.rshell_middle / self.roh)
            * self.k_pipe
            / self.k0
            * (2 * np.pi * self.roh * self.dz)
            / (2 * (self.Dpi2 / 2.0 + self.pipe_thick) * self.Lp2)
        )
        self.C_shell_middle = self.Cgeneric * self.capshellcoeff
        self.R_shell = (
            1
            / (2 * np.pi * self.dz * self.k0)
            * np.log((self.D0 / 2) / self.rshell_middle)
        )
        self.C_shell = self.Cgeneric * (1 - self.capshellcoeff)
        self.R_core1 = (
            1 / (2 * np.pi * self.dz * self.k0) * np.log(self.rcore / self.r1o)
        )
        self.R_core2 = (
            1
            / (2 * np.pi * self.dz * self.k0)
            * np.log(self.rih / self.rcore)
            * self.k_pipe
            / self.k0
            * (2 * np.pi * self.rih * self.dz)
            / (2.0 * (self.Dpi2 / 2 + self.pipe_thick) * self.Lp2)
        ) 
        self.R_core_shell_middle = (
            1
            / (2 * np.pi * self.P * self.k0)
            * np.log(self.rshell_middle / self.rcore)
            * 1
            / (1 + self.dz / self.P)
        )
        self.R_core_shell = (
            1
            / (2 * np.pi * self.P * self.k0)
            * np.log((self.D0 / 2.0) / self.rcore)
            * 1
            / (1 + self.dz / self.P)
        )
        self.C_core = (
            self.rho_0 * self.cp_0 * self.dz * self.S_core
        ) 
        self._bhe_res_axial()



    def _fluid_capacitance(self) -> tuple[float, float]:

        C_fluid1 = self.fluid.rho_w * self.fluid.cp_w * np.pi * self.dz * self.r1i**2
        C_fluid2 = (
            self.fluid.rho_w * self.fluid.cp_w * np.pi * self.dz * self.F * self.r2i**2
        )

        return C_fluid1, C_fluid2

    def _alpha_calculation(self, mw_tot: float) -> tuple[float, float]:

        mw = mw_tot
        Re1 = (4 * mw) / (np.pi * (self.Dpi1) * self.fluid.rho_w * self.fluid.ni_w)
        Pr1 = (self.fluid.cp_w * self.fluid.rho_w * self.fluid.ni_w) / self.fluid.k_w

        if Re1 <= 2000:
            Nu = 1.61 * (2000 * Pr1 * (self.Dpi1) / self.Lbore) ** (1 / 3)

        elif Re1 > 2000 and Re1 <= 10000:
            Nu = (
                0.116
                * (Re1 ** (2 / 3) - 125)
                * (Pr1 ** (1 / 3))
                * (1 + ((self.Dpi1) / self.Lbore) ** (2 / 3))
            )

        else:
            Nu = 0.023 * (Re1**0.8) * (Pr1 ** (1 / 3))

        alpha_w1 = (self.fluid.k_w * Nu) / (self.Dpi1)

        Re2 = (4 * mw) / (np.pi * (self.Dpi2) * self.fluid.rho_w * self.fluid.ni_w)
        Pr2 = (self.fluid.cp_w * self.fluid.rho_w * self.fluid.ni_w) / self.fluid.k_w

        if Re2 <= 2000:
            Nu = 1.61 * (2000 * Pr2 * (self.Dpi2) / self.Lp2tot) ** (1 / 3)

        elif Re2 > 2000 and Re2 <= 10000:
            Nu = (
                0.116
                * (Re2 ** (2 / 3) - 125)
                * (Pr2 ** (1 / 3))
                * (1 + ((self.Dpi2) / self.Lp2tot) ** (2 / 3))
            )

        else:
            Nu = 0.023 * (Re2**0.8) * (Pr2 ** (1 / 3))

        alpha_w2 = (self.fluid.k_w * Nu) / (self.Dpi2)
        return alpha_w1, alpha_w2

    def _borehole_resistances(self, mw_tot: float) -> tuple[float, float]:

        alpha_w1, alpha_w2 = self._alpha_calculation(mw_tot=mw_tot)

        R_conv1 = 1 / (2 * np.pi * self.r1i * self.dz * alpha_w1)
        R_conv2 = 1 / (2 * np.pi * self.r2i * self.dz * self.F * alpha_w2)

        return R_conv1, R_conv2

    def crossing_time_calculation(self, mw_tot: NDArray) -> NDArray:
        """
        Compute the fluid transit time through the helical pipe.

        Parameters
        ----------
        mw_tot : NDArray
            Total mass flow rate [kg/s].

        Returns
        -------
        NDArray
            Fluid transit time [s].
        """
        mw = mw_tot
        v_w1 = mw / (self.fluid.rho_w * np.pi * (self.Dpi1**2 / 4))
        crossing_time = 2 * self.Lbore / v_w1

        v_w2 = mw / (self.fluid.rho_w * np.pi * (self.Dpi2**2 / 4))
        crossing_time += 2 * self.Lp2tot / v_w2

        return crossing_time

    def _bhe_res_axial(self):
        self.R_axial_core = self.dz / (self.k0 * self.S_core)
        self.R_axial_shell = self.dz / (self.k0 * self.S_shell_middle)
