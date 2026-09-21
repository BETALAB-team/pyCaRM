# -*- coding: utf-8 -*-
"""
Borehole field geometry module.

Defines the spatial layout of the borehole field: coordinate input and validation
(``FieldInput``), Voronoi decomposition, distance matrix, and neighbor graph (``Field``).
"""
from typing import Sequence, Dict
from numpy.typing import NDArray
from matplotlib.figure import Figure
from matplotlib.axes import Axes

from shapely.geometry import MultiPoint, box, GeometryCollection, Polygon, LineString
from shapely.ops import voronoi_diagram
from shapely import voronoi_polygons, intersection

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class FieldInput:
    """
    Geometry container for the borehole field layout.

    Stores the field bounding box and borehole coordinates, with validation.
    Coordinates must be loaded explicitly via one of the ``from_*`` methods
    before the field can be used downstream.

    Attributes
    ----------
    n_bhes : int
        Number of boreholes in the field.
    xmin, xmax : float
        Field bounding box x-extent [m].
    ymin, ymax : float
        Field bounding box y-extent [m].
    borehole_coordinates : Sequence[tuple[float, float]]
        List of (x, y) coordinate pairs for each borehole [m].
        ``[]`` until populated via a ``from_*`` method.
    rb: float
        External Borehole radius [m] 
    layout: str
        Borehole spacing layout. The layout can be "regular"
        or "irregular". If regular, the adiabatic condition
        will be applied at mid-distancce between thoe adjacent
        boreholes. If irregular, the FLS calculation will be
        performed to account for penalty temperature at the
        ground maximum radius. By default it is "regular".
        It is important that even if regularly spaced, if 
        series connection is present, the layout must be set
        as irregular.
    """
    def __init__(
        self,
        *,
        n_bhes: int,  # number of boreholes
        xmin: float,  # min x field coordinate
        ymin: float,  # min y field coordinate
        xmax: float,  # max x field coordinate
        ymax: float,  # max y field coordinate
        rb: float, #borehole radius
        layout: str = "regular",  # regular | irregular spacing layout
    ) -> None:

        if (xmin > xmax) or (ymin > ymax):
            raise ValueError("xmin must be < xmax, ymin must be < ymax")

        self.n_bhes = n_bhes
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.rb = rb
        self._borehole_coordinates: Sequence[tuple[float, float]] = []

        layout = layout.strip().lower()
        if layout not in {"regular", "irregular"}:
            raise ValueError("layout must be 'regular' or 'irregular'")
        self.layout = layout

    @property
    def borehole_coordinates(self) -> Sequence[tuple[float, float]]:
        return self._borehole_coordinates

    def from_array(self, x: NDArray, y: NDArray) -> None:
        """
        Load borehole coordinates from two 1-D arrays.

        Parameters
        ----------
        x : NDArray
            x-coordinates of the boreholes [m].
        y : NDArray
            y-coordinates of the boreholes [m].

        Raises
        ------
        ValueError
            If ``x`` and ``y`` have different lengths, contain non-finite values,
            or coordinates fall outside the bounding box.

        Examples
        --------
        >>> fi = FieldInput(n_bhes=2, xmin=0, ymin=0, xmax=10, ymax=10)
        >>> fi.from_array(np.array([3.0, 7.0]), np.array([5.0, 5.0]))
        """
        if len(x) != len(y):
            raise ValueError("x and y arrays must have the same length")
        self._validate_and_set_coordinates(x, y)

    def from_matrix(self, matrix: NDArray) -> None:
        """
        Load borehole coordinates from a 2-D array of shape (n_bhes, 2).

        Parameters
        ----------
        matrix : NDArray
            Array with columns [x, y] for each borehole [m].

        Raises
        ------
        ValueError
            If the array is not 2-D, does not have exactly 2 columns,
            or coordinates fail validation.

        Examples
        --------
        >>> coords = np.array([[3.0, 5.0], [7.0, 5.0]])
        >>> fi = FieldInput(n_bhes=2, xmin=0, ymin=0, xmax=10, ymax=10)
        >>> fi.from_matrix(coords)
        """
        if matrix.ndim != 2:
            raise ValueError("Matrix dimensions must be = 2")

        if matrix.shape[1] != 2:
            raise ValueError("Matrix must have 2 columns: x and y")
        self._validate_and_set_coordinates(matrix[:, 0], matrix[:, 1])

    def _validate_and_set_coordinates(self, x: NDArray, y: NDArray) -> None:
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
            raise ValueError("x and y coordinates may contain invalid values")

        if self.n_bhes != len(x):
            raise ValueError("Each borehole must have a coordinate pair")

        if np.any(x <= self.xmin) or np.any(x >= self.xmax):
            raise ValueError(
                "The following relationship must be satisfied: xmin < x coordinates < xmax"
            )

        if np.any(y <= self.ymin) or np.any(y >= self.ymax):
            raise ValueError(
                "The following relationship must be satisfied: ymin < y coordinates < ymax"
            )

        self._borehole_coordinates = list(zip(x, y))


class Field:
    """
    Voronoi decomposition and interaction geometry of the borehole field.

    Built from a populated ``FieldInput``, computes the Voronoi cell for each
    borehole, the equivalent radius ``r_eq = sqrt(area / pi)``, the pairwise
    distance matrix (corrected for ``r_eq``), and the neighbor graph.

    Attributes
    ----------
    fieldinput : FieldInput
        Source geometry object (must have coordinates loaded).
    field_dict : dict
        Mapping from borehole index to its Voronoi cell data::

            {i: {"cell": Polygon, "area": float, "req": float, "coords": (x, y)}}

    distance_matrix : NDArray[np.float64]
        Symmetric matrix of shape (n_bhes, n_bhes) with inter-borehole distances
        corrected by ``r_eq`` [m]. Diagonal entries are zero.
    """
    def __init__(self, *, fieldinput: FieldInput) -> None:
        self.fieldinput = fieldinput
        self._field_dict: dict = self._matching_field()
        self._distance_matrix: NDArray[np.float64] = self._build_distance_matrix()
        self._borehole_graph: nx.Graph = self._compute_neighbors()

    @property
    def distance_matrix(self) -> NDArray:
        return self._distance_matrix

    @property
    def field_dict(self) -> dict:
        return self._field_dict

    def _build_domain(self) -> tuple[MultiPoint, Polygon]:
        points = MultiPoint(self.fieldinput._borehole_coordinates)
        domain = box(
            self.fieldinput.xmin,
            self.fieldinput.ymin,
            self.fieldinput.xmax,
            self.fieldinput.ymax,
        )

        return points, domain

    def _build_voronoi(self) -> GeometryCollection:
        points, domain = self._build_domain()
        try:
            raw = voronoi_polygons(points, extend_to=domain)
        except Exception:
            raw = voronoi_diagram(points, envelope=domain)

        geoms = list(raw.geoms) if hasattr(raw, "geoms") else [raw]

        clipped = []
        for g in geoms:

            if g.is_empty:
                continue

            c = g.intersection(domain)

            if not c.is_empty and c.area > 0:
                clipped.append(c)

        return GeometryCollection(clipped)

    def _matching_field(self) -> Dict:

        vor = self._build_voronoi()
        field_dict = {}

        for cell in vor.geoms:
            p = cell.representative_point()
            best_dist = np.inf
            for i, (x, y) in enumerate(self.fieldinput._borehole_coordinates):
                min_distance = (x - p.x) ** 2 + (y - p.y) ** 2
                if min_distance < best_dist: 
                    best_dist = min_distance
                    best_idx = i

            if best_idx in field_dict:
                raise ValueError("Same index appeares multiple times in the field")

            field_dict[best_idx] = {
                "cell": cell,
                "area": cell.area,
                "req": np.sqrt(cell.area / np.pi),
                "coords": self.fieldinput._borehole_coordinates[best_idx],
            }

        if len(field_dict) != self.fieldinput.n_bhes:
            raise ValueError("Some values are missing")

        idx_present = [i for i in range(len(self.fieldinput._borehole_coordinates))]

        if sorted(field_dict.keys()) != idx_present:
            raise ValueError("Some idx are missing")

        return field_dict

    def _build_distance_matrix(self) -> NDArray:
        n = self.fieldinput.n_bhes
        req = np.array([self.field_dict[j]["req"] for j in range(n)])

        coords = np.array(self.fieldinput._borehole_coordinates)
        x = coords[:, 0]
        y = coords[:, 1]

        x_j = x[None, :]
        x_i = x[:, None]
        y_j = y[None, :]
        y_i = y[:, None]

        distance_matrix = np.sqrt(np.sqrt((x_j - x_i)**2 + (y_j - y_i)**2)**2 + req[:, None]**2)

        return distance_matrix

    def _compute_neighbors(self) -> nx.Graph:
        n_bhes = self.fieldinput.n_bhes

        borehole_graph = nx.Graph()
        borehole_graph.add_nodes_from(list(self.field_dict.keys()))

        for i in range(n_bhes):
            cell = self.field_dict[i]["cell"]
            for j in range(i + 1, n_bhes):
                potential_neighbor = self.field_dict[j]["cell"]
                if isinstance(intersection(cell, potential_neighbor), LineString):
                    borehole_graph.add_edge(i, j)
        return borehole_graph

    def plot_field(
        self,
        ax=None,
        show_points=True,
        show_ids= False,
        show_area=False,
        show_req=False,
        color_by_area=False,
        alpha=0.35,
        linewidth=0.5,
        point_size=8,
        save_path=None,
        show=True,
        show_graph: bool = False,
    ) -> tuple[Figure, Axes]:
        """
        Plot the Voronoi field decomposition.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None
            Axes to draw on. If ``None``, a new figure is created.
        show_points : bool
            If ``True``, draw borehole generator points.
        show_ids : bool
            If ``True``, annotate each point with its borehole index.
        show_area : bool
            If ``True``, label each cell with its area.
        show_req : bool
            If ``True``, label each cell with its equivalent radius ``r_eq``.
        color_by_area : bool
            If ``True``, shade cells according to their area.
        alpha : float
            Cell fill transparency.
        linewidth : float
            Line width for cell edges and domain boundary.
        point_size : float
            Marker size for borehole points.
        save_path : str or None
            If provided, save the figure to this path at 300 dpi.
        show : bool
            If ``True``, call ``plt.show()``.
        show_graph : bool
            If ``True``, draw edges of the Voronoi neighbor graph.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.

        Examples
        --------
        >>> fi = FieldInput(n_bhes=4, xmin=0, ymin=0, xmax=10, ymax=10)
        >>> fi.from_array(np.array([2.5, 7.5, 2.5, 7.5]),
        ...               np.array([2.5, 2.5, 7.5, 7.5]))
        >>> f = Field(fieldinput=fi)
        >>> fig, ax = f.plot_field(show=False)
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

    
        if not hasattr(self, "field_dict") or self.field_dict is None:
            raise ValueError("field_dict non found: call _matching_field() first.")


        areas = None
        if color_by_area or show_area or show_req:

            areas = np.empty(self.fieldinput.n_bhes, dtype=float)
            for i, item in self.field_dict.items():
                areas[i] = item["area"]

        if color_by_area:
            a_min = np.nanmin(areas)
            a_max = np.nanmax(areas)
            denom = (a_max - a_min) if (a_max > a_min) else 1.0

        for i, poly in self.field_dict.items():
            if poly["cell"].is_empty:
                continue

            x, y = poly["cell"].exterior.xy

            if color_by_area:
                t = (areas[i] - a_min) / denom
                ax.fill(
                    x,
                    y,
                    alpha=alpha,
                    linewidth=linewidth,
                    edgecolor="k",
                    facecolor=plt.cm.viridis(t),
                )
            else:
                ax.fill(
                    x,
                    y,
                    alpha=alpha,
                    linewidth=linewidth,
                    edgecolor="k",
                    facecolor="whitesmoke",
                )

            
            if show_area or show_req:
                rp = poly["cell"].representative_point()
                lines = []
                if show_area:
                    lines.append(f"A={poly['area']:.3g}")
                if show_req:
                    req = np.sqrt(poly['area'] / np.pi)
                    lines.append(f"r_eq={req:.3g}")
                rp = poly["cell"].representative_point()
                ax.text(
                    rp.x, rp.y, "\n".join(lines), ha="center", va="center", fontsize=8
                )


        if show_points:
            xs = [c[0] for c in self.fieldinput._borehole_coordinates]
            ys = [c[1] for c in self.fieldinput._borehole_coordinates]
            ax.scatter(xs, ys, s=point_size, marker="o", c="k")

            if show_ids:
                for i, (x, y) in enumerate(self.fieldinput._borehole_coordinates):
                    ax.text(
                        x + 0.3, y + 0.3, f"{i}", fontsize=8, ha="left", va="bottom"
                    )

        if show_graph:
            for i, j in self._borehole_graph.edges():
                x1, y1 = self.fieldinput._borehole_coordinates[i]
                x2, y2 = self.fieldinput._borehole_coordinates[j]
                ax.plot([x1, x2], [y1, y2], color="lightcoral", linewidth=1.0, zorder=2)

        xmin, ymin, xmax, ymax = (
            self.fieldinput.xmin,
            self.fieldinput.ymin,
            self.fieldinput.xmax,
            self.fieldinput.ymax,
        )
        rect_x = [xmin, xmax, xmax, xmin, xmin]
        rect_y = [ymin, ymin, ymax, ymax, ymin]
        ax.plot(rect_x, rect_y, linewidth=linewidth)

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        if show:
            plt.show()

        return fig, ax
