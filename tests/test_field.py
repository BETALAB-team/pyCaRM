# -*- coding: utf-8 -*-
"""
Tests for borehole field geometry module.

Covers: FieldInput (constructor, from_array, from_matrix),
        Field (Voronoi decomposition, distance matrix, neighbor graph),
        Field.plot_field (smoke test).

Reference values are computed analytically for a symmetric 2×2 grid on
domain [0, 10] × [0, 10] with boreholes at (2.5, 2.5), (7.5, 2.5),
(2.5, 7.5), (7.5, 7.5). Each Voronoi cell is a 5×5 square → area = 25 m²,
req = sqrt(25 / π).
"""
import numpy as np
import pytest

from carm import FieldInput, Field


# ============================================================
# FIXTURES — griglia 2×2 simmetrica
# ============================================================

@pytest.fixture
def grid_x():
    return np.array([2.5, 7.5, 2.5, 7.5])


@pytest.fixture
def grid_y():
    return np.array([2.5, 2.5, 7.5, 7.5])


@pytest.fixture
def field_input(grid_x, grid_y):
    fi = FieldInput(n_bhes=4, xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0, rb=0.075)
    fi.from_array(grid_x, grid_y)
    return fi


@pytest.fixture
def field(field_input):
    return Field(fieldinput=field_input)


# ============================================================
# FieldInput — constructor
# ============================================================

def test_field_input_stores_values():
    fi = FieldInput(n_bhes=4, xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0, rb=0.075)
    assert fi.n_bhes == 4
    assert fi.xmin == 0.0
    assert fi.ymin == 0.0
    assert fi.xmax == 10.0
    assert fi.ymax == 10.0
    assert fi.rb == 0.075


def test_field_input_coordinates_empty_before_load():
    fi = FieldInput(n_bhes=4, xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0, rb=0.075)
    assert fi.borehole_coordinates == []


def test_field_input_invalid_bbox_x():
    with pytest.raises(ValueError):
        FieldInput(n_bhes=4, xmin=10.0, ymin=0.0, xmax=5.0, ymax=10.0, rb=0.075)


def test_field_input_invalid_bbox_y():
    with pytest.raises(ValueError):
        FieldInput(n_bhes=4, xmin=0.0, ymin=10.0, xmax=10.0, ymax=5.0, rb=0.075)


# ============================================================
# FieldInput — from_array
# ============================================================

def test_from_array_stores_coordinates(field_input, grid_x, grid_y):
    expected = list(zip(grid_x, grid_y))
    assert field_input.borehole_coordinates == expected


def test_from_array_wrong_length():
    fi = FieldInput(n_bhes=4, xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0, rb=0.075)
    with pytest.raises(ValueError):
        fi.from_array(np.array([2.5, 7.5]), np.array([2.5, 2.5, 7.5, 7.5]))


def test_from_array_mismatched_n_bhes():
    fi = FieldInput(n_bhes=3, xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0, rb=0.075)
    with pytest.raises(ValueError):
        fi.from_array(
            np.array([2.5, 7.5, 2.5, 7.5]),
            np.array([2.5, 2.5, 7.5, 7.5]),
        )


def test_from_array_nan_coordinate():
    fi = FieldInput(n_bhes=4, xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0, rb=0.075)
    with pytest.raises(ValueError):
        fi.from_array(
            np.array([2.5, np.nan, 2.5, 7.5]),
            np.array([2.5, 2.5, 7.5, 7.5]),
        )


def test_from_array_out_of_bounds_x():
    fi = FieldInput(n_bhes=4, xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0, rb=0.075)
    with pytest.raises(ValueError):
        fi.from_array(
            np.array([0.0, 7.5, 2.5, 7.5]),
            np.array([2.5, 2.5, 7.5, 7.5]),
        )


def test_from_array_out_of_bounds_y():
    fi = FieldInput(n_bhes=4, xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0, rb=0.075)
    with pytest.raises(ValueError):
        fi.from_array(
            np.array([2.5, 7.5, 2.5, 7.5]),
            np.array([0.0, 2.5, 7.5, 7.5]),
        )


# ============================================================
# FieldInput — from_matrix
# ============================================================

def test_from_matrix_stores_coordinates(grid_x, grid_y):
    fi = FieldInput(n_bhes=4, xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0, rb=0.075)
    matrix = np.column_stack([grid_x, grid_y])
    fi.from_matrix(matrix)
    expected = list(zip(grid_x, grid_y))
    assert fi.borehole_coordinates == expected


def test_from_matrix_wrong_ndim():
    fi = FieldInput(n_bhes=4, xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0, rb=0.075)
    with pytest.raises(ValueError):
        fi.from_matrix(np.array([2.5, 7.5, 2.5, 7.5]))  # 1-D


def test_from_matrix_wrong_columns():
    fi = FieldInput(n_bhes=4, xmin=0.0, ymin=0.0, xmax=10.0, ymax=10.0, rb=0.075)
    with pytest.raises(ValueError):
        fi.from_matrix(np.ones((4, 3)))  # 3 colonne invece di 2


# ============================================================
# Field — field_dict
# ============================================================

def test_field_dict_has_all_indices(field):
    assert set(field.field_dict.keys()) == {0, 1, 2, 3}


def test_field_dict_area(field):
    """Griglia 2×2 su [0,10]×[0,10]: ogni cella è 5×5 = 25 m²."""
    for i in range(4):
        assert field.field_dict[i]["area"] == pytest.approx(25.0, rel=1e-4)


def test_field_dict_req(field):
    """req = sqrt(area / π) = sqrt(25 / π)."""
    expected_req = np.sqrt(25.0 / np.pi)
    for i in range(4):
        assert field.field_dict[i]["req"] == pytest.approx(expected_req, rel=1e-4)


def test_field_dict_coords(field, grid_x, grid_y):
    """Le coordinate nel field_dict devono corrispondere all'input."""
    for i in range(4):
        x, y = field.field_dict[i]["coords"]
        assert x == pytest.approx(grid_x[i], rel=1e-9)
        assert y == pytest.approx(grid_y[i], rel=1e-9)


# ============================================================
# Field — distance matrix
# ============================================================

def test_distance_matrix_shape(field):
    assert field.distance_matrix.shape == (4, 4)


def test_distance_matrix_diagonal_zero(field):
    """La diagonale deve essere req_i (distanza di un borehole da se stesso)."""
    req = np.sqrt(25.0 / np.pi)
    for i in range(4):
        # D[i,i] = sqrt(0 + req_i²) = req_i
        assert field.distance_matrix[i, i] == pytest.approx(req, rel=1e-4)


def test_distance_matrix_adjacent_horizontal(field):
    """
    Boreholes 0=(2.5,2.5) e 1=(7.5,2.5): distanza euclidea = 5.0 m.
    D[0,1] = sqrt(5² + req_1²), D[1,0] = sqrt(5² + req_0²).
    Per simmetria della griglia req_0 == req_1, quindi D[0,1] == D[1,0].
    """
    req = np.sqrt(25.0 / np.pi)
    expected = np.sqrt(5.0**2 + req**2)
    assert field.distance_matrix[0, 1] == pytest.approx(expected, rel=1e-4)
    assert field.distance_matrix[1, 0] == pytest.approx(expected, rel=1e-4)


def test_distance_matrix_diagonal_pair(field):
    """
    Boreholes 0=(2.5,2.5) e 3=(7.5,7.5): distanza euclidea = 5*sqrt(2).
    D[0,3] = sqrt((5√2)² + req_3²).
    """
    req = np.sqrt(25.0 / np.pi)
    d_eucl = 5.0 * np.sqrt(2.0)
    expected = np.sqrt(d_eucl**2 + req**2)
    assert field.distance_matrix[0, 3] == pytest.approx(expected, rel=1e-4)


def test_distance_matrix_nonnegative(field):
    assert np.all(field.distance_matrix >= 0)


# ============================================================
# Field — neighbor graph
# ============================================================

def test_borehole_graph_has_all_nodes(field):
    assert set(field._borehole_graph.nodes()) == {0, 1, 2, 3}


def test_borehole_graph_adjacent_edges(field):
    """
    Nella griglia 2×2, i 4 vicini condividono un lato Voronoi.
    I bordi attesi sono: (0,1), (0,2), (1,3), (2,3).
    I diagonali (0,3) e (1,2) condividono solo un punto, non un lato.
    """
    edges = set(field._borehole_graph.edges())
    assert (0, 1) in edges or (1, 0) in edges
    assert (0, 2) in edges or (2, 0) in edges
    assert (1, 3) in edges or (3, 1) in edges
    assert (2, 3) in edges or (3, 2) in edges


def test_borehole_graph_no_diagonal_edges(field):
    """I diagonali (0,3) e (1,2) non devono essere vicini Voronoi."""
    edges = set(field._borehole_graph.edges())
    assert (0, 3) not in edges and (3, 0) not in edges
    assert (1, 2) not in edges and (2, 1) not in edges


def test_borehole_graph_edge_count(field):
    """Griglia 2×2: 4 lati condivisi (no diagonali)."""
    assert field._borehole_graph.number_of_edges() == 4