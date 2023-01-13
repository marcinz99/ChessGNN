import numpy as np


def test_get_adjacency_matrix_from_edge_list():
    from chessgnn.utils.graph_processing import get_adjacency_matrix_from_edge_list

    # Create just self-connections
    np.testing.assert_equal(
        get_adjacency_matrix_from_edge_list(n=5, edges=[], self_connections=True),
        np.eye(5, dtype=np.int32)
    )

    # No connections at all
    np.testing.assert_equal(
        get_adjacency_matrix_from_edge_list(n=5, edges=[]),
        np.zeros([5, 5], dtype=np.int32)
    )

    # Empty matrix (0 vertices)
    np.testing.assert_equal(
        get_adjacency_matrix_from_edge_list(n=0, edges=[]),
        np.array([], dtype=np.int32).reshape(0, 0)
    )

    # Out of range values (negative)
    with np.testing.assert_raises(AssertionError):
        get_adjacency_matrix_from_edge_list(n=3, edges=[(1, 2), (-1, 0)])

    # Out of range values (too high)
    with np.testing.assert_raises(AssertionError):
        get_adjacency_matrix_from_edge_list(n=3, edges=[(1, 2), (1, 3)])

    # Regular input - directed
    np.testing.assert_equal(
        get_adjacency_matrix_from_edge_list(
            n=5, edges=[(0, 2), (0, 4), (1, 3), (1, 4), (2, 3), (3, 4)], directed=True),
        np.array([
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0]], dtype=np.int32)
    )

    # Regular input - undirected
    np.testing.assert_equal(
        get_adjacency_matrix_from_edge_list(
            n=5, edges=[(0, 2), (0, 4), (1, 3), (1, 4), (2, 3), (3, 4)]),
        np.array([
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [1, 1, 0, 1, 0]], dtype=np.int32)
    )

    # Regular input - undirected and with self-connections
    np.testing.assert_equal(
        get_adjacency_matrix_from_edge_list(
            n=5, edges=[(0, 2), (0, 4), (1, 3), (1, 4), (2, 3), (3, 4)], self_connections=True),
        np.array([
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 1],
            [1, 0, 1, 1, 0],
            [0, 1, 1, 1, 1],
            [1, 1, 0, 1, 1]], dtype=np.int32)
    )
