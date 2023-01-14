import numpy as np


def test_get_adjacency_matrix_from_edge_list():
    from chessgnn.utils.graph_processing import get_adjacency_matrix_from_edge_list

    # Create just self-connections
    np.testing.assert_array_equal(
        get_adjacency_matrix_from_edge_list(n=5, edges=[], self_connections=True),
        np.eye(5, dtype=np.int32)
    )

    # No connections at all
    np.testing.assert_array_equal(
        get_adjacency_matrix_from_edge_list(n=5, edges=[]),
        np.zeros([5, 5], dtype=np.int32)
    )

    # Empty matrix (0 vertices)
    np.testing.assert_array_equal(
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
    np.testing.assert_array_equal(
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
    np.testing.assert_array_equal(
        get_adjacency_matrix_from_edge_list(
            n=5, edges=[(0, 2), (0, 4), (1, 3), (1, 4), (2, 3), (3, 4)]),
        np.array([
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [1, 1, 0, 1, 0]], dtype=np.int32)
    )

    # Regular input - undirected (from np.ndarray)
    np.testing.assert_array_equal(
        get_adjacency_matrix_from_edge_list(
            n=5, edges=np.array([[0, 2], [0, 4], [1, 3], [1, 4], [2, 3], [3, 4]])),
        np.array([
            [0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1],
            [1, 0, 0, 1, 0],
            [0, 1, 1, 0, 1],
            [1, 1, 0, 1, 0]], dtype=np.int32)
    )

    # Regular input - undirected and with self-connections
    np.testing.assert_array_equal(
        get_adjacency_matrix_from_edge_list(
            n=5, edges=[(0, 2), (0, 4), (1, 3), (1, 4), (2, 3), (3, 4)], self_connections=True),
        np.array([
            [1, 0, 1, 0, 1],
            [0, 1, 0, 1, 1],
            [1, 0, 1, 1, 0],
            [0, 1, 1, 1, 1],
            [1, 1, 0, 1, 1]], dtype=np.int32)
    )


def test_get_neighbors_graph_with_kdtree():
    from sklearn.neighbors import KDTree
    from chessgnn.utils.graph_processing import get_neighbors_graph_with_kdtree

    space = np.array(  # Points to be retrieved.
        [[0.6, 0.5], [-0.7, 0.1], [0.1, 0.5], [0.4, 0.4], [-0.4, 0.7],
         [0.4, -2.0], [0.2, -1.0], [-0.4, 0.3], [-0.1, 0.4], [-0.4, 1.1]])
    kdt = KDTree(space)

    X = np.array([  # Query points.
        [[1.31, -0.57], [0.04, -0.1], [0.7, 0.49], [2.05, 1.15],
         [0.93, -1.32], [1.38, -0.38], [-0.27, -0.06], [-0.2, -0.25]],
        [[-1.01, -0.14], [-1.36, -0.59], [2.02, -0.37], [0.29, 0.14],
         [-1.87, -1.1], [-1.01, -0.58], [0.43, 0.4], [0.74, 0.34]]
    ])

    # Note: The solution below seems to be nice and easy (and is!), but scales rather poorly.
    diffs = (X[:, :, np.newaxis, :] - space[np.newaxis, np.newaxis, :, :])  # Shape = [2, 8, 10, 2]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))  # Shape = [2, 8, 10]

    idxs_sorted = np.argsort(dists, axis=-1)
    dists_sorted = np.take_along_axis(dists, idxs_sorted, axis=-1)

    # Basic test - limit
    np.testing.assert_array_equal(
        get_neighbors_graph_with_kdtree(X, kdt, limit=2)[0],
        np.array([
            [0, 6], [0, 0], [1, 8], [1, 7], [2, 0], [2, 3], [3, 0], [3, 3],
            [4, 6], [4, 5], [5, 0], [5, 3], [6, 7], [6, 1], [7, 7], [7, 1]])
    )

    # Basic test - radius
    np.testing.assert_array_equal(
        get_neighbors_graph_with_kdtree(X, kdt, radius=0.5)[0],
        np.array([[2, 0], [2, 3], [6, 7], [6, 1], [6, 8]])
    )

    # Basic test - 2D and 3D single-example batch should return the same output.
    np.testing.assert_array_equal(
        get_neighbors_graph_with_kdtree(X[0], kdt, limit=4)[0],
        get_neighbors_graph_with_kdtree(X[:1], kdt, limit=4)[0]
    )

    # Just limit the number of neighbors.
    reference = idxs_sorted[:, :, :5]
    tested_results = get_neighbors_graph_with_kdtree(X, kdt, limit=5)

    for idx_ref, coo_graph in zip(reference, tested_results):
        ref_graph = []
        for i, row in enumerate(idx_ref):
            ref_graph += list(map(lambda x: (i, x), row))
        ref_graph = np.array(ref_graph)

        np.testing.assert_array_equal(ref_graph, coo_graph)

    # Just in a radius.
    reference = np.where(dists_sorted <= 1.0 + 1e-5, idxs_sorted, -1)
    tested_results = get_neighbors_graph_with_kdtree(X, kdt, radius=1)

    for idx_ref, coo_graph in zip(reference, tested_results):
        ref_graph = []
        for i, row in enumerate(idx_ref):
            ref_graph += list(filter(lambda x: x[1] >= 0, (map(lambda x: (i, x), row))))
        ref_graph = np.array(ref_graph)

        np.testing.assert_array_equal(ref_graph, coo_graph)

    # Limit the number of neighbors in a radius.
    reference = np.where(dists_sorted <= 1.0 + 1e-5, idxs_sorted, -1)[:, :, :3]
    tested_results = get_neighbors_graph_with_kdtree(X, kdt, radius=1, limit=3)

    for idx_ref, coo_graph in zip(reference, tested_results):
        ref_graph = []
        for i, row in enumerate(idx_ref):
            ref_graph += list(filter(lambda x: x[1] >= 0, (map(lambda x: (i, x), row))))
        ref_graph = np.array(ref_graph)

        np.testing.assert_array_equal(ref_graph, coo_graph)

    # If randomized output is expected, two consecutive run should not be equal by design (usually).
    np.testing.assert_equal(
        np.array_equal(
            get_neighbors_graph_with_kdtree(X[0], kdt, limit=7, randomized=True),
            get_neighbors_graph_with_kdtree(X[0], kdt, limit=7, randomized=True)
        ),
        False)
