import numpy as np
from sklearn.neighbors import KDTree
from typing import List, Optional, Tuple, Union


def get_adjacency_matrix_from_edge_list(
        n: int,
        edges: Optional[Union[List[Tuple[int, int]], np.ndarray]] = None,
        directed: bool = False,
        self_connections: bool = False,
) -> np.ndarray:
    """
    Create an adjacency matrix for `n` vertices, given the edge list.

    Args:
        n: Number of vertices.
        edges: List of tuples defining the vertices connected by the edges - (v_from, v_to) if directed. Alternatively,
            2D array of shape [n, 2] with edges in rows can also be passed.
        directed: If graph is undirected, opposite edges will also be included.
        self_connections: Whether to include the self-connections of vertices.

    Returns:
        Two-dimensional array with integer zeros (no connection) and ones (connection).
    """
    # Initialize adjacency matrix with zeros.
    adj = np.zeros([n, n], dtype=np.int32)

    # Process the edges.
    edges = np.array(edges if (isinstance(edges, np.ndarray) or edges) else [], dtype=np.int32).reshape(-1, 2)

    if edges.shape[0] > 0:
        assert np.min(edges) >= 0 and np.max(edges) < n, "Edges out of range"

    # Put ones where the edges belong.
    # [OBFUSCATED CODE CLARIFICATION] The operation below performs low-level at-index addition in a loop through the
    # pairs of indices.
    np.add.at(adj, tuple(edges.T), 1)  # Original direction

    if not directed:
        np.add.at(adj, tuple(np.flip(edges.T, axis=0)), 1)  # Opposite direction

    # Self-connections are included through the addition of a diagonal.
    if self_connections:
        adj += np.eye(n, dtype=np.int32)

    # If any edge appeared more than once, no problem, just clip to one.
    # Should the warning be shown, this is a right place to do it in the future.
    adj = np.minimum(adj, 1)

    return adj


def get_neighbors_graph_with_kdtree(
        X: np.ndarray,
        kdtree: KDTree,
        radius: Optional[Union[int, float]] = None,
        limit: Optional[int] = None,
        randomized: bool = False,
) -> List[np.ndarray]:
    """
    Constructs a neighboring graphs using a KD-Tree in a batch setting.

    KD-Tree is used specifically for faster processing of large inputs (high number of vertices).

    Args:
        X: Query points organized in a 3D or 2D array. If 2D array is passed, each row is a single point of a given
            dimensionality; in case of 3D input, the outermost dimension is for batch.
        kdtree: KDTree with the points to be extracted.
        radius: Maximum interaction distance for neighbors.
        limit: Maximum number of neighbors per vertex.
        randomized: If both limit and radius are set, you may want to randomize the results before limiting. Otherwise,
            the results are returned sorted.

    Returns:
        A list of graphs in COOrdinate format, organized as a 2D array, where first column corresponds to the edge
        sources (from X) and the second one to the edge sinks (from KD-Tree). Keep in mind, however, that the same index
        in X and KD-Tree MAY NOT correspond to the same point, unless you made it so.
    """
    assert radius or limit, "Either `radius` or `limit` has to be set."

    # If there is a single query matrix, parse it as a single-example batch anyway.
    if len(X.shape) == 2:
        X = X[np.newaxis, ...]

    def shuffle(x: np.ndarray) -> np.ndarray:
        # Supplementary function for inline shuffle. Maybe a better solution can be found.
        np.random.shuffle(x)
        return x

    # Initialize the (yet empty) list of graphs in COO format.
    coo_graphs = []

    # Prepare proper mapping for parsing the neighbors pairs.
    # * Pair the source vertex (x[0]: int) with the neighbors (x[1]: array 1D).
    # * Parse as 2D array of shape [_, 2].
    if randomized:
        # Shuffle and limit the results, if exceeding.
        def mapping(x: Tuple[int, np.ndarray]) -> np.ndarray:
            return shuffle(
                np.pad(x[1].reshape(-1, 1), pad_width=[(0, 0), (1, 0)], constant_values=x[0])
            )[:limit]
    else:
        # Just limit the results, if exceeding.
        def mapping(x: Tuple[int, np.ndarray]) -> np.ndarray:
            return np.pad(x[1].reshape(-1, 1), pad_width=[(0, 0), (1, 0)], constant_values=x[0])[:limit]

    # Prepare a proper neighbor retrieval method.
    if radius:
        # Search inside a given radius.
        def obtain_neighbors(X_in_batch: np.ndarray) -> np.ndarray:
            # Note: In query_radius `sort_results=True` requires `return_distance=True`.
            return kdtree.query_radius(X_in_batch, r=radius, sort_results=True, return_distance=True)[0]
    else:
        # Unconstrained search.
        def obtain_neighbors(X_in_batch: np.ndarray) -> np.ndarray:
            return kdtree.query(X_in_batch, k=limit, sort_results=True, return_distance=False)

    for X_in_batch in X:
        # Repeat over each element of the batch.

        neighbors = list(map(
            # Mapping prepared earlier.
            mapping,

            # Enumerate over the neighbors returned by KD-Tree query.
            enumerate(obtain_neighbors(X_in_batch)),
        ))

        # Flatten the neighbors list to a single array of shape [_, 2].
        neighbors = np.concatenate(neighbors, axis=0)

        # Append the COO representation to the list.
        coo_graphs.append(neighbors)

    return coo_graphs


def get_neighbors_graph_pairwise(
        X: np.ndarray,
        X_ext: np.ndarray,
        radius: Optional[Union[int, float]] = None,
        limit: Optional[int] = None,
        randomized: bool = False,
) -> List[np.ndarray]:
    """
    Constructs a neighboring graphs using mostly vectorized all-vs-all exhaustive pairwise distance comparison.

    Might be quick for input of moderate size, but for better performance with large inputs consider using KD-Tree
    equivalent of this function.

    Args:
        X: Query points organized in a 3D or 2D array. If 2D array is passed, each row is a single point of a given
            dimensionality; in case of 3D input, the outermost dimension is for batch.
        X_ext: 2D array with the points to be extracted.
        radius: Maximum interaction distance for neighbors.
        limit: Maximum number of neighbors per vertex.
        randomized: If both limit and radius are set, you may want to randomize the results before limiting. Otherwise,
            the results are returned sorted.

    Returns:
        A list of graphs in COOrdinate format, organized as a 2D array, where first column corresponds to the edge
        sources (from X) and the second one to the edge sinks (from KD-Tree). Keep in mind, however, that the same index
        in X and KD-Tree MAY NOT correspond to the same point, unless you made it so.
    """
    assert radius or limit, "Either `radius` or `limit` has to be set."

    # If there is a single query matrix, parse it as a single-example batch anyway.
    if len(X.shape) == 2:
        X = X[np.newaxis, ...]

    def shuffle(x: np.ndarray) -> np.ndarray:
        # Supplementary function for inline shuffle. Maybe a better solution can be found.
        np.random.shuffle(x)
        return x

    # Calculate pairwise differences.
    diffs = (X[:, :, np.newaxis, :] - X_ext[np.newaxis, np.newaxis, :, :])  # Shape = [b, n_x, n_ext, 2]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))  # Shape = [b, n_x, n_ext]

    # Sort the pairs by their distance.
    idxs_sorted = np.argsort(dists, axis=-1)
    dists_sorted = np.take_along_axis(dists, idxs_sorted, axis=-1)

    if radius:
        # If radius is specified, find the indices of points inside it. The remaining ones mark with -1.
        neighbors_idx = np.where(dists_sorted <= radius * (1 + 1e-5), idxs_sorted, -1)
    else:
        # If radius is not specified, one can already apply the limit.
        neighbors_idx = idxs_sorted[..., :limit]

    # Pair the source vertices with their neighbors by appending source indices in front of them.
    neighbors_idx = np.pad(neighbors_idx[..., np.newaxis], [(0, 0), (0, 0), (0, 0), (1, 0)])
    neighbors_idx[:, :, :, 0] += np.arange(X.shape[1])[np.newaxis, :, np.newaxis]

    # Initialize the (yet empty) list of graphs in COO format.
    coo_graphs = []

    if randomized:
        # Shuffle and limit the results, if exceeding.
        def mapping(x: np.ndarray) -> np.ndarray:
            return shuffle(x[x[:, 1] >= 0])[:limit]
    else:
        # Just limit the results, if exceeding.
        def mapping(x: np.ndarray) -> np.ndarray:
            return x[x[:, 1] >= 0][:limit]

    for neighbors_idx_in_batch in neighbors_idx:
        # Repeat over each element of the batch.

        neighbors = list(map(
            # Mapping prepared earlier.
            mapping,

            # Relevant neighbors' indices.
            neighbors_idx_in_batch
        ))

        # Flatten the neighbors list to a single array of shape [_, 2].
        neighbors = np.concatenate(neighbors, axis=0)

        # Append the COO representation to the list.
        coo_graphs.append(neighbors)

    return coo_graphs
