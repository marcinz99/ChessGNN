import numpy as np
from typing import List, Optional, Tuple


def get_adjacency_matrix_from_edge_list(
        n: int,
        edges: Optional[List[Tuple[int, int]]] = None,
        directed: bool = False,
        self_connections: bool = False,
) -> np.ndarray:
    """
    Create an adjacency matrix for `n` vertices, given the edge list.

    Args:
        n: Number of vertices.
        edges: List of tuples defining the vertices connected by the edges - (v_from, v_to) if directed.
        directed: If graph is undirected, opposite edges will also be included.
        self_connections: Whether to include the self-connections of vertices.

    Returns:
        Two-dimensional array with integer zeros (no connection) and ones (connection).
    """
    # Initialize adjacency matrix with zeros
    adj = np.zeros([n, n], dtype=np.int32)

    # Process the edges
    edges = np.array(edges if edges else [], dtype=np.int32).reshape(-1, 2)

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
