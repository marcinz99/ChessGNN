import numpy as np
from typing import Optional, List, Union


def sample_graph_configuration_model(
        k_out: Union[List[int], np.ndarray],
        k_in: Optional[Union[List[int], np.ndarray]] = None,
        self_connections: bool = False,
        directed: bool = False,
) -> np.ndarray:
    """
    Generate a random graph according to configuration model.
    """
    sources = np.zeros(np.sum(k_out) + 1, dtype=int)
    np.add.at(sources, np.cumsum(k_out)[:-1], 1)
    sources = np.cumsum(sources)[:-1]

    if k_in is not None:
        sinks = np.zeros(np.sum(k_in) + 1, dtype=int)
        np.add.at(sinks, np.cumsum(k_in)[:-1], 1)
        sinks = np.cumsum(sinks)[:-1]
    else:
        sinks = sources.copy()
        k_in = k_out.copy()

    np.random.shuffle(sinks)
    sinks = sinks.tolist()

    adj = set()
    k_out_c = k_out.copy()
    k_in_c = k_in.copy()

    for i in sources:
        if k_out_c[i] < 1:
            continue
        tries_left = len(sinks)

        while sinks and tries_left:
            j = sinks.pop(0)
            tries_left -= 1

            if i == j or (i, j) in adj:
                sinks.append(j)

            elif k_in_c[j] >= 1:
                adj.add((i, j))
                k_out_c[i] -= 1
                k_in_c[j] -= 1

                if not directed:
                    adj.add((j, i))
                    k_out_c[j] -= 1
                    k_in_c[i] -= 1

                break

    return np.array(sorted(adj))


def scale_free_cauchy_degree_distribution(n: int, min_k: int, max_k: int, cauchy_scale: float = 1.0) -> np.ndarray:
    k_dist = np.floor((cauchy_scale * np.abs(np.random.standard_cauchy(n))) % (max_k - min_k)).astype(int) + min_k

    if np.sum(k_dist) % 2 == 1:
        for i, is_ok in enumerate(k_dist != max_k):
            if is_ok:
                k_dist[i] += 1
                break

    return k_dist


def sample_graph_scale_free(n: int, min_k: int, max_k: int, cauchy_scale: float = 1.0) -> np.ndarray:
    """
    Generate a random graph with scale-free property.
    """
    k_dist = scale_free_cauchy_degree_distribution(n, min_k, max_k, cauchy_scale)
    coo_graph = sample_graph_configuration_model(k_dist, k_dist)

    return coo_graph
