import pytest
import numpy as np
import tensorflow as tf


@pytest.fixture
def random_graph():
    from chessgnn.utils.random_graphs import sample_graph_scale_free

    coo = sample_graph_scale_free(n=20, min_k=0, max_k=5, cauchy_scale=2.5)

    rec = tf.reshape(coo[:, 0].astype(np.int32), [1, -1])
    sen = tf.reshape(coo[:, 1].astype(np.int32), [1, -1])

    V = tf.random.normal(shape=[1, 20, 2], dtype=tf.float32)
    E = tf.random.normal(shape=[1, len(coo), 2], dtype=tf.float32)
    u = tf.random.normal(shape=[1, 1, 3], dtype=tf.float32)

    return {'V_set': V, 'E_set': E, 'u_supernode': u, 'receivers': rec, 'senders': sen}


def test_gather_nodes(random_graph):
    from chessgnn.gnn.layers import GatherNodesLayer
    graph = random_graph

    gather_nodes = GatherNodesLayer()
    gather_nodes.build(input_shape=[[1, None, 2], [1, None]])

    def dumbed_down_gather_nodes(V_set, node_ids):
        result = [V_set[0, i] for i in node_ids[0]]
        result = np.stack(result)[np.newaxis, ...]
        return tf.identity(result)

    np.testing.assert_array_equal(
        gather_nodes([graph['V_set'], graph['senders']]),
        dumbed_down_gather_nodes(graph['V_set'], graph['senders']))


def test_filter_nodes(random_graph):
    from chessgnn.gnn.layers import FilterNodesLayer
    graph = random_graph

    filter_nodes = FilterNodesLayer()
    filter_nodes.build(input_shape=[[1, None, 2], [1, None]])

    def dumbed_down_filter_nodes(V_set, node_ids):
        result = [V_set[0, i] for i in node_ids[0] if i != -1]
        result = np.stack(result)[np.newaxis, ...]
        return tf.identity(result)

    mask = tf.where(tf.range(20) % 4 != 1, tf.range(20), -1)[tf.newaxis, :]
    np.testing.assert_array_equal(
        filter_nodes([graph['V_set'], mask]),
        dumbed_down_filter_nodes(graph['V_set'], mask))


def test_scatter_and_aggregate_sum(random_graph):
    from chessgnn.gnn.layers import ScatterAndAggregateLayer
    graph = random_graph

    scatter_and_aggregate_sum = ScatterAndAggregateLayer(agg_method='sum')
    scatter_and_aggregate_sum.build(input_shape=[[1, None, 2], [1, None, 2], [1, None]])

    def dumbed_down_scatter_and_aggregate_sum(V_set, E_set, node_ids):
        result = np.zeros([V_set.shape[1], E_set.shape[2]], dtype=np.float32)
        for i_e, i_vrec in enumerate(node_ids[0]):
            result[i_vrec] += E_set[0, i_e]
        result = result[np.newaxis, ...]
        return tf.identity(result)

    np.testing.assert_array_equal(
        scatter_and_aggregate_sum([graph['V_set'], graph['E_set'], graph['receivers']]),
        dumbed_down_scatter_and_aggregate_sum(graph['V_set'], graph['E_set'], graph['receivers']))


def test_scatter_and_aggregate_mean(random_graph):
    from chessgnn.gnn.layers import ScatterAndAggregateLayer
    graph = random_graph

    scatter_and_aggregate_mean = ScatterAndAggregateLayer(agg_method='mean')
    scatter_and_aggregate_mean.build(input_shape=[[1, None, 2], [1, None, 2], [1, None]])

    def dumbed_down_scatter_and_aggregate_mean(V_set, E_set, node_ids):
        result = np.zeros([V_set.shape[1], E_set.shape[2]], dtype=np.float32)
        counts = np.zeros([V_set.shape[1], 1], dtype=np.int32)
        for i_e, i_vrec in enumerate(node_ids[0]):
            result[i_vrec] += E_set[0, i_e]
            counts[i_vrec] += 1
        result = result / np.maximum(counts, 1)
        result = result[np.newaxis, ...]
        return tf.identity(result)

    np.testing.assert_array_almost_equal(
        scatter_and_aggregate_mean([graph['V_set'], graph['E_set'], graph['receivers']]),
        dumbed_down_scatter_and_aggregate_mean(graph['V_set'], graph['E_set'], graph['receivers']))


def test_scatter_and_aggregate_max(random_graph):
    from chessgnn.gnn.layers import ScatterAndAggregateLayer
    graph = random_graph

    scatter_and_aggregate_max = ScatterAndAggregateLayer(agg_method='max')
    scatter_and_aggregate_max.build(input_shape=[[1, None, 2], [1, None, 2], [1, None]])

    def dumbed_down_scatter_and_aggregate_max(V_set, E_set, node_ids):
        result = np.zeros([V_set.shape[1], E_set.shape[2]], dtype=np.float32) - np.inf
        counts = np.zeros([V_set.shape[1], 1], dtype=np.int32)
        for i_e, i_vrec in enumerate(node_ids[0]):
            result[i_vrec] = np.maximum(result[i_vrec], E_set[0, i_e])
            counts[i_vrec] += 1
        result = np.where(counts > 0, result, 0.0)
        result = result[np.newaxis, ...]
        return tf.identity(result)

    np.testing.assert_array_equal(
        scatter_and_aggregate_max([graph['V_set'], graph['E_set'], graph['receivers']]),
        dumbed_down_scatter_and_aggregate_max(graph['V_set'], graph['E_set'], graph['receivers']))


def test_scatter_and_softmax(random_graph):
    from chessgnn.gnn.layers import ScatterAndSoftmaxLayer
    graph = random_graph

    scatter_and_softmax = ScatterAndSoftmaxLayer()
    scatter_and_softmax.build(input_shape=[[1, None, 2], [1, None, 2], [1, None]])

    def dumbed_down_scatter_and_softmax(V_set, E_set, node_ids):
        from scipy.special import softmax
        result = np.zeros_like(E_set[0])
        for i in np.unique(node_ids):
            result += softmax(np.where(node_ids[0, :, np.newaxis] == i, E_set[0], -np.inf), axis=0)
        result = result[np.newaxis, ...]
        return tf.identity(result)

    np.testing.assert_array_almost_equal(
        scatter_and_softmax([graph['V_set'], graph['E_set'], graph['receivers']]),
        dumbed_down_scatter_and_softmax(graph['V_set'], graph['E_set'], graph['receivers']))


def test_concat_edges_ends_only(random_graph):
    from chessgnn.gnn.layers import ConcatEdgesEndsOnlyLayer
    graph = random_graph

    concat_edges_ends_only = ConcatEdgesEndsOnlyLayer()
    concat_edges_ends_only.build(input_shape=[[1, None, 2], [1, None], [1, None]])

    def dumbed_down_concat_edges_ends_only(V_set, a_node_ids, b_node_ids):
        subres_a = [V_set[0, i] for i in a_node_ids[0]]
        subres_a = np.stack(subres_a)
        subres_b = [V_set[0, i] for i in b_node_ids[0]]
        subres_b = np.stack(subres_b)
        result = np.concatenate([subres_a, subres_b], axis=-1)
        result = result[np.newaxis, ...]
        return tf.identity(result)

    np.testing.assert_array_equal(
        concat_edges_ends_only([graph['V_set'], graph['receivers'], graph['senders']]),
        dumbed_down_concat_edges_ends_only(graph['V_set'], graph['receivers'], graph['senders']))


def test_concat_edge_with_ends(random_graph):
    from chessgnn.gnn.layers import ConcatEdgeWithEndsLayer
    graph = random_graph

    concat_edge_with_ends = ConcatEdgeWithEndsLayer()
    concat_edge_with_ends.build(input_shape=[[1, None, 2], [1, None, 2], [1, None], [1, None]])

    def dumbed_down_concat_edge_with_ends(V_set, E_set, a_node_ids, b_node_ids):
        subres_a = [V_set[0, i] for i in a_node_ids[0]]
        subres_a = np.stack(subres_a)
        subres_b = [V_set[0, i] for i in b_node_ids[0]]
        subres_b = np.stack(subres_b)
        result = np.concatenate([E_set[0], subres_a, subres_b], axis=-1)
        result = result[np.newaxis, ...]
        return tf.identity(result)

    np.testing.assert_array_equal(
        concat_edge_with_ends([graph['V_set'], graph['E_set'], graph['receivers'], graph['senders']]),
        dumbed_down_concat_edge_with_ends(graph['V_set'], graph['E_set'], graph['receivers'], graph['senders']))


def test_concat_edge_with_single_end(random_graph):
    from chessgnn.gnn.layers import ConcatEdgeWithSingleEndLayer
    graph = random_graph

    concat_edge_with_single_end = ConcatEdgeWithSingleEndLayer()
    concat_edge_with_single_end.build(input_shape=[[1, None, 2], [1, None, 2], [1, None]])

    def dumbed_down_concat_edge_with_single_end(V_set, E_set, node_ids):
        subres = [V_set[0, i] for i in node_ids[0]]
        subres = np.stack(subres)
        result = np.concatenate([E_set[0], subres], axis=-1)
        result = result[np.newaxis, ...]
        return tf.identity(result)

    np.testing.assert_array_equal(
        concat_edge_with_single_end([graph['V_set'], graph['E_set'], graph['receivers']]),
        dumbed_down_concat_edge_with_single_end(graph['V_set'], graph['E_set'], graph['receivers']))


def test_concat_globals_with_anything(random_graph):
    from chessgnn.gnn.layers import ConcatGlobalsWithAnythingLayer
    graph = random_graph

    concat_globals_with_anything = ConcatGlobalsWithAnythingLayer()
    concat_globals_with_anything.build(input_shape=[[1, 3], [1, None, 2]])

    def dumbed_down_concat_globals_with_anything(u_supernode, X):
        subres = np.tile(u_supernode[0], reps=[X.shape[1], 1])
        result = np.concatenate([subres, X[0]], axis=-1)
        result = result[np.newaxis, ...]
        return tf.identity(result)

    np.testing.assert_array_equal(
        concat_globals_with_anything([graph['u_supernode'], graph['V_set']]),
        dumbed_down_concat_globals_with_anything(graph['u_supernode'], graph['V_set']))
    np.testing.assert_array_equal(
        concat_globals_with_anything([graph['u_supernode'], graph['E_set']]),
        dumbed_down_concat_globals_with_anything(graph['u_supernode'], graph['E_set']))


def test_cluster_pooling_new_coo(random_graph):
    from chessgnn.gnn.layers import ClusterPoolingNewCOOLayer
    graph = random_graph

    cluster_pooling_new_coo = ClusterPoolingNewCOOLayer()
    cluster_pooling_new_coo.build(input_shape=[[1, None], [1, None], [1, None]])

    def dumbed_down_cluster_pooling_new_coo(rec_node_ids, sen_node_ids, clusters):
        a = np.zeros([clusters.shape[1], clusters.shape[1]], dtype=np.int32)
        for i, j in zip(rec_node_ids[0], sen_node_ids[0]):
            a[i, j] = 1
        b = np.zeros([np.max(clusters[0]) + 1, clusters.shape[1]], dtype=np.int32)
        for i, j in enumerate(clusters[0]):
            if j >= 0:
                b[j] = np.maximum(b[j], a[i])
        c = np.zeros([np.max(clusters[0]) + 1, np.max(clusters[0]) + 1], dtype=np.int32)
        for i, j in enumerate(clusters[0]):
            if j >= 0:
                c[:, j] = np.maximum(c[:, j], b[:, i])
        new_rec, new_sen = np.nonzero(c)
        new_rec = new_rec[np.newaxis, ...]
        new_sen = new_sen[np.newaxis, ...]
        return tf.identity(new_rec), tf.identity(new_sen)

    mask2 = tf.where(tf.range(20) % 7 != 3, tf.range(20) % 5, -1)[tf.newaxis, :]
    np.testing.assert_array_equal(
        cluster_pooling_new_coo([graph['receivers'], graph['senders'], mask2]),
        dumbed_down_cluster_pooling_new_coo(graph['receivers'], graph['senders'], mask2))
