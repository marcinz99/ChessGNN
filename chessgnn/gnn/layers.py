import tensorflow as tf


class GatherNodesLayer(tf.keras.layers.Layer):
    """
    OP: v -> e'
    IN: V_set, node_ids

    Transition from V-set to E-set through node gathering.
    """
    def __init__(self, **kwargs):
        super(GatherNodesLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        V_set, node_ids = inputs

        return tf.gather(V_set[0], node_ids[0])[tf.newaxis, ...]


class FilterNodesLayer(tf.keras.layers.Layer):
    """
    OP: v -> v'
    IN: V_set, node_ids

    Transition from V-set to V-set through node filtering (discards nodes with id -1).
    """
    def __init__(self, **kwargs):
        super(FilterNodesLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        V_set, node_ids = inputs

        return tf.boolean_mask(V_set[0], node_ids[0] != -1)[tf.newaxis, ...]


class ScatterAndAggregateLayer(tf.keras.layers.Layer):
    """
    OP: e -> v'
    IN: V_set, E_set, node_ids

    Scatter E-set entries into buckets defined by node_ids and aggregate accordingly. V-set is passed only for shape
    preservation.
    """
    def __init__(self, agg_method='sum', **kwargs):
        super(ScatterAndAggregateLayer, self).__init__(**kwargs)
        self.agg_method = agg_method
        self.agg_method_func = None
        self.exdtended_pad = None

    def build(self, input_shape):
        self.agg_method_func = {
            'sum': tf.math.segment_sum,
            'mean': tf.math.segment_mean,
            'max': tf.math.segment_max,
        }[self.agg_method]

        self.exdtended_pad = [[0, 0] for _ in range(max(0, len(input_shape[1]) - 3))]

    def call(self, inputs):
        V_set, E_set, node_ids = inputs

        residue_pad = (
                tf.reduce_sum(tf.ones_like(V_set[0, ..., :1], dtype=tf.int32))
                - tf.maximum(tf.reduce_max(node_ids[0]) + 1, 0)
        )
        return tf.pad(
            self.agg_method_func(E_set[0], node_ids[0]),
            paddings=[[0, residue_pad], [0, 0]] + self.exdtended_pad,
        )[tf.newaxis, ...]


class ScatterAndSoftmaxLayer(tf.keras.layers.Layer):
    """
    OP: e -> e'
    IN: V_set, E_set, node_ids

    Segment softmax operation over E-set. V-set is passed only for shape preservation.
    """
    def __init__(self, **kwargs):
        super(ScatterAndSoftmaxLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        V_set, E_set, node_ids = inputs

        residue_pad = (
                tf.reduce_sum(tf.ones_like(V_set[0, :, :1], dtype=tf.int32))
                - tf.maximum(tf.reduce_max(node_ids[0]) + 1, 0)
        )
        segment_maxes = tf.pad(tf.math.segment_max(E_set[0], node_ids[0]), paddings=[[0, residue_pad], [0, 0]])
        adjusted_E_set = tf.exp(E_set[0] - tf.gather(segment_maxes, node_ids[0]))

        segment_expsums = tf.pad(tf.math.segment_sum(adjusted_E_set, node_ids[0]), paddings=[[0, residue_pad], [0, 0]])
        return (adjusted_E_set / tf.gather(segment_expsums, node_ids[0]))[tf.newaxis, ...]


class ConcatEdgesEndsOnlyLayer(tf.keras.layers.Layer):
    """
    OP: [v1 || v2] -> e'
    IN: V_set, a_node_ids, b_node_ids

    Transition from V-set to E-set through concatenation of edge endnodes (e.g. a_node_ids = receivers and
    b_node_ids = senders).
    """
    def __init__(self, **kwargs):
        super(ConcatEdgesEndsOnlyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        V_set, a_node_ids, b_node_ids = inputs

        return tf.concat([
            tf.gather(V_set[0], a_node_ids[0]),
            tf.gather(V_set[0], b_node_ids[0]),
        ], axis=-1)[tf.newaxis, ...]


class ConcatEdgeWithEndsLayer(tf.keras.layers.Layer):
    """
    OP: [e || v1 || v2] -> e'
    IN: V_set, E_set, a_node_ids, b_node_ids

    Transition from V-set + E-set to E-set through concatenation of edges with their endnodes.
    """
    def __init__(self, **kwargs):
        super(ConcatEdgeWithEndsLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        V_set, E_set, a_node_ids, b_node_ids = inputs

        return tf.concat([
            E_set[0],
            tf.gather(V_set[0], a_node_ids[0]),
            tf.gather(V_set[0], b_node_ids[0]),
        ], axis=-1)[tf.newaxis, ...]


class ConcatEdgeWithSingleEndLayer(tf.keras.layers.Layer):
    """
    OP: [e || v] -> e'
    IN: V_set, E_set, node_ids

    Transition from V-set + E-set to E-set through concatenation of edges with one of their endnodes.
    """
    def __init__(self, **kwargs):
        super(ConcatEdgeWithSingleEndLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        V_set, E_set, node_ids = inputs

        return tf.concat([
            E_set[0],
            tf.gather(V_set[0], node_ids[0]),
        ], axis=-1)[tf.newaxis, ...]


class ConcatGlobalsWithAnythingLayer(tf.keras.layers.Layer):
    """
    OP: [u || e] -> e' or [u || v] -> v'
    IN: u_supernode, X

    Append u-supernode globals to either V-set or E-set.
    """
    def __init__(self, **kwargs):
        super(ConcatGlobalsWithAnythingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        u_supernode, X = inputs

        u_broadcast = tf.zeros_like(X[0, :, 0])[..., tf.newaxis] + u_supernode[0]

        return tf.concat([u_broadcast, X[0]], axis=-1)[tf.newaxis, ...]


class ClusterPoolingNewCOOLayer(tf.keras.layers.Layer):
    """
    OP: [receivers, senders, clusters] -> [receivers', senders']
    IN: rec_node_ids, sen_node_ids, clusters

    Any cluster that is given id -1, will be discarded, therefore resulting in drop pooling.
    """
    def __init__(self, **kwargs):
        super(ClusterPoolingNewCOOLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, inputs):
        rec_node_ids, sen_node_ids, clusters = inputs

        new_receivers = tf.gather(clusters[0], rec_node_ids)
        new_senders = tf.gather(clusters[0], sen_node_ids)

        paired_new_clusters = tf.transpose(tf.stack([new_senders, new_receivers]))
        mask = tf.reduce_all(paired_new_clusters != -1, axis=-1)
        paired_new_clusters = tf.boolean_mask(paired_new_clusters, mask)

        new_receivers, new_senders = tf.transpose(tf.bitcast(
            tf.sort(tf.unique(tf.bitcast(paired_new_clusters, type=tf.int64)).y),
            type=tf.int32
        )[:, ::-1])

        return new_receivers[tf.newaxis, ...], new_senders[tf.newaxis, ...]
