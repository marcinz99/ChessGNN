import tensorflow as tf
from chessgnn.gnn.layers import (
    GatherNodesLayer,
    ScatterAndAggregateLayer,
    ConcatEdgesEndsOnlyLayer,
    ScatterAndSoftmaxLayer,
)


def plain_gcn(v_set, receivers, senders, f_out, activation, agg_method):
    e = GatherNodesLayer()([v_set, senders])
    v = ScatterAndAggregateLayer(agg_method=agg_method)([v_set, e, receivers])
    v = tf.keras.layers.Dense(units=f_out, activation=activation)(v)
    return v


def basic_gat(v_set, receivers, senders, f_mid, activation):
    e = ConcatEdgesEndsOnlyLayer()([v_set, receivers, senders])
    e_a = tf.keras.layers.Dense(units=f_mid, activation=activation)(e)
    e_a = tf.keras.layers.Dense(units=1, activation=activation)(e_a)
    e_a = ScatterAndSoftmaxLayer()([v_set, e_a, receivers])
    e = tf.keras.layers.Multiply()([e, e_a])
    v = ScatterAndAggregateLayer(agg_method='sum')([v_set, e, receivers])
    v = tf.keras.layers.Dense(units=f_mid, activation=activation)(v)
    return v


def multihead_gat(v_set, receivers, senders, f_mid, k_heads, head_f_out, activation):
    e = ConcatEdgesEndsOnlyLayer()([v_set, receivers, senders])
    e_a = tf.keras.layers.Dense(units=f_mid, activation=activation)(e)
    e_a = tf.keras.layers.Dense(units=k_heads, activation=activation)(e_a)
    e_a = ScatterAndSoftmaxLayer()([v_set, e_a, receivers])
    e = tf.keras.layers.Multiply()([e[..., :, tf.newaxis], e_a[..., tf.newaxis, :]])
    v = ScatterAndAggregateLayer(agg_method='sum')([v_set, e, receivers])
    v = tf.keras.layers.DepthwiseConv2D(
        kernel_size=[1, e.shape[2]], depth_multiplier=head_f_out, activation=activation)(v)
    v = v[..., 0, :]
    return v


def leaky_relu():
    return tf.keras.layers.LeakyReLU(alpha=0.3)
