import tensorflow as tf
from chessgnn.gnn.layers import (
    GatherNodesLayer,
    ScatterAndAggregateLayer,
    ConcatEdgesEndsOnlyLayer,
    ScatterAndSoftmaxLayer,
    ConcatEdgeWithEndsLayer,
    ConcatGlobalsWithAnythingLayer,
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


def full_block(v_prev, e_prev, u_prev, receivers, senders, name):
    e = ConcatEdgeWithEndsLayer()([v_prev, e_prev, receivers, senders])
    e = tf.keras.layers.Dense(units=32, activation=leaky_relu())(e)
    e = ConcatGlobalsWithAnythingLayer()([u_prev, e])
    e = tf.keras.layers.Dense(units=48, activation=leaky_relu())(e)
    v = ScatterAndAggregateLayer(agg_method='mean')([v_prev, e, receivers])
    v = multihead_gat(v, receivers, senders, f_mid=48, k_heads=24, head_f_out=2, activation=leaky_relu())
    u = tf.keras.layers.GlobalAveragePooling1D(keepdims=True)(v)

    v = tf.keras.layers.Add()([v_prev, v])
    e = tf.keras.layers.Add()([e_prev, e])
    u = tf.keras.layers.Add()([u_prev, u])

    v = tf.keras.layers.LayerNormalization(name=f'{name}_V')(v)
    e = tf.keras.layers.LayerNormalization(name=f'{name}_E')(e)
    u = tf.keras.layers.LayerNormalization(name=f'{name}_U')(u)

    return v, e, u
