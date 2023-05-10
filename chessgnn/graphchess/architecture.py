import tensorflow as tf
from gnn.layers import ConcatEdgesEndsOnlyLayer, ConcatGlobalsWithAnythingLayer
from gnn.structures import leaky_relu


def input_block(pos_feats_input, glob_feats_input, legal_moves_input, receivers, senders):
    """
    Establish the info-flow paths for graph network model (node, edge and graph level paths).

    :param pos_feats_input:
    :param glob_feats_input:
    :param legal_moves_input:
    :param receivers:
    :param senders:
    :return: Tuple (V_set, E_set, u_supernode)
    """
    v = tf.keras.layers.Dense(units=48, activation=leaky_relu())(pos_feats_input)
    v = tf.keras.layers.Dense(units=48, activation=leaky_relu())(v)

    e = tf.keras.layers.Dense(units=48, activation=leaky_relu())(pos_feats_input)
    e = ConcatEdgesEndsOnlyLayer()([e, receivers, senders])
    e = tf.keras.layers.Concatenate()([e, legal_moves_input[..., tf.newaxis]])
    e = tf.keras.layers.Dense(units=48, activation=leaky_relu())(e)

    u = tf.keras.layers.Dense(units=48, activation=leaky_relu())(pos_feats_input)
    u = tf.keras.layers.GlobalAveragePooling1D(keepdims=True)(u)
    u = ConcatGlobalsWithAnythingLayer()([u, glob_feats_input[..., tf.newaxis, :]])
    u = tf.keras.layers.Dense(units=48, activation=leaky_relu())(u)

    v = tf.keras.layers.LayerNormalization(name='BLOCK_0_V')(v)
    e = tf.keras.layers.LayerNormalization(name='BLOCK_0_E')(e)
    u = tf.keras.layers.LayerNormalization(name='BLOCK_0_U')(u)

    return v, e, u


def regular_mlp_head(x, units, name, final_activation):
    x = tf.keras.layers.Dense(units=units[0], activation=leaky_relu())(x)
    x = tf.keras.layers.Dense(units=units[1], activation=leaky_relu())(x)
    x = tf.keras.layers.Dense(units=units[2], activation=final_activation)(x)
    return tf.keras.layers.Flatten(name=name)(x)


def softmax_mlp_head(x, units, name):
    x = tf.keras.layers.Dense(units=units[0], activation=leaky_relu())(x)
    x = tf.keras.layers.Dense(units=units[1], activation=leaky_relu())(x)
    x = tf.keras.layers.Dense(units=units[2], activation=None)(x)
    x = tf.keras.layers.Flatten()(x)
    return tf.keras.layers.Softmax(name=name)(x)
