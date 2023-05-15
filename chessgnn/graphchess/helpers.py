import tensorflow as tf
from chessgnn.gnn.layers import (
    ConcatEdgesEndsOnlyLayer,
    ConcatEdgeWithEndsLayer,
    ConcatGlobalsWithAnythingLayer,
    ScatterAndAggregateLayer,
    ScatterAndSoftmaxLayer,
)


def load_model(path):
    return tf.keras.models.load_model(path, custom_objects={
        'ConcatEdgesEndsOnlyLayer': ConcatEdgesEndsOnlyLayer,
        'ConcatEdgeWithEndsLayer': ConcatEdgeWithEndsLayer,
        'ConcatGlobalsWithAnythingLayer': ConcatGlobalsWithAnythingLayer,
        'ScatterAndAggregateLayer': ScatterAndAggregateLayer,
        'ScatterAndSoftmaxLayer': ScatterAndSoftmaxLayer,
    })
