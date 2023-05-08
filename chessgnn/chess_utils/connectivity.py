import tensorflow as tf


def obtain_rook_mobility(rows, cols):
    return tf.cast(
        tf.logical_or(
            (tf.range(8) == tf.reshape(cols, [-1, 1, 1])),
            (tf.reshape(tf.range(8), [-1, 1]) == tf.reshape(rows, [-1, 1, 1]))
        ), dtype=tf.int32
    )


def obtain_knight_mobility(rows, cols):
    return tf.cast(tf.logical_or(
        (
            (tf.range(8) - tf.reshape(cols, [-1, 1, 1])) ** 2
            + (tf.reshape(tf.range(8), [-1, 1]) - tf.reshape(rows, [-1, 1, 1])) ** 2 == 5
        ),
        (
            tf.abs(tf.range(8) - tf.reshape(cols, [-1, 1, 1]))
            + tf.abs(tf.reshape(tf.range(8), [-1, 1]) - tf.reshape(rows, [-1, 1, 1])) == 0
        )
    ), dtype=tf.int32)


def obtain_bishop_mobility(rows, cols):
    return tf.cast(
        (
            tf.abs(tf.range(8) - tf.reshape(cols, [-1, 1, 1]))
            == tf.abs(tf.reshape(tf.range(8), [-1, 1]) - tf.reshape(rows, [-1, 1, 1]))
        ),
        dtype=tf.int32
    )


def obtain_queen_mobility(rows, cols):
    return tf.cast(tf.logical_or(
        (
            tf.abs(tf.range(8) - tf.reshape(cols, [-1, 1, 1]))
            == tf.abs(tf.reshape(tf.range(8), [-1, 1]) - tf.reshape(rows, [-1, 1, 1]))
        ),
        tf.logical_or(
            (tf.range(8) == tf.reshape(cols, [-1, 1, 1])),
            (tf.reshape(tf.range(8), [-1, 1]) == tf.reshape(rows, [-1, 1, 1]))
        )
    ), dtype=tf.int32)


def obtain_king_mobility(rows, cols):
    return tf.cast(
        (
            (tf.range(8) - tf.reshape(cols, [-1, 1, 1])) ** 2
            + 2 * (tf.reshape(tf.range(8), [-1, 1]) - tf.reshape(rows, [-1, 1, 1])) ** 2
        ) < 5,
        dtype=tf.int32
    )


def obtain_pawn_mobility(rows, cols):
    return tf.cast(
        (
            2 * (tf.range(8) - tf.reshape(cols, [-1, 1, 1])) ** 2
            + (tf.reshape(tf.range(8), [-1, 1]) - tf.reshape(rows, [-1, 1, 1])) ** 2
        ) < 5,
        dtype=tf.int32
    )


def create_adjacency_matrix(piece_tracking, piece_typing, self_connections=False, undirected=False):
    adjacency = tf.zeros([64, 64], dtype=tf.int32)

    for i, mobility_function in enumerate([
            obtain_rook_mobility, obtain_knight_mobility, obtain_bishop_mobility,
            obtain_queen_mobility, obtain_king_mobility, obtain_pawn_mobility]):
        fields = piece_tracking[piece_typing == i + 1]

        adjacency = tf.tensor_scatter_nd_max(
            tensor=adjacency,
            indices=tf.reshape((fields - 1), [-1, 1]),
            updates=tf.reshape(mobility_function((fields - 1) // 8, (fields - 1) % 8), [-1, 64]))

    if self_connections:
        adjacency = tf.maximum(adjacency, tf.eye(64, dtype=tf.int32))
    if undirected:
        adjacency = tf.maximum(adjacency, tf.transpose(adjacency))

    return adjacency
