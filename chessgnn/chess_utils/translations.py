import itertools
import numpy as np
import tensorflow as tf
from chessgnn.chess_utils.connectivity import create_adjacency_matrix


# UCI square code to node index
field_to_int = {
    field: i + 1
    for i, field
    in enumerate(list(map(
        lambda x: f"{x[1]}{x[0]}",
        itertools.product([8, 7, 6, 5, 4, 3, 2, 1], ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    )))
}
field_to_int['00'] = 0
field_to_int['--'] = 0
field_to_int['-'] = 0

# Node index to UCI square code
int_to_field = np.array(['--'] + list(map(
        lambda x: f"{x[1]}{x[0]}",
        itertools.product([8, 7, 6, 5, 4, 3, 2, 1], ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
)))

# - - - Board representation encoding/decoding - - -
spaces = {ord(str(i)): " " * i for i in range(1, 9)}
transitions = {ord(letter): i for i, letter in enumerate(" " + "rnbqkp" + "rnbqkp".upper())}
transitions[ord('/')] = None
inv_transitions = {v: k for k, v in transitions.items()}


def piece_placement_from_fen_code(fen_code: str) -> np.ndarray:
    return np.array(list(bytes(fen_code.translate(spaces).translate(transitions), 'ASCII')), dtype=np.uint8)


def piece_placement_to_fen_code(piece_placement: np.ndarray) -> str:
    return (
        bytes(
            np.concatenate([
                np.array(list(map(lambda x: inv_transitions[x], piece_placement))).reshape(8, 8),
                np.ones([8, 1], dtype=int) * ord('/')
            ], axis=1)
            .reshape(-1)[:-1]
            .astype(np.int8))
        .decode()
        .replace(' ', '1').replace('11', '2').replace('22', '4').replace('42', '6')
        .replace('21', '3').replace('41', '5').replace('61', '7').replace('44', '8')
    )


def piece_typing_and_tracking_to_fen_code(piece_typing: np.ndarray, piece_tracking: np.ndarray) -> str:
    piece_placement = np.zeros(64, dtype=int)
    for typing, tracking in zip(piece_typing + np.where(np.arange(32) >= 16, 6, 0), piece_tracking):
        if tracking > 0:
            piece_placement[tracking - 1] = typing

    return piece_placement_to_fen_code(piece_placement)


def piece_encoding_to_fen_code(piece_encoding: np.ndarray) -> str:
    piece_placement = np.sum(piece_encoding[..., :8] * np.array([1, 2, 3, 4, 5, 6, 0, 6]), axis=1).astype(int)

    return piece_placement_to_fen_code(piece_placement)


def piece_encoding_from_fen_code(fen_code: str, en_passant_target: int = 0) -> np.ndarray:
    placement_rec = piece_placement_from_fen_code(fen_code)

    return np.concatenate([
        (
                np.c_[np.eye(7, dtype=int)[:, 1:], np.arange(14).reshape(2, 7)[::-1].T > 7]
                [np.where(placement_rec <= 6, placement_rec, 0)]
                + np.c_[np.eye(7, dtype=int)[:, 1:], np.arange(14).reshape(2, 7).T > 7]
                [np.where(placement_rec > 6, placement_rec - 6, 0)]
        ),
        (
            (np.transpose(np.mgrid[:8, :8], axes=[1, 2, 0]).reshape(-1, 2).sum(axis=-1) % 2 == 1)
            .astype(int)
            .reshape(-1, 1),
        ),
        0.3 * (np.transpose(np.mgrid[:8, :8], axes=[1, 2, 0]).reshape(-1, 2) - 3.5).astype(np.float32),
        (np.arange(64) == (en_passant_target - 1)).reshape(-1, 1)
    ], axis=-1).astype(np.float32)


def coo_graph_from_fen_code(fen_code: str) -> np.ndarray:
    piece_placement = piece_placement_from_fen_code(fen_code)

    positions = tf.range(1, 65)[piece_placement > 0]
    types = piece_placement[piece_placement > 0]
    types = tf.where(types > 6, types - 6, types)

    adj = create_adjacency_matrix(positions, types, self_connections=True, undirected=True)

    return tf.cast(tf.transpose(tf.stack(tf.experimental.numpy.nonzero(adj))), dtype=tf.int32)
