import os
import pandas as pd
import tensorflow as tf
from tensorflow.data import Dataset
from chessgnn.chess_utils.connectivity import create_adjacency_matrix

feature_specs = {
    # Inputs
    'piece_tracking': tf.TensorSpec([32], dtype=tf.int32),
    'piece_typing': tf.TensorSpec([32], dtype=tf.int32),
    'who_moves_next': tf.TensorSpec([], dtype=tf.string),
    'en_passant_target': tf.TensorSpec([], dtype=tf.int32),
    'K_castling_ability': tf.TensorSpec([], dtype=tf.bool),
    'Q_castling_ability': tf.TensorSpec([], dtype=tf.bool),
    'k_castling_ability': tf.TensorSpec([], dtype=tf.bool),
    'q_castling_ability': tf.TensorSpec([], dtype=tf.bool),
    'halfmove_clock': tf.TensorSpec([], dtype=tf.int32),
    'state_repetition': tf.TensorSpec([], dtype=tf.int32),
    # Outputs
    'final_result': tf.TensorSpec([], dtype=tf.string),
    'possible_moves': tf.TensorSpec([None, 2], dtype=tf.int32),
    'possible_captures': tf.TensorSpec([None, 2], dtype=tf.int32),
    'plies_till_end': tf.TensorSpec([], dtype=tf.int32),
    'future_idx_moved': tf.TensorSpec([None], dtype=tf.int32),
    'future_idx_captured': tf.TensorSpec([None], dtype=tf.int32),
    'future_move': tf.TensorSpec([1, 2], dtype=tf.int32),
    'future_give_check_in_1': tf.TensorSpec([], dtype=tf.bool),
    'future_give_check_in_3': tf.TensorSpec([], dtype=tf.bool),
    'future_give_check_in_5': tf.TensorSpec([], dtype=tf.bool),
    'future_get_checked_in_1': tf.TensorSpec([], dtype=tf.bool),
    'future_get_checked_in_3': tf.TensorSpec([], dtype=tf.bool),
    'future_get_checked_in_5': tf.TensorSpec([], dtype=tf.bool),
    # Other
    'ply_counter': tf.TensorSpec([], dtype=tf.int32),
    'game_index': tf.TensorSpec([], dtype=tf.int32),
}


def get_dataset(data_file_path=None, data_folder_path=None, shuffle=False):
    def data_generator():
        filelist = [data_file_path] if data_file_path else os.listdir(data_folder_path)

        for filename in filelist:
            if data_folder_path:
                data = pd.read_parquet(f'{data_folder_path}/{filename}', columns=list(feature_specs.keys()))
            else:
                data = pd.read_parquet(filename, columns=list(feature_specs.keys()))

            data = data[data['plies_till_end'] != 0]

            if shuffle:
                data = data.sample(frac=1).reset_index(drop=True)

            for _, row in data.iterrows():
                yield {
                    feat: tf.reshape(tf.cast(row[feat], dtype=spec.dtype), [x if x else -1 for x in spec.shape])
                    for feat, spec in feature_specs.items()
                }

    ds = Dataset.from_generator(data_generator, output_signature=feature_specs)
    return ds


def get_squarewise_features(piece_typing, piece_tracking, en_passant_target):
    """
    > 6x binary - piece type
    > 2x binary - piece black/white (first is black)
    > 1x binary - square color (white - 0, black - 1)
    > 2x numeric - ranks and files (first are ranks)
    > 1x binary - is en passant target?
    """
    zipped_pieces = tf.transpose(tf.stack([tf.range(32, dtype=tf.int32), piece_tracking - 1, piece_typing - 1]))
    zipped_pieces = tf.boolean_mask(zipped_pieces, zipped_pieces[:, 1] >= 0)

    return tf.concat([
        # Pieces
        tf.tensor_scatter_nd_add(
            tf.zeros([64, 6], dtype=tf.float32),
            zipped_pieces[:, 1:3],
            tf.ones_like(zipped_pieces[:, 0], dtype=tf.float32)
        ),
        # Piece color - is black
        tf.reshape(tf.tensor_scatter_nd_add(
            tf.zeros([64], dtype=tf.float32),
            zipped_pieces[:, 1][:, tf.newaxis],
            tf.cast(zipped_pieces[:, 0] < 16, dtype=tf.float32)
        ), [-1, 1]),
        # Piece color - is white
        tf.reshape(tf.tensor_scatter_nd_add(
            tf.zeros([64], dtype=tf.float32),
            zipped_pieces[:, 1][:, tf.newaxis],
            tf.cast(zipped_pieces[:, 0] >= 16, dtype=tf.float32)
        ), [-1, 1]),
        # Square color - is black
        tf.cast(tf.reshape(tf.reduce_sum(
            tf.reshape(tf.transpose(tf.stack(tf.meshgrid(tf.range(8), tf.range(8))[::-1]), perm=[1, 2, 0]), [-1, 2]),
            axis=-1
        ) % 2, [-1, 1]), dtype=tf.float32),
        # Ranks and files (first are ranks)
        (0.3 * (tf.cast(
            tf.reshape(tf.transpose(tf.stack(tf.meshgrid(tf.range(8), tf.range(8))[::-1]), perm=[1, 2, 0]), [-1, 2]),
            dtype=tf.float32
        ) - 3.5)),
        # Is susceptible to en passant?
        tf.cast(tf.reshape(tf.range(64, dtype=tf.int32) == (en_passant_target - 1), [-1, 1]), dtype=tf.float32),
    ], axis=-1)


@tf.function
def plies_till_end_transform(x):
    return tf.exp(-0.2 * (tf.cast(x, dtype=tf.float32) - 1.0))


def prepare_features(example):
    adj = create_adjacency_matrix(
        example['piece_tracking'], example['piece_typing'],
        self_connections=True, undirected=True)
    coo_graph = tf.cast(tf.transpose(tf.stack(tf.experimental.numpy.nonzero(adj))), dtype=tf.int32)

    pos_feats = get_squarewise_features(
        example['piece_typing'], example['piece_tracking'], example['en_passant_target'])

    glob_feats = tf.concat([
        tf.cast(example['who_moves_next'] == tf.constant(['w', 'b']), dtype=tf.float32),
        tf.cast([
            example['K_castling_ability'], example['Q_castling_ability'],
            example['k_castling_ability'], example['q_castling_ability']], dtype=tf.float32),
        tf.cast([example['halfmove_clock']], dtype=tf.float32) * 0.04,
        tf.cast([example['state_repetition']], dtype=tf.float32) * 0.5,
    ], axis=0)

    final_result = tf.cast(example['final_result'] == tf.constant(['W', 'D', 'B']), dtype=tf.int32)
    plies_till_end = plies_till_end_transform(example['plies_till_end'])
    next_move = tf.cast(
        tf.reduce_all(coo_graph == (example['future_move'] - 1), axis=-1),
        dtype=tf.int32)
    legal_moves = tf.cast(
        tf.reduce_any(
            tf.reduce_all(coo_graph[:, tf.newaxis, :] == (example['possible_moves'] - 1), axis=-1), axis=-1),
        dtype=tf.int32)
    legal_captures = tf.cast(
        tf.reduce_any(
            tf.reduce_all(coo_graph[:, tf.newaxis, :] == (example['possible_captures'] - 1), axis=-1), axis=-1),
        dtype=tf.int32)
    next_pieces_moved = tf.tensor_scatter_nd_max(
        tf.zeros(64, dtype=tf.int32),
        tf.gather(example['piece_tracking'] - 1, example['future_idx_moved'])[:, tf.newaxis],
        tf.ones_like(example['future_idx_moved'], dtype=tf.int32))
    next_pieces_captured = tf.tensor_scatter_nd_max(
        tf.zeros(64, dtype=tf.int32),
        tf.gather(example['piece_tracking'] - 1, example['future_idx_captured'])[:, tf.newaxis],
        tf.ones_like(example['future_idx_captured'], dtype=tf.int32))

    return {
        'coo_graph': coo_graph,
        'pos_feats': pos_feats,
        'glob_feats': glob_feats,
        'final_result': final_result,
        'plies_till_end': plies_till_end,
        'next_move': next_move,
        'legal_moves': legal_moves,
        'legal_captures': legal_captures,
        'next_pieces_moved': next_pieces_moved,
        'next_pieces_captured': next_pieces_captured,
        'checking_in_1': tf.cast(example['future_give_check_in_1'], dtype=tf.int32),
        'checking_in_3': tf.cast(example['future_give_check_in_3'], dtype=tf.int32),
        'checking_in_5': tf.cast(example['future_give_check_in_5'], dtype=tf.int32),
        'checked_in_1': tf.cast(example['future_get_checked_in_1'], dtype=tf.int32),
        'checked_in_3': tf.cast(example['future_get_checked_in_3'], dtype=tf.int32),
        'checked_in_5': tf.cast(example['future_get_checked_in_5'], dtype=tf.int32),
    }
