import pytest
import numpy as np
import tensorflow as tf


def test_field_to_int_and_reverse():
    from chessgnn.chess_utils.translations import int_to_field, field_to_int
    assert int_to_field[0] == '--'
    assert field_to_int['-'] == 0
    assert field_to_int['--'] == 0
    assert int_to_field[4] == 'd8'
    assert field_to_int['c4'] == 35
    for i in range(1, 65):
        assert field_to_int[int_to_field[i]] == i


@pytest.fixture
def moves_and_features():
    from chessgnn.chess_utils.connectivity import create_adjacency_matrix
    from chessgnn.graphchess.data_ingestion import get_squarewise_features

    moves = {
        'piece_typing': np.array([
                1, 0, 3, 4, 5, 3, 2, 1, 6, 6, 6, 6, 6, 0, 6, 6,
                6, 6, 6, 6, 0, 6, 6, 6, 1, 2, 3, 4, 5, 3, 0, 1,
            ], dtype=np.int32),
        'piece_tracking': np.array([
                1,   0, 30, 11,  5, 27,  7,  8,  9, 10, 19, 28, 21, 0, 15, 16,
                49, 50, 43, 29,  0, 54, 55, 56, 57, 58, 59, 60, 63, 53, 0, 61,
            ], dtype=np.int32),
        'piece_placement': np.array([
                1, 0, 0, 0, 5, 0, 2, 1, 6, 6, 4, 0, 0, 0, 6, 6, 0,
                0, 6, 0, 6, 0, 0, 0, 0, 0, 3, 6, 12, 3, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 12, 12, 0,
                0, 9, 12, 12, 12, 7, 8, 9, 10, 7, 0, 11, 0,
            ], dtype=np.int32),
        'en_passant_target': 0,
    }

    adj = create_adjacency_matrix(
        moves['piece_tracking'], moves['piece_typing'], self_connections=True, undirected=True)

    features = {
        'pos_feats': get_squarewise_features(
            moves['piece_typing'], moves['piece_tracking'], moves['en_passant_target']),
        'coo_graph': tf.cast(tf.transpose(tf.stack(tf.experimental.numpy.nonzero(adj))), dtype=tf.int32),
    }

    return {'moves': moves, 'features': features}


def test_piece_placement_from_and_to_fen_translations():
    from chessgnn.chess_utils.translations import piece_placement_from_fen_code, piece_placement_to_fen_code
    fen_ex = 'r1b1k1nr/p2p1pNp/n2B4/1p1NP2P/6P1/3P1Q2/P1P1K3/q5b1'

    np.testing.assert_equal(
        fen_ex,
        piece_placement_to_fen_code(piece_placement_from_fen_code(fen_ex)))


def test_piece_tracking_typing_placing_translations(moves_and_features):
    from chessgnn.chess_utils.translations import (
        piece_placement_from_fen_code,
        piece_placement_to_fen_code,
        piece_typing_and_tracking_to_fen_code,
    )
    moves = moves_and_features['moves']

    np.testing.assert_array_equal(
        moves['piece_placement'],
        piece_placement_from_fen_code(piece_placement_to_fen_code(moves['piece_placement']))
    )
    np.testing.assert_array_equal(
        moves['piece_placement'],
        piece_placement_from_fen_code(
            piece_typing_and_tracking_to_fen_code(moves['piece_typing'], moves['piece_tracking']))
    )


def test_piece_encoding_piece_placement_and_fen_translations(moves_and_features):
    from chessgnn.chess_utils.translations import (
        piece_encoding_from_fen_code,
        piece_encoding_to_fen_code,
        piece_placement_from_fen_code,
        coo_graph_from_fen_code,
    )
    features = moves_and_features['features']

    np.testing.assert_array_equal(
        features['pos_feats'][:, :-1],
        piece_encoding_from_fen_code(piece_encoding_to_fen_code(features['pos_feats']))[:, :-1]
    )

    placement_rec = piece_placement_from_fen_code(piece_encoding_to_fen_code(features['pos_feats']))

    encoding_rec = (
        np.c_[np.eye(7, dtype=int)[:, 1:], np.arange(14).reshape(2, 7)[::-1].T > 7][
            np.where(placement_rec <= 6, placement_rec, 0)]
        + np.c_[np.eye(7, dtype=int)[:, 1:], np.arange(14).reshape(2, 7).T > 7][
            np.where(placement_rec > 6, placement_rec - 6, 0)])

    np.testing.assert_array_equal(features['pos_feats'][:, :8], encoding_rec)

    np.testing.assert_array_equal(
        features['coo_graph'],
        coo_graph_from_fen_code(piece_encoding_to_fen_code(features['pos_feats'])))
