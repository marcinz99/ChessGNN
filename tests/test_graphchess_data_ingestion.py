import pytest
import numpy as np
import tensorflow as tf


@pytest.fixture
def moves():
    return {
        'piece_typing': tf.cast(np.array([
                1, 0, 3, 4, 5, 3, 2, 1, 6, 6, 6, 6, 6, 0, 6, 6,
                6, 6, 6, 6, 0, 6, 6, 6, 1, 2, 3, 4, 5, 3, 0, 1,
            ]), dtype=tf.int32),
        'piece_tracking': tf.cast(np.array([
                1,   0, 30, 11,  5, 27,  7,  8,  9, 10, 19, 28, 21, 0, 15, 16,
                49, 50, 43, 29,  0, 54, 55, 56, 57, 58, 59, 60, 63, 53, 0, 61,
            ]), dtype=tf.int32),
        'en_passant_target': 0,
    }


def test_get_squarewise_features(moves):
    from chessgnn.graphchess.data_ingestion import get_squarewise_features

    piece_typing = moves['piece_typing']
    piece_tracking = moves['piece_tracking']
    en_passant_target = moves['en_passant_target']
    feats = get_squarewise_features(piece_typing, piece_tracking, en_passant_target)

    # Assert number of each pieces
    np.testing.assert_array_equal(
        np.sum(piece_typing.numpy() == np.array([[1, 2, 3, 4, 5, 6]]).T, axis=1),
        feats.numpy()[:, :6].sum(0))

    # Assess colors of pieces
    np.testing.assert_array_equal(
        sorted(np.nonzero(feats.numpy()[:, 6])[0] + 1),
        sorted([i for i in piece_tracking.numpy()[:16] if i > 0]))
    np.testing.assert_array_equal(
        sorted(np.nonzero(feats.numpy()[:, 7])[0] + 1),
        sorted([i for i in piece_tracking.numpy()[16:] if i > 0]))

    # Assert alternating color grid pattern
    np.testing.assert_equal(feats.numpy()[0, 8], 0)
    np.testing.assert_equal(
        0.0,
        np.std(2 * feats.numpy()[:, 8].reshape(8, 8) - 2
               + feats.numpy()[:, 8].reshape(8, 8)[::-1, :] + feats.numpy()[:, 8].reshape(8, 8)[:, ::-1]))

    # Assert ranks and files
    np.testing.assert_almost_equal(0.0, feats.numpy()[:, 9].reshape(8, 8).std(axis=1).sum())
    np.testing.assert_almost_equal(0.0, feats.numpy()[:, 10].reshape(8, 8).std(axis=0).sum())
