import chess
import numpy as np
import pandas as pd
from chessgnn.chess.translations import field_to_int
from typing import List, Tuple


starting_positions = np.array(list(map(lambda x: field_to_int[x], np.array([
    'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8',
    'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7',
    'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2',
    'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
]))), dtype=np.uint8)

starting_typing = np.array([
    1, 2, 3, 4, 5, 3, 2, 1,
    6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6,
    1, 2, 3, 4, 5, 3, 2, 1,], dtype=np.uint8)

numeric_types = {
    'game_index': np.int32, 'total_len': np.int32, 'halfmove_clock': np.int32, 'fullmove_number': np.int32,
    'ply_counter': np.int32, 'state_repetition': np.int32, 'en_passant_target': np.int32,
    'place_from': np.int32, 'place_to': np.int32, 'lm_piece_type': np.int32, 'en_passant_target_prev': np.int32}


def move_list_encode(move_generator) -> np.ndarray:
    return np.array(
        [(field_to_int[str(i)[0:2]], field_to_int[str(i)[2:4]]) for i in move_generator]
    ).astype(np.uint8).reshape(-1)


def find_idx(positions, piece):
    idx = np.argwhere(positions == piece)[:, 0]
    return None if idx.shape == (0,) else int(idx[0])


def apply_castling(positions, is_white, is_kingside):
    rook = is_white * 24 + is_kingside * 7
    king = is_white * 24 + 4

    positions[rook] = 56 * is_white + 4 + 2 * is_kingside
    positions[king] = 56 * is_white + 3 + 4 * is_kingside

    return np.array([king, rook], dtype=np.uint8), np.array([], dtype=np.uint8)


def apply_non_castling_move(positions, typings, field_from, field_to, en_passant, new_typing):
    idx_from = find_idx(positions, field_from)
    idx_to = find_idx(positions, field_to)

    positions[idx_from] = field_to
    typings[idx_from] = new_typing

    moved = np.array([idx_from], dtype=np.uint8)
    captured = np.array([], dtype=np.uint8)

    if idx_to is not None:
        positions[idx_to] = 0
        typings[idx_to] = 0
        captured = np.array([idx_to], dtype=np.uint8)

    if en_passant:
        idx_ep = find_idx(positions, field_to + 8 if field_to < 32 else field_to - 8)
        positions[idx_ep] = 0
        typings[idx_ep] = 0
        captured = np.array([idx_ep], dtype=np.uint8)

    return moved, captured


def apply_move(positions, typings, move_uci, is_castling, is_white, is_kingside, en_passant, new_typing):
    if is_castling:
        moved, captured = apply_castling(positions, is_white, is_kingside)
    else:
        moved, captured = apply_non_castling_move(
            positions, typings, field_to_int[move_uci[:2]], field_to_int[move_uci[2:4]], en_passant, new_typing)

    return moved, captured


def parse_and_replay(games_df: pd.DataFrame, n_from: int, n_to: int) -> List[Tuple]:
    board = chess.Board()
    states_all = []

    for i, (game, game_len, game_result) in games_df[['game', 'len', 'result']][n_from:n_to].iterrows():
        print(f"\rGame: {i + 1}", end='')
        board.reset()
        game_san = [i.split('.')[1] for i in game.split(' ')]

        states_current = []

        for move_san in game_san:
            move = board.parse_san(move_san)
            move_uci = move.uci()
            board.push(move)

            possible_moves = move_list_encode(board.generate_legal_moves())
            possible_attacks = move_list_encode(board.generate_legal_captures())

            states_current += [
                (i, game_len, game_result, move_san, move_uci, board.fen(), possible_moves, possible_attacks)]

        states_all += states_current

    print()
    return states_all
