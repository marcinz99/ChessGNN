import chess
import numpy as np
import pandas as pd
from chessgnn.chess_utils.translations import field_to_int, piece_placement_from_fen_code
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


def states_all_to_moves_dataframe(states_all: List[Tuple]) -> pd.DataFrame:
    moves = pd.DataFrame(states_all, columns=[
        "game_index", "total_len", "final_result", "latest_move_san", "latest_move_uci", "board_fen",
        "possible_moves", "possible_captures"])
    moves['latest_move'] = moves['latest_move_uci'].apply(lambda x: move_list_encode([x]))

    moves = pd.concat([
        moves.drop("board_fen", axis=1),
        pd.DataFrame(
            moves['board_fen'].apply(lambda x: x.split(' ')).tolist(),
            columns=["piece_placement", "who_moves_next", "castling_rights", "en_passant_targets",
                     "halfmove_clock", "fullmove_number"]
        )
    ], axis=1)
    moves['halfmove_clock'] = moves['halfmove_clock'].astype(np.int32)
    moves['fullmove_number'] = moves['fullmove_number'].astype(np.int32)

    moves['ply_counter'] = (moves.groupby(['game_index']).cumcount() + 1).astype(np.int32)
    moves['state_repetition'] = moves.groupby(
        ['game_index', "piece_placement", "castling_rights", "en_passant_targets"]).cumcount()

    moves['piece_placement'] = moves['piece_placement'].apply(piece_placement_from_fen_code)
    moves['lm_capture'] = moves['latest_move_san'].apply(lambda x: 'x' in x)
    moves['lm_check'] = moves['latest_move_san'].apply(lambda x: '+' in x)
    moves['lm_mate'] = moves['latest_move_san'].apply(lambda x: '#' in x)
    moves['lm_kingside_castling'] = moves['latest_move_san'].apply(lambda x: "O-O" in x)
    moves['lm_queenside_castling'] = moves['latest_move_san'].apply(lambda x: "O-O-O" in x)
    moves['lm_kingside_castling'] = (moves['lm_kingside_castling'] & (~moves['lm_queenside_castling']))
    moves['lm_promotion'] = moves['latest_move_san'].apply(lambda x: '=' in x)

    moves['en_passant_target'] = moves['en_passant_targets'].apply(lambda x: field_to_int[x])
    moves['K_castling_ability'] = moves['castling_rights'].apply(lambda x: 'K' in x)
    moves['Q_castling_ability'] = moves['castling_rights'].apply(lambda x: 'Q' in x)
    moves['k_castling_ability'] = moves['castling_rights'].apply(lambda x: 'k' in x)
    moves['q_castling_ability'] = moves['castling_rights'].apply(lambda x: 'q' in x)

    moves = moves.drop(["castling_rights", "en_passant_targets"], axis=1)

    moves['place_from'] = moves['latest_move_uci'].apply(lambda x: field_to_int[x[0:2]])
    moves['place_to'] = moves['latest_move_uci'].apply(lambda x: field_to_int[x[2:4]])
    moves['promoted_to'] = moves['latest_move_uci'].apply(lambda x: x[4:])
    moves['lm_piece_type'] = (
        moves[['piece_placement', 'place_to']]
        .apply(lambda x: (x['piece_placement'][x['place_to'] - 1] - 1) % 6 + 1, axis='columns'))

    moves['en_passant_target_prev'] = np.where(
        moves['game_index'].shift(1, fill_value=-1) == moves['game_index'],
        moves['en_passant_target'].shift(1, fill_value=0),
        0).reshape(-1, 1)
    moves['lm_en_passant'] = ((moves['en_passant_target_prev'] == moves['place_to']) & (moves['lm_piece_type'] == 6))

    return moves


def calculate_tracking(moves: pd.DataFrame) -> pd.DataFrame:
    latest_idx = -1

    pos_list = []
    typ_list = []
    moved_list = []
    captured_list = []

    for i, (idx, uci, color, kingside, queenside, en_passant, new_typing) in moves[
        ['game_index', 'latest_move_uci', 'who_moves_next', 'lm_kingside_castling', 'lm_queenside_castling',
         'lm_en_passant', 'lm_piece_type']
    ].iterrows():
        print(f"\rGame: {idx + 1}", end='')
        if idx != latest_idx:
            positions = starting_positions.copy()
            typings = starting_typing.copy()
        latest_idx = idx

        moved, captured = apply_move(
            positions, typings, uci, kingside or queenside, color == 'b', kingside, en_passant, new_typing)
        pos_list += [positions.copy().astype(np.uint8)]
        typ_list += [typings.copy().astype(np.uint8)]
        moved_list += [moved]
        captured_list += [captured]

    moves['piece_tracking'] = pd.Series(pos_list)
    moves['piece_typing'] = pd.Series(typ_list)
    moves['lm_idx_moved'] = pd.Series(moved_list)
    moves['lm_idx_captured'] = pd.Series(captured_list)

    moves = moves.astype(numeric_types)

    print()
    return moves


def concat_initial_position(moves: pd.DataFrame) -> pd.DataFrame:
    board = chess.Board()
    initial_pos_encoding = pd.DataFrame([{
        'latest_move_san': '--',
        'latest_move_uci': '----',
        'possible_moves': move_list_encode(board.generate_legal_moves()),
        'possible_captures': move_list_encode(board.generate_legal_captures()),
        'latest_move': np.array([], dtype=np.uint8),
        'piece_placement': piece_placement_from_fen_code(board.board_fen()),
        'who_moves_next': 'w',
        'halfmove_clock': 0,
        'fullmove_number': 1,
        'ply_counter': 0,
        'state_repetition': 0,
        'lm_capture': False,
        'lm_check': False,
        'lm_mate': False,
        'lm_kingside_castling': False,
        'lm_queenside_castling': False,
        'lm_promotion': False,
        'en_passant_target': 0,
        'K_castling_ability': True,
        'Q_castling_ability': True,
        'k_castling_ability': True,
        'q_castling_ability': True,
        'place_from': 0,
        'place_to': 0,
        'promoted_to': '',
        'lm_piece_type':  0,
        'en_passant_target_prev': 0,
        'lm_en_passant': False,
        'piece_tracking': starting_positions,
        'piece_typing': starting_typing,
        'lm_idx_moved': np.array([], dtype=np.uint8),
        'lm_idx_captured': np.array([], dtype=np.uint8),
    }]).astype({k: v for k, v in numeric_types.items() if k not in ['game_index', 'total_len', 'final_result']})

    initial_states = (
        moves[['game_index', 'total_len', 'final_result']]
        .drop_duplicates()
        .merge(initial_pos_encoding, how='cross')
    )

    moves = (
        pd.concat([moves, initial_states], axis='rows')
        .sort_values(['game_index', 'ply_counter'])
        .reset_index(drop=True)
    )

    moves['plies_till_end'] = moves['total_len'] - moves['ply_counter']

    return moves


def concat_shifted_features(moves: pd.DataFrame) -> pd.DataFrame:
    def shifted_col(game_index_col, target_col, n_shift):
        return pd.Series(np.where(
            game_index_col == game_index_col.shift(-n_shift),
            target_col.shift(-n_shift),
            pd.Series(list(np.zeros([moves.shape[0], 0], dtype=np.uint8)))
        ))

    def future_checker(game_index_col, target_col, n_shift):
        return (game_index_col == game_index_col.shift(-n_shift)) & target_col.shift(-n_shift)

    moves['future_idx_moved'] = (
        pd.DataFrame(pd.concat(
            [shifted_col(moves['game_index'], moves['lm_idx_moved'], n_shift=i) for i in range(1, 11)],
            axis='columns'))
        .apply(lambda x: np.concatenate(x), axis='columns')
    )
    moves['future_idx_captured'] = (
        pd.DataFrame(pd.concat(
            [shifted_col(moves['game_index'], moves['lm_idx_captured'], n_shift=i) for i in range(1, 11)],
            axis='columns'))
        .apply(lambda x: np.concatenate(x), axis='columns')
    )
    moves['future_move'] = shifted_col(moves['game_index'], moves['latest_move'], n_shift=1)

    checks_in_future = pd.DataFrame(pd.concat(
        [future_checker(moves['game_index'], moves['lm_check'], n_shift=i) for i in range(1, 11)],
        axis='columns'))

    moves['future_give_check_in_1'] = checks_in_future[[0]].sum(axis='columns').astype(bool)
    moves['future_give_check_in_3'] = checks_in_future[[0, 2, 4]].sum(axis='columns').astype(bool)
    moves['future_give_check_in_5'] = checks_in_future[[0, 2, 4, 6, 8]].sum(axis='columns').astype(bool)
    moves['future_get_checked_in_1'] = checks_in_future[[1]].sum(axis='columns').astype(bool)
    moves['future_get_checked_in_3'] = checks_in_future[[1, 3, 5]].sum(axis='columns').astype(bool)
    moves['future_get_checked_in_5'] = checks_in_future[[1, 3, 5, 7, 9]].sum(axis='columns').astype(bool)

    return moves


def generate_boards_daraframe(games_df: pd.DataFrame, n_from: int, n_to: int) -> pd.DataFrame:
    print(f"[{n_from}-{n_to}] Generating board features")
    print("> Parsings plays and replaying")
    states_all = parse_and_replay(games_df, n_from, n_to)
    print("> Processing gather metadata into dataframe")
    moves = states_all_to_moves_dataframe(states_all)
    print("> Tracking pieces")
    moves = calculate_tracking(moves)
    print("> Appending initial position")
    moves = concat_initial_position(moves)
    print("> Appending shifted features")
    moves = concat_shifted_features(moves)
    print("> Done!")

    return moves


def games_consistency_check(moves: pd.DataFrame) -> None:
    player_mask = np.where(np.arange(32) < 16, 0, 6)

    for k in range(moves.shape[0]):
        print(f"\rIteration {k + 1}", end='')
        recreated = np.zeros(64, dtype=np.uint8)
        for i, j in zip(moves['piece_tracking'][k], moves['piece_typing'][k] + player_mask):
            if i > 0:
                recreated[i - 1] = j

        np.testing.assert_array_equal(moves['piece_placement'][k], recreated, str(k))

    print("\nAll tested passed successfully")
