import pandas as pd


def test_generate_boards_daraframe():
    from chessgnn.chess_utils.data_preparation import generate_boards_daraframe, games_consistency_check
    games_df = pd.DataFrame([
        {'result': 'D', 'len': 64,
         'game': 'W1.e4 B1.c6 W2.d4 B2.d5 W3.e5 B3.Bf5 W4.Nf3 B4.e6 W5.c3 B5.Nd7 W6.Be2 B6.f6 W7.O-O B7.fxe5 W8.Nxe5'},
        {'result': 'W', 'len': 144,
         'game': 'W1.e4 B1.c5 W2.Nf3 B2.Nc6 W3.Bb5 B3.g6 W4.Bxc6 B4.dxc6 W5.d3 B5.Bg7 W6.h3 B6.Nf6 W7.Nc3'},
        {'result': 'D', 'len': 122,
         'game': 'W1.e4 B1.e5 W2.Nf3 B2.Nc6 W3.Bb5 B3.a6 W4.Ba4 B4.Nf6 W5.O-O B5.Be7 W6.Re1 B6.b5 W7.Bb3 B7.O-O W8.a4'},
        {'result': 'W', 'len': 149,
         'game': 'W1.e4 B1.e5 W2.Nf3 B2.Nc6 W3.Bb5 B3.a6 W4.Ba4 B4.Nf6 W5.O-O B5.Be7 W6.Re1 B6.b5 W7.Bb3 B7.d6 W8.c3'},
        {'result': 'D', 'len': 98,
         'game': 'W1.d4 B1.d5 W2.c4 B2.dxc4 W3.Nf3 B3.Nf6 W4.e3 B4.e6 W5.Bxc4 B5.c5 W6.O-O B6.a6 W7.a4 B7.Nc6 W8.Nc3'},
    ])
    moves = generate_boards_daraframe(games_df=games_df, n_from=0, n_to=5)
    games_consistency_check(moves)
