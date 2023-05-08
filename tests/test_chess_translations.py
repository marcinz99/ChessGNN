def test_field_to_int_and_reverse():
    from chessgnn.chess_utils.translations import int_to_field, field_to_int
    assert int_to_field[0] == '--'
    assert field_to_int['-'] == 0
    assert field_to_int['--'] == 0
    assert int_to_field[4] == 'd8'
    assert field_to_int['c4'] == 35
    for i in range(1, 65):
        assert field_to_int[int_to_field[i]] == i
