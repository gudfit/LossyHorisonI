from lossy_horizon.bits.bit_cost import positional_bits_cost_from_indices, comb_log2_nCk


def test_positional_bits_enum_preferred_when_k0():
    n = 32
    keep = []
    bits_pos, pos_method, bits_enum, bits_rle = positional_bits_cost_from_indices(
        n, keep
    )
    assert pos_method == "enumerative"
    assert bits_pos == bits_enum
    assert bits_enum <= bits_rle


def test_positional_bits_rle_preferred_for_single_long_run():
    n = 128
    keep = list(range(40, 56))
    bits_pos, pos_method, bits_enum, bits_rle = positional_bits_cost_from_indices(
        n, keep
    )
    assert pos_method == "rle"
    assert bits_pos == bits_rle
    assert bits_rle <= bits_enum


def test_positional_bits_all_kept_equals_enumerative():
    n = 64
    keep = list(range(n))
    bits_pos, pos_method, bits_enum, bits_rle = positional_bits_cost_from_indices(
        n, keep
    )
    assert pos_method == "enumerative"
    assert bits_enum == comb_log2_nCk(n, n) == 0.0
    assert bits_pos == bits_enum
