from lossy_horizon.bits.bit_cost import positional_bits_cost_from_indices


def test_positional_bits_enum_preferred_when_k0():
    n = 32
    keep = []
    bits_pos, pos_method, bits_enum, bits_rle = positional_bits_cost_from_indices(n, keep)
    assert pos_method == "enumerative"
    assert bits_pos == bits_enum
    assert bits_enum <= bits_rle


def test_positional_bits_rle_preferred_for_single_long_run():
    n = 128
    # Single contiguous keep run in the middle
    keep = list(range(40, 56))  # 16 tokens kept
    bits_pos, pos_method, bits_enum, bits_rle = positional_bits_cost_from_indices(n, keep)
    assert pos_method == "rle"
    assert bits_pos == bits_rle
    assert bits_rle <= bits_enum

