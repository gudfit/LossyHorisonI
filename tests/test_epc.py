from lossy_horizon.codecs import EPCEncoder, EPCDecoder


def test_epc_fallback_is_lossless():
    token_ids = [10, 11, 12, 13, 14]
    n = len(token_ids)
    mask_idx = [1, 3]
    keep_idx = [i for i in range(n) if i not in mask_idx]
    K = 4
    ranks_by_index = {1: 100, 3: 100}
    enc = EPCEncoder(rank_threshold=K, fallback=True, vocab_size=50000)
    payload = enc.encode(token_ids, keep_idx, mask_idx, ranks_by_index)
    topk_by_index = {
        1: list(range(100, 100 + K)),
        3: list(range(200, 200 + K)),
    }
    predicted_top1 = {1: 999, 3: 999}

    dec = EPCDecoder(rank_threshold=K)
    full = dec.decode(payload, mask_idx, topk_by_index, predicted_top1)
    assert full == token_ids
