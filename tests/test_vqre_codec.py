from lossy_horizon.codecs.vqre_codec import VQREEncoder, VQREDecoder


def _build_topk(start: int, K: int):
    return list(range(start, start + K))


def test_vqre_fallback_lossless_with_anchors():
    token_ids = [10, 11, 12, 13, 14, 15, 16, 17]
    n = len(token_ids)
    anchors_every = 2
    mask_idx = [i for i in range(n) if i % 2 == 1]
    K = 4
    ranks_by_index = {i: K + 100 for i in mask_idx}
    enc = VQREEncoder(rank_threshold=K, anchors_every=anchors_every, vocab_size=50000)
    payload = enc.encode(token_ids, mask_idx, ranks_by_index, topk_by_index={})
    topk_by_index = {i: _build_topk(100 + i * 10, K) for i in mask_idx}
    predicted_top1 = {i: 999 for i in mask_idx}
    dec = VQREDecoder(rank_threshold=K)
    full = dec.decode(payload, mask_idx, topk_by_index, predicted_top1)
    assert full == token_ids


def test_vqre_rank2_recovers_second_candidate_and_anchors_survive():
    token_ids = [101, 102, 103, 104, 105, 106]
    n = len(token_ids)
    anchors_every = 2
    mask_idx = [i for i in range(n) if i % 2 == 1]
    K = 4
    ranks_by_index = {i: 2 for i in mask_idx}
    topk_by_index = {i: [1000 + i, 2000 + i, 3000 + i, 4000 + i] for i in mask_idx}
    predicted_top1 = {i: topk_by_index[i][0] for i in mask_idx}
    enc = VQREEncoder(rank_threshold=K, anchors_every=anchors_every, vocab_size=50000)
    payload = enc.encode(token_ids, mask_idx, ranks_by_index, topk_by_index)
    dec = VQREDecoder(rank_threshold=K)
    full = dec.decode(payload, mask_idx, topk_by_index, predicted_top1)
    for i in range(0, n, anchors_every):
        assert full[i] == token_ids[i]
    for i in mask_idx:
        assert full[i] == topk_by_index[i][1]
