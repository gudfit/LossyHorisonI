from lossy_horizon.policies import select_mask_set
from lossy_horizon.policies.masking import window_partition


def _max_run_length(mask_indices, n):
    mask = [0] * n
    for i in mask_indices:
        mask[i] = 1
    best = cur = 0
    for v in mask:
        if v == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def test_special_token_exclusion():
    n = 20
    surprisals = [float(i % 7) for i in range(n)]
    forbidden = {0, 5, 7, 11, 19}
    mask_idx, _ = select_mask_set(
        surprisals,
        p_mask=0.5,
        selection="global",
        forbidden=forbidden,
    )
    assert set(mask_idx).isdisjoint(forbidden)


def test_run_cap_enforced():
    n = 50
    surprisals = [float(i) for i in range(n)]
    p_mask = 0.9
    cap = 3
    mask_idx, _ = select_mask_set(
        surprisals,
        p_mask=p_mask,
        max_mask_run=cap,
        selection="window",
        window_size=25,
        forbidden=set(),
    )
    assert _max_run_length(mask_idx, n) <= cap


def test_per_window_rate_within_one():
    n = 100
    surprisals = [float((i * 7) % 13) for i in range(n)]
    p_mask = 0.5
    window_size = 20
    cap = 8
    mask_idx, _ = select_mask_set(
        surprisals,
        p_mask=p_mask,
        max_mask_run=cap,
        selection="window",
        window_size=window_size,
        forbidden=set(),
    )
    mask_set = set(mask_idx)
    for s, e in window_partition(n, window_size):
        idxs = list(range(s, e))
        target = int(round(p_mask * len(idxs)))
        selected = sum(1 for i in idxs if i in mask_set)
        assert abs(selected - target) <= 1
