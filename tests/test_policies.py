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
    surprisals = [float(i % 7) for i in range(20)]
    forbidden = {0, 5, 7, 11, 19}
    mask_idx, _ = select_mask_set(
        surprisals,
        p_mask=0.5,
        selection="global",
        forbidden=forbidden,
    )
    assert set(mask_idx).isdisjoint(forbidden)


def test_run_cap_enforced():
    surprisals = [float(i) for i in range(50)]
    mask_idx, _ = select_mask_set(
        surprisals,
        p_mask=0.9,
        max_mask_run=3,
        selection="window",
        window_size=25,
        forbidden=set(),
    )
    assert _max_run_length(mask_idx, 50) <= 3


def test_per_window_rate_within_one():
    surprisals = [float((i * 7) % 13) for i in range(100)]
    window_size = 20
    mask_idx, _ = select_mask_set(
        surprisals,
        p_mask=0.5,
        max_mask_run=8,
        selection="window",
        window_size=window_size,
        forbidden=set(),
    )
    for s, e in window_partition(100, window_size):
        idxs = list(range(s, e))
        target = int(round(0.5 * len(idxs)))
        selected = sum(1 for i in idxs if i in set(mask_idx))
        assert abs(selected - target) <= 1
