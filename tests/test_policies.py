from lossy_horizon.policies import select_mask_set


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
    # Encourage masking many positions
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

