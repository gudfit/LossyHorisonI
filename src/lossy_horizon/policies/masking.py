from __future__ import annotations

from typing import List, Tuple, Optional, Set


def window_partition(n: int, window_size: int) -> List[Tuple[int, int]]:
    return [(i, min(i + window_size, n)) for i in range(0, n, window_size)]


def select_mask_set(
    surprisals: List[float],
    p_mask: float,
    window_size: int = 64,
    max_mask_run: int = 16,
    selection: str = "window",
    forbidden: Optional[Set[int]] = None,
    eligible: Optional[List[int]] = None,
) -> Tuple[List[int], List[int]]:
    n = len(surprisals)
    mask = [0] * n
    forb = forbidden or set()

    def can_add(idx: int) -> bool:
        if idx in forb:
            return False
        if max_mask_run <= 0:
            return True
        if mask[idx] == 1:
            return True
        l = 0
        j = idx - 1
        while j >= 0 and mask[j] == 1:
            l += 1
            j -= 1
        r = 0
        j = idx + 1
        while j < n and mask[j] == 1:
            r += 1
            j += 1
        return (l + 1 + r) <= max_mask_run

    def greedy_fill(candidates: List[int], target: int) -> int:
        selected = 0
        for idx in candidates:
            if selected >= target:
                break
            if mask[idx] == 0 and can_add(idx):
                mask[idx] = 1
                selected += 1
        return selected

    base_pool = eligible if eligible is not None else list(range(n))
    eligible_all = [i for i in base_pool if i not in forb]
    total_target = int(round(p_mask * len(eligible_all)))

    if selection == "global":
        sorted_global = sorted(eligible_all, key=lambda i: (surprisals[i], i))
        greedy_fill(sorted_global, total_target)
    else:
        windows = window_partition(n, window_size)
        selected_total = 0
        for s, e in windows:
            idxs = [i for i in range(s, e) if (i not in forb) and (i in eligible_all)]
            k = int(round(p_mask * len(idxs)))
            sorted_by_s = sorted(idxs, key=lambda i: (surprisals[i], i))
            # First pass
            selected_total += greedy_fill(sorted_by_s, k)
            # Per-window top-up
            window_selected = sum(mask[i] for i in idxs)
            if window_selected < k:
                remaining = [i for i in sorted_by_s if mask[i] == 0]
                added = greedy_fill(remaining, k - window_selected)
                selected_total += added
        # Top up globally
        if selected_total < total_target:
            remaining = [i for i in eligible_all if mask[i] == 0]
            remaining_sorted = sorted(remaining, key=lambda i: (surprisals[i], i))
            greedy_fill(remaining_sorted, total_target - selected_total)

    mask_indices = [i for i, m in enumerate(mask) if m == 1]
    keep_indices = [i for i, m in enumerate(mask) if m == 0]
    return mask_indices, keep_indices
