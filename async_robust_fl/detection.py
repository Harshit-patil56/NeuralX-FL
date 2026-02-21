"""
detection.py — Byzantine-client detection filters.

Two complementary filters catch different attack classes:
  1. filter_by_norm    — catches scaling attacks (abnormally large updates)
  2. filter_by_cosine  — catches sign-flip and directional poisoning

Pipeline (enforced in strategy.aggregate_fit):
    clean, norm_flagged   = filter_by_norm(all_updates, threshold)
    clean, cosine_flagged = filter_by_cosine(clean, threshold)

Both functions are pure: they do not modify the input dicts in-place but
may add computed fields ('norm', 'cosine_sim') to each dict as annotations.

Design note on cosine reference direction:
    We use the coordinate-wise MEDIAN of all updates, not the mean.
    If malicious clients dominate the buffer, the mean shifts toward the
    attack — making mean-based filtering self-defeating. The median is
    provably robust as long as fewer than 50% of clients are Byzantine.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
UpdateDict = dict   # keys: 'params', 'num_examples', 'staleness_weight',
                    #       'client_id', 'is_malicious', 'simulated_delay'


# ---------------------------------------------------------------------------
# Filter 1: L2-norm filter
# ---------------------------------------------------------------------------

def filter_by_norm(
    updates: List[UpdateDict],
    threshold_multiplier: float = 3.0,
) -> Tuple[List[UpdateDict], List[int]]:
    """Reject updates whose L2 norm exceeds `threshold_multiplier × median_norm`.

    Scaling attacks multiply gradients by a large factor (e.g. 10×), which
    inflates the L2 norm far beyond honest updates.

    Args:
        updates              : All update dicts from submitted clients.
        threshold_multiplier : Reject if norm > this × median norm.
                               3.0 is robust against 20% adversaries.

    Returns:
        (clean_updates, flagged_client_ids)
    """
    if not updates:
        return [], []

    # Compute L2 norm for each client (flat concatenation of all layers)
    for update in updates:
        flat = np.concatenate([p.flatten() for p in update["params"]])
        update["norm"] = float(np.linalg.norm(flat))

    norms = [u["norm"] for u in updates]
    median_norm = float(np.median(norms))
    cutoff = threshold_multiplier * median_norm

    clean_updates: List[UpdateDict] = []
    flagged: List[int] = []

    for update in updates:
        if update["norm"] <= cutoff:
            clean_updates.append(update)
        else:
            flagged.append(int(update["client_id"]))

    return clean_updates, flagged


# ---------------------------------------------------------------------------
# Filter 2: Cosine-similarity filter
# ---------------------------------------------------------------------------

def filter_by_cosine(
    updates: List[UpdateDict],
    similarity_threshold: float = 0.0,
) -> Tuple[List[UpdateDict], List[int]]:
    """Reject updates whose cosine similarity with the MEDIAN direction is below threshold.

    Catches:
      - Sign-flip attacks (cosine ≈ -1)
      - Adversarial directional deviations
      - Noisy clients whose updates consistently diverge from the consensus

    The reference direction is the coordinate-wise median of all flat updates —
    not the mean.  This makes the reference itself robust to outlier directions.

    Args:
        updates              : Surviving updates after norm filtering.
        similarity_threshold : Reject if cosine_sim < this value.
                               0.0 rejects updates pointing in the opposite half-space.

    Returns:
        (clean_updates, flagged_client_ids)
    """
    if len(updates) < 2:
        # Cannot compute a meaningful cosine reference with fewer than 2 updates
        return updates, []

    # Flatten every client's parameters to a single 1-D vector
    flat_updates: List[np.ndarray] = [
        np.concatenate([p.flatten() for p in u["params"]]).astype(np.float64)
        for u in updates
    ]

    stacked = np.stack(flat_updates, axis=0)            # (n_clients, n_params)
    median_direction = np.median(stacked, axis=0)       # coordinate-wise median
    median_norm = float(np.linalg.norm(median_direction))

    # Degenerate case: all updates near zero (e.g. round 0 with random weights)
    if median_norm < 1e-10:
        return updates, []

    clean_updates: List[UpdateDict] = []
    flagged: List[int] = []

    for i, update in enumerate(updates):
        flat = flat_updates[i]
        flat_norm = float(np.linalg.norm(flat))

        if flat_norm < 1e-10:
            # Near-zero update: suspicious, flag it
            flagged.append(int(update["client_id"]))
            continue

        cosine_sim = float(
            np.dot(flat, median_direction) / (flat_norm * median_norm)
        )
        update["cosine_sim"] = cosine_sim

        if cosine_sim >= similarity_threshold:
            clean_updates.append(update)
        else:
            flagged.append(int(update["client_id"]))

    return clean_updates, flagged
