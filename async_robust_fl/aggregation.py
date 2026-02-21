"""
aggregation.py — Byzantine-robust aggregation algorithms.

All functions are pure (no side effects, no global state).
The strategy passes method='trimmed_mean' | 'median' | 'krum' | 'fedavg'
and calls aggregate_robust(), which dispatches to the right implementation.

References:
  - Trimmed Mean / Median: Yin et al., ICML 2018
    "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"
  - Krum: Blanchard et al., NeurIPS 2017
    "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
  - FedAvg: McMahan et al., AISTATS 2017
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Low-level algorithms (operate on a single layer / 1-D arrays)
# ---------------------------------------------------------------------------

def trimmed_mean(layer_updates: List[np.ndarray], trim_fraction: float = 0.1) -> np.ndarray:
    """Coordinate-wise trimmed mean across client updates for one layer.

    For each coordinate, sort values, discard bottom and top `trim_fraction`
    fraction, then average the remaining values.

    Robust against up to `trim_fraction` fraction of Byzantine clients.

    Args:
        layer_updates  : List of 1-D float32 arrays, one per client.
        trim_fraction  : Fraction to remove from each tail (0.1 removes 10%
                         from each end, tolerating 10% adversarial fraction).

    Returns:
        1-D float32 array of aggregated values.
    """
    stacked = np.stack(layer_updates, axis=0)   # (n_clients, n_params)
    n = len(layer_updates)
    k = int(np.floor(trim_fraction * n))

    if k == 0:
        return np.mean(stacked, axis=0).astype(np.float32)

    # Guard: trim_fraction >= 0.5 would remove everything → fall back to median
    if n - 2 * k <= 0:
        return np.median(stacked, axis=0).astype(np.float32)

    sorted_stacked = np.sort(stacked, axis=0)
    trimmed = sorted_stacked[k : n - k, :]
    return np.mean(trimmed, axis=0).astype(np.float32)


def coordinate_median(layer_updates: List[np.ndarray]) -> np.ndarray:
    """Coordinate-wise median across client updates for one layer.

    The median is the most robust of the three aggregators — it is
    provably resilient against fewer than 50% Byzantine clients.

    Args:
        layer_updates: List of 1-D float32 arrays, one per client.

    Returns:
        1-D float32 array of per-coordinate medians.
    """
    stacked = np.stack(layer_updates, axis=0)   # (n_clients, n_params)
    return np.median(stacked, axis=0).astype(np.float32)


def krum(
    layer_updates: List[np.ndarray],
    num_byzantine: int,
    multi: bool = False,
    m: Optional[int] = None,
) -> np.ndarray:
    """Krum / Multi-Krum aggregation for one layer.

    Krum selects the single update whose sum of squared distances to its
    (n - num_byzantine - 2) nearest neighbours is smallest.

    Multi-Krum selects the `m` updates with the lowest Krum scores and
    then averages them, providing a trade-off between robustness and accuracy.

    Args:
        layer_updates  : List of 1-D float32 arrays, one per client.
        num_byzantine  : Assumed upper-bound on Byzantine clients (f).
        multi          : If True, use Multi-Krum (average top-m).
        m              : Number of updates to select for Multi-Krum.
                         Defaults to n - num_byzantine.

    Returns:
        1-D float32 array.
    """
    n = len(layer_updates)
    f = num_byzantine
    neighbours = n - f - 2   # Krum neighbourhood size

    if neighbours <= 0:
        # Not enough clients to run Krum; fall back to median
        return coordinate_median(layer_updates)

    scores: List[float] = []
    for i in range(n):
        distances = sorted(
            float(np.sum((layer_updates[i] - layer_updates[j]) ** 2))
            for j in range(n)
            if j != i
        )
        scores.append(sum(distances[:neighbours]))

    if multi:
        k = m if m is not None else (n - f)
        k = max(1, min(k, n))
        selected_idx = np.argsort(scores)[:k]
        selected = [layer_updates[i] for i in selected_idx]
        return np.mean(np.stack(selected, axis=0), axis=0).astype(np.float32)
    else:
        return layer_updates[int(np.argmin(scores))].astype(np.float32)


# ---------------------------------------------------------------------------
# Top-level dispatcher — strategy calls this
# ---------------------------------------------------------------------------

def aggregate_robust(
    all_client_params: List[List[np.ndarray]],
    method: str = "trimmed_mean",
    weights: Optional[List[float]] = None,
    trim_fraction: float = 0.1,
    num_byzantine: int = 2,
) -> List[np.ndarray]:
    """Aggregate model parameters from multiple clients using `method`.

    Each element of `all_client_params` is a list of numpy arrays (one per
    layer / parameter tensor) for a single client.

    Args:
        all_client_params : List[List[np.ndarray]] — outer = clients, inner = layers.
        method            : Aggregation algorithm to use.
        weights           : Per-client weights (used only for 'fedavg').
                            Typically = staleness_weight × num_examples.
        trim_fraction     : Passed to trimmed_mean.
        num_byzantine     : Passed to krum.

    Returns:
        List[np.ndarray] — one float32 array per layer, same shapes as input.

    Raises:
        ValueError: if `method` is not recognised.
    """
    if not all_client_params:
        raise ValueError("aggregate_robust received an empty list of client params.")

    num_layers = len(all_client_params[0])
    aggregated: List[np.ndarray] = []

    for layer_idx in range(num_layers):
        # Flatten each client's layer to 1-D for the aggregation algorithms
        layer_updates: List[np.ndarray] = [
            cp[layer_idx].flatten().astype(np.float32)
            for cp in all_client_params
        ]

        if method == "trimmed_mean":
            agg_layer = trimmed_mean(layer_updates, trim_fraction=trim_fraction)

        elif method == "median":
            agg_layer = coordinate_median(layer_updates)

        elif method == "krum":
            agg_layer = krum(layer_updates, num_byzantine=num_byzantine)

        elif method == "fedavg":
            if weights is not None:
                w = np.array(weights, dtype=np.float64)
                w = w / w.sum()                              # normalise to sum=1
                stacked = np.stack(layer_updates, axis=0)   # (n, params)
                agg_layer = np.sum(stacked * w[:, np.newaxis], axis=0).astype(np.float32)
            else:
                stacked = np.stack(layer_updates, axis=0)
                agg_layer = np.mean(stacked, axis=0).astype(np.float32)

        else:
            raise ValueError(
                f"Unknown aggregation method '{method}'. "
                "Choose from: 'trimmed_mean', 'median', 'krum', 'fedavg'."
            )

        # Restore original layer shape and ensure float32 throughout
        original_shape = all_client_params[0][layer_idx].shape
        aggregated.append(agg_layer.reshape(original_shape).astype(np.float32))

    return aggregated
