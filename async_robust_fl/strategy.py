"""
strategy.py — AsyncRobustFLStrategy: custom Flower Strategy.

Core behaviours implemented here:
  1. Versioned global model  — global_version counter sent to each client;
       echoed back as client_model_version; staleness = version gap.
  2. Filter-first async selection:
       ALL results are filtered (norm + cosine) BEFORE the fastest K are
       selected.  Filtering after selection allows attackers to exploit
       fast response times.
  3. Robust aggregation — delegates to aggregation.aggregate_robust().
  4. Dropout tracking — every round's (submitted, dropped) counts are logged
       in self.dropout_history for later visualisation.
  5. Server-side evaluation — uses an injected evaluate_fn for global accuracy.

No direct imports from Flower internals beyond the public API.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np

import flwr as fl
from flwr.common import (
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from aggregation import aggregate_robust
from detection import filter_by_norm, filter_by_cosine
from config import (
    CLIENTS_PER_ROUND,
    ASYNC_BUFFER_SIZE,
    AGGREGATION_METHOD,
    TRIM_FRACTION,
    NORM_THRESHOLD,
    COSINE_THRESHOLD,
    ATTACK_TYPE,
    ATTACK_SCALE,
    NOISE_RATE,
    DROPOUT_PROB,
    DELAY_SCALE,
    LOCAL_EPOCHS_WARMUP,
    LOCAL_EPOCHS_MAIN,
)


class AsyncRobustFLStrategy(Strategy):
    """Asynchronous, Byzantine-robust Federated Learning Strategy.

    Implements all six abstract methods required by the Flower Strategy API.

    Key design:
        - Filter → Select (not Select → Filter).  Attackers cannot game the
          async buffer by responding fast.
        - Staleness weight = 1 / (1 + staleness_rounds), so old updates
          contribute less to the global model.
        - All configuration (thresholds, client taxonomy) is injected at
          construction time — no global state dependencies.
    """

    def __init__(
        self,
        initial_parameters: Parameters,
        num_clients_per_round: int = CLIENTS_PER_ROUND,
        async_buffer_size: int = ASYNC_BUFFER_SIZE,
        aggregation_method: str = AGGREGATION_METHOD,
        trim_fraction: float = TRIM_FRACTION,
        norm_threshold: float = NORM_THRESHOLD,
        cosine_threshold: float = COSINE_THRESHOLD,
        evaluate_fn: Optional[Callable] = None,
        malicious_client_ids: Optional[frozenset] = None,
        noisy_client_ids: Optional[frozenset] = None,
        unreliable_client_ids: Optional[frozenset] = None,
    ) -> None:
        self.current_parameters     = initial_parameters
        self.num_clients_per_round  = num_clients_per_round
        self.async_buffer_size      = async_buffer_size
        self.aggregation_method     = aggregation_method
        self.trim_fraction          = trim_fraction
        self.norm_threshold         = norm_threshold
        self.cosine_threshold       = cosine_threshold
        self.evaluate_fn            = evaluate_fn
        self.malicious_client_ids   = malicious_client_ids  or frozenset()
        self.noisy_client_ids       = noisy_client_ids      or frozenset()
        self.unreliable_client_ids  = unreliable_client_ids or frozenset()

        # ---- Telemetry (persisted for evaluation.py) ----
        self.round_metrics:   List[Dict]  = []
        self.flagged_history: List[Dict]  = []   # [{'round': r, 'flagged': [ids]}]
        self.dropout_history: List[Dict]  = []   # [{'round': r, 'dropped': n, ...}]

        # Versioned model counter: incremented once per aggregation.
        # configure_fit() sends this to every client; client returns it as
        # 'client_model_version' so the strategy can compute staleness.
        self.global_version: int = 0

    # ------------------------------------------------------------------
    # Strategy API — initialisation
    # ------------------------------------------------------------------

    def initialize_parameters(self, client_manager) -> Parameters:
        return self.current_parameters

    # ------------------------------------------------------------------
    # Strategy API — configure training
    # ------------------------------------------------------------------

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Sample hospitals and build per-client training configs."""
        available   = client_manager.num_available()
        sample_size = min(self.num_clients_per_round, available)
        clients     = client_manager.sample(num_clients=sample_size)

        # Local epochs: short warm-up, then longer for stable rounds
        local_epochs = LOCAL_EPOCHS_WARMUP if server_round <= 3 else LOCAL_EPOCHS_MAIN

        fit_configurations: List[Tuple[ClientProxy, FitIns]] = []
        for client in clients:
            cid = int(client.cid)
            is_malicious  = cid in self.malicious_client_ids
            is_noisy      = cid in self.noisy_client_ids
            is_unreliable = cid in self.unreliable_client_ids

            config: Dict[str, Scalar] = {
                "local_epochs":  local_epochs,
                "current_round": server_round,
                "model_version": self.global_version,   # for staleness
                "delay_scale":   DELAY_SCALE,
                # --- Adversarial behaviour ---
                "is_malicious":  is_malicious,
                "attack_type":   ATTACK_TYPE if is_malicious else "none",
                "attack_scale":  ATTACK_SCALE,
                # --- Data quality ---
                "is_noisy":      is_noisy,
                "noise_rate":    NOISE_RATE if is_noisy else 0.0,
                # --- Network reliability ---
                "dropout_prob":  DROPOUT_PROB if is_unreliable else 0.0,
            }
            fit_configurations.append((client, FitIns(parameters, config)))

        return fit_configurations

    # ------------------------------------------------------------------
    # Strategy API — aggregate training results (core async logic)
    # ------------------------------------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Filter → Select → Aggregate, with staleness weighting."""

        # --- Track dropouts (unreliable clients + any other Flower failures) ---
        self.dropout_history.append({
            "round":          server_round,
            "submitted":      len(results),
            "dropped":        len(failures),
            "total_selected": len(results) + len(failures),
        })

        if not results:
            return self.current_parameters, {
                "error":   "no_results",
                "dropped": len(failures),
            }

        # STEP 1 — Build update dicts from ALL submitted results
        all_updates: List[Dict] = []
        for proxy, res in results:
            client_ver      = int(res.metrics.get("client_model_version", self.global_version))
            staleness       = self.global_version - client_ver
            staleness_weight = 1.0 / (1.0 + max(staleness, 0))

            all_updates.append({
                "params":          parameters_to_ndarrays(res.parameters),
                "num_examples":    res.num_examples,
                "staleness_weight": staleness_weight,
                "client_id":       int(proxy.cid),
                "is_malicious":    bool(res.metrics.get("is_malicious", False)),
                "simulated_delay": float(res.metrics.get("simulated_delay", 0.0)),
            })

        # STEP 2 — Apply detection filters on ALL results BEFORE async selection
        all_updates, norm_flagged   = filter_by_norm(all_updates, self.norm_threshold)
        all_updates, cosine_flagged = filter_by_cosine(all_updates, self.cosine_threshold)
        all_flagged = norm_flagged + cosine_flagged

        self.flagged_history.append({
            "round":   server_round,
            "flagged": all_flagged,
        })

        # STEP 3 — Sort clean survivors by simulated arrival time, take buffer_size
        all_updates_sorted = sorted(all_updates, key=lambda u: u["simulated_delay"])
        async_updates = all_updates_sorted[: self.async_buffer_size]

        if not async_updates:
            # All surviving updates were filtered — return current model unchanged
            return self.current_parameters, {
                "clients_filtered": len(all_flagged),
                "clients_dropped":  len(failures),
                "global_version":   self.global_version,
            }

        # STEP 4 — Robust aggregation on clean, staleness-weighted async updates
        weights = [u["staleness_weight"] * u["num_examples"] for u in async_updates]
        aggregated = aggregate_robust(
            [u["params"] for u in async_updates],
            method     = self.aggregation_method,
            weights    = weights,
            trim_fraction = self.trim_fraction,
        )

        self.current_parameters = ndarrays_to_parameters(aggregated)
        self.global_version += 1   # advance versioned model counter

        metrics: Dict[str, Scalar] = {
            "async_clients_used": len(async_updates),
            "total_submitted":    len(results),
            "clients_filtered":   len(all_flagged),
            "clients_dropped":    len(failures),
            "flagged_ids":        str(all_flagged),
            "global_version":     self.global_version,
        }
        self.round_metrics.append(metrics)
        return self.current_parameters, metrics

    # ------------------------------------------------------------------
    # Strategy API — configure / aggregate evaluation (server-side only)
    # ------------------------------------------------------------------

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        # Evaluation is done server-side via evaluate_fn; no client-side eval needed.
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        return None, {}

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side global evaluation using the injected evaluate_fn."""
        if self.evaluate_fn is None:
            return None
        return self.evaluate_fn(
            server_round,
            parameters_to_ndarrays(parameters),
            {},
        )
