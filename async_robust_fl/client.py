"""
client.py — AsyncFLClient and make_client_fn.

Each Flower client represents one hospital.  The client:
  1. Receives the current global model weights.
  2. Trains locally on its private slide data (never shared).
  3. Optionally simulates adversarial, noisy, or unreliable behaviour.
  4. Returns updated weights + metadata for staleness / detection.

make_client_fn(dirichlet_alpha) is the public factory consumed by
Flower's ClientApp.  It returns a closure-based client_fn so that
Experiment E can override dirichlet_alpha without relying on
run_simulation's run_config argument (which is not stable across
Flower minor releases).
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import flwr as fl
from flwr.common import NDArrays, Scalar

from model import PathologyNet, train, evaluate_model, get_weights, set_weights
from data import load_data
from config import (
    DIRICHLET_ALPHA,
    NOISY_CLIENT_IDS,
)

logger = logging.getLogger(__name__)


class AsyncFLClient(fl.client.NumPyClient):
    """Flower NumPyClient representing a single hospital in the FL system.

    Responsibilities:
      - Local training on private pathology slides.
      - Adversarial injection (if this client is in MALICIOUS_CLIENT_IDS).
      - Simulated latency via exponential random delay (async simulation).
      - Dropout simulation by raising RuntimeError before training.

    The strategy detects malicious behaviour via norm / cosine filters;
    the client does NOT suppress or hide its identity — it is the server's
    defence layer that catches it.
    """

    def __init__(
        self,
        partition_id: int,
        net: PathologyNet,
        trainloader,
        testloader,
    ) -> None:
        self.partition_id = partition_id
        self.net          = net
        self.trainloader  = trainloader
        self.testloader   = testloader

    # ------------------------------------------------------------------
    # Flower API — training
    # ------------------------------------------------------------------

    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train locally and return updated weights.

        Execution order:
          1. Dropout check  — raise before doing any work (saves GPU time).
          2. Load global model weights.
          3. Local training.
          4. Adversarial injection  — applied AFTER honest training so that
             even malicious hospitals produce a valid base update first;
             the scaling factor then amplifies it.
          5. Simulate async arrival delay (metadata only — does not block).
        """

        # --- 1. Unreliable client: simulate dropout / network timeout ---
        # Determine role from the ID lists passed by the strategy.
        # client.cid is a UUID in Flower 1.8+ simulation, so we use our own
        # partition_id (available at construction time) to check membership.
        def _parse_ids(key: str) -> frozenset:
            raw = str(config.get(key, ""))
            return frozenset(int(x) for x in raw.split(",") if x.strip().isdigit())

        is_malicious  = self.partition_id in _parse_ids("malicious_ids")
        is_unreliable = self.partition_id in _parse_ids("unreliable_ids")

        dropout_prob = float(config.get("dropout_prob", 0.0)) if is_unreliable else 0.0
        if dropout_prob > 0.0 and np.random.random() < dropout_prob:
            raise RuntimeError(
                f"Hospital-{self.partition_id}: simulated dropout / network timeout."
            )

        # --- 2. Receive global model ---
        set_weights(self.net, parameters)

        # --- 3. Local training ---
        train(self.net, self.trainloader, epochs=int(config.get("local_epochs", 2)))
        updated_params = get_weights(self.net)

        # --- 4. Adversarial injection ---
        attack_type  = str(config.get("attack_type", "none"))

        if is_malicious:
            if attack_type == "scaling":
                scale = float(config.get("attack_scale", 10.0))
                updated_params = [p * scale for p in updated_params]

            elif attack_type == "random":
                updated_params = [
                    np.random.randn(*p.shape).astype(np.float32)
                    for p in updated_params
                ]

            elif attack_type == "sign_flip":
                updated_params = [-p for p in updated_params]

            else:
                logger.warning(
                    "Hospital-%d: unrecognised attack_type=%r — using honest update.",
                    self.partition_id, attack_type,
                )

        # --- 5. Simulated async arrival delay (metadata carried in metrics) ---
        simulated_delay = float(
            np.random.exponential(scale=float(config.get("delay_scale", 1.0)))
        )

        return updated_params, len(self.trainloader.dataset), {
            "client_id":           self.partition_id,
            "is_malicious":        is_malicious,
            "client_round":        int(config.get("current_round", 0)),
            # Echo model version back so strategy can compute staleness
            "client_model_version": int(config.get("model_version", 0)),
            "simulated_delay":     simulated_delay,
        }

    # ------------------------------------------------------------------
    # Flower API — evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar],
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the global model on local held-out slides."""
        set_weights(self.net, parameters)
        accuracy, loss = evaluate_model(self.net, self.testloader)
        return float(loss), len(self.testloader.dataset), {
            "accuracy": float(accuracy),
        }


# ---------------------------------------------------------------------------
# Flower ClientApp factory
# ---------------------------------------------------------------------------

def make_client_fn(dirichlet_alpha: float = DIRICHLET_ALPHA):
    """Return a Flower-compatible client_fn with ``dirichlet_alpha`` baked in.

    Using a closure avoids relying on ``run_simulation``'s ``run_config``
    keyword argument, whose availability varies across Flower minor releases.
    Each call to ``make_client_fn`` produces an independent factory so that
    different experiments (e.g. Exp E IID vs non-IID) do not share state.

    Hospital taxonomy (defined in config.py)::
        {0, 1}    → Malicious   (strategy injects attack via FitIns config)
        {4, 5}    → Noisy       (30 % label noise applied to training data)
        {6, 7}    → Unreliable  (dropout_prob = 0.4 per round)
        {2,3,8,9} → Honest

    Args:
        dirichlet_alpha: Heterogeneity parameter forwarded to
                         :func:`data.load_data`.  0.5 = non-IID (default),
                         1000.0 = near-IID (Experiment E).

    Returns:
        A ``client_fn(context) → fl.client.Client`` callable ready to be
        passed to ``flwr.client.ClientApp``.
    """
    _alpha = float(dirichlet_alpha)   # snapshot at factory creation time

    def client_fn(context) -> fl.client.Client:
        partition_id = int(context.node_config["partition-id"])

        # Noisy hospitals receive mislabelled training data
        noise_rate = 0.3 if partition_id in NOISY_CLIENT_IDS else 0.0

        net = PathologyNet()
        trainloader, testloader = load_data(
            partition_id,
            noise_rate      = noise_rate,
            dirichlet_alpha = _alpha,
        )

        return AsyncFLClient(partition_id, net, trainloader, testloader).to_client()

    return client_fn
