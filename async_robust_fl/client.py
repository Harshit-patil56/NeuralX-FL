"""
client.py — AsyncFLClient and client_fn.

Each Flower client represents one hospital.  The client:
  1. Receives the current global model weights.
  2. Trains locally on its private slide data (never shared).
  3. Optionally simulates adversarial, noisy, or unreliable behaviour.
  4. Returns updated weights + metadata for staleness / detection.

client_fn() is the factory consumed by Flower's ClientApp.
It reads dirichlet_alpha from context.run_config so Experiment E can
override the partitioning without touching any other code.
"""

from __future__ import annotations

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
        dropout_prob = float(config.get("dropout_prob", 0.0))
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
        is_malicious = bool(config.get("is_malicious", False))
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

            # Unknown attack type → honest update (safe fallback, avoid silent error)

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

def client_fn(context) -> fl.client.Client:
    """Create and return an AsyncFLClient for the given hospital (context).

    Reads `dirichlet_alpha` from context.run_config so that Experiment E
    can override partitioning without modifying this function.

    Hospital taxonomy (defined in config.py):
        {0,1}    → Malicious   (strategy injects attack via config)
        {4,5}    → Noisy       (data partitioned with 30% label noise)
        {6,7}    → Unreliable  (dropout_prob=0.4 per round)
        {2,3,8,9}→ Honest
    """
    partition_id = int(context.node_config["partition-id"])

    # Allow alpha override for Experiment E (IID vs non-IID isolation)
    alpha = float(context.run_config.get("dirichlet_alpha", DIRICHLET_ALPHA))

    # Noisy hospitals receive mislabelled training data
    noise_rate = 0.3 if partition_id in NOISY_CLIENT_IDS else 0.0

    net = PathologyNet()
    trainloader, testloader = load_data(
        partition_id,
        noise_rate      = noise_rate,
        dirichlet_alpha = alpha,
    )

    return AsyncFLClient(partition_id, net, trainloader, testloader).to_client()
