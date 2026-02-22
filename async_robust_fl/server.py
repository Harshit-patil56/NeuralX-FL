"""
server.py — Real-network FL server.  Run this on YOUR laptop.

Usage:
    python server.py

Requirements:
  - ngrok installed: https://ngrok.com/download  (free account is enough)
  - Both friends running run_client.py on their laptops.

ngrok workflow (do in this order):
  1. python server.py          ← start server first
  2. ngrok tcp 8080            ← in a second terminal
  3. ngrok prints:  Forwarding  tcp://0.tcp.ngrok.io:12345 -> localhost:8080
  4. Give  0.tcp.ngrok.io:12345  to friends (WhatsApp, chat, etc.)
  5. Friends run:
       python run_client.py --partition-id 0 --server-address 0.tcp.ngrok.io:12345
       python run_client.py --partition-id 1 --server-address 0.tcp.ngrok.io:12345

What this does:
  - Starts a real Flower gRPC server on 0.0.0.0:8080.
  - Uses the same AsyncRobustFLStrategy (detection, trust scoring) as simulation.
  - Broadcasts global model to friends, receives weight gradients (NOT images).
  - Evaluates the global model on YOUR local test set after each round.

Wireshark proof:
  - On a friend's laptop: open Wireshark → capture on the network interface.
  - Filter: tcp.port == 8080
  - You will see gRPC frames containing float arrays (model weights).
  - You will NOT see any image data, patient IDs, or raw pixels.
  - This proves federated learning — data never leaves the client machine.
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import torch
import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerConfig

from config import (
    USE_REAL_NETWORK,
    SERVER_BIND,
    SERVER_HOST,
    REAL_NUM_CLIENTS,
    REAL_CLIENTS_PER_ROUND,
    REAL_ASYNC_BUFFER_SIZE,
    NUM_ROUNDS,
    TRIM_FRACTION,
    NORM_THRESHOLD,
    COSINE_THRESHOLD,
    RESULTS_DIR,
)
from model import PathologyNet, get_weights
from data import load_global_test
from strategy import AsyncRobustFLStrategy
from evaluation import make_evaluate_fn


def main() -> None:
    if not USE_REAL_NETWORK:
        print(
            "\n[ERROR] USE_REAL_NETWORK is False in config.py.\n"
            "        Set USE_REAL_NETWORK = True before running server.py.\n"
        )
        sys.exit(1)

    if "REPLACE_WITH_YOUR_TAILSCALE_IP" in SERVER_HOST:
        print(
            "\n[ERROR] SERVER_HOST is not set in config.py.\n"
            "        Run:  tailscale ip -4\n"
            "        Then set SERVER_HOST = '<that_ip>:8080' in config.py.\n"
        )
        sys.exit(1)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    logger.info("======================================================")
    logger.info("  NeuralX-FL  — Real Network Server")
    logger.info("  Listening on : %s", SERVER_BIND)
    logger.info("  Clients connect to: %s", SERVER_HOST)
    logger.info("  Rounds   : %d", NUM_ROUNDS)
    logger.info("  Clients  : %d real laptops", REAL_NUM_CLIENTS)
    logger.info("======================================================")
    logger.info("")
    logger.info("  WIRESHARK PROOF: Capture tcp.port == 8080 on a friend's")
    logger.info("  laptop. Traffic = float arrays (gradients). No images.")
    logger.info("")

    initial_params = ndarrays_to_parameters(get_weights(PathologyNet()))
    eval_fn        = make_evaluate_fn(load_global_test())

    strategy = AsyncRobustFLStrategy(
        initial_parameters    = initial_params,
        num_clients_per_round = REAL_CLIENTS_PER_ROUND,
        async_buffer_size     = REAL_ASYNC_BUFFER_SIZE,
        aggregation_method    = "trimmed_mean",
        trim_fraction         = TRIM_FRACTION,
        norm_threshold        = NORM_THRESHOLD,
        cosine_threshold      = COSINE_THRESHOLD,
        evaluate_fn           = eval_fn,
        # Real clients are honest — no simulated malicious / noisy / unreliable
        malicious_client_ids  = frozenset(),
        noisy_client_ids      = frozenset(),
        unreliable_client_ids = frozenset(),
        use_detection         = True,
    )

    fl.server.start_server(
        server_address = SERVER_BIND,
        strategy       = strategy,
        config         = ServerConfig(num_rounds=NUM_ROUNDS),
    )

    logger.info("Training complete.")
    logger.info("Accuracy history : %s", [round(a, 4) for a in eval_fn.accuracy_history])


if __name__ == "__main__":
    main()
