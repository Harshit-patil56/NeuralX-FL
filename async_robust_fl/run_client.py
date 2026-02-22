"""
run_client.py — Real-network FL client.  Run this on your FRIEND'S laptop.

Usage:
    Friend 1:  python run_client.py --partition-id 0
    Friend 2:  python run_client.py --partition-id 1

Setup (do this ONCE before the demo):
  1. Install Python 3.10 and run:  pip install -r requirements.txt
  2. Get pathmnist.npz from the organiser (USB / Google Drive).
  3. Put pathmnist.npz inside the  shared_data/  folder in this project.
  4. Make sure SERVER_HOST in config.py matches the organiser's Tailscale IP.
  5. Run the command above.

What happens:
  - This laptop downloads the current global model from the server.
  - Trains locally on YOUR local data slice (images never leave this laptop).
  - Sends only the updated MODEL WEIGHTS back to the server.
  - Repeat for every round.

What data stays here:
  - All images in shared_data/pathmnist.npz stay on this machine.
  - Only floating-point weight arrays (no pixels, no labels) go to the server.
  - You can verify this in Wireshark: capture tcp.port == 8080 and inspect frames.
"""

from __future__ import annotations

import argparse
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

import flwr as fl

from config import (
    USE_REAL_NETWORK,
    SERVER_HOST,
    SHARED_DATA_DIR,
    DIRICHLET_ALPHA,
)
from model import PathologyNet
from data import load_data
from client import AsyncFLClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NeuralX-FL real client")
    parser.add_argument(
        "--partition-id",
        type=int,
        required=True,
        choices=[0, 1],
        metavar="{0,1}",
        help="Which data partition this laptop owns. Friend 1 = 0, Friend 2 = 1.",
    )
    parser.add_argument(
        "--server-address",
        type=str,
        default=None,
        metavar="HOST:PORT",
        help="ngrok forwarding address e.g. 0.tcp.ngrok.io:12345  (overrides SERVER_HOST in config.py)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    partition_id: int = args.partition_id

    # --server-address on the command line takes priority over config.py
    server_address: str = args.server_address if args.server_address else SERVER_HOST

    if not USE_REAL_NETWORK:
        print(
            "\n[ERROR] USE_REAL_NETWORK is False in config.py.\n"
            "        Set USE_REAL_NETWORK = True before running run_client.py.\n"
        )
        sys.exit(1)

    if "NGROK_ADDRESS_GOES_HERE" in server_address:
        print(
            "\n[ERROR] No server address provided.\n"
            "        Pass the ngrok address like this:\n"
            "        python run_client.py --partition-id 0 --server-address 0.tcp.ngrok.io:12345\n"
        )
        sys.exit(1)

    npz_path = os.path.join(SHARED_DATA_DIR, "pathmnist.npz")
    if not os.path.exists(npz_path):
        print(
            f"\n[ERROR] Data file not found: {npz_path}\n"
            f"        Copy pathmnist.npz into the shared_data/ folder first.\n"
        )
        sys.exit(1)

    logger.info("======================================================")
    logger.info("  NeuralX-FL  — Real Network Client")
    logger.info("  Partition ID : %d", partition_id)
    logger.info("  Connecting to: %s", server_address)
    logger.info("  Data from    : %s", SHARED_DATA_DIR)
    logger.info("======================================================")
    logger.info("")
    logger.info("  DATA PRIVACY: Images in shared_data/ stay on THIS machine.")
    logger.info("  Only model weight arrays are sent to the server.")
    logger.info("")

    # Real clients are honest — no noise injection (that is simulation-only)
    trainloader, testloader = load_data(
        partition_id    = partition_id,
        noise_rate      = 0.0,       # real hospital — no artificial noise
        dirichlet_alpha = DIRICHLET_ALPHA,
    )

    net    = PathologyNet()
    client = AsyncFLClient(partition_id, net, trainloader, testloader)

    logger.info("Loaded %d training samples for partition %d.",
                len(trainloader.dataset), partition_id)
    logger.info("Connecting to server at %s …", server_address)

    fl.client.start_numpy_client(
        server_address = server_address,
        client         = client,
    )

    logger.info("Done. Disconnected from server.")


if __name__ == "__main__":
    main()
