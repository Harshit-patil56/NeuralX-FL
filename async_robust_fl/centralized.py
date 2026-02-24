"""
centralized.py — Centralised training baseline for scientific comparison.

Purpose:
    Train PathologyNet on the full PathMNIST training set (all 89,996 images)
    without any federation, privacy, or attack simulation.  This represents
    the theoretical performance upper bound achievable when all hospital data
    is pooled in one place — i.e., what FL is *trying to approximate* without
    data sharing.

Comparison use cases:
    1. Centralised accuracy vs FL accuracy per round (convergence gap).
    2. Centralised convergence speed vs FL communication cost.
    3. Illustrates the privacy-utility trade-off quantitatively.

Usage:
    python centralized.py                   # standalone
    # or import run_centralized() from main.py for integrated comparison.

Output:
    results/centralized_accuracy.csv        — per-epoch accuracy + loss
    results/centralized_vs_fl.png           — convergence overlay plot
"""

from __future__ import annotations

import csv
import logging
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

from config import (
    SEED,
    CENTRALIZED_EPOCHS,
    LEARNING_RATE,
    NUM_CLASSES,
    RESULTS_DIR,
)
from data import load_global_test, load_global_train
from model import PathologyNet, evaluate_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def run_centralized(
    num_epochs: int = CENTRALIZED_EPOCHS,
    seed: int = SEED,
) -> Tuple[List[float], List[float], List[float]]:
    """Train PathologyNet centrally and record per-epoch metrics.

    All data is used (no Dirichlet split) and no attacks are injected.
    This is the strongest possible non-federated baseline.

    Args:
        num_epochs: Number of full-dataset training epochs.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of three equal-length lists:
            accuracy_history : global test accuracy after each epoch.
            loss_history     : global test loss after each epoch.
            epoch_time_secs  : wall-clock time for each training epoch.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = PathologyNet().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    trainloader = load_global_train()
    testloader  = load_global_test()

    accuracy_history: List[float] = []
    loss_history:     List[float] = []
    epoch_times:      List[float] = []

    logger.info(
        "Centralised baseline: %d epochs, lr=%.4f, device=%s",
        num_epochs, LEARNING_RATE, device,
    )

    for epoch in range(1, num_epochs + 1):
        t_start = time.perf_counter()
        net.train()

        for batch in trainloader:
            images, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

        epoch_time = time.perf_counter() - t_start
        accuracy, test_loss = evaluate_model(net, testloader, device=device)

        accuracy_history.append(accuracy)
        loss_history.append(test_loss)
        epoch_times.append(epoch_time)

        logger.info(
            "[Epoch %03d/%03d]  acc=%.4f  loss=%.4f  time=%.1fs",
            epoch, num_epochs, accuracy, test_loss, epoch_time,
        )

    return accuracy_history, loss_history, epoch_times


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_centralized_csv(
    accuracy_history: List[float],
    loss_history: List[float],
    epoch_times: List[float],
    filename: str = "centralized_accuracy.csv",
) -> str:
    """Write per-epoch centralised metrics to a CSV file.

    Columns: epoch, accuracy, loss, epoch_time_secs

    Args:
        accuracy_history : Per-epoch accuracy values.
        loss_history     : Per-epoch loss values.
        epoch_times      : Per-epoch wall-clock training time.
        filename         : Output filename inside RESULTS_DIR.

    Returns:
        Absolute path of the written CSV file.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "accuracy", "loss", "epoch_time_secs"])
        for epoch, (acc, loss, t) in enumerate(
            zip(accuracy_history, loss_history, epoch_times), start=1
        ):
            writer.writerow([epoch, f"{acc:.6f}", f"{loss:.6f}", f"{t:.3f}"])

    logger.info("Centralised CSV saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    acc_hist, loss_hist, times_hist = run_centralized()
    export_centralized_csv(acc_hist, loss_hist, times_hist)
    logger.info(
        "Final centralised accuracy: %.4f  |  Total time: %.0fs",
        acc_hist[-1],
        sum(times_hist),
    )
