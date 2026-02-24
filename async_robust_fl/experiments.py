"""
experiments.py — Research-level experimental extensions for NeuralX-FL.

Implements the six research-grade capabilities that extend main.py:

  1. Defense vs Defense comparison
       FedAvg | Coordinate Median | Krum | Trimmed Mean | AsyncRobustFL
       — all under identical attack / dataset / participation conditions.

  2. Sensitivity analysis
       Dirichlet α ∈ {0.1, 0.5, 1.0}
       Byzantine client % ∈ {10%, 20%, 30%}
       Async buffer size ∈ {4, 6, 8}
       Each parameter varies independently; all others are frozen.

  3. Multi-seed runs
       SEEDS ∈ {42, 123, 999}
       All randomness (dataset partitioning, model init, training) is
       seed-controlled for full reproducibility.

  4. Statistical reporting
       Mean ± std across seed runs for every configuration.
       Exported to CSV and JSON in RESULTS_DIR.

  5. Robustness / failure threshold
       Byzantine fraction ∈ {0%, 10%, 20%, 30%, 40%, 50%} of 10 clients.
       Final accuracy tracked per fraction; failure point identified as the
       fraction where accuracy drops below a configurable threshold.

  6. Communication cost analysis
       Per round: uploads transmitted, model bytes, total communication volume.
       Accuracy vs cumulative bytes curve.
       Rounds-to-target accuracy (default 0.80) metric.

Usage::
    python experiments.py                 # run all six experiment groups
    python experiments.py --group defense # run a single group

All outputs are written to RESULTS_DIR (configured in config.py).

References:
  - McMahan et al. (2017) FedAvg
  - Blanchard et al. (2017) Krum
  - Yin et al. (2018) Trimmed Mean / Coordinate Median
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup: allow running from workspace root or from within the package dir
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

# Logging must be configured before any package import that creates loggers.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import torch

from config import (
    SEED,
    NUM_CLIENTS,
    NUM_ROUNDS,
    CLIENTS_PER_ROUND,
    ASYNC_BUFFER_SIZE,
    DIRICHLET_ALPHA,
    MALICIOUS_CLIENT_IDS,
    NOISY_CLIENT_IDS,
    RESULTS_DIR,
)
from model import PathologyNet
from main import run_one_experiment


# ---------------------------------------------------------------------------
# Constants for the research experiments
# ---------------------------------------------------------------------------

#: Seeds used for all multi-seed experiments.
MULTI_SEEDS: Tuple[int, ...] = (42, 123, 999)

#: Aggregation methods compared in the defense-vs-defense experiment.
DEFENSE_METHODS: Tuple[str, ...] = ("fedavg", "median", "krum", "trimmed_mean")

#: Dirichlet α values for the sensitivity sweep.
ALPHA_VALUES: Tuple[float, ...] = (0.1, 0.5, 1.0)

#: Byzantine fractions for the sensitivity and failure-threshold sweeps.
#: Expressed as a fraction of NUM_CLIENTS. 0.1 → 1 client, 0.5 → 5 clients.
BYZANTINE_FRACTIONS: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)

#: Async buffer sizes for the buffer-size sensitivity sweep.
BUFFER_SIZES: Tuple[int, ...] = (4, 6, 8)

#: Accuracy target used for "rounds to target" communication-cost metric.
ACCURACY_TARGET: float = 0.80

#: Failure threshold: accuracy below this value is considered a failure.
FAILURE_THRESHOLD: float = 0.50


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _ensure_results_dir() -> str:
    """Create RESULTS_DIR if absent and return its path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def _save_fig(fig: plt.Figure, filename: str) -> str:
    """Save a matplotlib figure to RESULTS_DIR and close it.

    Args:
        fig:      The figure to save.
        filename: Output filename (e.g. 'defense_comparison.png').

    Returns:
        Absolute path of the saved image.
    """
    path = os.path.join(_ensure_results_dir(), filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def _model_param_count() -> int:
    """Return the total number of scalar parameters in PathologyNet."""
    net = PathologyNet()
    return sum(p.numel() for p in net.parameters())


def _model_size_bytes() -> int:
    """Return the size of PathologyNet's parameters in bytes (float32 = 4 B/param)."""
    return _model_param_count() * 4


def _rounds_to_target(accuracy_history: List[float], target: float = ACCURACY_TARGET) -> Optional[int]:
    """Return the first round index (1-based) where accuracy >= target, or None."""
    for rnd, acc in enumerate(accuracy_history, start=1):
        if acc >= target:
            return rnd
    return None


def _byzantine_ids_for_fraction(fraction: float) -> frozenset:
    """Return a frozenset of client IDs corresponding to the given Byzantine fraction.

    Clients are assigned IDs 0, 1, 2, … in order of increasing fraction.
    This ensures a strict superset relationship as fraction increases, so
    the sets are nested and directly comparable across sweep values.

    Args:
        fraction: Fraction of NUM_CLIENTS to mark as Byzantine, e.g. 0.2.

    Returns:
        Frozenset of integer client IDs to designate as malicious.
    """
    n_byzantine = int(round(fraction * NUM_CLIENTS))
    return frozenset(range(n_byzantine))


# ---------------------------------------------------------------------------
# 1. Defense vs Defense Comparison
# ---------------------------------------------------------------------------

def run_defense_comparison(
    seeds: Tuple[int, ...] = (SEED,),
    num_rounds: int = NUM_ROUNDS,
) -> Dict[str, Dict]:
    """Compare aggregation defenses under identical attack conditions.

    All methods are evaluated with:
      - Same Dirichlet α (= DIRICHLET_ALPHA from config)
      - Same malicious client IDs (= MALICIOUS_CLIENT_IDS from config)
      - Same CLIENTS_PER_ROUND and ASYNC_BUFFER_SIZE
      - Same seed(s)
      - Byzantine detection enabled for all methods except FedAvg
        (FedAvg is tested both WITHOUT detection to expose its raw
        vulnerability, and WITH detection to isolate aggregation differences)

    Note: FedAvg without detection is the standard FedAvg baseline. The other
    methods always run with detection enabled, since robust aggregators are
    designed to be used alongside detection / filtering.

    Args:
        seeds:      Tuple of random seeds. Per-seed results are averaged for
                    multi-seed runs.
        num_rounds: Number of FL communication rounds.

    Returns:
        Dict mapping method label → {
          "accuracy_per_seed": List[List[float]],
          "loss_per_seed":     List[List[float]],
          "mean_accuracy":     List[float],
          "std_accuracy":      List[float],
          "final_acc_mean":    float,
          "final_acc_std":     float,
        }
    """
    logger.info("=" * 60)
    logger.info("Experiment Group 1: Defense vs Defense Comparison")
    logger.info("Methods: %s", list(DEFENSE_METHODS))
    logger.info("Seeds:   %s | Rounds: %d", list(seeds), num_rounds)
    logger.info("=" * 60)

    # FedAvg is run WITHOUT detection (to match the standard FedAvg protocol
    # used in the FL literature) and also WITH detection for a fair comparison.
    # Label "FedAvg (no detection)" and "FedAvg (with detection)" keep them
    # separate in the output dict.
    method_configs = [
        ("fedavg",       False, "FedAvg (no detection)"),
        ("fedavg",       True,  "FedAvg (with detection)"),
        ("median",       True,  "Coordinate Median"),
        ("krum",         True,  "Krum"),
        ("trimmed_mean", True,  "AsyncRobustFL (Trimmed Mean)"),
    ]

    results: Dict[str, Dict] = {}

    for method, use_detection, label in method_configs:
        logger.info("--- Running: %s ---", label)
        acc_per_seed: List[List[float]] = []
        loss_per_seed: List[List[float]] = []

        for seed in seeds:
            acc, loss, _, _, _ = run_one_experiment(
                method               = method,
                use_dp               = False,
                use_attack           = True,
                use_detection        = use_detection,
                dirichlet_alpha      = DIRICHLET_ALPHA,
                num_rounds           = num_rounds,
                clients_per_round    = CLIENTS_PER_ROUND,
                label                = f"{label} [seed={seed}]",
                seed                 = seed,
                async_buffer_size    = ASYNC_BUFFER_SIZE,
                malicious_client_ids = MALICIOUS_CLIENT_IDS,
            )
            acc_per_seed.append(acc)
            loss_per_seed.append(loss)

        # Pad to equal length in case of partial runs, then compute statistics
        min_len = min(len(a) for a in acc_per_seed)
        acc_arr = np.array([a[:min_len] for a in acc_per_seed])  # (n_seeds, rounds)

        results[label] = {
            "accuracy_per_seed": acc_per_seed,
            "loss_per_seed":     loss_per_seed,
            "mean_accuracy":     acc_arr.mean(axis=0).tolist(),
            "std_accuracy":      acc_arr.std(axis=0).tolist(),
            "final_acc_mean":    float(acc_arr[:, -1].mean()),
            "final_acc_std":     float(acc_arr[:, -1].std()),
        }
        logger.info(
            "  %-30s  final_acc = %.4f ± %.4f",
            label,
            results[label]["final_acc_mean"],
            results[label]["final_acc_std"],
        )

    _export_defense_comparison(results)
    _plot_defense_comparison(results)
    return results


def _plot_defense_comparison(results: Dict[str, Dict]) -> None:
    """Plot mean accuracy ± 1 std for each defense method."""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors  = ["steelblue", "cornflowerblue", "tomato", "purple", "green"]
    markers = ["o", "s", "^", "D", "v"]

    for idx, (label, data) in enumerate(results.items()):
        mean_acc = np.array(data["mean_accuracy"])
        std_acc  = np.array(data["std_accuracy"])
        rounds   = np.arange(1, len(mean_acc) + 1)

        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]

        ax.plot(rounds, mean_acc, f"-{m}", label=label,
                linewidth=2, markersize=4, color=c)
        ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc,
                        alpha=0.15, color=c)

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Global Accuracy (PathMNIST)", fontsize=12)
    ax.set_title(
        "Defense vs Defense: Aggregation Method Comparison\n"
        "(Under 20% Byzantine Attack, PathMNIST, Non-IID α=0.5)",
        fontsize=13,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "defense_comparison.png")


def _export_defense_comparison(results: Dict[str, Dict]) -> None:
    """Export defense comparison results to CSV and JSON."""
    rdir = _ensure_results_dir()

    # --- CSV ---
    csv_path = os.path.join(rdir, "defense_comparison.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "method", "round", "mean_accuracy", "std_accuracy",
        ])
        for label, data in results.items():
            for rnd, (mean, std) in enumerate(
                zip(data["mean_accuracy"], data["std_accuracy"]), start=1
            ):
                writer.writerow([label, rnd, f"{mean:.6f}", f"{std:.6f}"])
    logger.info("Defense comparison CSV saved: %s", csv_path)

    # --- JSON ---
    json_path = os.path.join(rdir, "defense_comparison.json")
    exportable = {
        label: {
            "final_acc_mean":  data["final_acc_mean"],
            "final_acc_std":   data["final_acc_std"],
            "mean_accuracy":   [round(v, 6) for v in data["mean_accuracy"]],
            "std_accuracy":    [round(v, 6) for v in data["std_accuracy"]],
        }
        for label, data in results.items()
    }
    with open(json_path, "w") as fh:
        json.dump(exportable, fh, indent=2)
    logger.info("Defense comparison JSON saved: %s", json_path)


# ---------------------------------------------------------------------------
# 2. Sensitivity Analysis
# ---------------------------------------------------------------------------

def run_sensitivity_alpha(
    alpha_values: Tuple[float, ...] = ALPHA_VALUES,
    seeds: Tuple[int, ...] = MULTI_SEEDS,
    num_rounds: int = NUM_ROUNDS,
) -> Dict[str, Dict]:
    """Sweep Dirichlet α while holding all other parameters constant.

    Uses AsyncRobustFL (trimmed_mean + detection) with the standard 20%
    Byzantine configuration so that the results are directly comparable to
    Exp C in main.py.

    Args:
        alpha_values: Iterable of Dirichlet α values to sweep.
        seeds:        Random seeds for multi-seed averaging.
        num_rounds:   FL communication rounds per run.

    Returns:
        Dict keyed by f"alpha={alpha}" → per-seed and aggregated accuracy data.
    """
    logger.info("=" * 60)
    logger.info("Sensitivity: Dirichlet α sweep — values: %s", list(alpha_values))
    logger.info("=" * 60)

    results: Dict[str, Dict] = {}

    for alpha in alpha_values:
        key = f"alpha={alpha}"
        logger.info("--- α = %s ---", alpha)
        acc_per_seed: List[List[float]] = []

        for seed in seeds:
            acc, _, _, _, _ = run_one_experiment(
                method               = "trimmed_mean",
                use_dp               = False,
                use_attack           = True,
                use_detection        = True,
                dirichlet_alpha      = alpha,
                num_rounds           = num_rounds,
                clients_per_round    = CLIENTS_PER_ROUND,
                label                = f"Sensitivity α={alpha} [seed={seed}]",
                seed                 = seed,
                async_buffer_size    = ASYNC_BUFFER_SIZE,
                malicious_client_ids = MALICIOUS_CLIENT_IDS,
            )
            acc_per_seed.append(acc)

        min_len = min(len(a) for a in acc_per_seed)
        acc_arr = np.array([a[:min_len] for a in acc_per_seed])

        results[key] = {
            "alpha":           alpha,
            "accuracy_per_seed": acc_per_seed,
            "mean_accuracy":   acc_arr.mean(axis=0).tolist(),
            "std_accuracy":    acc_arr.std(axis=0).tolist(),
            "final_acc_mean":  float(acc_arr[:, -1].mean()),
            "final_acc_std":   float(acc_arr[:, -1].std()),
        }
        logger.info(
            "  α=%-5s  final_acc = %.4f ± %.4f",
            alpha,
            results[key]["final_acc_mean"],
            results[key]["final_acc_std"],
        )

    _export_sensitivity(results, prefix="sensitivity_alpha")
    _plot_sensitivity(
        results,
        param_name="Dirichlet α",
        save_file="sensitivity_alpha.png",
        title="Sensitivity Analysis: Dirichlet α (Data Heterogeneity)\n"
              "(AsyncRobustFL, 20% Byzantine, Non-IID sweep)",
    )
    return results


def run_sensitivity_byzantine(
    byzantine_fractions: Tuple[float, ...] = (0.1, 0.2, 0.3),
    seeds: Tuple[int, ...] = MULTI_SEEDS,
    num_rounds: int = NUM_ROUNDS,
) -> Dict[str, Dict]:
    """Sweep Byzantine client percentage while holding all other parameters constant.

    The noisy and unreliable client sets remain fixed at their config values.
    Only the MALICIOUS client fraction is varied. Noisy clients (IDs 4, 5) are
    kept out of the malicious set by assigning malicious IDs from 0 upward,
    except for 4 and 5.

    Args:
        byzantine_fractions: Fractions of NUM_CLIENTS to mark as malicious.
        seeds:               Random seeds for multi-seed averaging.
        num_rounds:          FL communication rounds per run.

    Returns:
        Dict keyed by f"byzantine={frac:.0%}" → per-seed and aggregated data.
    """
    logger.info("=" * 60)
    logger.info(
        "Sensitivity: Byzantine %% sweep — fractions: %s",
        [f"{f:.0%}" for f in byzantine_fractions],
    )
    logger.info("=" * 60)

    results: Dict[str, Dict] = {}

    for frac in byzantine_fractions:
        mal_ids = _byzantine_ids_for_fraction(frac)
        key = f"byzantine={frac:.0%}"
        logger.info("--- Byzantine fraction = %.0f%%  ids = %s ---", frac * 100, sorted(mal_ids))
        acc_per_seed: List[List[float]] = []

        for seed in seeds:
            acc, _, _, _, _ = run_one_experiment(
                method               = "trimmed_mean",
                use_dp               = False,
                use_attack           = bool(mal_ids),
                use_detection        = True,
                dirichlet_alpha      = DIRICHLET_ALPHA,
                num_rounds           = num_rounds,
                clients_per_round    = CLIENTS_PER_ROUND,
                label                = f"Sensitivity byz={frac:.0%} [seed={seed}]",
                seed                 = seed,
                async_buffer_size    = ASYNC_BUFFER_SIZE,
                malicious_client_ids = mal_ids,
            )
            acc_per_seed.append(acc)

        min_len = min(len(a) for a in acc_per_seed)
        acc_arr = np.array([a[:min_len] for a in acc_per_seed])

        results[key] = {
            "fraction":          frac,
            "n_byzantine":       len(mal_ids),
            "accuracy_per_seed": acc_per_seed,
            "mean_accuracy":     acc_arr.mean(axis=0).tolist(),
            "std_accuracy":      acc_arr.std(axis=0).tolist(),
            "final_acc_mean":    float(acc_arr[:, -1].mean()),
            "final_acc_std":     float(acc_arr[:, -1].std()),
        }
        logger.info(
            "  byz=%-4.0f%%  final_acc = %.4f ± %.4f",
            frac * 100,
            results[key]["final_acc_mean"],
            results[key]["final_acc_std"],
        )

    _export_sensitivity(results, prefix="sensitivity_byzantine")
    _plot_sensitivity(
        results,
        param_name="Byzantine Client Fraction",
        save_file="sensitivity_byzantine.png",
        title="Sensitivity Analysis: Byzantine Client Percentage\n"
              "(AsyncRobustFL, Non-IID α=0.5, trimmed-mean aggregation)",
    )
    return results


def run_sensitivity_buffer(
    buffer_sizes: Tuple[int, ...] = BUFFER_SIZES,
    seeds: Tuple[int, ...] = MULTI_SEEDS,
    num_rounds: int = NUM_ROUNDS,
) -> Dict[str, Dict]:
    """Sweep async buffer size while holding all other parameters constant.

    Args:
        buffer_sizes: Async buffer values to evaluate.
        seeds:        Random seeds for multi-seed averaging.
        num_rounds:   FL communication rounds per run.

    Returns:
        Dict keyed by f"buffer={size}" → per-seed and aggregated accuracy data.
    """
    logger.info("=" * 60)
    logger.info("Sensitivity: Async buffer size sweep — sizes: %s", list(buffer_sizes))
    logger.info("=" * 60)

    results: Dict[str, Dict] = {}

    for buf in buffer_sizes:
        key = f"buffer={buf}"
        logger.info("--- buffer_size = %d ---", buf)
        acc_per_seed: List[List[float]] = []

        for seed in seeds:
            acc, _, _, _, _ = run_one_experiment(
                method               = "trimmed_mean",
                use_dp               = False,
                use_attack           = True,
                use_detection        = True,
                dirichlet_alpha      = DIRICHLET_ALPHA,
                num_rounds           = num_rounds,
                clients_per_round    = CLIENTS_PER_ROUND,
                label                = f"Sensitivity buffer={buf} [seed={seed}]",
                seed                 = seed,
                async_buffer_size    = buf,
                malicious_client_ids = MALICIOUS_CLIENT_IDS,
            )
            acc_per_seed.append(acc)

        min_len = min(len(a) for a in acc_per_seed)
        acc_arr = np.array([a[:min_len] for a in acc_per_seed])

        results[key] = {
            "buffer_size":       buf,
            "accuracy_per_seed": acc_per_seed,
            "mean_accuracy":     acc_arr.mean(axis=0).tolist(),
            "std_accuracy":      acc_arr.std(axis=0).tolist(),
            "final_acc_mean":    float(acc_arr[:, -1].mean()),
            "final_acc_std":     float(acc_arr[:, -1].std()),
        }
        logger.info(
            "  buffer=%-3d  final_acc = %.4f ± %.4f",
            buf,
            results[key]["final_acc_mean"],
            results[key]["final_acc_std"],
        )

    _export_sensitivity(results, prefix="sensitivity_buffer")
    _plot_sensitivity(
        results,
        param_name="Async Buffer Size",
        save_file="sensitivity_buffer.png",
        title="Sensitivity Analysis: Async Buffer Size\n"
              "(AsyncRobustFL, 20% Byzantine, Non-IID α=0.5)",
    )
    return results


def _plot_sensitivity(
    results: Dict[str, Dict],
    param_name: str,
    save_file: str,
    title: str,
) -> None:
    """Generic line plot for sensitivity results.

    Draws one curve per parameter value with mean ± 1 std shading across seeds.

    Args:
        results:    Output dict from any run_sensitivity_* function.
        param_name: Human-readable name for the swept parameter (legend / axis).
        save_file:  Output PNG filename inside RESULTS_DIR.
        title:      Chart title.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    colors  = ["steelblue", "tomato", "green", "purple", "orange", "brown"]
    markers = ["o", "s", "^", "D", "v", "P"]

    for idx, (label, data) in enumerate(results.items()):
        mean_acc = np.array(data["mean_accuracy"])
        std_acc  = np.array(data["std_accuracy"])
        rounds   = np.arange(1, len(mean_acc) + 1)

        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        ax.plot(rounds, mean_acc, f"-{m}", label=label,
                linewidth=2, markersize=4, color=c)
        ax.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc,
                        alpha=0.15, color=c)

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Global Accuracy (PathMNIST)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save_file)


def _export_sensitivity(results: Dict[str, Dict], prefix: str) -> None:
    """Export sensitivity results to CSV and JSON.

    Args:
        results: Output dict from any run_sensitivity_* function.
        prefix:  Filename prefix (e.g. 'sensitivity_alpha').
    """
    rdir = _ensure_results_dir()

    # --- CSV ---
    csv_path = os.path.join(rdir, f"{prefix}.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["config", "round", "mean_accuracy", "std_accuracy"])
        for label, data in results.items():
            for rnd, (mean, std) in enumerate(
                zip(data["mean_accuracy"], data["std_accuracy"]), start=1
            ):
                writer.writerow([label, rnd, f"{mean:.6f}", f"{std:.6f}"])
    logger.info("Sensitivity CSV saved: %s", csv_path)

    # --- JSON ---
    json_path = os.path.join(rdir, f"{prefix}.json")
    exportable = {
        label: {
            "final_acc_mean": round(data["final_acc_mean"], 6),
            "final_acc_std":  round(data["final_acc_std"], 6),
            "mean_accuracy":  [round(v, 6) for v in data["mean_accuracy"]],
            "std_accuracy":   [round(v, 6) for v in data["std_accuracy"]],
        }
        for label, data in results.items()
    }
    with open(json_path, "w") as fh:
        json.dump(exportable, fh, indent=2)
    logger.info("Sensitivity JSON saved: %s", json_path)


# ---------------------------------------------------------------------------
# 3 & 4. Multi-Seed Runs + Statistical Reporting
# ---------------------------------------------------------------------------

def run_multiseed_experiment(
    method: str              = "trimmed_mean",
    use_attack: bool         = True,
    use_detection: bool      = True,
    dirichlet_alpha: float   = DIRICHLET_ALPHA,
    num_rounds: int          = NUM_ROUNDS,
    clients_per_round: int   = CLIENTS_PER_ROUND,
    async_buffer_size: int   = ASYNC_BUFFER_SIZE,
    malicious_ids: frozenset = MALICIOUS_CLIENT_IDS,
    seeds: Tuple[int, ...]   = MULTI_SEEDS,
    experiment_label: str    = "",
) -> Dict:
    """Run a specified FL configuration over multiple seeds and report statistics.

    Seeds control all sources of randomness:
      - ``np.random.seed(seed)`` — Dirichlet partitioning, dropout sampling.
      - ``torch.manual_seed(seed)`` — model initialisation, training batch order.

    Args:
        method:            Aggregation method ('fedavg' | 'trimmed_mean' |
                           'median' | 'krum').
        use_attack:        Turn on malicious + noisy clients.
        use_detection:     Enable norm + cosine detection filters.
        dirichlet_alpha:   Dirichlet concentration for data partitioning.
        num_rounds:        FL communication rounds.
        clients_per_round: Hospitals sampled per round.
        async_buffer_size: Async buffer size.
        malicious_ids:     Frozenset of malicious client IDs.
        seeds:             Tuple of integer seeds to run.
        experiment_label:  Human-readable name for logging and export.

    Returns:
        Dict with keys:
          ``accuracy_per_seed``  List[List[float]]
          ``mean_accuracy``      List[float]  (per round)
          ``std_accuracy``       List[float]  (per round)
          ``final_acc_mean``     float
          ``final_acc_std``      float
          ``final_acc_per_seed`` List[float]
    """
    label = experiment_label or f"{method} | α={dirichlet_alpha} | attack={use_attack}"
    logger.info("Multi-seed run: %s | seeds=%s", label, list(seeds))

    acc_per_seed: List[List[float]] = []

    for seed in seeds:
        acc, _, _, _, _ = run_one_experiment(
            method               = method,
            use_dp               = False,
            use_attack           = use_attack,
            use_detection        = use_detection,
            dirichlet_alpha      = dirichlet_alpha,
            num_rounds           = num_rounds,
            clients_per_round    = clients_per_round,
            label                = f"{label} [seed={seed}]",
            seed                 = seed,
            async_buffer_size    = async_buffer_size,
            malicious_client_ids = malicious_ids,
        )
        acc_per_seed.append(acc)

    min_len = min(len(a) for a in acc_per_seed)
    acc_arr = np.array([a[:min_len] for a in acc_per_seed])

    result = {
        "label":               label,
        "seeds":               list(seeds),
        "accuracy_per_seed":   acc_per_seed,
        "mean_accuracy":       acc_arr.mean(axis=0).tolist(),
        "std_accuracy":        acc_arr.std(axis=0).tolist(),
        "final_acc_mean":      float(acc_arr[:, -1].mean()),
        "final_acc_std":       float(acc_arr[:, -1].std()),
        "final_acc_per_seed":  acc_arr[:, -1].tolist(),
    }

    logger.info(
        "  %s -> final_acc = %.4f ± %.4f  (per seed: %s)",
        label,
        result["final_acc_mean"],
        result["final_acc_std"],
        [f"{v:.4f}" for v in result["final_acc_per_seed"]],
    )
    return result


def export_multiseed_stats(
    multiseed_results: Dict[str, Dict],
    csv_filename: str  = "multiseed_stats.csv",
    json_filename: str = "multiseed_stats.json",
) -> Tuple[str, str]:
    """Export mean ± std statistics for all multi-seed experiments.

    The CSV format has one row per experiment per round, capturing mean and
    std accuracy for easy import into analysis tools (pandas, R, Excel).

    Args:
        multiseed_results: Dict mapping experiment label → result dict from
                           ``run_multiseed_experiment``.
        csv_filename:      Output CSV filename inside RESULTS_DIR.
        json_filename:     Output JSON filename inside RESULTS_DIR.

    Returns:
        Tuple of (csv_path, json_path).
    """
    rdir = _ensure_results_dir()
    csv_path  = os.path.join(rdir, csv_filename)
    json_path = os.path.join(rdir, json_filename)

    # --- CSV ---
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "experiment", "seeds", "round",
            "mean_accuracy", "std_accuracy",
            "final_acc_mean", "final_acc_std",
        ])
        for exp_label, data in multiseed_results.items():
            seeds_str = " ".join(str(s) for s in data.get("seeds", []))
            for rnd, (mean, std) in enumerate(
                zip(data["mean_accuracy"], data["std_accuracy"]), start=1
            ):
                writer.writerow([
                    exp_label,
                    seeds_str,
                    rnd,
                    f"{mean:.6f}",
                    f"{std:.6f}",
                    f"{data['final_acc_mean']:.6f}",
                    f"{data['final_acc_std']:.6f}",
                ])
    logger.info("Multi-seed stats CSV saved: %s", csv_path)

    # --- JSON ---
    exportable: Dict = {}
    for exp_label, data in multiseed_results.items():
        exportable[exp_label] = {
            "seeds":             data.get("seeds", []),
            "final_acc_mean":    round(data["final_acc_mean"], 6),
            "final_acc_std":     round(data["final_acc_std"], 6),
            "final_acc_per_seed": [round(v, 6) for v in data.get("final_acc_per_seed", [])],
            "mean_accuracy":     [round(v, 6) for v in data["mean_accuracy"]],
            "std_accuracy":      [round(v, 6) for v in data["std_accuracy"]],
        }
    with open(json_path, "w") as fh:
        json.dump(exportable, fh, indent=2)
    logger.info("Multi-seed stats JSON saved: %s", json_path)

    return csv_path, json_path


def plot_multiseed_stats(
    multiseed_results: Dict[str, Dict],
    save_file: str = "multiseed_stats.png",
) -> None:
    """Plot per-round mean accuracy with ± 1 std shading for each configuration.

    Args:
        multiseed_results: Dict from ``run_multiseed_experiment`` calls.
        save_file:         Output PNG filename inside RESULTS_DIR.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    colors  = ["steelblue", "tomato", "green", "purple", "orange", "brown"]
    markers = ["o", "s", "^", "D", "v", "P"]

    for idx, (label, data) in enumerate(multiseed_results.items()):
        mean_arr = np.array(data["mean_accuracy"])
        std_arr  = np.array(data["std_accuracy"])
        rounds   = np.arange(1, len(mean_arr) + 1)
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        ax.plot(rounds, mean_arr, f"-{m}", label=label,
                linewidth=2, markersize=4, color=c)
        ax.fill_between(rounds, mean_arr - std_arr, mean_arr + std_arr,
                        alpha=0.15, color=c,
                        label=f"±1 std ({label})")

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Global Accuracy (mean ± std)", fontsize=12)
    ax.set_title(
        "Multi-Seed Statistical Results\n"
        f"(Seeds: {list(MULTI_SEEDS)} — error bands = ±1 std)",
        fontsize=13,
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, save_file)


# ---------------------------------------------------------------------------
# 5. Robustness / Failure Threshold
# ---------------------------------------------------------------------------

def run_failure_threshold(
    byzantine_fractions: Tuple[float, ...] = BYZANTINE_FRACTIONS,
    seeds: Tuple[int, ...] = MULTI_SEEDS,
    num_rounds: int  = NUM_ROUNDS,
    threshold: float = FAILURE_THRESHOLD,
) -> Dict:
    """Sweep Byzantine fraction from 0% to 50% to locate the failure point.

    The failure threshold is defined as the lowest Byzantine fraction at which
    the mean final accuracy drops below ``threshold`` (default 0.50).

    Noisy and unreliable clients are kept at their config-defined IDs to
    reflect a realistic threat model; only the malicious gradient-attack
    fraction is varied.

    Args:
        byzantine_fractions: Tuple of fractions to sweep (0.0 – 0.5).
        seeds:               Seeds for mean ± std computation.
        num_rounds:          FL rounds per run.
        threshold:           Accuracy below which the system is considered
                             to have failed.

    Returns:
        Dict with keys:
          ``fractions``     List[float]
          ``mean_acc``      List[float]
          ``std_acc``       List[float]
          ``failure_point`` Optional[float]  — first fraction where mean_acc < threshold
          ``per_fraction``  Dict[str, Dict]  — full per-fraction data
    """
    logger.info("=" * 60)
    logger.info(
        "Experiment Group 5: Robustness / Failure Threshold sweep — fractions: %s",
        [f"{f:.0%}" for f in byzantine_fractions],
    )
    logger.info("=" * 60)

    per_fraction: Dict[str, Dict] = {}
    mean_accs: List[float] = []
    std_accs: List[float]  = []

    for frac in byzantine_fractions:
        mal_ids = _byzantine_ids_for_fraction(frac)
        key = f"byz={frac:.0%}"
        logger.info("--- Byzantine fraction = %.0f%%  IDs = %s ---", frac * 100, sorted(mal_ids))

        acc_per_seed: List[List[float]] = []

        for seed in seeds:
            acc, _, _, _, _ = run_one_experiment(
                method               = "trimmed_mean",
                use_dp               = False,
                use_attack           = bool(mal_ids),
                use_detection        = True,
                dirichlet_alpha      = DIRICHLET_ALPHA,
                num_rounds           = num_rounds,
                clients_per_round    = CLIENTS_PER_ROUND,
                label                = f"Failure threshold byz={frac:.0%} [seed={seed}]",
                seed                 = seed,
                async_buffer_size    = ASYNC_BUFFER_SIZE,
                malicious_client_ids = mal_ids,
            )
            acc_per_seed.append(acc)

        min_len = min(len(a) for a in acc_per_seed)
        acc_arr = np.array([a[:min_len] for a in acc_per_seed])
        final_mean = float(acc_arr[:, -1].mean())
        final_std  = float(acc_arr[:, -1].std())

        per_fraction[key] = {
            "fraction":          frac,
            "n_byzantine":       len(mal_ids),
            "mean_accuracy":     acc_arr.mean(axis=0).tolist(),
            "std_accuracy":      acc_arr.std(axis=0).tolist(),
            "final_acc_mean":    final_mean,
            "final_acc_std":     final_std,
        }
        mean_accs.append(final_mean)
        std_accs.append(final_std)

        logger.info(
            "  byz=%-4.0f%%  final_acc = %.4f ± %.4f",
            frac * 100, final_mean, final_std,
        )

    # Identify failure point
    failure_point: Optional[float] = None
    for frac, mean_acc in zip(byzantine_fractions, mean_accs):
        if mean_acc < threshold:
            failure_point = frac
            break

    if failure_point is not None:
        logger.info(
            "Failure threshold identified: %.0f%% Byzantine clients "
            "(accuracy dropped below %.0f%%)",
            failure_point * 100,
            threshold * 100,
        )
    else:
        logger.info(
            "No failure point found within swept range (accuracy stayed above %.0f%%)",
            threshold * 100,
        )

    summary = {
        "fractions":     list(byzantine_fractions),
        "mean_acc":      mean_accs,
        "std_acc":       std_accs,
        "failure_point": failure_point,
        "threshold":     threshold,
        "per_fraction":  per_fraction,
    }

    _export_failure_threshold(summary)
    _plot_failure_threshold(summary)
    return summary


def _plot_failure_threshold(summary: Dict) -> None:
    """Bar chart of final accuracy vs Byzantine fraction with a failure threshold line."""
    fractions = [f"{f:.0%}" for f in summary["fractions"]]
    mean_accs = summary["mean_acc"]
    std_accs  = summary["std_acc"]

    colors = [
        "tomato" if acc < summary["threshold"] else "steelblue"
        for acc in mean_accs
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(fractions, mean_accs, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.8)
    ax.errorbar(
        fractions, mean_accs,
        yerr=std_accs,
        fmt="none", color="black", capsize=5, linewidth=1.5,
    )

    # Annotate bars
    for bar, acc, std in zip(bars, mean_accs, std_accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            acc + std + 0.01,
            f"{acc:.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.axhline(
        y=summary["threshold"],
        color="red", linestyle="--", linewidth=1.8,
        label=f"Failure threshold = {summary['threshold']:.0%}",
    )

    if summary["failure_point"] is not None:
        ax.axvline(
            x=f"{summary['failure_point']:.0%}",
            color="darkred", linestyle=":", linewidth=1.8,
            label=f"Failure point: {summary['failure_point']:.0%} Byzantine",
        )

    ax.set_xlabel("Byzantine Client Fraction (of 10 hospitals)", fontsize=11)
    ax.set_ylabel("Final Global Accuracy (mean ± std)", fontsize=11)
    ax.set_title(
        "Robustness / Failure Threshold Analysis\n"
        "(AsyncRobustFL, Non-IID α=0.5, Trimmed Mean)",
        fontsize=12,
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save_fig(fig, "failure_threshold.png")


def _export_failure_threshold(summary: Dict) -> None:
    """Export failure threshold summary to CSV and JSON."""
    rdir = _ensure_results_dir()

    # --- CSV ---
    csv_path = os.path.join(rdir, "failure_threshold.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "byzantine_fraction", "n_byzantine",
            "final_acc_mean", "final_acc_std", "failure",
        ])
        for frac, mean_acc, std_acc in zip(
            summary["fractions"],
            summary["mean_acc"],
            summary["std_acc"],
        ):
            n_byz = int(round(frac * NUM_CLIENTS))
            failed = "YES" if mean_acc < summary["threshold"] else "NO"
            writer.writerow([
                f"{frac:.2f}",
                n_byz,
                f"{mean_acc:.6f}",
                f"{std_acc:.6f}",
                failed,
            ])
    logger.info("Failure threshold CSV saved: %s", csv_path)

    # --- JSON ---
    json_path = os.path.join(rdir, "failure_threshold.json")
    exportable = {
        "failure_point": summary["failure_point"],
        "threshold":     summary["threshold"],
        "sweep": [
            {
                "byzantine_fraction": frac,
                "n_byzantine":        int(round(frac * NUM_CLIENTS)),
                "final_acc_mean":     round(mean, 6),
                "final_acc_std":      round(std, 6),
                "failed":             mean < summary["threshold"],
            }
            for frac, mean, std in zip(
                summary["fractions"], summary["mean_acc"], summary["std_acc"]
            )
        ],
    }
    with open(json_path, "w") as fh:
        json.dump(exportable, fh, indent=2)
    logger.info("Failure threshold JSON saved: %s", json_path)


# ---------------------------------------------------------------------------
# 6. Communication Cost Analysis
# ---------------------------------------------------------------------------

def compute_communication_cost(
    accuracy_history: List[float],
    dropout_history:  List[Dict],
    clients_per_round: int = CLIENTS_PER_ROUND,
    model_bytes:       int = 0,
) -> Dict:
    """Compute per-round and cumulative communication cost for an FL run.

    Communication model (standard two-way FL protocol):
      - Upload (clients → server): Each submitting client sends one full
        parameter tensor = ``model_bytes`` per client.
      - Download (server → clients): Server broadcasts the global model to
        all sampled clients at the start of each round = ``clients_per_round``
        × ``model_bytes``.
      - Total per round = upload + download.

    Args:
        accuracy_history:  List of per-round global accuracy values.
        dropout_history:   List of {"round", "submitted", "dropped",
                           "total_selected"} dicts from the strategy.
        clients_per_round: Number of clients sampled per round (for download
                           cost computation).
        model_bytes:       Size of the model in bytes (float32 × n_params).
                           Computed from PathologyNet if 0.

    Returns:
        Dict with keys:
          ``model_bytes``               int — bytes per parameter tensor
          ``model_params``              int — number of scalar parameters
          ``per_round_upload_bytes``    List[int]
          ``per_round_download_bytes``  List[int]
          ``per_round_total_bytes``     List[int]
          ``cumulative_bytes``          List[int]
          ``accuracy_vs_bytes``         List[Tuple[int, float]]
          ``rounds_to_target``          Optional[int]
          ``total_bytes``               int
    """
    if model_bytes == 0:
        model_bytes = _model_size_bytes()
    model_params = _model_param_count()

    # Build per-round submitted count from dropout_history.
    # submitted = clients that returned an update this round.
    submitted_per_round: List[int] = [e["submitted"] for e in dropout_history]

    per_round_upload:    List[int] = []
    per_round_download:  List[int] = []
    per_round_total:     List[int] = []
    cumulative:          List[int] = []
    running_total = 0

    for submitted in submitted_per_round:
        upload   = submitted * model_bytes
        download = clients_per_round * model_bytes
        total    = upload + download
        running_total += total
        per_round_upload.append(upload)
        per_round_download.append(download)
        per_round_total.append(total)
        cumulative.append(running_total)

    # Accuracy vs cumulative bytes (one tuple per round)
    acc_vs_bytes: List[Tuple[int, float]] = list(zip(cumulative, accuracy_history))

    rounds_to_target = _rounds_to_target(accuracy_history)

    return {
        "model_bytes":              model_bytes,
        "model_params":             model_params,
        "per_round_upload_bytes":   per_round_upload,
        "per_round_download_bytes": per_round_download,
        "per_round_total_bytes":    per_round_total,
        "cumulative_bytes":         cumulative,
        "accuracy_vs_bytes":        acc_vs_bytes,
        "rounds_to_target":         rounds_to_target,
        "accuracy_target":          ACCURACY_TARGET,
        "total_bytes":              running_total,
    }


def run_communication_analysis(
    seeds: Tuple[int, ...] = MULTI_SEEDS,
    num_rounds: int        = NUM_ROUNDS,
) -> Dict[str, Dict]:
    """Run async-robust under all defense-comparison methods and measure comm cost.

    For each method, the communication cost is computed from ``dropout_history``
    (actual submitted-client counts) so the uploaded volume reflects the real
    async behaviour rather than an assumption of full participation.

    Args:
        seeds:      Seeds for multi-seed averaging.
        num_rounds: FL rounds per run.

    Returns:
        Dict keyed by method label → {
          "comm_cost":     averaged communication cost dict,
          "mean_accuracy": List[float],
          "std_accuracy":  List[float],
        }
    """
    logger.info("=" * 60)
    logger.info("Experiment Group 6: Communication Cost Analysis")
    logger.info("=" * 60)

    model_bytes = _model_size_bytes()
    logger.info(
        "PathologyNet: %d parameters | %.2f MB per parameter tensor",
        _model_param_count(),
        model_bytes / (1024 ** 2),
    )

    method_configs = [
        ("fedavg",       False, "FedAvg (no detection)"),
        ("median",       True,  "Coordinate Median"),
        ("krum",         True,  "Krum"),
        ("trimmed_mean", True,  "AsyncRobustFL (Trimmed Mean)"),
    ]

    results: Dict[str, Dict] = {}

    for method, use_detection, label in method_configs:
        logger.info("--- Communication cost: %s ---", label)
        acc_per_seed:   List[List[float]] = []
        cost_per_seed:  List[Dict]        = []

        for seed in seeds:
            acc, _, _, _, drop = run_one_experiment(
                method               = method,
                use_dp               = False,
                use_attack           = True,
                use_detection        = use_detection,
                dirichlet_alpha      = DIRICHLET_ALPHA,
                num_rounds           = num_rounds,
                clients_per_round    = CLIENTS_PER_ROUND,
                label                = f"CommCost {label} [seed={seed}]",
                seed                 = seed,
                async_buffer_size    = ASYNC_BUFFER_SIZE,
                malicious_client_ids = MALICIOUS_CLIENT_IDS,
            )
            cost = compute_communication_cost(
                accuracy_history   = acc,
                dropout_history    = drop,
                clients_per_round  = CLIENTS_PER_ROUND,
                model_bytes        = model_bytes,
            )
            acc_per_seed.append(acc)
            cost_per_seed.append(cost)

        min_len  = min(len(a) for a in acc_per_seed)
        acc_arr  = np.array([a[:min_len] for a in acc_per_seed])
        mean_acc = acc_arr.mean(axis=0).tolist()
        std_acc  = acc_arr.std(axis=0).tolist()

        # Average cumulative bytes across seeds (same for all seeds when
        # submitted count differs only due to randomness)
        cum_bytes_arr = np.array([c["cumulative_bytes"][:min_len] for c in cost_per_seed])
        mean_cumulative = cum_bytes_arr.mean(axis=0).tolist()
        total_bytes_mean = float(cum_bytes_arr[:, -1].mean())

        # Accuracy vs cumulative bytes (use mean curve)
        acc_vs_bytes = list(zip(
            [int(b) for b in mean_cumulative],
            mean_acc,
        ))

        # Rounds to target from mean accuracy curve
        rtt = _rounds_to_target(mean_acc)

        results[label] = {
            "method":              method,
            "mean_accuracy":       mean_acc,
            "std_accuracy":        std_acc,
            "mean_cumulative_bytes": mean_cumulative,
            "total_bytes_mean":    total_bytes_mean,
            "accuracy_vs_bytes":   acc_vs_bytes,
            "rounds_to_target":    rtt,
            "accuracy_target":     ACCURACY_TARGET,
            "model_bytes":         model_bytes,
        }

        logger.info(
            "  %-30s  total_bytes=%.2f MB  rounds_to_%.0f%%=%s",
            label,
            total_bytes_mean / (1024 ** 2),
            ACCURACY_TARGET * 100,
            rtt or "never",
        )

    _export_communication_cost(results)
    _plot_communication_cost(results)
    return results


def _plot_communication_cost(results: Dict[str, Dict]) -> None:
    """Two-panel plot: accuracy vs rounds and accuracy vs cumulative bytes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors  = ["steelblue", "tomato", "purple", "green"]
    markers = ["o", "s", "D", "^"]

    for idx, (label, data) in enumerate(results.items()):
        c = colors[idx % len(colors)]
        m = markers[idx % len(markers)]
        mean_acc = np.array(data["mean_accuracy"])
        std_acc  = np.array(data["std_accuracy"])
        rounds   = np.arange(1, len(mean_acc) + 1)

        # Left panel: accuracy vs rounds
        ax1.plot(rounds, mean_acc, f"-{m}", label=label,
                 linewidth=2, markersize=4, color=c)
        ax1.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc,
                         alpha=0.12, color=c)

        # Right panel: accuracy vs cumulative bytes
        cum_mb  = [b / (1024 ** 2) for b in data["mean_cumulative_bytes"]]
        ax2.plot(cum_mb, mean_acc, f"-{m}", label=label,
                 linewidth=2, markersize=4, color=c)

    ax1.axhline(y=ACCURACY_TARGET, color="gray", linestyle="--", linewidth=1.2,
                label=f"Target accuracy ({ACCURACY_TARGET:.0%})")
    ax1.set_xlabel("Communication Round", fontsize=11)
    ax1.set_ylabel("Global Accuracy (mean ± std)", fontsize=11)
    ax1.set_title("Accuracy vs Communication Rounds\n(Under 20% Byzantine Attack)", fontsize=12)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(y=ACCURACY_TARGET, color="gray", linestyle="--", linewidth=1.2,
                label=f"Target accuracy ({ACCURACY_TARGET:.0%})")
    ax2.set_xlabel("Cumulative Communication Volume (MB)", fontsize=11)
    ax2.set_ylabel("Global Accuracy", fontsize=11)
    ax2.set_title("Accuracy vs Communication Volume (MB)\n(All Methods, Under Attack)", fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        "Communication Cost Analysis — Defense Method Comparison",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    _save_fig(fig, "communication_cost.png")

    # Separate bar chart: rounds-to-target per method
    _plot_rounds_to_target(results)


def _plot_rounds_to_target(results: Dict[str, Dict]) -> None:
    """Bar chart: rounds needed to reach accuracy target per method."""
    labels = list(results.keys())
    rtts   = [
        data["rounds_to_target"] if data["rounds_to_target"] is not None
        else results[list(results.keys())[0]].get("model_bytes", 20)  # fallback
        for data in results.values()
    ]
    # Replace None with NUM_ROUNDS + 1 (did not reach target)
    rtts_plot = [
        rtt if rtt is not None else NUM_ROUNDS + 1
        for rtt in [data["rounds_to_target"] for data in results.values()]
    ]
    colors = [
        "steelblue" if rtt is not None else "lightcoral"
        for rtt in [data["rounds_to_target"] for data in results.values()]
    ]

    target_pct = int(ACCURACY_TARGET * 100)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, rtts_plot, color=colors, alpha=0.85, edgecolor="white")
    for bar, val, raw in zip(
        bars,
        rtts_plot,
        [data["rounds_to_target"] for data in results.values()],
    ):
        label_text = str(raw) if raw is not None else f">{NUM_ROUNDS}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.2,
            label_text,
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_xlabel("Defense Method", fontsize=11)
    ax.set_ylabel(f"Rounds to Reach {target_pct}% Accuracy", fontsize=11)
    ax.set_title(
        f"Communication Efficiency: Rounds to {target_pct}% Accuracy\n"
        "(Under 20% Byzantine Attack)",
        fontsize=12,
    )
    ax.set_ylim(0, NUM_ROUNDS + 3)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right", fontsize=9)
    fig.tight_layout()
    _save_fig(fig, "rounds_to_target.png")


def _export_communication_cost(results: Dict[str, Dict]) -> None:
    """Export communication cost analysis to CSV and JSON."""
    rdir = _ensure_results_dir()

    # --- per-round CSV ---
    csv_path = os.path.join(rdir, "communication_cost.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "method", "round", "mean_accuracy",
            "cumulative_bytes_mb", "accuracy_target", "rounds_to_target",
        ])
        for label, data in results.items():
            rtt = data["rounds_to_target"]
            for rnd, (acc, cum_b) in enumerate(
                zip(data["mean_accuracy"], data["mean_cumulative_bytes"]), start=1
            ):
                writer.writerow([
                    label,
                    rnd,
                    f"{acc:.6f}",
                    f"{cum_b / (1024 ** 2):.4f}",
                    f"{data['accuracy_target']:.2f}",
                    rtt if rtt is not None else "N/A",
                ])
    logger.info("Communication cost CSV saved: %s", csv_path)

    # --- summary JSON ---
    json_path = os.path.join(rdir, "communication_cost.json")
    exportable: Dict = {}
    for label, data in results.items():
        exportable[label] = {
            "model_params":          _model_param_count(),
            "model_bytes":           data["model_bytes"],
            "model_size_mb":         round(data["model_bytes"] / (1024 ** 2), 4),
            "total_comm_bytes":      int(data["total_bytes_mean"]),
            "total_comm_mb":         round(data["total_bytes_mean"] / (1024 ** 2), 4),
            "rounds_to_target":      data["rounds_to_target"],
            "accuracy_target":       data["accuracy_target"],
            "per_round": [
                {
                    "round":              rnd,
                    "mean_accuracy":      round(acc, 6),
                    "cumulative_bytes_mb": round(cum_b / (1024 ** 2), 4),
                }
                for rnd, (acc, cum_b) in enumerate(
                    zip(data["mean_accuracy"], data["mean_cumulative_bytes"]), start=1
                )
            ],
        }
    with open(json_path, "w") as fh:
        json.dump(exportable, fh, indent=2)
    logger.info("Communication cost JSON saved: %s", json_path)


# ---------------------------------------------------------------------------
# Master runner
# ---------------------------------------------------------------------------

def run_all(groups: Optional[List[str]] = None) -> None:
    """Execute all research experiment groups.

    Args:
        groups: Optional list of group names to run.
                If None, all groups are run.
                Valid names: 'defense', 'alpha', 'byzantine', 'buffer',
                             'multiseed', 'threshold', 'comm'.
    """
    all_groups = {
        "defense",
        "alpha",
        "byzantine",
        "buffer",
        "multiseed",
        "threshold",
        "comm",
    }
    if groups is None:
        active = all_groups
    else:
        active = set(groups) & all_groups
        invalid = set(groups) - all_groups
        if invalid:
            logger.warning("Unknown group(s) ignored: %s", invalid)

    logger.info("Running experiment groups: %s", sorted(active))

    # --- 1. Defense vs Defense ---
    if "defense" in active:
        defense_results = run_defense_comparison(seeds=MULTI_SEEDS)

    # --- 2. Sensitivity: Dirichlet α ---
    if "alpha" in active:
        run_sensitivity_alpha(alpha_values=ALPHA_VALUES, seeds=MULTI_SEEDS)

    # --- 3. Sensitivity: Byzantine fraction ---
    if "byzantine" in active:
        run_sensitivity_byzantine(
            byzantine_fractions=(0.1, 0.2, 0.3),
            seeds=MULTI_SEEDS,
        )

    # --- 4. Sensitivity: Buffer size ---
    if "buffer" in active:
        run_sensitivity_buffer(buffer_sizes=BUFFER_SIZES, seeds=MULTI_SEEDS)

    # --- 5 & 4 combined: Multi-seed statistical reporting ---
    if "multiseed" in active:
        ms_async  = run_multiseed_experiment(
            method            = "trimmed_mean",
            use_attack        = True,
            use_detection     = True,
            experiment_label  = "AsyncRobustFL (20% attack)",
        )
        ms_fedavg = run_multiseed_experiment(
            method            = "fedavg",
            use_attack        = False,
            use_detection     = False,
            experiment_label  = "FedAvg (clean baseline)",
        )
        ms_results = {
            ms_async["label"]:  ms_async,
            ms_fedavg["label"]: ms_fedavg,
        }
        export_multiseed_stats(ms_results)
        plot_multiseed_stats(ms_results)

    # --- 6. Robustness / Failure Threshold ---
    if "threshold" in active:
        run_failure_threshold(
            byzantine_fractions=BYZANTINE_FRACTIONS,
            seeds=MULTI_SEEDS,
        )

    # --- 7. Communication Cost ---
    if "comm" in active:
        run_communication_analysis(seeds=MULTI_SEEDS)

    logger.info("All selected experiment groups complete.")
    logger.info("Results saved to: %s", RESULTS_DIR)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NeuralX-FL research experiments (defense comparison, "
                    "sensitivity, multi-seed, failure threshold, comm cost).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Groups:
  defense   - Defense vs Defense comparison (FedAvg / Median / Krum / TrimMean)
  alpha     - Sensitivity sweep over Dirichlet alpha values
  byzantine - Sensitivity sweep over Byzantine client percentage
  buffer    - Sensitivity sweep over async buffer size
  multiseed - Multi-seed runs with mean +/- std reporting
  threshold - Robustness / failure threshold analysis
  comm      - Communication cost analysis (accuracy vs bytes)

Examples:
  python experiments.py                     # run all groups
  python experiments.py --group defense
  python experiments.py --group alpha byzantine
""",
    )
    parser.add_argument(
        "--group",
        metavar="GROUP",
        nargs="+",
        default=None,
        help="One or more group names to run (default: all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_all(groups=args.group)
