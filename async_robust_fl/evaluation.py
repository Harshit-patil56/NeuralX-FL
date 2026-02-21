"""
evaluation.py — Server evaluation factory + all 6 required plots + summary.

All plotting functions:
  - Save to RESULTS_DIR (configurable in config.py)
  - Return the matplotlib Figure so callers can embed or close it
  - Work with the lists collected by make_evaluate_fn and the strategy

References to mandatory features:
  Mandatory #4 (Privacy)    → plot_dp_tradeoff
  Mandatory #5 (Evaluation) → plot_convergence, plot_attack_impact,
                               plot_dropout_reliability, plot_detection
  Mandatory #3 (Detection)  → plot_detection, compute_attack_metrics
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for simulation runs
import matplotlib.pyplot as plt

from model import PathologyNet, set_weights, evaluate_model
from config import NUM_CLIENTS, RESULTS_DIR


# ---------------------------------------------------------------------------
# Ensure results directory exists
# ---------------------------------------------------------------------------

def _ensure_results_dir() -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def _save(fig: plt.Figure, filename: str) -> str:
    path = os.path.join(_ensure_results_dir(), filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Server-side evaluation factory
# ---------------------------------------------------------------------------

def make_evaluate_fn(global_testloader) -> Callable:
    """Return a Flower-compatible evaluate_fn that logs accuracy per round.

    The returned callable is stateful: it accumulates accuracy_history and
    loss_history as lists.  These are attached as attributes so main.py
    can retrieve them after run_simulation completes.

    Usage:
        eval_fn = make_evaluate_fn(load_global_test())
        strategy = AsyncRobustFLStrategy(evaluate_fn=eval_fn, ...)
        # after run_simulation():
        accuracies = eval_fn.accuracy_history
    """
    accuracy_history: List[float] = []
    loss_history:     List[float] = []

    def evaluate_fn(
        server_round: int,
        parameters: List[np.ndarray],
        config: Dict,
    ) -> Tuple[float, Dict]:
        net = PathologyNet()
        set_weights(net, parameters)
        accuracy, loss = evaluate_model(net, global_testloader)
        accuracy_history.append(accuracy)
        loss_history.append(loss)
        print(
            f"[Round {server_round:02d}]  "
            f"PathMNIST Accuracy: {accuracy:.4f}  |  Loss: {loss:.4f}"
        )
        return float(loss), {"accuracy": float(accuracy), "round": server_round}

    evaluate_fn.accuracy_history = accuracy_history
    evaluate_fn.loss_history     = loss_history
    return evaluate_fn


# ---------------------------------------------------------------------------
# Plot 1 — Convergence comparison (Experiments A–D on one graph)
# ---------------------------------------------------------------------------

def plot_convergence(
    results_dict: Dict[str, List[float]],
    save_path: str = "convergence.png",
) -> plt.Figure:
    """Line chart of per-round global accuracy for multiple experiments.

    Args:
        results_dict: {label: accuracy_history}  (e.g. 'Exp A: FedAvg clean')
        save_path   : filename inside RESULTS_DIR.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    markers = ["o", "s", "^", "D", "v", "P"]
    colors  = ["steelblue", "tomato", "green", "purple", "orange", "brown"]

    for idx, (label, accuracies) in enumerate(results_dict.items()):
        rounds = range(1, len(accuracies) + 1)
        ax.plot(
            rounds, accuracies,
            f"-{markers[idx % len(markers)]}",
            label     = label,
            linewidth = 2,
            markersize= 4,
            color     = colors[idx % len(colors)],
        )

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Global Accuracy (PathMNIST)", fontsize=12)
    ax.set_title("Convergence: AsyncRobustFL vs Baselines\n(10 Hospitals, PathMNIST)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Plot 2 — DP trade-off (Experiment C vs D)  [Mandatory #4]
# ---------------------------------------------------------------------------

def plot_dp_tradeoff(
    acc_no_dp:   List[float],
    acc_with_dp: List[float],
    epsilon:     float,
    save_path:   str = "dp_tradeoff.png",
) -> plt.Figure:
    """Overlay Exp C (no DP) and Exp D (with DP) accuracy curves.

    The visual gap demonstrates the privacy-utility trade-off.
    Epsilon is shown in the title so the privacy budget is immediately visible.
    """
    rounds = range(1, len(acc_no_dp) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, acc_no_dp,   "-o", label="AsyncRobust (no DP)",  linewidth=2, color="steelblue")
    ax.plot(rounds, acc_with_dp, "-s", label=f"AsyncRobust + DP (ε≈{epsilon:.1f})",
            linewidth=2, linestyle="--", color="darkorange")
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Global Accuracy", fontsize=12)
    ax.set_title(
        f"Privacy-Utility Trade-off: DP vs No-DP\n"
        f"(ε ≈ {epsilon:.2f}, δ = 1e-5  —  higher ε = less privacy)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Plot 3 — Attack impact (Exp A vs B vs C)
# ---------------------------------------------------------------------------

def plot_attack_impact(
    acc_clean:    List[float],
    acc_attacked: List[float],
    acc_defended: List[float],
    save_path:    str = "attack_impact.png",
) -> plt.Figure:
    """Show attack damage (A→B gap) and defence recovery (B→C gap).

    Shaded regions make the magnitudes immediately legible to reviewers.
    """
    rounds = list(range(1, len(acc_clean) + 1))
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(rounds, acc_clean,    "-o", label="FedAvg, no attack (Exp A)",   linewidth=2, color="steelblue")
    ax.plot(rounds, acc_attacked, "-s", label="FedAvg, under attack (Exp B)", linewidth=2, color="tomato")
    ax.plot(rounds, acc_defended, "-^", label="AsyncRobust defense (Exp C)",  linewidth=2, color="green")

    # Shade the attack damage region (between clean and attacked)
    ax.fill_between(rounds, acc_attacked, acc_clean,
                    alpha=0.15, color="red",   label="Attack damage")
    # Shade the defence recovery region (between attacked and defended)
    ax.fill_between(rounds, acc_attacked, acc_defended,
                    alpha=0.15, color="green", label="Defense recovery")

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Global Accuracy", fontsize=12)
    ax.set_title("Attack Success vs Defense Effectiveness\n(10 Hospitals, PathMNIST)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


def compute_attack_metrics(
    acc_clean:    List[float],
    acc_attacked: List[float],
    acc_defended: List[float],
) -> Dict[str, float]:
    """Compute scalar attack / defence metrics for the summary table."""
    attack_damage    = [c - a for c, a in zip(acc_clean, acc_attacked)]
    defense_recovery = [d - a for d, a in zip(acc_defended, acc_attacked)]
    return {
        "avg_attack_damage":    float(np.mean(attack_damage)),
        "max_attack_damage":    float(np.max(attack_damage)),
        "avg_defense_recovery": float(np.mean(defense_recovery)),
        "final_clean_acc":      float(acc_clean[-1]),
        "final_attacked_acc":   float(acc_attacked[-1]),
        "final_defended_acc":   float(acc_defended[-1]),
    }


# ---------------------------------------------------------------------------
# Plot 4 — Dropout reliability (async tolerance to unreliable hospitals)
# ---------------------------------------------------------------------------

def plot_dropout_reliability(
    dropout_history: List[Dict],
    save_path: str = "dropout_reliability.png",
) -> plt.Figure:
    """Stacked bar: submitted vs dropped per round.

    Demonstrates that the async system aggregates successfully even when
    hospitals drop out, satisfying 'handles unreliable clients'.
    """
    rounds    = [e["round"]     for e in dropout_history]
    submitted = [e["submitted"] for e in dropout_history]
    dropped   = [e["dropped"]   for e in dropout_history]

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(rounds, submitted, label="Submitted updates",  color="steelblue", alpha=0.85)
    ax.bar(rounds, dropped,   label="Dropped / timed out", color="tomato",    alpha=0.85,
           bottom=submitted)

    ax.axhline(y=4, color="black", linestyle="--", linewidth=1.2,
               label="Async buffer threshold (4)")

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Hospital Count", fontsize=12)
    ax.set_title("Per-Round Hospital Participation: Submitted vs Dropped", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Plot 5 — Detection rate (true positives vs false positives per round)
# ---------------------------------------------------------------------------

def plot_detection(
    flagged_history: List[Dict],
    all_bad_ids:     frozenset,
    save_path:       str = "detection_rate.png",
) -> plt.Figure:
    """Two-panel: true detection rate and false positive rate per round.

    all_bad_ids = MALICIOUS_CLIENT_IDS | NOISY_CLIENT_IDS
    """
    rounds              = [r["round"] for r in flagged_history]
    detection_rates:     List[float] = []
    false_positive_rates: List[float] = []

    honest_count = NUM_CLIENTS - len(all_bad_ids)

    for entry in flagged_history:
        flagged_set = set(entry["flagged"])
        tp  = len(flagged_set & all_bad_ids)
        fp  = len(flagged_set - all_bad_ids)
        dr  = tp / len(all_bad_ids) if all_bad_ids else 0.0
        fpr = fp / honest_count     if honest_count > 0 else 0.0
        detection_rates.append(dr)
        false_positive_rates.append(fpr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.bar(rounds, detection_rates, color="green", alpha=0.75)
    ax1.set_ylabel("True Detection Rate", fontsize=11)
    ax1.set_title(
        "Malicious + Noisy Client Detection per Round\n"
        f"(Target IDs: {sorted(all_bad_ids)})",
        fontsize=12,
    )
    ax1.set_ylim(0, 1.15)
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    ax2.bar(rounds, false_positive_rates, color="tomato", alpha=0.75)
    ax2.set_ylabel("False Positive Rate", fontsize=11)
    ax2.set_xlabel("Round", fontsize=11)
    ax2.set_ylim(0, 1.15)

    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Plot 6 — Heterogeneity isolation (Experiment E: IID vs non-IID)
# ---------------------------------------------------------------------------

def plot_heterogeneity(
    acc_non_iid: List[float],
    acc_iid:     List[float],
    save_path:   str = "heterogeneity.png",
) -> plt.Figure:
    """Compare same AsyncRobust system under IID vs non-IID data distribution.

    ONLY dirichlet_alpha differs — all other settings are identical.
    This isolates heterogeneity as the single controlled variable.
    """
    rounds = range(1, len(acc_non_iid) + 1)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, acc_non_iid, "-o", label="Non-IID (α=0.5)",   linewidth=2, color="steelblue")
    ax.plot(rounds, acc_iid,     "-s", label="IID    (α=1000.0)", linewidth=2, color="green")
    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Global Accuracy", fontsize=12)
    ax.set_title(
        "Data Heterogeneity Impact: Non-IID vs IID\n"
        "(Exp E — all other settings identical)",
        fontsize=13,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Privacy budget estimation
# ---------------------------------------------------------------------------

def estimate_epsilon(
    noise_multiplier: float,
    sampling_rate:    float,
    num_rounds:       int,
    delta:            float = 1e-5,
) -> float:
    """Simplified Gaussian mechanism privacy budget estimate.

    Sufficient for demonstration purposes; not a tight RDP bound.
    For production use, apply the google/dp-accounting library.
    """
    sigma = noise_multiplier
    q     = sampling_rate
    T     = num_rounds
    epsilon = (q * np.sqrt(T * 2.0 * np.log(1.25 / delta))) / sigma
    return float(epsilon)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(
    experiment_name: str,
    accuracy_history: List[float],
    flagged_history:  List[Dict],
    dropout_history:  List[Dict],
    malicious_ids:    frozenset,
    noisy_ids:        frozenset,
) -> None:
    """Print a structured experiment summary to stdout."""
    all_bad_ids   = malicious_ids | noisy_ids
    final_acc     = accuracy_history[-1] if accuracy_history else 0.0

    convergence_round = next(
        (i + 1 for i, a in enumerate(accuracy_history) if a >= 0.90),
        None,
    )

    total_flagged  = sum(len(e["flagged"]) for e in flagged_history)
    true_positives = sum(
        len(set(e["flagged"]) & all_bad_ids) for e in flagged_history
    )
    possible_tp    = len(all_bad_ids) * len(flagged_history)
    detection_rate = true_positives / (possible_tp + 1e-10)

    total_dropped  = sum(e["dropped"]        for e in dropout_history)
    total_selected = sum(e["total_selected"] for e in dropout_history)
    dropout_rate   = total_dropped / (total_selected + 1e-10)

    print(f"\n{'=' * 55}")
    print(f"  Experiment : {experiment_name}")
    print(f"{'=' * 55}")
    print(f"  Final Accuracy       : {final_acc:.4f}")
    print(f"  Convergence Round    : {convergence_round or 'Did not reach 90%'}")
    print(f"  Total Flagged Events : {total_flagged}")
    print(f"  Avg Detection Rate   : {detection_rate:.2%}")
    print(f"  Client Dropout Rate  : {dropout_rate:.2%}  ({total_dropped}/{total_selected})")
    print(f"{'=' * 55}\n")
