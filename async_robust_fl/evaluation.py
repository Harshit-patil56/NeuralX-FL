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
  Research extension        → export_metrics_csv, export_metrics_json,
                               plot_participation_rate, plot_centralized_vs_fl
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for simulation runs
import matplotlib.pyplot as plt

from model import PathologyNet, set_weights, evaluate_model
from config import NUM_CLIENTS, RESULTS_DIR, ASYNC_BUFFER_SIZE

logger = logging.getLogger(__name__)


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
# Server-side evaluation — callable class
# ---------------------------------------------------------------------------

class EvaluateFn:
    """Stateful, callable server evaluation function.

    Flower's ``evaluate_fn`` interface requires a plain callable; using a class
    satisfies that interface while keeping ``accuracy_history`` and
    ``loss_history`` as proper instance attributes instead of attaching mutable
    lists to a function object (which is not type-safe or picklable).

    Usage::
        eval_fn = make_evaluate_fn(load_global_test())
        strategy = AsyncRobustFLStrategy(evaluate_fn=eval_fn, ...)
        # after run_simulation():
        accuracies = eval_fn.accuracy_history
    """

    def __init__(self, global_testloader) -> None:
        self._testloader      = global_testloader
        self.accuracy_history: List[float] = []
        self.loss_history:     List[float] = []
        self.round_times:      List[float] = []   # wall-clock seconds per round
        self._round_start:     float       = time.perf_counter()

    def __call__(
        self,
        server_round: int,
        parameters:   List[np.ndarray],
        config:       Dict,
    ) -> Tuple[float, Dict]:
        round_time = time.perf_counter() - self._round_start
        self._round_start = time.perf_counter()   # reset for next round

        net = PathologyNet()
        set_weights(net, parameters)
        accuracy, loss = evaluate_model(net, self._testloader)
        self.accuracy_history.append(accuracy)
        self.loss_history.append(loss)
        self.round_times.append(round_time)
        logger.info(
            "[Round %02d]  PathMNIST Accuracy: %.4f  |  Loss: %.4f  |  Round time: %.1fs",
            server_round, accuracy, loss, round_time,
        )
        return float(loss), {"accuracy": float(accuracy), "round": server_round}


def make_evaluate_fn(global_testloader) -> EvaluateFn:
    """Return an EvaluateFn instance configured for server-side evaluation."""
    return EvaluateFn(global_testloader)


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
# Plot 1b — Loss (cost) curves (Experiments A–D on one graph)
# ---------------------------------------------------------------------------

def plot_loss_curves(
    results_dict: Dict[str, List[float]],
    save_path: str = "loss_curves.png",
) -> plt.Figure:
    """Line chart of per-round global cross-entropy loss for multiple experiments.

    Companion plot to plot_convergence: shows the cost (loss) trajectory so
    reviewers can see both accuracy improvement *and* loss reduction in one
    glance.

    Args:
        results_dict: {label: loss_history}  (e.g. 'Exp A: FedAvg clean')
        save_path   : filename inside RESULTS_DIR.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    markers = ["o", "s", "^", "D", "v", "P"]
    colors  = ["steelblue", "tomato", "green", "purple", "orange", "brown"]

    for idx, (label, losses) in enumerate(results_dict.items()):
        rounds = range(1, len(losses) + 1)
        ax.plot(
            rounds, losses,
            marker    = markers[idx % len(markers)],
            linestyle = "-",
            label     = label,
            linewidth = 2,
            markersize= 4,
            color     = colors[idx % len(colors)],
        )

    ax.set_xlabel("Communication Round", fontsize=12)
    ax.set_ylabel("Global Cross-Entropy Loss (PathMNIST)", fontsize=12)
    ax.set_title(
        "Training Cost (Loss) Curves: AsyncRobustFL vs Baselines\n"
        "(10 Hospitals, PathMNIST — lower is better)",
        fontsize=13,
    )
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
    ax.plot(rounds, acc_no_dp,   marker="o", label="AsyncRobust (no DP)",  linewidth=2, linestyle="-",  color="steelblue")
    ax.plot(rounds, acc_with_dp, marker="s", label=f"AsyncRobust + DP (ε≈{epsilon:.1f})",
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

    ax.axhline(y=ASYNC_BUFFER_SIZE, color="black", linestyle="--", linewidth=1.2,
               label=f"Async buffer threshold ({ASYNC_BUFFER_SIZE})")

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
    malicious_ids:   frozenset,
    noisy_ids:       frozenset = frozenset(),
    save_path:       str       = "detection_rate.png",
) -> plt.Figure:
    """Two-panel chart: Byzantine detection rate and false positive rate per round.

    Detection rate is computed **only for malicious clients** (gradient attacks)
    because the norm/cosine filters are designed to catch gradient corruption,
    not label-noise corruption.  Noisy clients (who corrupt training labels but
    may still produce plausible gradient directions) are shown separately with
    a distinct colour so their incidental flagging is visible but is not folded
    into the primary detection metric.

    Args:
        flagged_history : List of {"round": int, "flagged": List[int]} dicts
                          collected by the strategy.
        malicious_ids   : Client IDs known to submit adversarial gradient updates.
        noisy_ids       : Client IDs with label noise (not gradient attacks).
        save_path       : Filename inside RESULTS_DIR.
    """
    rounds:               List[int]   = [r["round"]   for r in flagged_history]
    detection_rates:      List[float] = []
    false_positive_rates: List[float] = []
    noisy_detect_rates:   List[float] = []

    honest_count = NUM_CLIENTS - len(malicious_ids) - len(noisy_ids)

    for entry in flagged_history:
        flagged_set = set(entry["flagged"])
        tp_mal  = len(flagged_set & malicious_ids)
        tp_noisy = len(flagged_set & noisy_ids)
        fp      = len(flagged_set - malicious_ids - noisy_ids)

        dr   = tp_mal   / len(malicious_ids) if malicious_ids else 0.0
        ndr  = tp_noisy / len(noisy_ids)     if noisy_ids     else 0.0
        fpr  = fp       / honest_count       if honest_count  > 0 else 0.0

        detection_rates.append(dr)
        noisy_detect_rates.append(ndr)
        false_positive_rates.append(fpr)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.bar(rounds, detection_rates,   color="green",     alpha=0.80,
            label=f"Malicious clients {sorted(malicious_ids)}")
    if noisy_ids:
        ax1.bar(rounds, noisy_detect_rates, color="goldenrod", alpha=0.65,
                label=f"Noisy clients {sorted(noisy_ids)} (incidental)")
    ax1.set_ylabel("Detection Rate", fontsize=11)
    ax1.set_title(
        "Byzantine Detection per Round\n"
        f"(Gradient attacks: IDs {sorted(malicious_ids)}  |"
        f"  Label-noise: IDs {sorted(noisy_ids)})",
        fontsize=12,
    )
    ax1.set_ylim(0, 1.15)
    ax1.axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax1.legend(fontsize=9)

    ax2.bar(rounds, false_positive_rates, color="tomato", alpha=0.75)
    ax2.set_ylabel("False Positive Rate\n(Honest Hospitals Flagged)", fontsize=11)
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
    """Compute (\u03b5, \u03b4)-DP budget via R\u00e9nyi DP composition of the subsampled
    Gaussian mechanism, optimised over R\u00e9nyi orders \u03b1 \u2208 [2, 512].

    Based on:
      - Mironov (2017) “R\u00e9nyi Differential Privacy of the Gaussian Mechanism”
      - Wang et al. (2019) “Subsampled R\u00e9nyi DP and Analytical Moments Accountant”

    The per-step RDP of the Poisson-subsampled Gaussian mechanism
    (leading-order in q for small sampling rate q) is::

        \u03b5_RDP(\u03b1) = log(1 + \u03b1(\u03b1-1)q\u00b2 / (2\u03c3\u00b2)) / (\u03b1 - 1)

    After T rounds of composition:  \u03b5_RDP_total(\u03b1) = T \u00d7 \u03b5_RDP(\u03b1)

    Conversion to (\u03b5, \u03b4)-DP (Proposition 3, Balle et al. 2020 / standard RDP)::

        \u03b5(\u03b4) = \u03b5_RDP_total(\u03b1) + log(1/\u03b4) / (\u03b1 - 1)

    The function minimises over \u03b1 to return the tightest achievable \u03b5.
    This bound is tighter than the classical strong composition theorem and
    does not require any external DP-accounting library.

    Args:
        noise_multiplier : \u03c3 (noise standard deviation relative to clipping norm).
        sampling_rate    : q = clients_per_round / total_clients.
        num_rounds       : T (number of FL rounds with DP applied).
        delta            : \u03b4 failure probability; typically 1/n where n = dataset size.

    Returns:
        Estimated privacy budget \u03b5 (lower = stronger privacy).
    """
    sigma = noise_multiplier
    q     = sampling_rate
    T     = num_rounds

    best_eps = float("inf")
    for alpha in range(2, 513):
        a = float(alpha)
        # Per-step RDP of Poisson-subsampled Gaussian (small-q leading order)
        rdp_per_step = np.log1p(a * (a - 1.0) * q ** 2 / (2.0 * sigma ** 2)) / (a - 1.0)
        rdp_total    = T * rdp_per_step
        # Standard RDP → (\u03b5, \u03b4)-DP conversion, minimised over \u03b1
        eps = rdp_total + np.log(1.0 / delta) / (a - 1.0)
        if eps < best_eps:
            best_eps = eps

    return float(best_eps)


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

    lines = [
        f"\n{'=' * 55}",
        f"  Experiment : {experiment_name}",
        f"{'=' * 55}",
        f"  Final Accuracy       : {final_acc:.4f}",
        f"  Convergence Round    : {convergence_round or 'Did not reach 90%'}",
        f"  Total Flagged Events : {total_flagged}",
        f"  Avg Detection Rate   : {detection_rate:.2%}",
        f"  Client Dropout Rate  : {dropout_rate:.2%}  ({total_dropped}/{total_selected})",
        f"{'=' * 55}",
    ]
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Metrics export — CSV
# ---------------------------------------------------------------------------

def export_metrics_csv(
    experiment_results: Dict[str, Dict],
    filename: str = "fl_metrics.csv",
) -> str:
    """Export per-round FL metrics for all experiments to a single CSV file.

    Each row represents one communication round of one experiment.

    Args:
        experiment_results: Mapping of experiment label to a dict with keys:
            ``accuracy`` (List[float]), ``loss`` (List[float]),
            ``round_times`` (List[float]), ``participation_rate`` (float).
        filename: Output filename inside RESULTS_DIR.

    Returns:
        Absolute path of the written CSV file.

    Example CSV columns:
        experiment, round, accuracy, loss, round_time_secs, participation_rate
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "experiment",
            "round",
            "accuracy",
            "loss",
            "round_time_secs",
            "participation_rate",
        ])
        for label, data in experiment_results.items():
            acc_list   = data.get("accuracy", [])
            loss_list  = data.get("loss", [])
            time_list  = data.get("round_times", [None] * len(acc_list))
            part_rate  = data.get("participation_rate", float("nan"))
            for rnd, (acc, loss, t) in enumerate(
                zip(acc_list, loss_list, time_list), start=1
            ):
                writer.writerow([
                    label,
                    rnd,
                    f"{acc:.6f}",
                    f"{loss:.6f}",
                    f"{t:.3f}" if t is not None else "",
                    f"{part_rate:.4f}",
                ])

    logger.info("FL metrics CSV saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Metrics export — JSON
# ---------------------------------------------------------------------------

def export_metrics_json(
    experiment_results: Dict[str, Dict],
    filename: str = "fl_metrics.json",
) -> str:
    """Export per-round FL metrics for all experiments to a JSON file.

    Provides a machine-readable structured format suitable for automated
    analysis pipelines.

    Args:
        experiment_results: Same structure as ``export_metrics_csv``.
        filename: Output filename inside RESULTS_DIR.

    Returns:
        Absolute path of the written JSON file.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)

    serialisable: Dict = {}
    for label, data in experiment_results.items():
        acc_list  = data.get("accuracy",         [])
        loss_list = data.get("loss",             [])
        time_list = data.get("round_times",      [None] * len(acc_list))
        part_rate = data.get("participation_rate", None)

        serialisable[label] = {
            "participation_rate": part_rate,
            "rounds": [
                {
                    "round":          rnd,
                    "accuracy":       round(acc, 6),
                    "loss":           round(loss, 6),
                    "round_time_secs": round(t, 3) if t is not None else None,
                }
                for rnd, (acc, loss, t) in enumerate(
                    zip(acc_list, loss_list, time_list), start=1
                )
            ],
        }

    with open(path, "w") as fh:
        json.dump(serialisable, fh, indent=2)

    logger.info("FL metrics JSON saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Plot 7 — Accuracy vs client participation rate (Experiment F)
# ---------------------------------------------------------------------------

def plot_participation_rate(
    participation_rates: List[float],
    final_accuracies:    List[float],
    save_path: str = "participation_rate.png",
) -> plt.Figure:
    """Bar chart of final-round accuracy vs client participation rate.

    Isolates participation as the single controlled variable — all other
    settings (method, attack, detection) are held constant.

    Args:
        participation_rates: Fraction of NUM_CLIENTS used per round,
                             e.g. [0.2, 0.4, 0.6, 0.8, 1.0].
        final_accuracies:    Final-round global accuracy for each rate.
        save_path:           Filename inside RESULTS_DIR.
    """
    rate_labels = [f"{int(r * 100)}%" for r in participation_rates]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(
        rate_labels,
        final_accuracies,
        color="steelblue",
        alpha=0.85,
        edgecolor="white",
        linewidth=0.8,
    )

    # Annotate each bar with its exact accuracy value.
    for bar, acc in zip(bars, final_accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{acc:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_xlabel("Client Participation Rate (clients_per_round / total_clients)", fontsize=11)
    ax.set_ylabel("Final Global Accuracy (PathMNIST)", fontsize=11)
    ax.set_title(
        "Accuracy vs Client Participation Rate\n"
        "(Exp F — AsyncRobust, 20 rounds, non-IID α=0.5)",
        fontsize=12,
    )
    ax.set_ylim(0, min(1.0, max(final_accuracies) + 0.08))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Plot 8 — Centralised baseline vs FL overlay
# ---------------------------------------------------------------------------

def plot_centralized_vs_fl(
    centralized_accuracy: List[float],
    fl_accuracy:          List[float],
    fl_label:             str = "Exp C: AsyncRobust (FL)",
    save_path:            str = "centralized_vs_fl.png",
) -> plt.Figure:
    """Overlay centralised accuracy (per epoch) with FL accuracy (per round).

    The x-axis represents training steps in equivalent units so the
    convergence rates are directly comparable.

    Args:
        centralized_accuracy: Per-epoch test accuracy from ``centralized.py``.
        fl_accuracy:          Per-round test accuracy from an FL experiment.
        fl_label:             Legend label for the FL curve.
        save_path:            Filename inside RESULTS_DIR.
    """
    # Down-sample centralised curve to the same number of points as FL
    # so both curves share the same x-axis scale (communication rounds).
    n_fl = len(fl_accuracy)
    n_cen = len(centralized_accuracy)

    if n_cen >= n_fl:
        # Sample n_fl evenly-spaced epochs from the centralised history
        indices_cen = [int(round(i * (n_cen - 1) / (n_fl - 1))) for i in range(n_fl)]
        cen_sampled = [centralized_accuracy[i] for i in indices_cen]
    else:
        cen_sampled = centralized_accuracy

    x_axis = list(range(1, len(cen_sampled) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        x_axis, cen_sampled,
        "-D", label="Centralised (full data, no privacy)",
        linewidth=2, color="darkorange", markersize=4,
    )
    ax.plot(
        list(range(1, n_fl + 1)), fl_accuracy,
        "-o", label=fl_label,
        linewidth=2, color="green", markersize=4,
    )

    ax.set_xlabel("Communication Round (FL) / Equivalent Epoch (Centralised)", fontsize=11)
    ax.set_ylabel("Global Accuracy (PathMNIST)", fontsize=11)
    ax.set_title(
        "Centralised Baseline vs Federated Learning\n"
        "(PathMNIST, 10 Hospitals, Non-IID α=0.5)",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, save_path)
    return fig
