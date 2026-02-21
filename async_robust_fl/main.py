"""
main.py — Entry point.  Runs all 5 experiments and generates 6 plots.

Usage:
    python main.py

Output (written to ./results/):
    convergence.png          — Exp A–D on one graph
    dp_tradeoff.png          — Exp C vs D privacy-utility trade-off
    attack_impact.png        — Exp A vs B vs C attack/defence
    dropout_reliability.png  — Async tolerance to hospital dropouts
    detection_rate.png       — Per-round true-positive / false-positive
    heterogeneity.png        — Exp E: IID vs non-IID

Experiment map:
    A — FedAvg, no attack        (clean non-IID baseline)
    B — FedAvg, 20% attack       (shows FedAvg vulnerability)
    C — AsyncRobust, 20% attack  (trimmed mean + detection defence)
    D — AsyncRobust + DP, 20%    (full privacy-preserving system)
    E — AsyncRobust non-IID vs IID (heterogeneity isolation)
"""

from __future__ import annotations

import os
import sys

# Ensure the package root is on the path when running from the workspace root
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

import flwr as fl
from flwr.client import ClientApp
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import DifferentialPrivacyClientSideAdaptiveClipping
from flwr.client.mod import adaptiveclipping_mod
from flwr.simulation import run_simulation

from config import (
    SEED,
    NUM_CLIENTS,
    NUM_ROUNDS,
    CLIENTS_PER_ROUND,
    ASYNC_BUFFER_SIZE,
    DIRICHLET_ALPHA,
    MALICIOUS_CLIENT_IDS,
    NOISY_CLIENT_IDS,
    UNRELIABLE_CLIENT_IDS,
    DP_NOISE_MULTIPLIER,
    DP_CLIPPING_NORM,
    DP_DELTA,
    TRIM_FRACTION,
    NORM_THRESHOLD,
    COSINE_THRESHOLD,
)
from model import PathologyNet, get_weights
from data import load_global_test
from client import client_fn
from strategy import AsyncRobustFLStrategy
from evaluation import (
    make_evaluate_fn,
    plot_convergence,
    plot_dp_tradeoff,
    plot_attack_impact,
    plot_dropout_reliability,
    plot_detection,
    plot_heterogeneity,
    compute_attack_metrics,
    estimate_epsilon,
    print_summary,
)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_one_experiment(
    method:          str   = "trimmed_mean",
    use_dp:          bool  = False,
    use_attack:      bool  = True,
    dirichlet_alpha: float = DIRICHLET_ALPHA,
    num_rounds:      int   = NUM_ROUNDS,
    label:           str   = "",
) -> tuple:
    """Run a single FL simulation and return collected telemetry.

    Args:
        method          : Aggregation algorithm ('fedavg' | 'trimmed_mean' | 'median').
        use_dp          : Wrap strategy with DP adaptive clipping if True.
        use_attack      : Activate malicious + noisy clients if True.
        dirichlet_alpha : Data heterogeneity (0.5=non-IID, 1000=IID).
        num_rounds      : Number of FL rounds.
        label           : Human-readable experiment name for logging.

    Returns:
        (accuracy_history, flagged_history, dropout_history)
    """
    # Reproducibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"\n{'#' * 60}")
    print(f"  Running: {label or method}")
    print(f"  method={method}  dp={use_dp}  attack={use_attack}  alpha={dirichlet_alpha}")
    print(f"{'#' * 60}\n")

    mal_ids   = MALICIOUS_CLIENT_IDS   if use_attack else frozenset()
    noisy_ids = NOISY_CLIENT_IDS       if use_attack else frozenset()

    # Fresh model + eval function for each experiment (no state leakage)
    initial_params = ndarrays_to_parameters(get_weights(PathologyNet()))
    eval_fn        = make_evaluate_fn(load_global_test())

    strategy: fl.server.strategy.Strategy = AsyncRobustFLStrategy(
        initial_parameters    = initial_params,
        num_clients_per_round = CLIENTS_PER_ROUND,
        async_buffer_size     = ASYNC_BUFFER_SIZE,
        aggregation_method    = method,
        trim_fraction         = TRIM_FRACTION,
        norm_threshold        = NORM_THRESHOLD,
        cosine_threshold      = COSINE_THRESHOLD,
        evaluate_fn           = eval_fn,
        malicious_client_ids  = mal_ids,
        noisy_client_ids      = noisy_ids,
        unreliable_client_ids = UNRELIABLE_CLIENT_IDS,
    )

    base_strategy = strategy   # keep reference for telemetry

    if use_dp:
        strategy = DifferentialPrivacyClientSideAdaptiveClipping(
            strategy              = strategy,
            noise_multiplier      = DP_NOISE_MULTIPLIER,
            num_sampled_clients   = CLIENTS_PER_ROUND,
            initial_clipping_norm = DP_CLIPPING_NORM,
        )

    server_app = ServerApp(
        config   = ServerConfig(num_rounds=num_rounds),
        strategy = strategy,
    )

    mods       = [adaptiveclipping_mod] if use_dp else []
    client_app = ClientApp(client_fn=client_fn, mods=mods)

    run_simulation(
        server_app     = server_app,
        client_app     = client_app,
        num_supernodes = NUM_CLIENTS,
        backend_config = {
            "client_resources": {
                "num_cpus": 1,
                # 0.5 allows 2 clients to share one GPU; set 0.0 for CPU-only
                "num_gpus": 0.5 if torch.cuda.is_available() else 0.0,
            }
        },
        run_config = {"dirichlet_alpha": dirichlet_alpha},
    )

    return (
        list(eval_fn.accuracy_history),
        list(base_strategy.flagged_history),
        list(base_strategy.dropout_history),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs("results", exist_ok=True)

    # ------------------------------------------------------------------
    # Experiment A — FedAvg, no attack (clean non-IID baseline)
    # ------------------------------------------------------------------
    acc_A, flagged_A, dropout_A = run_one_experiment(
        method      = "fedavg",
        use_dp      = False,
        use_attack  = False,
        label       = "Exp A: FedAvg, no attack",
    )
    print_summary(
        "Exp A: FedAvg no attack",
        acc_A, flagged_A, dropout_A,
        frozenset(), frozenset(),
    )

    # ------------------------------------------------------------------
    # Experiment B — FedAvg under 20% malicious + noisy attack
    # ------------------------------------------------------------------
    acc_B, flagged_B, dropout_B = run_one_experiment(
        method      = "fedavg",
        use_dp      = False,
        use_attack  = True,
        label       = "Exp B: FedAvg, under attack",
    )
    print_summary(
        "Exp B: FedAvg attacked",
        acc_B, flagged_B, dropout_B,
        MALICIOUS_CLIENT_IDS, NOISY_CLIENT_IDS,
    )

    # ------------------------------------------------------------------
    # Experiment C — AsyncRobust (TrimmedMean + detection), 20% attack
    # ------------------------------------------------------------------
    acc_C, flagged_C, dropout_C = run_one_experiment(
        method      = "trimmed_mean",
        use_dp      = False,
        use_attack  = True,
        label       = "Exp C: AsyncRobust, no DP",
    )
    print_summary(
        "Exp C: AsyncRobust no DP",
        acc_C, flagged_C, dropout_C,
        MALICIOUS_CLIENT_IDS, NOISY_CLIENT_IDS,
    )

    # ------------------------------------------------------------------
    # Experiment D — AsyncRobust + DP (full system)
    # ------------------------------------------------------------------
    acc_D, flagged_D, dropout_D = run_one_experiment(
        method      = "trimmed_mean",
        use_dp      = True,
        use_attack  = True,
        label       = "Exp D: AsyncRobust + DP",
    )
    print_summary(
        "Exp D: AsyncRobust + DP",
        acc_D, flagged_D, dropout_D,
        MALICIOUS_CLIENT_IDS, NOISY_CLIENT_IDS,
    )

    # ------------------------------------------------------------------
    # Experiment E — Heterogeneity isolation
    #   E-nonIID : same as Exp C (reuse acc_C)
    #   E-IID    : identical settings, alpha=1000.0
    # ------------------------------------------------------------------
    acc_E_iid, _, _ = run_one_experiment(
        method          = "trimmed_mean",
        use_dp          = False,
        use_attack      = True,
        dirichlet_alpha = 1000.0,
        label           = "Exp E: AsyncRobust, IID (alpha=1000)",
    )

    # ------------------------------------------------------------------
    # Privacy budget (Experiment D)
    # ------------------------------------------------------------------
    epsilon = estimate_epsilon(
        noise_multiplier = DP_NOISE_MULTIPLIER,
        sampling_rate    = CLIENTS_PER_ROUND / NUM_CLIENTS,
        num_rounds       = NUM_ROUNDS,
        delta            = DP_DELTA,
    )
    print(f"\nPrivacy budget (Exp D): ε ≈ {epsilon:.2f}, δ = {DP_DELTA}")
    print(f"Interpretation: ε={epsilon:.2f} means an attacker cannot distinguish")
    print(f"individual training samples with probability better than e^{epsilon:.2f} ≈ {np.exp(epsilon):.1f}×.\n")

    # ------------------------------------------------------------------
    # All 6 plots
    # ------------------------------------------------------------------
    plot_convergence(
        {
            "Exp A: FedAvg clean":       acc_A,
            "Exp B: FedAvg attacked":    acc_B,
            "Exp C: AsyncRobust":        acc_C,
            "Exp D: AsyncRobust + DP":   acc_D,
        },
        save_path = "convergence.png",
    )

    plot_dp_tradeoff(acc_C, acc_D, epsilon, save_path="dp_tradeoff.png")

    plot_attack_impact(acc_A, acc_B, acc_C, save_path="attack_impact.png")

    plot_dropout_reliability(dropout_C, save_path="dropout_reliability.png")

    all_bad_ids = MALICIOUS_CLIENT_IDS | NOISY_CLIENT_IDS
    plot_detection(flagged_C, all_bad_ids, save_path="detection_rate.png")

    plot_heterogeneity(acc_C, acc_E_iid, save_path="heterogeneity.png")

    # ------------------------------------------------------------------
    # Attack metrics table
    # ------------------------------------------------------------------
    metrics = compute_attack_metrics(acc_A, acc_B, acc_C)
    print("\nAttack / Defence Metrics")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"  {k:<28}: {v:.4f}")
    print()

    print("All plots saved to ./results/")
    print("Done.")


if __name__ == "__main__":
    main()
