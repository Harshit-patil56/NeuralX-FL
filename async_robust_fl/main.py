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

import logging
import os
import sys
import time

# Ensure the package root is on the path when running from the workspace root
sys.path.insert(0, os.path.dirname(__file__))

# ---------------------------------------------------------------------------
# Logging — configured before any package import that might create loggers
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import numpy as np
import torch

import flwr as fl
from flwr.client import ClientApp
from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
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
    PARTICIPATION_RATES,
    RESULTS_DIR,
)
from model import PathologyNet, get_weights
from data import load_global_test
from client import make_client_fn
from strategy import AsyncRobustFLStrategy
from centralized import run_centralized, export_centralized_csv
from evaluation import (
    make_evaluate_fn,
    plot_convergence,
    plot_loss_curves,
    plot_dp_tradeoff,
    plot_attack_impact,
    plot_dropout_reliability,
    plot_detection,
    plot_heterogeneity,
    plot_participation_rate,
    plot_centralized_vs_fl,
    export_metrics_csv,
    export_metrics_json,
    compute_attack_metrics,
    estimate_epsilon,
    print_summary,
)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_one_experiment(
    method:              str        = "trimmed_mean",
    use_dp:              bool       = False,
    use_attack:          bool       = True,
    use_detection:       bool       = True,
    dirichlet_alpha:     float      = DIRICHLET_ALPHA,
    num_rounds:          int        = NUM_ROUNDS,
    clients_per_round:   int        = CLIENTS_PER_ROUND,
    label:               str        = "",
    seed:                int        = SEED,
    async_buffer_size:   int        = ASYNC_BUFFER_SIZE,
    malicious_client_ids: frozenset = MALICIOUS_CLIENT_IDS,
) -> tuple:
    """Run a single FL simulation and return collected telemetry.

    Args:
        method              : Aggregation algorithm
                              ('fedavg' | 'trimmed_mean' | 'median' | 'krum').
        use_dp              : Wrap strategy with DP adaptive clipping if True.
        use_attack          : Activate malicious + noisy clients if True.
        use_detection       : Enable norm + cosine Byzantine detection filters.
                              Set False for Exp B to expose raw FedAvg vulnerability.
        dirichlet_alpha     : Data heterogeneity (0.5=non-IID, 1000=IID).
        num_rounds          : Number of FL rounds.
        clients_per_round   : Clients sampled each round (controls participation
                              rate).
        label               : Human-readable experiment name for logging.
        seed                : Random seed for full reproducibility
                              (numpy, torch, partitioning).
        async_buffer_size   : Override the async buffer size from config.
        malicious_client_ids: Override the malicious client IDs from config.

    Returns:
        Tuple of five lists:
            (accuracy_history, loss_history, round_times,
             flagged_history, dropout_history)
    """
    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info("Starting: %s", label or method)
    logger.info(
        "method=%s  dp=%s  attack=%s  detection=%s  alpha=%s  "
        "clients_per_round=%d  seed=%d  buffer=%d",
        method, use_dp, use_attack, use_detection, dirichlet_alpha,
        clients_per_round, seed, async_buffer_size,
    )

    mal_ids   = malicious_client_ids   if use_attack else frozenset()
    noisy_ids = NOISY_CLIENT_IDS       if use_attack else frozenset()

    # Async buffer cannot exceed the number of clients participating.
    effective_buffer = min(async_buffer_size, clients_per_round)

    # Fresh model + eval function for each experiment (no state leakage)
    initial_params = ndarrays_to_parameters(get_weights(PathologyNet()))
    eval_fn        = make_evaluate_fn(load_global_test())

    strategy: fl.server.strategy.Strategy = AsyncRobustFLStrategy(
        initial_parameters    = initial_params,
        num_clients_per_round = clients_per_round,
        async_buffer_size     = effective_buffer,
        aggregation_method    = method,
        trim_fraction         = TRIM_FRACTION,
        norm_threshold        = NORM_THRESHOLD,
        cosine_threshold      = COSINE_THRESHOLD,
        evaluate_fn           = eval_fn,
        malicious_client_ids  = mal_ids,
        noisy_client_ids      = noisy_ids,
        unreliable_client_ids = UNRELIABLE_CLIENT_IDS,
        use_detection         = use_detection,
    )

    base_strategy = strategy   # keep reference for telemetry

    if use_dp:
        strategy = DifferentialPrivacyClientSideAdaptiveClipping(
            strategy              = strategy,
            noise_multiplier      = DP_NOISE_MULTIPLIER,
            num_sampled_clients   = clients_per_round,
            initial_clipping_norm = DP_CLIPPING_NORM,
        )

    # Use the new server_fn / ServerAppComponents pattern (avoid deprecation
    # warning from passing strategy/config directly to the ServerApp constructor)
    _strategy    = strategy          # captured by closure
    _num_rounds  = num_rounds

    def server_fn(context) -> ServerAppComponents:
        return ServerAppComponents(
            strategy = _strategy,
            config   = ServerConfig(num_rounds=_num_rounds),
        )

    server_app = ServerApp(server_fn=server_fn)

    # make_client_fn captures dirichlet_alpha via closure — no run_config needed
    mods       = [adaptiveclipping_mod] if use_dp else []
    client_app = ClientApp(client_fn=make_client_fn(dirichlet_alpha), mods=mods)

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
    )

    return (
        list(eval_fn.accuracy_history),
        list(eval_fn.loss_history),
        list(eval_fn.round_times),
        list(base_strategy.flagged_history),
        list(base_strategy.dropout_history),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Experiment A — FedAvg, no attack (clean non-IID baseline)
    # ------------------------------------------------------------------
    acc_A, loss_A, times_A, flagged_A, dropout_A = run_one_experiment(
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
    # use_detection=False: filters are OFF so FedAvg has no defence —
    # this is what makes Exp B show the raw attack damage.
    # ------------------------------------------------------------------
    acc_B, loss_B, times_B, flagged_B, dropout_B = run_one_experiment(
        method         = "fedavg",
        use_dp         = False,
        use_attack     = True,
        use_detection  = False,
        label          = "Exp B: FedAvg, under attack",
    )
    print_summary(
        "Exp B: FedAvg attacked",
        acc_B, flagged_B, dropout_B,
        MALICIOUS_CLIENT_IDS, NOISY_CLIENT_IDS,
    )

    # ------------------------------------------------------------------
    # Experiment C — AsyncRobust (TrimmedMean + detection), 20% attack
    # ------------------------------------------------------------------
    acc_C, loss_C, times_C, flagged_C, dropout_C = run_one_experiment(
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
    acc_D, loss_D, times_D, flagged_D, dropout_D = run_one_experiment(
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
    #   E-nonIID : identical settings to Exp C but run independently
    #              (ensures controlled comparison: ONLY alpha differs)
    #   E-IID    : same settings, alpha=1000.0
    # ------------------------------------------------------------------
    acc_E_noniid, _, _, _, _ = run_one_experiment(
        method          = "trimmed_mean",
        use_dp          = False,
        use_attack      = True,
        dirichlet_alpha = DIRICHLET_ALPHA,
        label           = "Exp E: AsyncRobust, non-IID (alpha=0.5)",
    )

    acc_E_iid, _, _, _, _ = run_one_experiment(
        method          = "trimmed_mean",
        use_dp          = False,
        use_attack      = True,
        dirichlet_alpha = 1000.0,
        label           = "Exp E: AsyncRobust, IID (alpha=1000)",
    )

    # ------------------------------------------------------------------
    # Experiment F — Participation rate sweep
    # Vary clients_per_round from 20% to 100% of NUM_CLIENTS,
    # holding all other settings constant (same as Exp C).
    # Reveals the accuracy–communication trade-off curve.
    # ------------------------------------------------------------------
    logger.info("Starting Experiment F: participation rate sweep")
    participation_final_accs: list = []
    for rate in PARTICIPATION_RATES:
        n_clients = max(2, int(round(rate * NUM_CLIENTS)))
        acc_F, _, _, _, _ = run_one_experiment(
            method             = "trimmed_mean",
            use_dp             = False,
            use_attack         = True,
            clients_per_round  = n_clients,
            label              = f"Exp F: rate={rate:.0%} ({n_clients} clients)",
        )
        participation_final_accs.append(acc_F[-1] if acc_F else 0.0)
        logger.info("  Rate %.0f%%  final_acc=%.4f", rate * 100, acc_F[-1] if acc_F else 0.0)

    # ------------------------------------------------------------------
    # Centralised baseline (runs standalone — slow on large datasets)
    # ------------------------------------------------------------------
    logger.info("Starting centralised baseline training (%d epochs)…", 100)
    acc_cen, loss_cen, times_cen = run_centralized()
    export_centralized_csv(acc_cen, loss_cen, times_cen)
    logger.info("Centralised final accuracy: %.4f", acc_cen[-1] if acc_cen else 0.0)

    # ------------------------------------------------------------------
    # Privacy budget (Experiment D)
    # ------------------------------------------------------------------
    epsilon = estimate_epsilon(
        noise_multiplier = DP_NOISE_MULTIPLIER,
        sampling_rate    = CLIENTS_PER_ROUND / NUM_CLIENTS,
        num_rounds       = NUM_ROUNDS,
        delta            = DP_DELTA,
    )
    logger.info(
        "Privacy budget (Exp D): ε ≈ %.4f, δ = %s\n"
        "  Lower ε = stronger privacy. Computed via Rényi DP accountant.",
        epsilon, DP_DELTA,
    )

    # ------------------------------------------------------------------
    # All plots
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

    plot_detection(
        flagged_C,
        malicious_ids = MALICIOUS_CLIENT_IDS,
        noisy_ids     = NOISY_CLIENT_IDS,
        save_path     = "detection_rate.png",
    )

    plot_heterogeneity(acc_E_noniid, acc_E_iid, save_path="heterogeneity.png")

    plot_loss_curves(
        {
            "Exp A: FedAvg clean":       loss_A,
            "Exp B: FedAvg attacked":    loss_B,
            "Exp C: AsyncRobust":        loss_C,
            "Exp D: AsyncRobust + DP":   loss_D,
        },
        save_path = "loss_curves.png",
    )

    plot_participation_rate(
        list(PARTICIPATION_RATES),
        participation_final_accs,
        save_path = "participation_rate.png",
    )

    plot_centralized_vs_fl(
        centralized_accuracy = acc_cen,
        fl_accuracy          = acc_C,
        fl_label             = "Exp C: AsyncRobust (FL)",
        save_path            = "centralized_vs_fl.png",
    )

    # ------------------------------------------------------------------
    # Structured metrics export (CSV + JSON)  — all FL experiments
    # ------------------------------------------------------------------
    experiment_results = {
        "Exp A: FedAvg clean": {
            "accuracy":           acc_A,
            "loss":               loss_A,
            "round_times":        times_A,
            "participation_rate": CLIENTS_PER_ROUND / NUM_CLIENTS,
        },
        "Exp B: FedAvg attacked": {
            "accuracy":           acc_B,
            "loss":               loss_B,
            "round_times":        times_B,
            "participation_rate": CLIENTS_PER_ROUND / NUM_CLIENTS,
        },
        "Exp C: AsyncRobust": {
            "accuracy":           acc_C,
            "loss":               loss_C,
            "round_times":        times_C,
            "participation_rate": CLIENTS_PER_ROUND / NUM_CLIENTS,
        },
        "Exp D: AsyncRobust + DP": {
            "accuracy":           acc_D,
            "loss":               loss_D,
            "round_times":        times_D,
            "participation_rate": CLIENTS_PER_ROUND / NUM_CLIENTS,
        },
    }
    export_metrics_csv(experiment_results)
    export_metrics_json(experiment_results)

    # ------------------------------------------------------------------
    # Attack metrics table
    # ------------------------------------------------------------------
    metrics = compute_attack_metrics(acc_A, acc_B, acc_C)
    logger.info("\nAttack / Defence Metrics")
    logger.info("-" * 40)
    for k, v in metrics.items():
        logger.info("  %-28s: %.4f", k, v)

    logger.info("All plots saved to %s", RESULTS_DIR)
    logger.info("Done.")


if __name__ == "__main__":
    main()
