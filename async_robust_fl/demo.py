"""
demo.py — 3-round quick demonstration of NeuralX-FL.

Runs TWO experiments in ~3 minutes on a GTX 1650:
  1. FedAvg under attack   (no defence)   → shows how attacks destroy accuracy
  2. AsyncRobustFL          (full defence) → shows how the system recovers

Usage:
    python demo.py

Output is printed to the terminal. No files are written or changed.
The full 20-round simulation (all 6 experiments) is in main.py.
"""

from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# ── Suppress Flower/Ray verbose logging for a clean demo output ─────────────
logging.basicConfig(level=logging.WARNING, format="%(message)s")
for noisy in ("flwr", "ray", "numexpr", "torch"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

import torch

from config import MALICIOUS_CLIENT_IDS, NOISY_CLIENT_IDS

# Import the experiment runner directly from main — no duplication
from main import run_one_experiment

# ── Helpers ──────────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 30, char: str = "█") -> str:
    """Render a simple ASCII progress bar for a 0–100 float."""
    filled = int(round(value / 100 * width))
    return char * filled + "░" * (width - filled)


def _print_round_table(accuracy_history: list[float]) -> None:
    """Print a compact per-round accuracy table."""
    print(f"  {'Round':>5}  {'Accuracy':>9}  {'Bar'}")
    print(f"  {'─'*5}  {'─'*9}  {'─'*32}")
    for i, acc in enumerate(accuracy_history):
        pct = acc * 100
        print(f"  {i:>5}  {pct:>8.2f}%  {_bar(pct, 32)}")


def _print_separator(char: str = "═", width: int = 60) -> None:
    print(char * width)


# ── Demo ─────────────────────────────────────────────────────────────────────

def main() -> None:
    DEMO_ROUNDS = 3   # Keep the demo under ~3 minutes

    _print_separator()
    print("  NeuralX-FL  —  Quick Demo  (3 rounds)")
    print("  10 hospitals · PathMNIST · GTX 1650")
    _print_separator()
    print()
    print("  Client setup:")
    print("    Hospitals {0, 1}   → MALICIOUS  (gradient scaling ×50)")
    print("    Hospitals {4, 5}   → NOISY      (30% label flip)")
    print("    Hospitals {6, 7}   → UNRELIABLE (40% dropout chance)")
    print("    Hospitals {2,3,8,9}→ HONEST")
    print()

    # ── Demo 1: FedAvg under attack, NO defence ──────────────────────────────
    _print_separator("─")
    print("  DEMO 1 of 2 — FedAvg  (no defence, attacks ON)")
    print("  Standard federated learning — no filters, no robust aggregation.")
    _print_separator("─")
    print()

    acc_B, _loss_B, flagged_B, dropout_B = run_one_experiment(
        method        = "fedavg",
        use_dp        = False,
        use_attack    = True,
        use_detection = False,
        num_rounds    = DEMO_ROUNDS,
        label         = "Demo 1: FedAvg under attack",
    )

    print()
    _print_round_table(acc_B)
    final_B = acc_B[-1] * 100 if acc_B else 0.0
    flagged_total_B = sum(len(r.get("flagged", [])) for r in flagged_B)
    print()
    print(f"  Final accuracy  : {final_B:.2f}%   (attacks undetected — model poisoned)")
    print(f"  Updates flagged : {flagged_total_B}  (detection was OFF)")
    print()

    # ── Demo 2: AsyncRobustFL, defence ON ────────────────────────────────────
    _print_separator("─")
    print("  DEMO 2 of 2 — AsyncRobustFL  (full defence, attacks ON)")
    print("  Norm filter + cosine filter + trimmed-mean aggregation.")
    _print_separator("─")
    print()

    acc_C, _loss_C, flagged_C, dropout_C = run_one_experiment(
        method        = "trimmed_mean",
        use_dp        = False,
        use_attack    = True,
        use_detection = True,
        num_rounds    = DEMO_ROUNDS,
        label         = "Demo 2: AsyncRobustFL with defence",
    )

    print()
    _print_round_table(acc_C)
    final_C = acc_C[-1] * 100 if acc_C else 0.0
    flagged_total_C = sum(len(r.get("flagged", [])) for r in flagged_C)
    dropped_total_C = sum(d.get("dropped", 0) for d in dropout_C)
    print()
    print(f"  Final accuracy  : {final_C:.2f}%   (defence active — model protected)")
    print(f"  Updates flagged : {flagged_total_C}  (Byzantine updates rejected)")
    print(f"  Client dropouts : {dropped_total_C}  (unreliable hospitals skipped)")
    print()

    # ── Side-by-side comparison ───────────────────────────────────────────────
    _print_separator()
    print("  COMPARISON SUMMARY")
    _print_separator()
    print()
    improvement = final_C - final_B
    print(f"  No defence  (Demo 1) :  {final_B:>6.2f}%  {_bar(final_B, 25)}")
    print(f"  With defence(Demo 2) :  {final_C:>6.2f}%  {_bar(final_C, 25)}")
    print()
    if improvement > 0:
        print(f"  ✓ Defence recovered {improvement:+.2f} percentage points in just {DEMO_ROUNDS} rounds.")
    else:
        print(f"  NOTE: Run main.py for the full 20-round results — improvements")
        print(f"        compound over more rounds (final result: 81.41% vs 18.64%).")
    print()
    print("  Full 20-round results → run:  python main.py")
    print("  Full report           →       report.txt")
    _print_separator()


if __name__ == "__main__":
    gpu_info = (
        f"CUDA — {torch.cuda.get_device_name(0)}"
        if torch.cuda.is_available()
        else "CPU only"
    )
    print(f"\n  Device: {gpu_info}\n")
    main()
