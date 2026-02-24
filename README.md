# NeuralX-FL: Asynchronous Byzantine-Robust Federated Learning for Pathology

> **Research Project** — End-to-end privacy-preserving FL system for multi-hospital colon pathology classification with Byzantine detection, robust asynchronous aggregation, differential privacy, and full empirical evaluation including a centralised non-federated baseline.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [System Architecture](#2-system-architecture)
3. [Defense Mechanisms](#3-defense-mechanisms)
4. [Experiment Design](#4-experiment-design)
5. [Results](#5-results)
6. [Project Structure](#6-project-structure)
7. [Setup & Installation](#7-setup--installation)
8. [Running the Project](#8-running-the-project)
9. [Test Suite](#9-test-suite)
10. [Technical Stack](#10-technical-stack)
11. [Key Design Decisions](#11-key-design-decisions)
12. [Output Files](#12-output-files)
13. [Experimental Methodology](#13-experimental-methodology)
14. [Reproducibility](#14-reproducibility)
15. [Limitations](#15-limitations)

---

## 1. Problem Statement

Hospitals cannot share patient pathology slide images due to legal and privacy constraints (HIPAA, GDPR). A centralized deep learning model is therefore infeasible.

**Federated Learning (FL)** allows hospitals to collaboratively train a shared model without any data leaving each site — only model weight updates are exchanged.

However, two real-world threats break standard FL:

| Threat | Description |
|--------|-------------|
| **Byzantine Attacks** | A subset of hospitals sends deliberately corrupted model updates to poison the global model |
| **Asynchronous Networks** | Hospitals have different hardware speeds and network reliability, causing updates to arrive at different times — standard FL stalls waiting for the slowest client |

This project builds a system that handles **both simultaneously**.

---

## 2. System Architecture

### Dataset — PathMNIST

| Property | Value |
|----------|-------|
| Source | Colon-tissue pathology patches |
| Image size | 28×28 RGB |
| Classes | 9 tissue types (adipose, background, debris, lymphocytes, mucus, normal colon mucosa, colorectal adenocarcinoma, smooth muscle, stroma) |
| Train split | 89,996 images |
| Test split | 7,180 images |
| Partitioning | Dirichlet (α = 0.5) — non-IID per hospital |

### Model — PathologyNet CNN

```
Input  →  [B, 3, 28, 28]
Conv1  →  3 → 32 filters, 3×3, pad=1, ReLU → MaxPool(2×2)
Conv2  →  32 → 64 filters, 3×3, pad=1, ReLU → MaxPool(2×2)
FC1    →  3136 → 128, ReLU, Dropout(0.25)
FC2    →  128 → 9 logits
Loss   →  CrossEntropyLoss
Device →  NVIDIA GTX 1650 (CUDA 12.4, PyTorch 2.6.0+cu124)
```

### Federation Topology

| Parameter | Value |
|-----------|-------|
| Total hospitals (clients) | 10 |
| Sampled per round | 6 |
| Async buffer size | 4 (fastest clean updates per round) |
| Training rounds | 20 |
| Local epochs (warm-up, rounds 1–3) | 2 |
| Local epochs (main, rounds 4–20) | 5 |
| Batch size | 32 |
| Data heterogeneity | Dirichlet α = 0.5 |

### Client Taxonomy

| Hospital IDs | Type | Behavior |
|---|---|---|
| {0, 1} | **Malicious** (20%) | Gradient scaling attack ×50 |
| {4, 5} | **Noisy** (20%) | 30% random label flipping |
| {6, 7} | **Unreliable** (20%) | 40% dropout probability per round |
| {2, 3, 8, 9} | **Honest** (40%) | Normal federated training |

---

## 3. Defense Mechanisms

The defense pipeline runs in 4 sequential steps:

### Step 1 — Detection Filters (before async selection)

**L2 Norm Filter**
- Computes L2 norm of each flattened update vector
- Rejects any update where `norm > 3 × median_norm`
- Rationale: scaling attack multiplies gradients by 50×, inflating the norm ~50× above honest updates

**Cosine Similarity Filter**
- Uses coordinate-wise median of all updates as the reference direction
- Rejects any update with cosine similarity < 0 (pointing to opposite half-space)
- Median is robust up to 50% Byzantine fraction; here only 20% are malicious

### Step 2 — Async Selection

After detection, surviving updates are sorted by simulated arrival delay (sampled from Exp(scale=2.0)). The **4 fastest survivors** are used; the remaining 2 are treated as stragglers.

> **Key property:** Detection runs *before* selection — attackers cannot game the async buffer by responding quickly.

### Step 3 — Robust Aggregation

| Setting | Value |
|---------|-------|
| Method | Coordinate-wise Trimmed Mean (Yin et al., ICML 2018) |
| Trim fraction | 10% from each tail per coordinate |
| Staleness weighting | `weight = 1 / (1 + staleness_rounds)` |

Staleness weighting penalizes stale updates proportionally — a client 3 rounds stale gets weight 0.25 vs. 1.0 for a fresh update.

### Step 4 — Differential Privacy (Experiment D only)

| Parameter | Value |
|-----------|-------|
| Mechanism | Client-side adaptive DP clipping (Flower built-in) |
| Noise multiplier | 0.1 |
| Initial clipping norm | 0.1 |
| Delta (δ) | 1×10⁻⁵ |
| Privacy budget (ε) | ≈ 0.6239 |

Privacy interpretation: any adversary's ability to distinguish a participating patient record from a non-participating one is bounded by $e^{0.6239} \approx 1.87\times$.

---

## 4. Experiment Design

| ID | Name | Attacks | Defense | Purpose |
|----|------|---------|---------|---------|
| **A** | FedAvg, no attack | None | OFF | Clean non-IID upper-bound baseline |
| **B** | FedAvg, under attack | Malicious + Noisy | OFF | Show raw vulnerability of standard FedAvg |
| **C** | AsyncRobustFL | Malicious + Noisy + Unreliable | ON | Show defense recovery |
| **D** | AsyncRobustFL + DP | Malicious + Noisy + Unreliable | ON + DP | Show privacy-utility trade-off |
| **E (i)** | Heterogeneity — non-IID | α = 0.5 | ON | Isolate effect of data heterogeneity |
| **E (ii)** | Heterogeneity — near-IID | α = 1000.0 | ON | Convergence under homogeneous data |

---

## 5. Results

### Final Accuracy per Experiment

| Experiment | Final Accuracy |
|------------|---------------|
| A — Clean baseline | **84.22%** |
| B — Under attack, no defense | **~18.6%** |
| C — With defense | **81.41%** |
| D — With defense + DP | **38.70%** |

### Attack & Defense Metrics (Exp B vs C)

| Metric | Value |
|--------|-------|
| Average attack damage | **39.68%** accuracy drop |
| Maximum attack damage | **78.05%** accuracy drop (single round) |
| Average defense recovery | **37.83%** accuracy recovered |
| Final clean accuracy (Exp A) | **83.51%** |
| Final attacked accuracy (Exp B) | **18.64%** |
| Final defended accuracy (Exp C) | **81.41%** |

### Detection Performance (Exp C)

| Metric | Value |
|--------|-------|
| Total flagged events (20 rounds) | 20 |
| Average detection rate per round | 25% |
| Average dropout rate (unreliable) | 5.83% per round |

### Differential Privacy Finding (Exp D)

Accuracy dropped from 81.41% (Exp C) to 38.70% (Exp D). This is a valid research finding demonstrating the **privacy-utility trade-off**: with `noise_multiplier=0.1` and `clipping_norm=0.1`, the effective noise scale equals 1.0, which overwhelms gradients. Additionally, DP clipping to norm ≤ 0.1 makes all updates have equal norm, defeating the L2 norm Byzantine filter. Privacy budget: **ε ≈ 0.6239** at δ = 1×10⁻⁵.

---

## 6. Project Structure

```
async_robust_fl/
├── config.py                   # All hyperparameters (LEARNING_RATE, PARTICIPATION_RATES, etc.)
├── model.py                    # PathologyNet CNN (GPU-only, raises RuntimeError without CUDA)
├── data.py                     # Memory-mapped PathMNIST loading + load_global_train()
├── client.py                   # Flower FlowerClient — local training, attack injection, dropout
├── strategy.py                 # AsyncRobustStrategy — detection, async selection, aggregation
├── aggregation.py              # trimmed_mean, coordinate_median, krum, multi_krum
├── detection.py                # filter_by_norm, filter_by_cosine, add_label_noise
├── evaluation.py               # EvaluateFn (with round-time tracking), 9 plotting/export functions
├── centralized.py              # Centralised baseline: run_centralized(), export_centralized_csv()
├── main.py                     # Experiment runner — Exp A–F + centralised + CSV/JSON export
├── experiments.py              # Research experiments: defense comparison, sensitivity,
│                               #   multi-seed stats, failure threshold, comm cost
├── generate_ieee_report.py     # IEEE-style Word report generator (python-docx)
├── requirements.txt            # Python dependencies
├── report.txt                  # Full technical report
├── tests/
│   └── test_algorithms.py      # 29 unit tests (pytest)
├── data/
│   └── pathmnist.npz           # Downloaded automatically on first run
└── results/
    ├── convergence.png
    ├── loss_curves.png
    ├── dp_tradeoff.png
    ├── attack_impact.png
    ├── dropout_reliability.png
    ├── detection_rate.png
    ├── heterogeneity.png
    ├── participation_rate.png       # Exp F — participation sweep
    ├── centralized_vs_fl.png        # Centralised baseline overlay
    ├── fl_metrics.csv               # Per-round FL metrics (all experiments)
    ├── fl_metrics.json              # Same data in JSON format
    ├── centralized_accuracy.csv     # Centralised per-epoch metrics
    │
    │   # ── experiments.py outputs ──────────────────────────────────────────
    ├── defense_comparison.png       # Exp G: all defense methods vs attack
    ├── defense_comparison.csv
    ├── defense_comparison.json
    ├── sensitivity_alpha.png        # Sensitivity: Dirichlet α sweep
    ├── sensitivity_alpha.csv
    ├── sensitivity_alpha.json
    ├── sensitivity_byzantine.png    # Sensitivity: Byzantine % sweep
    ├── sensitivity_byzantine.csv
    ├── sensitivity_byzantine.json
    ├── sensitivity_buffer.png       # Sensitivity: async buffer size sweep
    ├── sensitivity_buffer.csv
    ├── sensitivity_buffer.json
    ├── multiseed_stats.png          # Multi-seed mean ± std curves
    ├── multiseed_stats.csv
    ├── multiseed_stats.json
    ├── failure_threshold.png        # Robustness / failure point analysis
    ├── failure_threshold.csv
    ├── failure_threshold.json
    ├── communication_cost.png       # Accuracy vs comm volume (all methods)
    ├── rounds_to_target.png         # Rounds to 80% accuracy per method
    ├── communication_cost.csv
    └── communication_cost.json
```

---

## 7. Setup & Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA 12.4 (GTX 1650 or better recommended)
- Windows or Linux

### Step 1 — Create Virtual Environment

```powershell
cd C:\Hackthon\NeuralX-FL\async_robust_fl
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 2 — Install CUDA PyTorch

> **Important:** Install the CUDA build of PyTorch first, before the rest of the requirements.

```powershell
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```

### Step 3 — Install Remaining Dependencies

```powershell
pip install -r requirements.txt
```

### Step 4 — Verify GPU

```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Expected output:
```
CUDA: True
GPU: NVIDIA GeForce GTX 1650
```

---

## 8. Running the Project

### Run Core Experiments (A–F + Centralised baseline)

```powershell
python main.py
```

The first run automatically downloads PathMNIST (~16 MB) to `data/pathmnist.npz`.
All 6 experiments run sequentially. Results and plots are saved to `results/`.

**Estimated runtime:** ~22 seconds per round on GTX 1650 (~2.5 hours total for all experiments).

### Run Research Experiments

```powershell
# All research experiment groups (~12–18 hours on GTX 1650 with 3 seeds)
python experiments.py

# Run specific group(s) only
python experiments.py --group defense
python experiments.py --group alpha byzantine
python experiments.py --group multiseed threshold comm
```

Available groups:

| Group | Description | New Output Files |
|-------|-------------|------------------|
| `defense` | FedAvg / Median / Krum / TrimMean under same attack | `defense_comparison.*` |
| `alpha` | Dirichlet α ∈ {0.1, 0.5, 1.0} sweep | `sensitivity_alpha.*` |
| `byzantine` | Byzantine % ∈ {10%, 20%, 30%} sweep | `sensitivity_byzantine.*` |
| `buffer` | Async buffer size ∈ {4, 6, 8} sweep | `sensitivity_buffer.*` |
| `multiseed` | 3-seed runs with mean ± std | `multiseed_stats.*` |
| `threshold` | Byzantine fraction ∈ {0%–50%} failure point | `failure_threshold.*` |
| `comm` | Accuracy vs communication volume | `communication_cost.*`, `rounds_to_target.png` |

### Run Tests Only

```powershell
python -m pytest tests/ -v
```

---

## 9. Test Suite

**29 / 29 tests passing**

```
tests/test_algorithms.py
```

| Module | Tests |
|--------|-------|
| `trimmed_mean` | 3 (basic, all-same, trim-exceeds-half) |
| `coordinate_median` | 2 |
| `krum` / `multi_krum` | 3 |
| `aggregate_robust` dispatch | 4 (fedavg, trimmed_mean, median, krum) |
| `filter_by_norm` | 3 |
| `filter_by_cosine` | 3 |
| `add_label_noise` | 5 |
| Evaluation loss | 1 (unbiased avg for unequal batch sizes) |
| Epsilon (ε) estimation | 4 (including tighter-than-classic-Gaussian) |

---

## 10. Technical Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10 |
| Deep Learning | PyTorch 2.6.0+cu124 |
| GPU | NVIDIA GTX 1650, CUDA 12.4 |
| FL Framework | Flower (`flwr[simulation]`) ≥ 1.8.0 |
| Parallel Clients | Ray (Flower backend) |
| Dataset | medmnist ≥ 3.0 (PathMNIST) |
| Numerics | NumPy 2.2.6 |
| Plotting | Matplotlib 3.7+ |
| Privacy Accounting | SciPy 1.10+ (RDP accountant for ε) |
| Testing | pytest (29 tests) |

---

## 11. Key Design Decisions

### Filter-Before-Select (not Select-Then-Filter)

If detection ran *after* async selection, a malicious client could respond instantly to guarantee its poisoned update is always included. Running detection on all 6 submissions **before** selecting the fastest 4 removes this attack vector entirely.

### Coordinate-Wise Median as Cosine Reference

Using the mean as the cosine similarity reference is self-defeating under attack because malicious updates shift the mean toward the attack direction. The **coordinate-wise median** is Byzantine-robust as long as fewer than 50% of clients are adversarial — here only 20% are malicious.

### Memory-Mapped Data Loading

PathMNIST stored as NPZ. Each Ray actor opens the file with `mmap_mode="r"` — the OS shares physical memory pages across processes. Peak per-actor RAM is **~10 MB** (one client's ~900-sample slice) instead of **847 MB** (full dataset as tensor). This fix eliminated OOM crashes.

### Staleness Weighting Formula

$$\text{weight} = \frac{1}{1 + \Delta_{\text{rounds}}}$$

A client returning an update 3 rounds late gets weight 0.25 vs 1.0 for a fresh update — stale knowledge contributes less without being discarded entirely.

### Flower CID vs Partition ID

In Flower ≥ 1.8 simulation mode, `client.cid` is a large UUID integer (e.g., `13236999372251260638`), **not** the partition ID (0–9). Role assignment is done by passing CSV string ID sets in FitIns config (`"malicious_ids"="0,1"` etc.) and having each client self-identify using its own `partition_id` from `context.node_config`.

---

## 12. Output Files

### Core Experiment Plots (from `main.py`)

| File | Content |
|------|---------|
| `results/convergence.png` | Accuracy convergence — Exp A, B, C, D |
| `results/loss_curves.png` | Cross-entropy loss — Exp A, B, C, D |
| `results/dp_tradeoff.png` | Accuracy: Exp C (no DP) vs Exp D (DP) |
| `results/attack_impact.png` | Accuracy: clean vs attacked vs defended |
| `results/dropout_reliability.png` | Async buffer fill rate under unreliable clients |
| `results/detection_rate.png` | Per-round flagged client IDs |
| `results/heterogeneity.png` | non-IID (α=0.5) vs near-IID (α=1000) |
| `results/participation_rate.png` | Final accuracy vs client participation rate (Exp F) |
| `results/centralized_vs_fl.png` | Centralised baseline vs AsyncRobustFL overlay |

### Core Structured Metrics (from `main.py`)

| File | Content |
|------|---------|
| `results/fl_metrics.csv` | Per-round FL metrics: experiment, round, accuracy, loss, round_time_secs |
| `results/fl_metrics.json` | Same in JSON format |
| `results/centralized_accuracy.csv` | Centralised per-epoch metrics |

### Research Experiment Outputs (from `experiments.py`)

| File | Content |
|------|---------|
| `results/defense_comparison.png` | All defense methods vs attack (mean ± std) |
| `results/defense_comparison.csv` | Per-round mean/std per defense method |
| `results/defense_comparison.json` | Same in JSON |
| `results/sensitivity_alpha.png` | Accuracy curves for Dirichlet α sweep |
| `results/sensitivity_alpha.{csv,json}` | Per-round mean ± std per α value |
| `results/sensitivity_byzantine.png` | Accuracy curves for Byzantine % sweep |
| `results/sensitivity_byzantine.{csv,json}` | Per-round mean ± std per Byzantine fraction |
| `results/sensitivity_buffer.png` | Accuracy curves for async buffer size sweep |
| `results/sensitivity_buffer.{csv,json}` | Per-round mean ± std per buffer size |
| `results/multiseed_stats.png` | Mean ± std curves for 3-seed runs |
| `results/multiseed_stats.csv` | Per-round mean/std per experiment (seeds: 42, 123, 999) |
| `results/multiseed_stats.json` | Same in JSON |
| `results/failure_threshold.png` | Final accuracy vs Byzantine fraction (failure point marked) |
| `results/failure_threshold.csv` | Per-fraction final accuracy + failure flag |
| `results/failure_threshold.json` | Failure point + full sweep data |
| `results/communication_cost.png` | Accuracy vs comm rounds and vs cumulative MB |
| `results/rounds_to_target.png` | Rounds to reach 80% accuracy per defense method |
| `results/communication_cost.csv` | Per-round cumulative MB + accuracy per method |
| `results/communication_cost.json` | Model size, total bytes, rounds-to-target per method |

### Report

| File | Content |
|------|---------|
| `NeuralX_FL_IEEE_Report.docx` | IEEE-style 6-page Word document (run `generate_ieee_report.py`) |

---

## Summary

NeuralX-FL demonstrates that a carefully engineered FL system can:

1. **Withstand Byzantine scaling attacks** — detection filters reduce attack damage from 78% to near zero in the best rounds
2. **Operate asynchronously** — the buffer-based async protocol handles heterogeneous hospital speeds without stalling
3. **Recover 81.41% accuracy** under simultaneous attack + dropout (vs 83.51% clean baseline — only 2.1% gap)
4. **Quantify the privacy-utility trade-off** — DP with ε ≈ 0.6239 provides strong privacy guarantees but reduces accuracy to 38.7% at these parameter settings
5. **Scale without data leakage** — no raw images ever leave a hospital node; only compressed weight updates are shared

---

---

## 13. Experimental Methodology

All experiments follow a common protocol:

- **Controlled variable design**: Each experiment changes exactly one condition (attack presence, defence, DP, heterogeneity, participation rate) relative to a fixed baseline configuration.
- **Reproducibility**: SEED=42 is set for `random`, `numpy`, `torch` at the start of every experiment and simulation call. Dirichlet partitioning uses seed=42 (train) and seed=43 (test).
- **Evaluation**: A shared global test set (7,180 PathMNIST images) is evaluated after every round via the Flower `evaluate()` callback — no leakage from local test sets.
- **Centralised baseline**: `centralized.py` trains PathologyNet on the full 89,996-image train set (no federation, no privacy) to establish a theoretical accuracy upper bound.
- **Participation rate sweep (Exp F)**: `clients_per_round` is varied over 5 values (20%, 40%, 60%, 80%, 100% of 10 clients) while holding all other parameters fixed, to characterise the accuracy-communication trade-off.

### Metrics logged per round

| Metric | Source |
|--------|--------|
| Global test accuracy | `EvaluateFn.evaluate()` callback |
| Global test loss | `EvaluateFn.evaluate()` callback |
| Wall-clock time per round | `time.perf_counter()` delta in `EvaluateFn` |
| Flagged clients (Byzantine / noisy) | Returned by `strategy.aggregate_fit()` |
| Buffer dropout count | Returned by `strategy.aggregate_fit()` |

---

## 14. Reproducibility

### Steps to reproduce all results

```powershell
# 1. Clone and enter the project
git clone <repository-url>
cd async_robust_fl

# 2. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install CUDA PyTorch (must be first)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. IMPORTANT: Ensure simulation mode is active
# In config.py, confirm: USE_REAL_NETWORK = False

# 6. Run core experiments (~6 hours on GTX 1650)
python main.py

# 7. Run research experiments (all groups, ~12-18 hours for 3 seeds)
python experiments.py
# Or run a single group:
python experiments.py --group defense

# 8. Generate IEEE Word report
python generate_ieee_report.py
```

### Expected outputs after full run

```
results/
  convergence.png, loss_curves.png, dp_tradeoff.png,
  attack_impact.png, dropout_reliability.png, detection_rate.png,
  heterogeneity.png, participation_rate.png, centralized_vs_fl.png,
  fl_metrics.csv, fl_metrics.json, centralized_accuracy.csv

NeuralX_FL_IEEE_Report.docx
```

### Critical configuration note

`config.py` contains `USE_REAL_NETWORK: bool`. This must be `False` for simulation runs:

```python
# config.py
USE_REAL_NETWORK: bool = False   # ← required for main.py / centralized.py
```

Set to `True` only for real Tailscale multi-machine deployment (port 9092).

---

## 15. Limitations

| Limitation | Details | Workaround / Fix |
|------------|---------|------------------|
| Trimmed mean trims zero elements | `floor(0.1 × 4) = 0` — no trimming at buffer size 4 | Set `TRIM_FRACTION=0.25` or `ASYNC_BUFFER_SIZE≥8` |
| Krum invalid at n=4, f=1 | Krum requires `n > 2f+2 = 4`; not met at current buffer | Only use trimmed_mean/fedavg aggregation at BUFFER=4 |
| DP noise scale is small | `σ=0.1, C=0.1` gives effective noise=1.0; standard DP-FL uses `σ≥1.0` | Increase σ for stronger formal privacy |
| GPU-only model | PathologyNet raises `RuntimeError` if no CUDA device | Add CPU fallback path in `model.py` |
| 20-round limit | Full convergence may need ≥50 rounds on complex partitions | Increase `NUM_ROUNDS` in config.py |
| Centralised baseline has no privacy | Accesses full dataset — establishes ceiling only, not deployable | Use as reference, not production component |
| Research experiments runtime | 3 seeds × 7 groups ≈ 60+ FL simulation runs (~12–18 hours) | Run `--group` flag to execute one group at a time |

---

*Research project. All experiments validated. 29/29 tests passing. IEEE report: run `python generate_ieee_report.py`.*
