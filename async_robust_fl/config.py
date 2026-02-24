"""
config.py — Central configuration for the Async Robust FL system.

All tuneable constants live here. No magic numbers anywhere else.
To change any experiment parameter, edit ONLY this file.
"""

import os   # used only for RESULTS_DIR absolute path

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED: int = 42

# ---------------------------------------------------------------------------
# Federation topology
# ---------------------------------------------------------------------------
NUM_CLIENTS: int = 10          # Hospital-0 … Hospital-9
CLIENTS_PER_ROUND: int = 6     # How many hospitals are sampled each round
ASYNC_BUFFER_SIZE: int = 4     # Aggregate once this many clean updates arrive;
                                # the remaining 2 of 6 are treated as stragglers

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
NUM_ROUNDS: int = 20
LOCAL_EPOCHS_WARMUP: int = 2   # Rounds 1-3  — short local training while model
LOCAL_EPOCHS_MAIN: int = 5     # Rounds 4+   — longer once model has a base
LEARNING_RATE: float = 1e-3    # Adam optimiser learning rate (all clients + centralised)

# Centralised baseline: equivalent training budget to FL
# 20 rounds × 5 local epochs ≈ 100 total epochs when run centrally.
CENTRALIZED_EPOCHS: int = 100  # Full-data epochs for the centralised baseline run

# Participation rate sweep (Experiment F) — values expressed as fractions of NUM_CLIENTS.
# E.g. 0.2 means 2 out of 10 clients participate each round.
PARTICIPATION_RATES: tuple = (0.2, 0.4, 0.6, 0.8, 1.0)

# ---------------------------------------------------------------------------
# Data — PathMNIST
# ---------------------------------------------------------------------------
NUM_CLASSES: int = 9
BATCH_SIZE: int = 32
DIRICHLET_ALPHA: float = 0.5   # 0.5 = realistic non-IID; 1000.0 = near-IID

# Pre-computed channel statistics for PathMNIST (RGB)
PATHMNIST_MEAN: tuple = (0.7405, 0.5330, 0.7058)
PATHMNIST_STD: tuple  = (0.1270, 0.1770, 0.1256)

# ---------------------------------------------------------------------------
# Client taxonomy (10 hospitals total)
# ---------------------------------------------------------------------------
MALICIOUS_CLIENT_IDS: frozenset  = frozenset({0, 1})   # 20% — adversarial (scaling)
NOISY_CLIENT_IDS: frozenset      = frozenset({4, 5})   # 20% — 30% label noise
UNRELIABLE_CLIENT_IDS: frozenset = frozenset({6, 7})   # 20% — random dropout

# ---------------------------------------------------------------------------
# Attack simulation
# ---------------------------------------------------------------------------
ATTACK_TYPE: str     = "scaling"   # "scaling" | "random" | "sign_flip"
ATTACK_SCALE: float  = 50.0        # Multiplier for scaling attack — large enough
                                    # that FedAvg without filtering is visibly hurt
NOISE_RATE: float    = 0.3         # Fraction of labels flipped for noisy clients
DROPOUT_PROB: float  = 0.4         # Probability a unreliable client drops each round
DELAY_SCALE: float   = 2.0         # Exponential distribution scale for simulated delay

# ---------------------------------------------------------------------------
# Robust aggregation
# ---------------------------------------------------------------------------
AGGREGATION_METHOD: str = "trimmed_mean"   # "fedavg" | "trimmed_mean" | "median" | "krum"
TRIM_FRACTION: float    = 0.1              # Remove top/bottom 10% per dimension
NORM_THRESHOLD: float   = 3.0             # Reject if norm > 3× median norm
COSINE_THRESHOLD: float = 0.0             # Reject if cosine similarity < 0

# ---------------------------------------------------------------------------
# Trust Scoring & Dynamic Group Formation
# ---------------------------------------------------------------------------
# Each client starts with full trust. Score decays when flagged by detection
# filters and slowly recovers on honest submissions.

TRUST_SCORE_INIT: float         = 1.0   # Starting trust for every client
TRUST_DECAY: float              = 0.5   # score *= 0.5 when flagged this round
TRUST_GROWTH: float             = 0.1   # score += 0.1 for honest submission
TRUST_EXCLUSION_THRESHOLD: float = 0.3  # Below this → excluded from aggregation
                                         # (two flagged rounds from 1.0 → 0.5 → 0.25
                                         #  puts a client below threshold)

# Gradient-direction similarity required for two honest clients to be placed
# in the same sub-group (cosine ≥ 0.5 means directions within ~60° of each other)
GROUP_COSINE_THRESHOLD: float   = 0.5

# ---------------------------------------------------------------------------
# Real Network Mode
# ---------------------------------------------------------------------------
# Set USE_REAL_NETWORK = True  → your PC becomes the server, friends run
#                                run_client.py on their laptops.
# Set USE_REAL_NETWORK = False → standard Ray simulation (default).
# Everything else (strategy, detection, trust scoring) is identical.

USE_REAL_NETWORK: bool = False

# Your PC listens on this address (server side)
SERVER_BIND: str = "0.0.0.0:9092"

# Friends pass this address via --server-address when running run_client.py.
# With ngrok the address changes every session — no need to edit this file.
# After starting server.py, run:  ngrok tcp 9092
# ngrok will print something like:  tcp://0.tcp.ngrok.io:12345
# Give that string (without tcp://) to your friends.
SERVER_HOST: str = "100.73.144.110"  # fallback only — use --server-address

# Real mode has only 2 real client laptops (your 2 friends)
REAL_NUM_CLIENTS: int       = 2
REAL_CLIENTS_PER_ROUND: int = 2
REAL_ASYNC_BUFFER_SIZE: int = 2

# Data directory for files handed to friends.
# SEPARATE from data/ (simulation data) — keeps simulation and real-network data apart.
# Friends copy their pathmnist.npz here before running run_client.py.
SHARED_DATA_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shared_data")

# ---------------------------------------------------------------------------
# Differential Privacy
# ---------------------------------------------------------------------------
DP_NOISE_MULTIPLIER: float   = 0.1
DP_CLIPPING_NORM: float      = 0.1
DP_DELTA: float              = 1e-5

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
# Absolute path so results are always written next to this file regardless
# of the working directory from which main.py is invoked.
RESULTS_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
