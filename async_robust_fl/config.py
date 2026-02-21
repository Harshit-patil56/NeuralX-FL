"""
config.py — Central configuration for the Async Robust FL system.

All tuneable constants live here. No magic numbers anywhere else.
To change any experiment parameter, edit ONLY this file.
"""

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
ATTACK_SCALE: float  = 10.0        # Multiplier for scaling attack
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
# Differential Privacy
# ---------------------------------------------------------------------------
DP_NOISE_MULTIPLIER: float   = 0.1
DP_CLIPPING_NORM: float      = 0.1
DP_DELTA: float              = 1e-5

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
RESULTS_DIR: str = "results"    # All PNGs + summary text written here
