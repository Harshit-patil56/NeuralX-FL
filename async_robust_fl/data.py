"""
data.py — PathMNIST loading, Dirichlet partitioning, and label noise injection.

Design decisions:
  - PathMNIST downloaded once to ./data; kept in a module-level cache so every
    Flower client process reuses the same tensors without re-downloading.
  - Dirichlet split is deterministic given the same seed — guarantees
    reproducibility across all experiment runs.
  - Label noise is applied ONLY to the training labels of designated noisy
    hospitals; the validation/test split remains clean.
  - All batches returned are tuples (image_tensor, label_tensor) — i.e.
    TensorDataset format — so model.py's train() / evaluate_model() work
    without any dict-handling branch.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, ToTensor, Normalize

from config import (
    NUM_CLIENTS,
    NUM_CLASSES,
    BATCH_SIZE,
    DIRICHLET_ALPHA,
    PATHMNIST_MEAN,
    PATHMNIST_STD,
)

# ---------------------------------------------------------------------------
# Transform — applied once when building the cache
# ---------------------------------------------------------------------------
_TRANSFORM = Compose([ToTensor(), Normalize(PATHMNIST_MEAN, PATHMNIST_STD)])

# ---------------------------------------------------------------------------
# In-memory cache: { "train": (images, labels), "test": (images, labels) }
# Downloading happens at most once per Python process.
# ---------------------------------------------------------------------------
_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}


def _load_split(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Download (if needed) and cache PathMNIST for `split` ∈ {'train','test','val'}.

    Returns:
        images : Tensor[N, 3, 28, 28]
        labels : Tensor[N]            (dtype=torch.long)
    """
    if split in _CACHE:
        return _CACHE[split]

    try:
        from medmnist import PathMNIST
    except ImportError as exc:
        raise ImportError(
            "medmnist is not installed. Run: pip install medmnist>=3.0.0"
        ) from exc

    dataset = PathMNIST(
        split=split,
        transform=_TRANSFORM,
        download=True,
        root="./data",
    )

    imgs, lbls = [], []
    for img, lbl in dataset:
        imgs.append(img)
        # medmnist returns labels as 1-element arrays or scalars
        lbls.append(int(lbl.item() if hasattr(lbl, "item") else lbl[0]))

    result: tuple[torch.Tensor, torch.Tensor] = (
        torch.stack(imgs),
        torch.tensor(lbls, dtype=torch.long),
    )
    _CACHE[split] = result
    return result


# ---------------------------------------------------------------------------
# Dirichlet partitioner
# ---------------------------------------------------------------------------

def _dirichlet_split(
    labels: torch.Tensor,
    num_clients: int,
    alpha: float,
    seed: int,
) -> list[list[int]]:
    """Partition sample indices into `num_clients` subsets via Dirichlet(alpha).

    A low alpha (~0.5) creates strong class imbalance per hospital (non-IID).
    A high alpha (~1000) creates near-uniform class distribution (IID).

    Returns:
        client_idx: list[list[int]] where client_idx[i] are the indices for hospital i.
    """
    rng = np.random.default_rng(seed)
    n_classes = int(labels.max().item()) + 1
    label_np = labels.numpy()

    # Indices of each class across the full dataset
    class_indices = [np.where(label_np == c)[0] for c in range(n_classes)]

    client_idx: list[list[int]] = [[] for _ in range(num_clients)]

    for c_idx in class_indices:
        rng.shuffle(c_idx)
        # Sample proportions from Dirichlet for this class
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        sizes = (proportions * len(c_idx)).astype(int)
        # Fix rounding: last client absorbs the remainder
        sizes[-1] = len(c_idx) - sizes[:-1].sum()

        splits = np.split(c_idx, np.cumsum(sizes[:-1]))
        for cid, chunk in enumerate(splits):
            client_idx[cid].extend(chunk.tolist())

    return client_idx


# ---------------------------------------------------------------------------
# Label noise injection
# ---------------------------------------------------------------------------

def add_label_noise(
    labels: torch.Tensor,
    noise_rate: float,
    num_classes: int = NUM_CLASSES,
    seed: int = 0,
) -> torch.Tensor:
    """Randomly flip `noise_rate` fraction of labels to a *different* class.

    Simulates inter-annotator disagreement or transcription errors in hospital
    records — NOT an adversarial attack.

    Args:
        labels     : Original label tensor (will not be modified in-place).
        noise_rate : Fraction of labels to flip, e.g. 0.3 for 30%.
        num_classes: Total number of classes (determines the wrong-class pool).
        seed       : RNG seed for reproducibility.

    Returns:
        New label tensor with noise applied.
    """
    if noise_rate <= 0.0:
        return labels

    rng = torch.Generator()
    rng.manual_seed(seed)

    noisy = labels.clone()
    n_noisy = int(len(labels) * noise_rate)
    noisy_idx = torch.randperm(len(labels), generator=rng)[:n_noisy]

    for i in noisy_idx:
        original = noisy[i].item()
        # Choose uniformly from the OTHER classes
        wrong = (original + int(torch.randint(1, num_classes, (1,), generator=rng).item())) % num_classes
        noisy[i] = wrong

    return noisy


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(
    partition_id: int,
    num_clients: int = NUM_CLIENTS,
    noise_rate: float = 0.0,
    dirichlet_alpha: float = DIRICHLET_ALPHA,
) -> tuple[DataLoader, DataLoader]:
    """Return (trainloader, testloader) for a single hospital partition.

    Args:
        partition_id    : Hospital index in [0, num_clients).
        num_clients     : Total number of hospitals.
        noise_rate      : Fraction of training labels to flip (0 = clean).
        dirichlet_alpha : Heterogeneity parameter.
                          0.5 → realistic non-IID  (default)
                          1000.0 → near-IID  (Experiment E)

    Data never leaves this function unpartitioned — each hospital only
    receives its own slice.
    """
    train_imgs, train_labels = _load_split("train")
    test_imgs,  test_labels  = _load_split("test")

    train_idx = _dirichlet_split(train_labels, num_clients, dirichlet_alpha, seed=42)
    test_idx  = _dirichlet_split(test_labels,  num_clients, dirichlet_alpha, seed=43)

    t_imgs   = train_imgs[train_idx[partition_id]]
    t_labels = train_labels[train_idx[partition_id]]
    v_imgs   = test_imgs[test_idx[partition_id]]
    v_labels = test_labels[test_idx[partition_id]]

    if noise_rate > 0.0:
        t_labels = add_label_noise(t_labels, noise_rate, seed=partition_id)

    trainloader = DataLoader(
        TensorDataset(t_imgs, t_labels),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,   # keeps simulation stable across platforms
        pin_memory=False,
    )
    testloader = DataLoader(
        TensorDataset(v_imgs, v_labels),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return trainloader, testloader


def load_global_test() -> DataLoader:
    """Full PathMNIST test set for server-side global evaluation.

    This is NEVER shown to any individual hospital — it is only used by the
    server's evaluate_fn to measure the quality of the global model.
    """
    test_imgs, test_labels = _load_split("test")
    return DataLoader(
        TensorDataset(test_imgs, test_labels),
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
