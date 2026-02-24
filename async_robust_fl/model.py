"""
model.py — PathologyNet CNN + training / evaluation utilities.

Architecture: 3-channel RGB CNN for 28×28 PathMNIST slides → 9 tissue classes.
  Class 0: adipose tissue       Class 5: normal colon mucosa
  Class 1: background           Class 6: colorectal adenocarcinoma
  Class 2: debris               Class 7: smooth muscle
  Class 3: lymphocytes          Class 8: stroma
  Class 4: mucus

All functions are stateless and importable independently.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

from config import LEARNING_RATE

# ---------------------------------------------------------------------------
# Device — GPU only; raises immediately if no CUDA-capable device is found
# ---------------------------------------------------------------------------
if not torch.cuda.is_available():
    raise RuntimeError(
        "No CUDA-capable GPU found. "
        "Install a CUDA-enabled PyTorch build: "
        "pip install torch --extra-index-url https://download.pytorch.org/whl/cu124"
    )
DEVICE: torch.device = torch.device("cuda")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PathologyNet(nn.Module):
    """Lightweight CNN for colon-tissue classification on 28×28 RGB patches.

    Input : Tensor[B, 3, 28, 28]   (normalised PathMNIST slides)
    Output: Tensor[B, 9]           (raw logits for 9 tissue classes)
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1   = nn.Conv2d(3, 32, kernel_size=3, padding=1)   # → [B,32,28,28]
        self.conv2   = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # → [B,64,28,28]
        self.pool    = nn.MaxPool2d(2, 2)                            # halves spatial dims
        self.fc1     = nn.Linear(64 * 7 * 7, 128)
        self.fc2     = nn.Linear(128, 9)
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # [B,32,14,14]
        x = self.pool(F.relu(self.conv2(x)))   # [B,64,7,7]
        x = x.view(x.size(0), -1)             # [B,3136]
        x = F.relu(self.fc1(x))               # [B,128]
        x = self.dropout(x)
        return self.fc2(x)                    # [B,9]


# ---------------------------------------------------------------------------
# Parameter serialisation helpers (Flower protocol)
# ---------------------------------------------------------------------------

def get_weights(net: nn.Module) -> List:
    """Extract all parameters as a list of CPU numpy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters: List) -> None:
    """Load a list of numpy arrays back into the model in-place.

    Raises ValueError if the parameter count doesn't match the model.
    """
    keys = list(net.state_dict().keys())
    if len(keys) != len(parameters):
        raise ValueError(
            f"Parameter count mismatch: model has {len(keys)} tensors, "
            f"received {len(parameters)}."
        )
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, parameters)}
    net.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    net: nn.Module,
    trainloader: torch.utils.data.DataLoader,
    epochs: int,
    device: Optional[torch.device] = None,
) -> None:
    """Train `net` in-place for `epochs` local epochs.

    Supports both dict batches (HuggingFace datasets) and tuple batches
    (TensorDataset / torchvision).
    """
    device = device or DEVICE
    net.to(device)
    net.train()

    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for batch in trainloader:
            if isinstance(batch, dict):
                images, labels = batch["image"], batch["label"]
            else:
                images, labels = batch[0], batch[1]

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    net: nn.Module,
    testloader: torch.utils.data.DataLoader,
    device: Optional[torch.device] = None,
) -> tuple[float, float]:
    """Evaluate `net` and return (accuracy, avg_loss).

    Uses ``reduction='sum'`` on the criterion so ``loss_sum`` is the exact
    total unnormalised loss across all samples.  Dividing by ``total`` at the
    end gives a true per-sample mean that is unbiased for batches of any size,
    including the final short batch.

    Supports dict and tuple batch formats.
    """
    device = device or DEVICE
    net.to(device)
    net.eval()

    # reduction='sum' → each call returns the summed (not mean) loss for the
    # batch.  We accumulate directly and divide once at the end.
    criterion = nn.CrossEntropyLoss(reduction="sum")
    correct, total = 0, 0
    loss_sum = 0.0

    with torch.no_grad():
        for batch in testloader:
            if isinstance(batch, dict):
                images, labels = batch["image"], batch["label"]
            else:
                images, labels = batch[0], batch[1]

            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss_sum += criterion(outputs, labels).item()   # batch sum
            _, predicted = torch.max(outputs, dim=1)
            total   += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total if total > 0 else 0.0
    avg_loss = loss_sum / max(total, 1)   # true per-sample mean
    return accuracy, avg_loss
