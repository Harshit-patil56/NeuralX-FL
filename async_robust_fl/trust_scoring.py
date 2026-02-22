"""
trust_scoring.py — Per-client trust scoring and dynamic group formation.

Trust Score (per client, range [0.0, 1.0]):
    - Initialised at TRUST_SCORE_INIT (1.0 — benefit of the doubt).
    - Flagged by detection filter this round → score *= TRUST_DECAY (0.5).
    - Honest submission this round          → score += TRUST_GROWTH (0.1).
    - Did not submit this round             → score unchanged.
    - score < TRUST_EXCLUSION_THRESHOLD     → excluded from aggregation.

    Why exponential decay on penalty?  A single-round mis-classification should
    not permanently ban a hospital.  Starting at 1.0, two consecutive flaggings
    bring the score to 0.5 → 0.25, which is below TRUST_EXCLUSION_THRESHOLD
    (0.3).  One honest round after that: 0.25 + 0.1 = 0.35 → rehab in progress.
    Three more honest rounds → 0.65 — back in good standing.

Dynamic Group Formation:
    After each round the strategy calls compute_groups() with the set of clean
    update dicts that survived norm + cosine filtering.  Only clients whose
    trust score is at or above TRUST_EXCLUSION_THRESHOLD are considered.

    Algorithm: pairwise cosine similarity → adjacency graph →
    BFS connected components → sorted by size descending.

    Why connected components instead of k-means?
        - n per round is small (4–6 clients): O(n²) is negligible.
        - k is unknown and varies each round.
        - Threshold-based edges are directly interpretable and tied to the
          existing cosine values already computed in detection.py.
        - Group 0 is always the largest component → the main federation.
          Smaller groups suggest hospitals sharing similar data distributions
          (e.g., same tissue class) and could be candidates for personalised
          model branches in future work.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Set

import numpy as np

from config import (
    NUM_CLIENTS,
    TRUST_SCORE_INIT,
    TRUST_DECAY,
    TRUST_GROWTH,
    TRUST_EXCLUSION_THRESHOLD,
    GROUP_COSINE_THRESHOLD,
)


class TrustScoreTracker:
    """Maintains per-client trust scores and dynamic group assignments.

    Attributes:
        scores  : Dict[client_id -> float].  Values in [0.0, 1.0].
        history : List of per-round snapshots: scores, groups, excluded IDs.
    """

    def __init__(self, num_clients: int = NUM_CLIENTS) -> None:
        # Initialise all known clients with full trust
        self.scores:  Dict[int, float] = {
            i: TRUST_SCORE_INIT for i in range(num_clients)
        }
        self.history: List[Dict] = []

    # ------------------------------------------------------------------
    # Trust score update (called once per round)
    # ------------------------------------------------------------------

    def update(
        self,
        submitted_ids: List[int],
        flagged_ids:   List[int],
        server_round:  int,
    ) -> Dict[int, float]:
        """Update EMA trust scores for clients that participated this round.

        Clients that did not submit (dropouts) have their scores left
        unchanged — we do not penalise non-participation separately,
        since unreliable clients are not actively poisoning.

        Args:
            submitted_ids : Partition IDs of all clients that submitted an
                            update this round (including those later flagged).
            flagged_ids   : IDs caught by norm or cosine filter this round.
            server_round  : FL round number (stored in history).

        Returns:
            Copy of all trust scores after the update.
        """
        flagged_set   = set(flagged_ids)
        submitted_set = set(submitted_ids)

        for cid in submitted_set:
            if cid < 0:
                continue   # -1 sentinel for unknown partition ID — skip

            if cid in flagged_set:
                # Penalise: exponential decay
                self.scores[cid] = max(0.0, self.scores[cid] * TRUST_DECAY)
            else:
                # Reward: linear recovery
                self.scores[cid] = min(1.0, self.scores[cid] + TRUST_GROWTH)

        snapshot = dict(self.scores)
        self.history.append({
            "round":    server_round,
            "scores":   snapshot,
            "flagged":  sorted(flagged_set),
            "excluded": self.excluded_clients(),
        })
        return snapshot

    # ------------------------------------------------------------------
    # Dynamic group formation (called once per round)
    # ------------------------------------------------------------------

    def compute_groups(
        self,
        updates: List[Dict],
        server_round: int,
    ) -> Dict[int, List[int]]:
        """Cluster honest clients into sub-groups by gradient direction.

        Only clients whose current trust score is at or above
        TRUST_EXCLUSION_THRESHOLD are included in the clustering step.
        These are the clients that survived both the trust filter and the
        per-round norm/cosine filters, i.e. genuinely clean updates.

        Args:
            updates      : List of update dicts (post norm+cosine filter).
                           Must contain keys 'client_id' and 'params'.
            server_round : Round number (unused here; caller logs result).

        Returns:
            Dict mapping group_id -> [client_ids].
            Group 0 is always the largest component (main federation).
            Returns {} if no trusted updates are present.
        """
        # Keep only trusted clients (double-check — trust filter already
        # removed distrusted clients before detection in strategy.py)
        trusted = [
            u for u in updates
            if int(u["client_id"]) >= 0
            and self.scores.get(int(u["client_id"]), 0.0) >= TRUST_EXCLUSION_THRESHOLD
        ]

        if not trusted:
            return {}

        if len(trusted) == 1:
            return {0: [int(trusted[0]["client_id"])]}

        cids = [int(u["client_id"]) for u in trusted]
        n    = len(cids)

        # Compute unit vectors: flatten all layers → 1-D → normalise
        unit_vecs: List[np.ndarray] = []
        for u in trusted:
            flat = np.concatenate(
                [p.flatten() for p in u["params"]]
            ).astype(np.float64)
            norm = float(np.linalg.norm(flat))
            unit_vecs.append(flat / norm if norm > 1e-10 else flat)

        # Build adjacency: O(n²) — n is at most ASYNC_BUFFER_SIZE (4–6)
        adj: Dict[int, Set[int]] = {i: set() for i in range(n)}
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(np.dot(unit_vecs[i], unit_vecs[j]))
                if sim >= GROUP_COSINE_THRESHOLD:
                    adj[i].add(j)
                    adj[j].add(i)

        # BFS connected components
        visited:    List[bool]      = [False] * n
        components: List[List[int]] = []

        for start in range(n):
            if visited[start]:
                continue
            component: List[int] = []
            queue: deque[int] = deque([start])
            while queue:
                node = queue.popleft()
                if visited[node]:
                    continue
                visited[node] = True
                component.append(cids[node])
                for nb in adj[node]:
                    if not visited[nb]:
                        queue.append(nb)
            components.append(component)

        # Sort descending by size: group 0 = main federation
        components.sort(key=len, reverse=True)
        return {gid: comp for gid, comp in enumerate(components)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def excluded_clients(self) -> List[int]:
        """Return client IDs whose trust score is below the exclusion threshold."""
        return [
            cid for cid, score in self.scores.items()
            if score < TRUST_EXCLUSION_THRESHOLD
        ]

    def is_trusted(self, client_id: int) -> bool:
        """True if the client has sufficient trust to participate."""
        return (
            self.scores.get(client_id, TRUST_SCORE_INIT) >= TRUST_EXCLUSION_THRESHOLD
        )

    def score_snapshot(self) -> Dict[int, float]:
        """Return a copy of the current trust scores (all clients)."""
        return dict(self.scores)
