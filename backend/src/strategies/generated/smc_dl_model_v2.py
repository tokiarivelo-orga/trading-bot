"""Multi-head network for the three-outcome + give-back label set.

Heads, and what each one is for:

  ``secure``  2 sigmoids (long, short) — P(+secure_r x risk before the stop).
              This is the head that decides whether to trade at all: the
              engine's secure-base trailing books ~+0.2R, and break-even on
              that leg is 1/(1+0.2) = 0.833.
  ``target``  2 sigmoids (long, short) — P(+target_r x risk before the stop).
              Combined with ``secure`` it gives expected R exactly (see
              ``smc_dl_labels_v2.expected_r``); alone it is the quantity the
              v1 model predicted, which does not drive P&L on this engine.
  ``revert``  3-class logits {down-first, up-first, neither} over a short
              symmetric barrier — the give-back / change-of-character head
              that drives exit management.
  ``mfe``     2 softplus outputs (long, short) — expected favourable
              excursion in R, used to place a take-profit that the market
              can actually reach rather than a fixed ATR multiple.

LayerNorm, not BatchNorm
────────────────────────────────────────────────────────────────────────
``SmcMultiTaskNet`` (v1) uses ``BatchNorm1d``. Batch norm's running
statistics are estimated on the training distribution, so when gold moves
from 1,900 to 4,000 — or simply into a different volatility regime — every
activation is normalised by stale statistics and the whole network shifts.
LayerNorm normalises per sample, which makes single-sample live inference
identical in form to training and removes that particular train/serve
coupling. The features are scale-free for the same reason; this is the same
decision one layer down.

Sandbox: on ``ALLOWED_IMPORT_MODULES`` so the generated strategy can import
it. Pure ``torch.nn`` module definitions, no I/O — the weights are loaded by
the strategy, not here.
"""

from __future__ import annotations

import torch
from torch import nn

# Index meaning for the revert head's 3-way softmax. Mirrors
# `smc_dl_labels_v2._revert_labels`.
REVERT_DOWN_FIRST = 0
REVERT_UP_FIRST = 1
REVERT_NEITHER = 2


class SmcExitAwareNet(nn.Module):
    """Shared trunk, four task heads. Deliberately small: the honest sample
    count here is ~100k M5 bars with heavily autocorrelated labels, which is
    far less independent evidence than the row count suggests."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.25) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        head_dim = max(hidden_dim // 2, 16)
        self.head_secure = _head(hidden_dim, head_dim, 2)
        self.head_target = _head(hidden_dim, head_dim, 2)
        self.head_revert = _head(hidden_dim, head_dim, 3)
        self.head_mfe = _head(hidden_dim, head_dim, 2)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(secure_logits, target_logits, revert_logits, mfe_r)``.

        Logits, not probabilities — the loss functions want logits for
        numerical stability, and calibration (temperature scaling in the
        trainer) also operates on logits. Callers wanting probabilities
        apply ``sigmoid``/``softmax`` themselves; ``predict`` does it for
        them.
        """
        features = self.trunk(x)
        return (
            self.head_secure(features),
            self.head_target(features),
            self.head_revert(features),
            nn.functional.softplus(self.head_mfe(features)),
        )

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Probabilities and expected excursions, ready for the EV formula.

        ``p_target`` is *not* clamped to ``p_secure`` here — that clamp is
        part of the expected-R arithmetic and lives in
        ``smc_dl_labels_v2.expected_r`` so every consumer shares one copy.
        """
        secure_logits, target_logits, revert_logits, mfe_r = self(x)
        return {
            "p_secure": torch.sigmoid(secure_logits),
            "p_target": torch.sigmoid(target_logits),
            "p_revert": torch.softmax(revert_logits, dim=-1),
            "mfe_r": mfe_r,
        }


def _head(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )
