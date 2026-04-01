"""
Full DRC agent for Sokoban / Boxoban.

Architecture:
    obs (7, 8, 8) → encoder → (32, 8, 8) → DRC stack (D layers, N ticks)
                 → final h (last layer, last tick) → flatten → policy + value heads

Spatial correspondence is preserved throughout: every Conv2d uses same-padding.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from drc_sokoban.models.conv_lstm import DRCStack, LayerStates


class DRCAgent(nn.Module):
    """
    DRC-style agent: encoder + repeated ConvLSTM stack + policy/value heads.

    Spatial layout matches the paper (8×8, 32 ch, 3×3 same-padding), but this is a
    *minimal* stack — not the full Guez/Bush DRC (no pool-and-inject, no bottom-up
    / top-down skips, readout does not concat encoder with final h).  See
    PAPER_ALIGNMENT.md vs arXiv:2504.01871 Appendix E.3.

    Defaults: obs_channels=7, hidden_channels=32, num_layers=3, num_ticks=3,
    num_actions=4 (use num_layers=2 for faster PoC).
    """

    def __init__(
        self,
        obs_channels: int = 7,
        hidden_channels: int = 32,
        num_layers: int = 3,
        num_ticks: int = 3,
        num_actions: int = 4,
        H: int = 8,
        W: int = 8,
    ):
        super().__init__()
        self.H = H
        self.W = W
        self.hidden_channels = hidden_channels
        self.num_actions = num_actions

        # Encoder: (7, 8, 8) → (32, 8, 8) — NO striding, NO pooling
        self.encoder = nn.Sequential(
            nn.Conv2d(obs_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # DRC recurrent stack
        self.drc = DRCStack(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_ticks=num_ticks,
        )

        # Output heads: flatten final h then project
        flat_dim = hidden_channels * H * W  # 32 * 8 * 8 = 2048
        self.policy_head = nn.Linear(flat_dim, num_actions)
        self.value_head = nn.Linear(flat_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Policy head: smaller init so initial policy is more uniform
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor,
        hidden_states: LayerStates,
        return_all_ticks: bool = False,
    ):
        """
        Args:
            obs:              (B, 7, 8, 8) symbolic Sokoban observation
            hidden_states:    list of D tuples [(h, c), ...] from previous step
            return_all_ticks: if True return all intermediate tick states
                              (needed for probe data collection, not training)

        Returns (normal mode):
            logits:            (B, num_actions)
            value:             (B, 1)
            new_hidden_states: list of D tuples for next step

        Returns (return_all_ticks=True):
            logits, value, new_hidden_states, all_tick_hiddens
            all_tick_hiddens: list of N tick snapshots
                              all_tick_hiddens[tick][layer] = (h, c)
                              each h/c: (B, 32, 8, 8)
        """
        encoded = self.encoder(obs)  # (B, 32, 8, 8) — spatial dims preserved

        new_hidden_states, all_tick_hiddens = self.drc(encoded, hidden_states)

        # Use h from the last tick, last layer
        final_h = all_tick_hiddens[-1][-1][0]    # (B, 32, 8, 8)
        flat = final_h.flatten(start_dim=1)       # (B, 2048)

        logits = self.policy_head(flat)            # (B, num_actions)
        value = self.value_head(flat)              # (B, 1)

        if return_all_ticks:
            return logits, value, new_hidden_states, all_tick_hiddens
        return logits, value, new_hidden_states

    def init_hidden(self, batch_size: int, device: torch.device = "cpu") -> LayerStates:
        return self.drc.init_hidden(batch_size, self.H, self.W, device)

    def get_value(self, obs: torch.Tensor, hidden_states: LayerStates) -> torch.Tensor:
        """Convenience method returning only the value estimate."""
        _, value, _ = self.forward(obs, hidden_states)
        return value

    @staticmethod
    def mask_hidden(
        hidden_states: LayerStates, dones: torch.Tensor
    ) -> LayerStates:
        """
        Zero out hidden states for completed episodes.

        Args:
            hidden_states: list of D tuples [(h, c), ...]
            dones:         (B,) bool tensor — True for envs that just finished

        Returns:
            masked hidden states with zeroed entries for done envs
        """
        mask = (~dones).float().view(-1, 1, 1, 1)  # (B, 1, 1, 1)
        return [(h * mask, c * mask) for h, c in hidden_states]

    @staticmethod
    def detach_hidden(hidden_states: LayerStates) -> LayerStates:
        """Detach hidden states from the computation graph (for PPO updates)."""
        return [(h.detach(), c.detach()) for h, c in hidden_states]
