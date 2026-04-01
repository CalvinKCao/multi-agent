"""
DRC agent for Sokoban / Boxoban.

Architecture (matches Bush et al. 2025 / Guez et al. 2019 Appendix E.3):
    obs -> encoder -> (C, H, W) -> DRC stack (skip connections + pool-and-inject)
    -> cat(flatten(final_h), flatten(encoder)) -> MLP -> policy + value heads

Spatial correspondence preserved throughout: every Conv2d uses same-padding.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from drc_sokoban.models.conv_lstm import DRCStack, LayerStates


class DRCAgent(nn.Module):
    """
    DRC agent: encoder + repeated ConvLSTM stack + policy/value heads.

    With default settings (skip_connections=True, pool_and_inject=True,
    concat_encoder=True), matches Bush et al. 2025 Appendix E.3.
    Set all three to False to recover the original minimal stack.

    Defaults: obs_channels=7, hidden_channels=32, num_layers=3, num_ticks=3,
    num_actions=4.
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
        skip_connections: bool = True,
        pool_and_inject: bool = True,
        concat_encoder: bool = True,
    ):
        super().__init__()
        self.H = H
        self.W = W
        self.hidden_channels = hidden_channels
        self.num_actions = num_actions
        self.concat_encoder = concat_encoder

        self.encoder = nn.Sequential(
            nn.Conv2d(obs_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.drc = DRCStack(
            input_channels=hidden_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            num_ticks=num_ticks,
            skip_connections=skip_connections,
            pool_and_inject=pool_and_inject,
        )

        # paper: cat(h^D, i_t) -> FC -> ReLU -> policy / value
        flat_dim = hidden_channels * H * W
        if concat_encoder:
            flat_dim *= 2
        head_hidden = 256
        self.head_fc = nn.Sequential(
            nn.Linear(flat_dim, head_hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(head_hidden, num_actions)
        self.value_head = nn.Linear(head_hidden, 1)

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
        encoded = self.encoder(obs)
        new_hidden_states, all_tick_hiddens = self.drc(encoded, hidden_states)

        final_h = all_tick_hiddens[-1][-1][0]
        if self.concat_encoder:
            flat = torch.cat([final_h.flatten(1), encoded.flatten(1)], dim=1)
        else:
            flat = final_h.flatten(start_dim=1)

        features = self.head_fc(flat)
        logits = self.policy_head(features)
        value = self.value_head(features)

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
