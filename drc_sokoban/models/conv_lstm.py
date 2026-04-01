"""
ConvLSTM cell and DRC (Deep Repeated ConvLSTM) stack.

CRITICAL spatial invariant: every Conv2d uses kernel_size=3, padding=1
so that HxW dims are preserved throughout.  Hidden-state position (y, x)
corresponds exactly to board cell (y, x) — required for 1×1 probes.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


HiddenState = Tuple[torch.Tensor, torch.Tensor]   # (h, c) each (B, C, H, W)
LayerStates = List[HiddenState]                    # D elements


class PoolAndInject(nn.Module):
    """
    Global context injection (Guez et al. 2019).
    Mean+max pool h over spatial dims -> linear -> broadcast-add back to h.
    """
    def __init__(self, hidden_channels: int):
        super().__init__()
        self.fc = nn.Linear(2 * hidden_channels, hidden_channels)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, C = h.shape[:2]
        flat = h.reshape(B, C, -1)
        pooled = torch.cat([flat.mean(2), flat.amax(2)], dim=1)
        return h + self.fc(pooled).unsqueeze(-1).unsqueeze(-1)


class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell on spatial (B, C, H, W) tensors.
    Concatenates [x, h_prev] along channels, 4-gate LSTM update.
    3x3 same-padding preserves spatial dims.
    """

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels

        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.gates(combined)
        i_g, f_g, o_g, g_g = gates.chunk(4, dim=1)
        i_g = torch.sigmoid(i_g)
        f_g = torch.sigmoid(f_g)
        o_g = torch.sigmoid(o_g)
        g_g = torch.tanh(g_g)
        c_next = f_g * c_prev + i_g * g_g
        h_next = o_g * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size: int, H: int, W: int, device: torch.device):
        zeros = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        return (zeros, zeros.clone())


class DRCStack(nn.Module):
    """
    Deep Repeated ConvLSTM (DRC) stack -- D layers x N ticks per env step.

    Hidden states persist across ticks within a step AND across env steps;
    this temporal persistence is what enables planning behaviour.

    When skip_connections=True (paper-matching default):
      Layer 0 at tick t: cat(encoder, h_D from tick t-1)   [bottom-up + top-down]
      Layer d>0:         cat(encoder, h_{d-1} from tick t)  [bottom-up]
    When False: original minimal stack (encoder only feeds layer 0).

    all_tick_hiddens exposes every intermediate state for probe training.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_ticks: int,
        skip_connections: bool = True,
        pool_and_inject: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_ticks = num_ticks
        self.hidden_channels = hidden_channels
        self.skip_connections = skip_connections

        self.cells = nn.ModuleList()
        for d in range(num_layers):
            if skip_connections:
                in_ch = input_channels + hidden_channels
            else:
                in_ch = input_channels if d == 0 else hidden_channels
            self.cells.append(ConvLSTMCell(in_ch, hidden_channels))

        if pool_and_inject:
            self.pool_injects = nn.ModuleList(
                [PoolAndInject(hidden_channels) for _ in range(num_layers)]
            )
        else:
            self.pool_injects = None

    def forward(
        self,
        encoded_obs: torch.Tensor,
        hidden_states: LayerStates,
    ) -> Tuple[LayerStates, List[LayerStates]]:
        h_states = [hs[0] for hs in hidden_states]
        c_states = [hs[1] for hs in hidden_states]
        all_tick_hiddens: List[LayerStates] = []

        for _tick in range(self.num_ticks):
            tick_hiddens: LayerStates = []

            for d, cell in enumerate(self.cells):
                if self.skip_connections:
                    if d == 0:
                        prev_h = h_states[self.num_layers - 1]   # top-down
                        x = torch.cat([encoded_obs, prev_h], dim=1)
                    else:
                        x = torch.cat([encoded_obs, h_states[d - 1]], dim=1)
                else:
                    x = encoded_obs if d == 0 else h_states[d - 1]

                h_d = self.pool_injects[d](h_states[d]) if self.pool_injects else h_states[d]
                h_new, c_new = cell(x, h_d, c_states[d])
                h_states[d] = h_new
                c_states[d] = c_new
                tick_hiddens.append((h_new, c_new))

            all_tick_hiddens.append(tick_hiddens)

        new_hidden_states = [(h_states[d], c_states[d]) for d in range(self.num_layers)]
        return new_hidden_states, all_tick_hiddens

    def init_hidden(
        self, batch_size: int, H: int = 8, W: int = 8, device: torch.device = "cpu"
    ) -> LayerStates:
        return [cell.init_hidden(batch_size, H, W, device) for cell in self.cells]
