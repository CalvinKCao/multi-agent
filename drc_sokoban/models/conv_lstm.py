"""
ConvLSTM cell and DRC (Deep Repeated ConvLSTM) stack.

CRITICAL spatial invariant: every Conv2d uses kernel_size=3, padding=1
so that H=8 and W=8 are preserved throughout.  The hidden state position
(x, y) corresponds exactly to board cell (x, y) — required for 1×1 probes.
"""

import torch
import torch.nn as nn
from typing import List, Tuple


HiddenState = Tuple[torch.Tensor, torch.Tensor]   # (h, c) each (B, C, H, W)
LayerStates = List[HiddenState]                    # D elements


class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell operating on spatial (B, C, H, W) tensors.

    The combined gate convolution concatenates input and previous hidden state
    along the channel dimension, computes 4·hidden_channels feature maps, then
    splits them into i / f / o / g gates.

    Kernel 3×3 with same-padding preserves spatial dimensions identically.
    """

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2  # same-padding

        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        h_prev: torch.Tensor,
        c_prev: torch.Tensor,
    ) -> HiddenState:
        """
        Args:
            x:      (B, input_channels, H, W)
            h_prev: (B, hidden_channels, H, W)
            c_prev: (B, hidden_channels, H, W)
        Returns:
            h_next, c_next — each (B, hidden_channels, H, W)
        """
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

    def init_hidden(
        self, batch_size: int, H: int, W: int, device: torch.device
    ) -> HiddenState:
        zeros = torch.zeros(batch_size, self.hidden_channels, H, W, device=device)
        return (zeros, zeros.clone())


class DRCStack(nn.Module):
    """
    Deep Repeated ConvLSTM (DRC) stack.

    Performs N ticks of recurrent computation per environment step.
    Each tick passes through all D ConvLSTM layers in sequence.

    The hidden states (h, c) for every layer are carried across both ticks
    within a step AND environment steps across the episode — this temporal
    persistence is the key to planning behaviour.

    From Guez et al. 2019 / Bush et al. 2025:
        D = 3 layers  (use 2 for fast PoC)
        N = 3 ticks per step
        hidden_channels = 32

    The all_tick_hiddens output exposes every intermediate state so that
    probes can be trained at each (tick, layer) position.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_ticks: int,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_ticks = num_ticks
        self.hidden_channels = hidden_channels

        self.cells = nn.ModuleList()
        for d in range(num_layers):
            in_ch = input_channels if d == 0 else hidden_channels
            self.cells.append(ConvLSTMCell(in_ch, hidden_channels))

    def forward(
        self,
        encoded_obs: torch.Tensor,
        hidden_states: LayerStates,
    ) -> Tuple[LayerStates, List[LayerStates]]:
        """
        Args:
            encoded_obs:   (B, input_channels, 8, 8)
            hidden_states: list of D tuples [(h_d, c_d), ...]

        Returns:
            new_hidden_states: list of D tuples — carry to next env step
            all_tick_hiddens:  list of N tick snapshots;
                               all_tick_hiddens[tick][layer] = (h, c) tensor pair
                               Shape of each h/c: (B, hidden_channels, 8, 8)
        """
        h_states = [hs[0] for hs in hidden_states]
        c_states = [hs[1] for hs in hidden_states]

        all_tick_hiddens: List[LayerStates] = []

        for _ in range(self.num_ticks):
            tick_hiddens: LayerStates = []
            x = encoded_obs  # every tick sees fresh encoder output

            for d, cell in enumerate(self.cells):
                h_new, c_new = cell(x, h_states[d], c_states[d])
                h_states[d] = h_new
                c_states[d] = c_new
                tick_hiddens.append((h_new, c_new))
                x = h_new

            all_tick_hiddens.append(tick_hiddens)

        new_hidden_states = [(h_states[d], c_states[d]) for d in range(self.num_layers)]
        return new_hidden_states, all_tick_hiddens

    def init_hidden(
        self, batch_size: int, H: int = 8, W: int = 8, device: torch.device = "cpu"
    ) -> LayerStates:
        return [cell.init_hidden(batch_size, H, W, device) for cell in self.cells]
