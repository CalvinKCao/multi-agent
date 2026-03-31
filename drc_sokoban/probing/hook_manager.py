"""
Forward-hook manager for extracting ConvLSTM hidden states during inference.

The DRCAgent already exposes hidden states through its return_all_ticks=True
interface, so we primarily use that.  This module provides an alternative
hook-based approach for cases where you want to extract activations without
modifying the agent's forward() call signature.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


class HookManager:
    """
    Registers forward hooks on DRCStack cells to capture (h, c) tensors
    at every tick and every layer automatically.

    Usage:
        hooks = HookManager(agent.drc)
        logits, value, new_hidden = agent(obs, hidden)
        tick_layer_h = hooks.get_hidden_states()  # list[tick][layer] -> (B, C, H, W)
        hooks.clear()
    """

    def __init__(self, drc_stack):
        self._drc = drc_stack
        self._captures: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        self._tick_counter = 0
        self._hooks = []
        self._register()

    def _register(self):
        for d, cell in enumerate(self._drc.cells):
            h = cell.register_forward_hook(self._make_hook(d))
            self._hooks.append(h)

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            h_new, c_new = output
            tick = self._tick_counter // len(self._drc.cells)
            key = tick
            if key not in self._captures:
                self._captures[key] = []
            self._captures[key].append((h_new.detach(), c_new.detach()))
            self._tick_counter += 1
        return hook

    def get_hidden_states(self) -> List[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Returns captured hidden states as:
            list[tick_idx][layer_idx] -> (h, c) tensor pair
        """
        return [self._captures[t] for t in sorted(self._captures.keys())]

    def clear(self):
        """Reset captured states between forward passes."""
        self._captures.clear()
        self._tick_counter = 0

    def remove_hooks(self):
        """Call this when done to avoid memory leaks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
