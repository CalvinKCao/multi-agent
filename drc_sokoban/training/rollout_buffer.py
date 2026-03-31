"""
Rollout buffer for PPO with ConvLSTM hidden-state tracking.

Stores T steps × N environments of:
  - observations
  - actions
  - log-probabilities
  - rewards
  - dones
  - values
  - initial hidden states (one per env, at the start of each step)

Hidden states stored here are *detached* — they are carried across steps
for correct recurrence but NOT backpropagated across rollout boundaries.
"""

import numpy as np
import torch
from typing import List, Tuple, Generator


HiddenState = Tuple[torch.Tensor, torch.Tensor]
LayerStates = List[HiddenState]


class RolloutBuffer:
    """
    Fixed-length rollout buffer for vectorised PPO.

    Args:
        n_steps:         Rollout horizon T (steps per env before update).
        n_envs:          Number of parallel environments N.
        obs_shape:       Observation shape, e.g. (7, 8, 8).
        num_layers:      DRC depth D.
        hidden_channels: Channels per ConvLSTM layer.
        H, W:            Spatial dimensions of hidden state (8, 8 for Boxoban).
        gamma:           Discount factor.
        gae_lambda:      GAE lambda.
        device:          Torch device for returned tensors.
    """

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_shape: Tuple,
        num_layers: int,
        hidden_channels: int,
        H: int = 8,
        W: int = 8,
        gamma: float = 0.97,
        gae_lambda: float = 0.95,
        device: torch.device = "cpu",
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.obs_shape = obs_shape
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.H = H
        self.W = W
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device

        self._allocate()

    def _allocate(self):
        T, N = self.n_steps, self.n_envs
        C, H, W = self.obs_shape

        self.observations = np.zeros((T, N, C, H, W), dtype=np.float32)
        self.actions      = np.zeros((T, N), dtype=np.int64)
        self.log_probs    = np.zeros((T, N), dtype=np.float32)
        self.rewards      = np.zeros((T, N), dtype=np.float32)
        self.dones        = np.zeros((T, N), dtype=bool)
        self.values       = np.zeros((T, N), dtype=np.float32)

        # Hidden states at the START of each step (B, C_h, H, W)
        # hidden_states_h[t, d] = h-tensor for layer d at step t
        # Stored as numpy arrays for memory efficiency
        D = self.num_layers
        C_h = self.hidden_channels
        self.hidden_h = np.zeros((T, D, N, C_h, H, W), dtype=np.float32)
        self.hidden_c = np.zeros((T, D, N, C_h, H, W), dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        hidden_states: LayerStates,
    ):
        """Store one step across all N envs."""
        t = self.ptr
        self.observations[t] = obs
        self.actions[t]      = action
        self.log_probs[t]    = log_prob
        self.rewards[t]      = reward
        self.dones[t]        = done
        self.values[t]       = value

        for d, (h, c) in enumerate(hidden_states):
            self.hidden_h[t, d] = h.detach().cpu().numpy()
            self.hidden_c[t, d] = c.detach().cpu().numpy()

        self.ptr += 1
        if self.ptr >= self.n_steps:
            self.full = True

    def compute_returns_and_advantages(
        self, last_value: np.ndarray, last_done: np.ndarray
    ):
        """
        Compute GAE advantages and discounted returns in-place.

        Args:
            last_value: (N,) value estimate at the step after the last stored step.
            last_done:  (N,) done flags at that step.
        """
        advantages = np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        last_gae = np.zeros(self.n_envs, dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - last_done.astype(np.float32)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1].astype(np.float32)
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self.advantages = advantages
        self.returns = advantages + self.values

    def get_minibatches(
        self, minibatch_size: int
    ) -> Generator[dict, None, None]:
        """
        Yield minibatches over the flattened (T×N) rollout.

        Each minibatch is a dict of tensors on self.device.
        Hidden states are the *initial* hidden states for each sample
        (to be used as starting point with BPTT disabled across rollout boundaries).
        """
        T, N = self.n_steps, self.n_envs
        total = T * N
        indices = np.random.permutation(total)

        # Flatten buffer along T×N
        obs_flat    = self.observations.reshape(total, *self.obs_shape)
        act_flat    = self.actions.reshape(total)
        lp_flat     = self.log_probs.reshape(total)
        ret_flat    = self.returns.reshape(total)
        adv_flat    = self.advantages.reshape(total)
        val_flat    = self.values.reshape(total)

        # Hidden states: (T*N, D, C_h, H, W) — flatten T×N jointly
        D, C_h, H, W = self.num_layers, self.hidden_channels, self.H, self.W
        h_flat = self.hidden_h.transpose(0, 2, 1, 3, 4, 5).reshape(total, D, C_h, H, W)
        c_flat = self.hidden_c.transpose(0, 2, 1, 3, 4, 5).reshape(total, D, C_h, H, W)

        for start in range(0, total, minibatch_size):
            idx = indices[start : start + minibatch_size]
            yield {
                "obs":       torch.FloatTensor(obs_flat[idx]).to(self.device),
                "actions":   torch.LongTensor(act_flat[idx]).to(self.device),
                "log_probs": torch.FloatTensor(lp_flat[idx]).to(self.device),
                "returns":   torch.FloatTensor(ret_flat[idx]).to(self.device),
                "advantages":torch.FloatTensor(adv_flat[idx]).to(self.device),
                "values":    torch.FloatTensor(val_flat[idx]).to(self.device),
                # hidden states per sample: list of D tuples (B, C_h, H, W)
                "hidden_states": [
                    (
                        torch.FloatTensor(h_flat[idx, d]).to(self.device),
                        torch.FloatTensor(c_flat[idx, d]).to(self.device),
                    )
                    for d in range(D)
                ],
            }

    def reset(self):
        self.ptr = 0
        self.full = False
