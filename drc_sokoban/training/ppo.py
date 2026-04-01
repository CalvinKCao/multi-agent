"""
PPO trainer for the DRC agent.

Follows CleanRL's ppo_atari.py structure adapted for:
  - ConvLSTM hidden-state management across environment steps
  - Correct hidden-state masking at episode boundaries
  - No BPTT across rollout boundaries (detach at PPO update time)
  - Higher max-grad-norm (10.0) for LSTM stability

Reference: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Optional, Dict, Any

from drc_sokoban.models.agent import DRCAgent
from drc_sokoban.training.rollout_buffer import RolloutBuffer


class PPOTrainer:
    """
    PPO trainer with vectorised environments and ConvLSTM hidden-state handling.

    Hyperparameters follow Bush et al. 2025 / Guez et al. 2019 where specified.
    """

    DEFAULT_CFG: Dict[str, Any] = dict(
        # Environment
        num_envs        = 32,
        horizon         = 120,
        obs_shape       = (7, 8, 8),
        num_actions     = 4,
        # Architecture
        hidden_channels = 32,
        num_layers      = 3,
        num_ticks       = 3,
        H               = 8,
        W               = 8,
        skip_connections = True,
        pool_and_inject  = True,
        concat_encoder   = True,
        # PPO
        learning_rate   = 4e-4,    # paper: linear decay 4e-4 -> 0
        lr_decay        = True,
        gamma           = 0.97,
        gae_lambda      = 0.97,    # paper: 0.97 (was 0.95)
        clip_eps        = 0.1,
        value_coef      = 0.5,
        entropy_coef    = 0.01,
        max_grad_norm   = 10.0,
        ppo_epochs      = 4,
        minibatch_size  = 256,
        rollout_steps   = 20,
        # Training budget
        target_steps    = 50_000_000,
        save_every      = 5_000_000,
    )

    def __init__(self, env, cfg: Optional[Dict[str, Any]] = None, device: str = "auto"):
        self.env = env
        self.cfg = {**self.DEFAULT_CFG, **(cfg or {})}

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        c = self.cfg
        self.agent = DRCAgent(
            obs_channels     = c["obs_shape"][0],
            hidden_channels  = c["hidden_channels"],
            num_layers       = c["num_layers"],
            num_ticks        = c["num_ticks"],
            num_actions      = c["num_actions"],
            H                = c.get("H", 8),
            W                = c.get("W", 8),
            skip_connections = c.get("skip_connections", True),
            pool_and_inject  = c.get("pool_and_inject", True),
            concat_encoder   = c.get("concat_encoder", True),
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=c["learning_rate"], eps=1e-5
        )

        self.buffer = RolloutBuffer(
            n_steps         = c["rollout_steps"],
            n_envs          = c["num_envs"],
            obs_shape       = c["obs_shape"],
            num_layers      = c["num_layers"],
            hidden_channels = c["hidden_channels"],
            H               = c.get("H", 8),
            W               = c.get("W", 8),
            gamma           = c["gamma"],
            gae_lambda      = c["gae_lambda"],
            device          = self.device,
        )

        self.global_step = 0
        self._episode_rewards: list = []
        self._solve_tracker: list = []
        self._ep_lengths: list = []
        self._ep_timeouts: list = []

    # ── Main Training Loop ─────────────────────────────────────────────────────

    def _get_lr(self):
        """Linear decay from initial LR to 0 over target_steps."""
        if not self.cfg.get("lr_decay", True):
            return self.cfg["learning_rate"]
        frac = max(0.0, 1.0 - self.global_step / self.cfg["target_steps"])
        return self.cfg["learning_rate"] * frac

    def _set_lr(self):
        lr = self._get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr

    def train(self, save_path: Optional[str] = None, log_every: int = 1_000_000):
        cfg = self.cfg
        n_envs = cfg["num_envs"]
        T = cfg["rollout_steps"]
        target = cfg["target_steps"]

        obs = self.env.reset()
        hidden = self.agent.init_hidden(n_envs, self.device)
        ep_rewards = np.zeros(n_envs, dtype=np.float32)
        ep_solved = np.zeros(n_envs, dtype=bool)

        start_time = time.time()
        last_log_step = 0
        last_save_step = 0

        while self.global_step < target:
            cur_lr = self._set_lr()
            self.agent.eval()
            for _ in range(T):
                obs_t = torch.FloatTensor(obs).to(self.device)

                with torch.no_grad():
                    logits, value, new_hidden = self.agent(obs_t, hidden)

                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                action_np = action.cpu().numpy()
                obs_next, reward, done, info = self.env.step(action_np)

                self.buffer.add(
                    obs=obs, action=action_np,
                    log_prob=log_prob.detach().cpu().numpy(),
                    reward=reward, done=done,
                    value=value.squeeze(-1).detach().cpu().numpy(),
                    hidden_states=hidden,
                )

                ep_rewards += reward
                ep_solved |= np.array([info[i].get("solved", False) for i in range(n_envs)])

                for i in range(n_envs):
                    if done[i]:
                        self._episode_rewards.append(float(ep_rewards[i]))
                        self._solve_tracker.append(bool(ep_solved[i]))
                        self._ep_lengths.append(info[i].get("ep_length", 0))
                        self._ep_timeouts.append(info[i].get("timeout", False))
                        ep_rewards[i] = 0.0
                        ep_solved[i] = False

                dones_t = torch.as_tensor(done, device=self.device)
                hidden = DRCAgent.mask_hidden(new_hidden, dones_t)
                obs = obs_next
                self.global_step += n_envs

            with torch.no_grad():
                obs_t = torch.FloatTensor(obs).to(self.device)
                last_value = self.agent.get_value(obs_t, hidden)
                last_value = last_value.squeeze(-1).cpu().numpy()

            self.buffer.compute_returns_and_advantages(last_value, np.zeros(n_envs, dtype=bool))

            self.agent.train()
            losses = self._update()
            self.buffer.reset()

            if self.global_step - last_log_step >= log_every:
                elapsed = time.time() - start_time
                sps = self.global_step / elapsed
                W = 200
                solve = np.mean(self._solve_tracker[-W:]) if self._solve_tracker else 0.0
                rew = np.mean(self._episode_rewards[-W:]) if self._episode_rewards else 0.0
                ep_len = np.mean(self._ep_lengths[-W:]) if self._ep_lengths else 0.0
                timeout = np.mean(self._ep_timeouts[-W:]) if self._ep_timeouts else 0.0
                print(
                    f"Step {self.global_step/1e6:.1f}M | "
                    f"Solve {solve:.3f} | Rew {rew:.2f} | "
                    f"Len {ep_len:.0f} | TO {timeout:.2f} | "
                    f"Ent {losses['entropy']:.3f} | VF {losses['value_loss']:.4f} | "
                    f"Clip {losses['clip_frac']:.3f} | "
                    f"GNorm {losses['grad_norm']:.2f} | "
                    f"LR {cur_lr:.1e} | SPS {sps:.0f}",
                    flush=True,
                )
                last_log_step = self.global_step

            if save_path and self.global_step - last_save_step >= cfg["save_every"]:
                ckpt_path = f"{save_path}_{self.global_step // 1_000_000}M.pt"
                self.save(ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}", flush=True)
                last_save_step = self.global_step

        print(f"Training complete. Total steps: {self.global_step}", flush=True)
        if save_path:
            self.save(f"{save_path}_final.pt")

    # ── PPO Update Step ────────────────────────────────────────────────────────

    def _update(self) -> dict:
        cfg = self.cfg
        total_pg = total_vf = total_ent = total_clip = total_gnorm = 0.0
        n_batches = 0

        for _ in range(cfg["ppo_epochs"]):
            for batch in self.buffer.get_minibatches(cfg["minibatch_size"]):
                obs        = batch["obs"]
                actions    = batch["actions"]
                old_lp     = batch["log_probs"]
                returns    = batch["returns"]
                advantages = batch["advantages"]
                old_values = batch["values"]
                hidden     = DRCAgent.detach_hidden(batch["hidden_states"])

                logits, value, _ = self.agent(obs, hidden)
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = torch.exp(new_lp - old_lp)
                pg_loss1 = -adv_norm * ratio
                pg_loss2 = -adv_norm * torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                clip_frac = ((ratio - 1.0).abs() > cfg["clip_eps"]).float().mean()

                value = value.squeeze(-1)
                v_clipped = old_values + torch.clamp(
                    value - old_values, -cfg["clip_eps"], cfg["clip_eps"]
                )
                vf_loss = torch.max(
                    nn.functional.mse_loss(value, returns),
                    nn.functional.mse_loss(v_clipped, returns),
                )

                loss = pg_loss + cfg["value_coef"] * vf_loss - cfg["entropy_coef"] * entropy

                self.optimizer.zero_grad()
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(self.agent.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()

                total_pg    += pg_loss.item()
                total_vf    += vf_loss.item()
                total_ent   += entropy.item()
                total_clip  += clip_frac.item()
                total_gnorm += gnorm.item() if hasattr(gnorm, 'item') else float(gnorm)
                n_batches   += 1

        n = max(n_batches, 1)
        return {
            "policy_loss": total_pg / n,
            "value_loss":  total_vf / n,
            "entropy":     total_ent / n,
            "clip_frac":   total_clip / n,
            "grad_norm":   total_gnorm / n,
        }

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save(
            {
                "model_state":     self.agent.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "global_step":     self.global_step,
                "cfg":             self.cfg,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.global_step = ckpt.get("global_step", 0)
        print(f"Loaded checkpoint from {path} (step {self.global_step})")
        return self
