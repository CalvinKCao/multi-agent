"""
IPPO (Independent PPO) trainer with parameter sharing for the MA Boxoban ToM experiment.

Both agents use the same DRCAgent weights.  At each rollout step we stack
obs_A and obs_B along the batch dimension and run a single forward pass —
so the effective batch size is 2 * n_envs.  The buffer also stores 2 * n_envs
entries and mixes A and B data freely during PPO updates, which is exactly
what IPPO with parameter sharing requires.

Hidden-state management:
  - hidden_A, hidden_B: each LayerStates for n_envs envs
  - Combined on-the-fly for batched inference: first N slots = A, last N = B
  - Episode done in env i  ->  zero hidden_A[i] AND hidden_B[i]

WandB:
  Run id is persisted to <save_path>_wandb_run_id.txt so training can be
  resumed onto the same run via --resume.
"""

import time
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Optional, Dict, Any

from drc_sokoban.models.agent import DRCAgent
from drc_sokoban.training.rollout_buffer import RolloutBuffer

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


class IPPOTrainer:
    """
    IPPO trainer for the two-agent cooperative Boxoban setting.

    The partner policy can optionally come from a separate (fixed) checkpoint,
    enabling the "handicapped partner" condition for the ToM cross-policy test.
    If partner_ckpt is None both agents share live weights (self-play).
    """

    DEFAULT_CFG: Dict[str, Any] = dict(
        num_envs        = 64,
        horizon         = 400,
        obs_channels    = 10,
        num_actions     = 4,
        hidden_channels = 32,
        num_layers      = 2,
        num_ticks       = 3,
        learning_rate   = 3e-4,
        gamma           = 0.97,
        gae_lambda      = 0.95,
        clip_eps        = 0.1,
        value_coef      = 0.5,
        entropy_coef    = 0.01,
        max_grad_norm   = 10.0,
        ppo_epochs      = 4,
        minibatch_size  = 256,
        rollout_steps   = 20,
        target_steps    = 50_000_000,
        save_every      = 5_000_000,
        partner_noise_eps = 0.0,   # epsilon-greedy noise for partner (v2)
        wandb_project   = "drc-sokoban-ma-tom",
        wandb_run_name  = "ippo-selfplay-50M",
    )

    def __init__(
        self,
        env,
        cfg: Optional[Dict[str, Any]] = None,
        device: str = "auto",
        partner_ckpt: Optional[str] = None,
    ):
        self.env = env
        self.cfg = {**self.DEFAULT_CFG, **(cfg or {})}
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device == "auto" else torch.device(device)
        )

        N = self.cfg["num_envs"]
        self.agent = DRCAgent(
            obs_channels    = self.cfg["obs_channels"],
            hidden_channels = self.cfg["hidden_channels"],
            num_layers      = self.cfg["num_layers"],
            num_ticks       = self.cfg["num_ticks"],
            num_actions     = self.cfg["num_actions"],
        ).to(self.device)

        # Optional frozen partner policy (v2 handicapped condition)
        self._partner_fixed = False
        if partner_ckpt is not None:
            ckpt = torch.load(partner_ckpt, map_location=self.device, weights_only=False)
            partner_agent = DRCAgent(
                obs_channels    = self.cfg["obs_channels"],
                hidden_channels = self.cfg["hidden_channels"],
                num_layers      = self.cfg["num_layers"],
                num_ticks       = self.cfg["num_ticks"],
                num_actions     = self.cfg["num_actions"],
            ).to(self.device)
            partner_agent.load_state_dict(ckpt["model_state"])
            partner_agent.eval()
            self._partner_agent  = partner_agent
            self._partner_fixed  = True
        else:
            self._partner_agent = self.agent

        self.optimizer = optim.Adam(
            self.agent.parameters(), lr=self.cfg["learning_rate"], eps=1e-5
        )

        # Buffer uses 2*N slots: indices 0..N-1 = agent A, N..2N-1 = agent B
        obs_shape = (self.cfg["obs_channels"], 8, 8)
        self.buffer = RolloutBuffer(
            n_steps         = self.cfg["rollout_steps"],
            n_envs          = 2 * N,
            obs_shape       = obs_shape,
            num_layers      = self.cfg["num_layers"],
            hidden_channels = self.cfg["hidden_channels"],
            gamma           = self.cfg["gamma"],
            gae_lambda      = self.cfg["gae_lambda"],
            device          = self.device,
        )

        self.global_step = 0
        self._ep_rewards: list = []
        self._solve_tracker: list = []
        self._wandb_run = None

    # ── Training loop ──────────────────────────────────────────────────────────

    def train(self, save_path: Optional[str] = None, log_every: int = 1_000_000):
        cfg = self.cfg
        N   = cfg["num_envs"]
        T   = cfg["rollout_steps"]

        self._maybe_init_wandb(save_path)

        obs_A, obs_B = self.env.reset()
        hidden_A = self.agent.init_hidden(N, self.device)
        hidden_B = self._partner_agent.init_hidden(N, self.device)
        ep_rewards = np.zeros(N, dtype=np.float32)
        ep_solved  = np.zeros(N, dtype=bool)
        start_time = time.time()
        last_log_step  = 0
        last_save_step = 0

        while self.global_step < cfg["target_steps"]:
            self.agent.eval()
            if self._partner_fixed:
                self._partner_agent.eval()

            for _ in range(T):
                # ── Batched inference: stack A and B obs ───────────────────────
                obs_t = torch.FloatTensor(
                    np.concatenate([obs_A, obs_B], axis=0)
                ).to(self.device)
                hidden_both = [
                    (torch.cat([h_a, h_b], dim=0),
                     torch.cat([c_a, c_b], dim=0))
                    for (h_a, c_a), (h_b, c_b) in zip(hidden_A, hidden_B)
                ]

                with torch.no_grad():
                    # Use live agent for A; partner agent for B
                    logits_A_t, val_A_t, new_hA = self.agent(obs_t[:N], hidden_A)
                    logits_B_t, val_B_t, new_hB = self._partner_agent(obs_t[N:], hidden_B)

                dist_A = Categorical(logits=logits_A_t)
                dist_B = Categorical(logits=logits_B_t)
                act_A  = dist_A.sample()
                act_B  = dist_B.sample()

                # Apply partner noise for handicapped condition
                if cfg["partner_noise_eps"] > 0.0:
                    noise_mask = torch.rand(N, device=self.device) < cfg["partner_noise_eps"]
                    random_acts = torch.randint(0, 4, (N,), device=self.device)
                    act_B = torch.where(noise_mask, random_acts, act_B)

                lp_A  = dist_A.log_prob(act_A)
                lp_B  = dist_B.log_prob(act_B)

                act_A_np = act_A.cpu().numpy()
                act_B_np = act_B.cpu().numpy()
                obs_A_next, obs_B_next, reward, done, infos = self.env.step(act_A_np, act_B_np)

                # Combine into 2N buffer entry
                obs_both  = np.concatenate([obs_A, obs_B], axis=0)
                act_both  = np.concatenate([act_A_np, act_B_np], axis=0)
                lp_both   = np.concatenate(
                    [lp_A.detach().cpu().numpy(), lp_B.detach().cpu().numpy()], axis=0
                )
                rew_both  = np.concatenate([reward, reward], axis=0)
                done_both = np.concatenate([done, done], axis=0)
                val_both  = np.concatenate(
                    [val_A_t.squeeze(-1).detach().cpu().numpy(),
                     val_B_t.squeeze(-1).detach().cpu().numpy()], axis=0
                )
                hidden_both_stored = [
                    (torch.cat([h_a, h_b], dim=0),
                     torch.cat([c_a, c_b], dim=0))
                    for (h_a, c_a), (h_b, c_b) in zip(hidden_A, hidden_B)
                ]

                self.buffer.add(
                    obs=obs_both, action=act_both, log_prob=lp_both,
                    reward=rew_both, done=done_both, value=val_both,
                    hidden_states=hidden_both_stored,
                )

                ep_rewards += reward
                ep_solved  |= np.array([info.get("solved", False) for info in infos])
                for i in range(N):
                    if done[i]:
                        self._ep_rewards.append(float(ep_rewards[i]))
                        self._solve_tracker.append(bool(ep_solved[i]))
                        ep_rewards[i] = 0.0
                        ep_solved[i]  = False

                # Mask hidden states for finished episodes
                dones_both = np.concatenate([done, done], axis=0)
                dones_t    = torch.as_tensor(dones_both, device=self.device)
                combined_hidden = [
                    (torch.cat([h_a, h_b], dim=0), torch.cat([c_a, c_b], dim=0))
                    for (h_a, c_a), (h_b, c_b) in zip(new_hA, new_hB)
                ]
                masked = DRCAgent.mask_hidden(combined_hidden, dones_t)
                hidden_A = [(h[:N], c[:N]) for h, c in masked]
                hidden_B = [(h[N:], c[N:]) for h, c in masked]

                obs_A, obs_B = obs_A_next, obs_B_next
                self.global_step += N   # count unique env steps (N envs)

            # ── Bootstrap ──────────────────────────────────────────────────────
            with torch.no_grad():
                obs_both_boot = torch.FloatTensor(
                    np.concatenate([obs_A, obs_B], axis=0)
                ).to(self.device)
                hidden_boot = [
                    (torch.cat([h_a, h_b], dim=0), torch.cat([c_a, c_b], dim=0))
                    for (h_a, c_a), (h_b, c_b) in zip(hidden_A, hidden_B)
                ]
                _, val_boot, _ = self.agent(obs_both_boot[:N], hidden_A)
                _, val_boot_B, _ = self._partner_agent(obs_both_boot[N:], hidden_B)
                last_val = np.concatenate([
                    val_boot.squeeze(-1).cpu().numpy(),
                    val_boot_B.squeeze(-1).cpu().numpy(),
                ], axis=0)

            self.buffer.compute_returns_and_advantages(last_val, np.zeros(2 * N, dtype=bool))

            # ── PPO update ─────────────────────────────────────────────────────
            self.agent.train()
            losses = self._update()
            self.buffer.reset()

            # ── Logging ────────────────────────────────────────────────────────
            if self.global_step - last_log_step >= log_every:
                elapsed     = time.time() - start_time
                sps         = self.global_step / elapsed
                solve_rate  = np.mean(self._solve_tracker[-200:]) if self._solve_tracker else 0.0
                mean_rew    = np.mean(self._ep_rewards[-200:]) if self._ep_rewards else 0.0
                print(
                    f"Step {self.global_step/1e6:.1f}M | "
                    f"Solve: {solve_rate:.3f} | Rew: {mean_rew:.2f} | "
                    f"Ent: {losses['entropy']:.3f} | VF: {losses['value_loss']:.4f} | "
                    f"SPS: {sps:.0f}",
                    flush=True,
                )
                if self._wandb_run:
                    self._wandb_run.log({
                        "train/solve_rate": solve_rate,
                        "train/mean_reward": mean_rew,
                        "train/entropy": losses["entropy"],
                        "train/value_loss": losses["value_loss"],
                        "train/policy_loss": losses["policy_loss"],
                        "train/sps": sps,
                        "global_step": self.global_step,
                    })
                last_log_step = self.global_step

            # ── Checkpoint ─────────────────────────────────────────────────────
            if save_path and self.global_step - last_save_step >= cfg["save_every"]:
                ckpt_path = f"{save_path}_{self.global_step // 1_000_000}M.pt"
                self.save(ckpt_path)
                print(f"Checkpoint: {ckpt_path}", flush=True)
                last_save_step = self.global_step

        print(f"Training done. Steps: {self.global_step}", flush=True)
        if save_path:
            final = f"{save_path}_final.pt"
            self.save(final)
        if self._wandb_run:
            self._wandb_run.finish()

    # ── PPO update ─────────────────────────────────────────────────────────────

    def _update(self) -> dict:
        cfg = self.cfg
        total_pg = total_vf = total_ent = 0.0
        n_batches = 0

        for _ in range(cfg["ppo_epochs"]):
            for batch in self.buffer.get_minibatches(cfg["minibatch_size"]):
                obs        = batch["obs"]
                actions    = batch["actions"]
                old_lp     = batch["log_probs"]
                returns    = batch["returns"]
                advantages = batch["advantages"]
                old_vals   = batch["values"]
                hidden     = DRCAgent.detach_hidden(batch["hidden_states"])

                logits, value, _ = self.agent(obs, hidden)
                dist    = Categorical(logits=logits)
                new_lp  = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                adv_norm = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                ratio    = torch.exp(new_lp - old_lp)
                pg_loss  = torch.max(
                    -adv_norm * ratio,
                    -adv_norm * torch.clamp(ratio, 1 - cfg["clip_eps"], 1 + cfg["clip_eps"])
                ).mean()

                value = value.squeeze(-1)
                v_clip = old_vals + torch.clamp(value - old_vals, -cfg["clip_eps"], cfg["clip_eps"])
                vf_loss = torch.max(
                    nn.functional.mse_loss(value, returns),
                    nn.functional.mse_loss(v_clip, returns),
                )

                loss = pg_loss + cfg["value_coef"] * vf_loss - cfg["entropy_coef"] * entropy
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), cfg["max_grad_norm"])
                self.optimizer.step()

                total_pg  += pg_loss.item()
                total_vf  += vf_loss.item()
                total_ent += entropy.item()
                n_batches += 1

        n = max(n_batches, 1)
        return {
            "policy_loss": total_pg / n,
            "value_loss":  total_vf / n,
            "entropy":     total_ent / n,
        }

    # ── WandB helpers ──────────────────────────────────────────────────────────

    def _maybe_init_wandb(self, save_path):
        if not _WANDB:
            return
        from drc_sokoban.wandb_env import load_wandb_local_env

        load_wandb_local_env()
        cfg = self.cfg
        run_id_file = f"{save_path}_wandb_run_id.txt" if save_path else None
        run_id = None
        if run_id_file and os.path.exists(run_id_file):
            with open(run_id_file) as f:
                run_id = f.read().strip()

        try:
            self._wandb_run = wandb.init(
                project = cfg["wandb_project"],
                name    = cfg["wandb_run_name"],
                id      = run_id,
                resume  = "allow" if run_id else None,
                config  = {k: v for k, v in cfg.items()
                           if not k.startswith("wandb")},
            )
        except Exception as e:
            print(
                f"wandb.init failed ({e}); training continues without W&B.\n"
                f"  Repo root: copy wandb.local.example -> wandb.local (gitignored)\n"
                f"  Or:  wandb login   on the login node (~/.netrc)\n"
                f"  Or before the job:   export WANDB_API_KEY=...   (never commit)\n"
                f"  Or offline only:      export WANDB_MODE=offline",
                flush=True,
            )
            self._wandb_run = None
            return
        if run_id_file and not run_id and self._wandb_run is not None:
            os.makedirs(os.path.dirname(run_id_file) or ".", exist_ok=True)
            with open(run_id_file, "w") as f:
                f.write(self._wandb_run.id)

    # ── Checkpoint I/O ─────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "model_state":     self.agent.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "global_step":     self.global_step,
            "cfg":             self.cfg,
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.agent.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.global_step = ckpt.get("global_step", 0)
        print(f"Loaded {path} (step {self.global_step:,})")
        return self
