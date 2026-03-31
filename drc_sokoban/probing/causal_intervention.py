"""
Phase 5 — Causal Probes (Activation Steering).

Moves from CORRELATION (the probe can read the plan from the hidden state)
to CAUSATION (injecting a plan vector into the hidden state changes behavior).

Step 5.1  Extract "Plan Vector" from probe weights.
Step 5.2  Inject it at inference time via a forward hook.
Step 5.3  Dose-response analysis across injection strengths α ∈ [0, 5].
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# ── 5.1  Extract Plan Vector ──────────────────────────────────────────────────

def extract_plan_vector(probe: Pipeline, target_class: int) -> np.ndarray:
    """
    Extract the unit-normalised weight vector for `target_class` from a fitted
    logistic-regression probe.

    For a probe trained on {UP=0, DOWN=1, LEFT=2, RIGHT=3, NEVER=4},
    target_class=0 gives the "UP approach / push" vector.

    The vector lives in the 32-dimensional hidden-channel space at one (x, y)
    position.  We return it unit-normalised so α controls magnitude exactly.
    """
    clf = probe.named_steps["clf"]          # LogisticRegression
    # coef_ shape: (n_classes, n_features) or (1, n_features) for binary
    if clf.coef_.shape[0] == 1:
        w = clf.coef_[0]
    else:
        w = clf.coef_[target_class]

    norm = np.linalg.norm(w)
    if norm < 1e-9:
        return w
    return w / norm                         # (32,) unit vector


def extract_plan_vectors_all_classes(
    probe: Pipeline,
) -> Dict[int, np.ndarray]:
    """Return a dict {class_idx: unit_vector} for all 5 classes."""
    clf = probe.named_steps["clf"]
    n_classes = clf.coef_.shape[0]
    result = {}
    for c in range(n_classes):
        w = clf.coef_[c]
        n = np.linalg.norm(w)
        result[c] = w / n if n > 1e-9 else w
    return result


# ── 5.2  Intervention Protocol ────────────────────────────────────────────────

class HiddenStateInjector:
    """
    Registers a forward hook on a specific ConvLSTM cell layer and spatial
    position (x, y) that adds α × plan_vector to h at each forward pass.

    The injection is SPATIALLY LOCAL: only the 32-dim channel vector at
    position (y, x) in the hidden state is modified.

    Usage:
        injector = HiddenStateInjector(agent, layer=1, x=3, y=4)
        injector.set_vector(plan_vector, alpha=2.0)
        injector.enable()
        with torch.no_grad():
            logits, value, new_hidden = agent(obs_t, hidden)
        injector.disable()
    """

    def __init__(self, agent, layer: int, x: int, y: int):
        self.agent = agent
        self.layer = layer
        self.x = x
        self.y = y
        self._hook = None
        self._vector: Optional[np.ndarray] = None
        self._alpha: float = 0.0
        self._enabled: bool = False

    def set_vector(self, vector: np.ndarray, alpha: float):
        """Set the plan vector and injection strength."""
        self._vector = vector.copy()
        self._alpha = alpha

    def enable(self):
        """Register the hook on the target ConvLSTM cell."""
        cell = self.agent.drc.cells[self.layer]
        self._hook = cell.register_forward_hook(self._hook_fn)
        self._enabled = True

    def disable(self):
        """Remove the hook."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
        self._enabled = False

    def _hook_fn(self, module, inputs, output):
        """
        Post-forward hook: modifies h_next at position (y, x) by adding
        α × plan_vector (broadcast across batch).
        """
        if self._vector is None or not self._enabled:
            return output

        h_next, c_next = output
        device = h_next.device
        dtype  = h_next.dtype

        v = torch.tensor(self._vector, dtype=dtype, device=device)  # (32,)
        delta = self._alpha * v                                       # (32,)

        # h_next: (B, 32, H, W) — add delta at spatial position (y, x)
        h_modified = h_next.clone()
        h_modified[:, :, self.y, self.x] += delta.unsqueeze(0)       # broadcast over B

        return h_modified, c_next


class NullInjector(HiddenStateInjector):
    """
    Control condition: injects a RANDOM unit vector of the same magnitude.
    Proves that results aren't caused by "breaking" the hidden state.
    Each call to enable() samples a fresh random direction.
    """

    def enable(self):
        # Sample a new random unit vector each time
        rnd = np.random.randn(self._vector.shape[0] if self._vector is not None else 32)
        rnd /= np.linalg.norm(rnd)
        self._vector = rnd
        super().enable()


# ── 5.3  Dose-Response Analysis ───────────────────────────────────────────────

def run_dose_response(
    agent,
    env_factory,
    plan_vector: np.ndarray,
    layer: int,
    x: int,
    y: int,
    target_class: int,
    alphas: Optional[List[float]] = None,
    n_episodes_per_alpha: int = 50,
    device: str = "cuda",
) -> Dict:
    """
    For each α in alphas, run n_episodes and record:
      - action_freq: fraction of steps the agent took action=target_class
      - solve_rate: fraction of episodes solved
      - probe_decoded_frac: fraction of steps where a fresh probe predicts
                            target_class at position (x, y)

    Returns dict {alpha: {action_freq, solve_rate, mean_reward}}.
    """
    if alphas is None:
        alphas = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]

    torch_device = torch.device(device)
    agent = agent.to(torch_device)
    agent.eval()

    results = {}
    plan_injector = HiddenStateInjector(agent, layer, x, y)
    null_injector  = NullInjector(agent, layer, x, y)

    for alpha in alphas:
        plan_injector.set_vector(plan_vector, alpha)
        null_injector.set_vector(plan_vector, alpha)   # same magnitude

        for mode, injector in [("plan", plan_injector), ("null", null_injector)]:
            action_counts = np.zeros(4)
            solve_count   = 0
            total_reward  = 0.0
            total_steps   = 0

            for ep in range(n_episodes_per_alpha):
                env = env_factory()
                obs = env.reset()
                hidden = agent.init_hidden(1, torch_device)
                done = False

                injector.enable()
                while not done:
                    obs_t = torch.FloatTensor(obs).unsqueeze(0).to(torch_device)
                    with torch.no_grad():
                        logits, _, new_hidden = agent(obs_t, hidden)
                    injector.disable()

                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample().item()
                    action_counts[action] += 1
                    total_steps += 1

                    obs, rew, done, info = env.step(action)
                    total_reward += rew

                    mask = torch.FloatTensor([0.0 if done else 1.0]).view(1,1,1,1).to(torch_device)
                    hidden = [(h * mask, c * mask) for h, c in new_hidden]

                    if not done:
                        injector.enable()

                if info.get("solved", False):
                    solve_count += 1

            if alpha not in results:
                results[alpha] = {}

            results[alpha][f"{mode}_action_freq_target"] = (
                action_counts[target_class] / max(total_steps, 1)
            )
            results[alpha][f"{mode}_solve_rate"] = solve_count / n_episodes_per_alpha
            results[alpha][f"{mode}_mean_reward"] = total_reward / max(total_steps, 1)

    return results


def print_dose_response_table(results: Dict):
    """Print the dose-response results as a formatted table."""
    print("\n" + "="*70)
    print("PHASE 5 — DOSE-RESPONSE ANALYSIS")
    print("="*70)
    print(f"\n{'α':>5}  {'Plan→ActionFreq':>16}  {'Plan→Solve':>10}  "
          f"{'Null→ActionFreq':>16}  {'Null→Solve':>10}")
    print("-" * 64)

    for alpha in sorted(results.keys()):
        r = results[alpha]
        print(
            f"{alpha:>5.1f}  "
            f"{r.get('plan_action_freq_target', 0):>16.3f}  "
            f"{r.get('plan_solve_rate', 0):>10.3f}  "
            f"{r.get('null_action_freq_target', 0):>16.3f}  "
            f"{r.get('null_solve_rate', 0):>10.3f}"
        )
    print()


# ── Probe-free causal analysis using just action distribution shift ───────────

def measure_action_shift(
    agent,
    env_factory,
    plan_vector: np.ndarray,
    layer: int,
    x: int,
    y: int,
    alphas: List[float],
    n_episodes: int = 100,
    device: str = "cuda",
) -> Dict:
    """
    Simpler version: just measure how action distribution shifts with α.
    No dose-response loop — runs all alphas in one pass per episode by
    resetting to the same level seed each time.

    Returns {alpha: action_distribution (4,)} dicts.
    """
    torch_device = torch.device(device)
    agent = agent.to(torch_device).eval()

    action_dists = {}
    injector = HiddenStateInjector(agent, layer, x, y)

    for alpha in alphas:
        injector.set_vector(plan_vector, alpha)
        counts = np.zeros(4, dtype=np.float64)
        total  = 0

        for ep_seed in range(n_episodes):
            env = env_factory()
            obs = env.reset()
            hidden = agent.init_hidden(1, torch_device)
            done = False
            step = 0

            while not done and step < 200:
                injector.enable()
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(torch_device)
                with torch.no_grad():
                    logits, _, new_hidden = agent(obs_t, hidden)
                injector.disable()  # always disable immediately after inference

                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                counts += probs
                total  += 1

                action = int(np.random.choice(4, p=probs))
                obs, rew, done, info = env.step(action)
                mask = torch.FloatTensor([0.0 if done else 1.0]).view(1,1,1,1).to(torch_device)
                hidden = [(h*mask, c*mask) for h,c in new_hidden]
                step += 1

        action_dists[alpha] = counts / max(total, 1)

    return action_dists
