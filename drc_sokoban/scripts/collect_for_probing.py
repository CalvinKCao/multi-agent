"""
Run trained DRC agent and collect hidden states for probe training.

Saves per-episode data: observations, hidden states (all ticks & layers),
actions, agent positions, box positions, rewards, dones, solved flag.

Usage:
    python scripts/collect_for_probing.py \\
        --checkpoint checkpoints/agent_final.pt \\
        --n-episodes 3000 \\
        --save-path data/probe_data/train.pkl

    # Smoke test (10 episodes)
    python scripts/collect_for_probing.py \\
        --checkpoint checkpoints/smoke_test_final.pt \\
        --n-episodes 10 \\
        --save-path data/probe_data/smoke_test.pkl
"""

import argparse
import os
import sys
import pickle
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from drc_sokoban.envs.boxoban_env import BoxobanEnv
from drc_sokoban.models.agent import DRCAgent


def parse_args():
    p = argparse.ArgumentParser(description="Collect probe data from trained DRC agent")
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--n-episodes",  type=int, default=3000)
    p.add_argument("--save-path",   type=str, required=True)
    p.add_argument("--data-dir",    type=str, default=None)
    p.add_argument("--split",       type=str, default="valid",
                   help="Use validation levels for probe data (avoid train/test leakage)")
    p.add_argument("--difficulty",  type=str, default="unfiltered")
    p.add_argument("--device",      type=str, default="auto")
    p.add_argument("--seed",        type=int, default=0)
    return p.parse_args()


def collect_probe_data(agent, env, n_episodes, device):
    """
    Run agent for n_episodes, storing all data needed for probing.

    Returns list of episode dicts.
    """
    all_episodes = []
    solve_count = 0

    for ep in tqdm(range(n_episodes), desc="Collecting episodes"):
        obs = env.reset()
        hidden = agent.init_hidden(batch_size=1, device=device)

        episode = {
            "observations":    [],
            "hidden_states":   [],   # (T, num_ticks, num_layers, 32, 8, 8)
            "actions":         [],
            "agent_positions": [],
            "box_positions":   [],
            "rewards":         [],
            "dones":           [],
        }

        done = False
        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                logits, value, new_hidden, all_tick_hiddens = agent(
                    obs_t, hidden, return_all_ticks=True
                )

            # Pack hidden states: (num_ticks, num_layers, 32, 8, 8)
            # all_tick_hiddens[tick][layer] = (h, c) — each (1, 32, 8, 8)
            tick_layer_h = []
            for tick_hiddens in all_tick_hiddens:
                layer_h = []
                for h, _c in tick_hiddens:
                    layer_h.append(h.squeeze(0).cpu().numpy())  # (32, 8, 8)
                tick_layer_h.append(np.stack(layer_h, axis=0))  # (num_layers, 32, 8, 8)
            hs_arr = np.stack(tick_layer_h, axis=0)  # (num_ticks, num_layers, 32, 8, 8)

            # Greedy action during collection (deterministic)
            action = torch.argmax(logits, dim=-1).item()

            episode["observations"].append(obs.copy())
            episode["hidden_states"].append(hs_arr)
            episode["agent_positions"].append(env.get_agent_pos())
            episode["box_positions"].append(env.get_box_positions())
            episode["actions"].append(action)

            obs, reward, done, info = env.step(action)

            episode["rewards"].append(reward)
            episode["dones"].append(done)

            # Update hidden state; reset if done
            mask = 0.0 if done else 1.0
            mask_t = torch.FloatTensor([mask]).view(1, 1, 1, 1).to(device)
            hidden = [(h * mask_t, c * mask_t) for h, c in new_hidden]

        # Convert lists to arrays
        episode["observations"]  = np.stack(episode["observations"],  axis=0)
        episode["hidden_states"] = np.stack(episode["hidden_states"], axis=0)
        episode["actions"]       = np.array(episode["actions"],  dtype=np.int32)
        episode["rewards"]       = np.array(episode["rewards"],  dtype=np.float32)
        episode["dones"]         = np.array(episode["dones"],    dtype=bool)
        episode["solved"]        = bool(np.any(episode["rewards"] > 0.5))

        if episode["solved"]:
            solve_count += 1

        all_episodes.append(episode)

        if (ep + 1) % 100 == 0:
            sr = solve_count / (ep + 1)
            tqdm.write(f"Episode {ep+1}/{n_episodes} | Solve rate: {sr:.3f}")

    return all_episodes


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg  = ckpt.get("cfg", {})

    agent = DRCAgent(
        hidden_channels = cfg.get("hidden_channels", 32),
        num_layers      = cfg.get("num_layers", 2),
        num_ticks       = cfg.get("num_ticks", 3),
    ).to(device)
    agent.load_state_dict(ckpt["model_state"])
    agent.eval()
    print(f"Loaded agent from {args.checkpoint} "
          f"(step {ckpt.get('global_step', 0):,})")

    # Create environment
    env = BoxobanEnv(
        data_dir   = args.data_dir or cfg.get("data_dir"),
        split      = args.split,
        difficulty = args.difficulty,
        max_steps  = 400,
        seed       = args.seed,
    )

    # Collect data
    episodes = collect_probe_data(agent, env, args.n_episodes, device)

    # Save
    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    with open(args.save_path, "wb") as f:
        pickle.dump(episodes, f, protocol=4)

    solve_rate = np.mean([ep["solved"] for ep in episodes])
    print(f"\nCollected {len(episodes)} episodes | "
          f"Solve rate: {solve_rate:.3f}")
    print(f"Saved to {args.save_path}")


if __name__ == "__main__":
    main()
