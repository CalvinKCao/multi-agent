"""
Vectorised MA Boxoban factory.  Returns (obs_A, obs_B) at each step.
obs_A / obs_B each shape (N, 10, 8, 8).
"""

import multiprocessing as mp
import numpy as np
from typing import List, Optional
import cloudpickle
import pickle


def _ma_worker(remote, parent_remote, env_fn_pickle):
    parent_remote.close()
    env = pickle.loads(env_fn_pickle)()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                (obs_a, obs_b), rew, done, info = env.step(data)
                if done:
                    obs_a, obs_b = env.reset()
                remote.send(((obs_a, obs_b), rew, done, info))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "get_agent_a_pos":
                remote.send(env.get_agent_a_pos())
            elif cmd == "get_agent_b_pos":
                remote.send(env.get_agent_b_pos())
            elif cmd == "get_box_positions":
                remote.send(env.get_box_positions())
            elif cmd == "close":
                remote.close(); break
            else:
                raise RuntimeError(f"Unknown cmd: {cmd}")
    except EOFError:
        pass


class SubprocMAVecEnv:
    """N parallel MA envs in subprocesses, stepped in lock-step."""

    def __init__(self, env_fns):
        self.n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in env_fns])
        self.processes = []
        for wr, r, fn in zip(self.work_remotes, self.remotes, env_fns):
            p = mp.Process(
                target=_ma_worker,
                args=(wr, r, cloudpickle.dumps(fn)),
                daemon=True,
            )
            p.start()
            self.processes.append(p)
            wr.close()

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        pairs = [r.recv() for r in self.remotes]
        return (
            np.stack([p[0] for p in pairs], axis=0),
            np.stack([p[1] for p in pairs], axis=0),
        )

    def step(self, actions_A, actions_B):
        for r, a_a, a_b in zip(self.remotes, actions_A, actions_B):
            r.send(("step", (int(a_a), int(a_b))))
        results = [r.recv() for r in self.remotes]
        obs_A   = np.stack([res[0][0] for res in results], axis=0)
        obs_B   = np.stack([res[0][1] for res in results], axis=0)
        rewards = np.array([res[1] for res in results], dtype=np.float32)
        dones   = np.array([res[2] for res in results], dtype=bool)
        infos   = [res[3] for res in results]
        return obs_A, obs_B, rewards, dones, infos

    def get_agent_a_pos(self) -> List:
        for r in self.remotes: r.send(("get_agent_a_pos", None))
        return [r.recv() for r in self.remotes]

    def get_agent_b_pos(self) -> List:
        for r in self.remotes: r.send(("get_agent_b_pos", None))
        return [r.recv() for r in self.remotes]

    def get_box_positions(self) -> List:
        for r in self.remotes: r.send(("get_box_positions", None))
        return [r.recv() for r in self.remotes]

    def close(self):
        for r in self.remotes:
            try: r.send(("close", None))
            except BrokenPipeError: pass
        for p in self.processes: p.join()

    def __len__(self): return self.n_envs


class DummyMAVecEnv:
    """Single-process sequential version for debugging."""

    def __init__(self, env_fns):
        self.envs   = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)

    def reset(self):
        pairs = [env.reset() for env in self.envs]
        return (
            np.stack([p[0] for p in pairs], axis=0),
            np.stack([p[1] for p in pairs], axis=0),
        )

    def step(self, actions_A, actions_B):
        obs_A_l, obs_B_l, rews, dones, infos = [], [], [], [], []
        for env, a_a, a_b in zip(self.envs, actions_A, actions_B):
            (obs_a, obs_b), rew, done, info = env.step((int(a_a), int(a_b)))
            if done:
                obs_a, obs_b = env.reset()
            obs_A_l.append(obs_a); obs_B_l.append(obs_b)
            rews.append(rew); dones.append(done); infos.append(info)
        return (
            np.stack(obs_A_l, axis=0), np.stack(obs_B_l, axis=0),
            np.array(rews, dtype=np.float32), np.array(dones, dtype=bool), infos,
        )

    def get_agent_a_pos(self) -> List: return [e.get_agent_a_pos() for e in self.envs]
    def get_agent_b_pos(self) -> List: return [e.get_agent_b_pos() for e in self.envs]
    def get_box_positions(self) -> List: return [e.get_box_positions() for e in self.envs]
    def close(self): pass
    def __len__(self): return self.n_envs


def make_ma_env(
    n_envs: int = 64,
    data_dir: Optional[str] = None,
    split: str = "train",
    difficulty: str = "unfiltered",
    max_steps: int = 400,
    seed: Optional[int] = None,
    use_subproc: bool = True,
):
    """Create N parallel MA Boxoban envs."""
    from drc_sokoban.envs.ma_boxoban_env import MABoxobanEnv

    def _make(i):
        def _fn():
            return MABoxobanEnv(
                data_dir=data_dir, split=split, difficulty=difficulty,
                max_steps=max_steps,
                seed=(seed + i) if seed is not None else None,
            )
        return _fn

    fns = [_make(i) for i in range(n_envs)]
    return (SubprocMAVecEnv if use_subproc else DummyMAVecEnv)(fns)
