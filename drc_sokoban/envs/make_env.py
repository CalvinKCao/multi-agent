"""
Factory for creating vectorised (parallel) Boxoban environments.

Uses multiprocessing via a simple SubprocVecEnv implementation so that
environment stepping is truly parallel and doesn't block the training loop.
"""

import multiprocessing as mp
import numpy as np
from typing import Optional, List
import cloudpickle
import pickle


# ── Worker process ─────────────────────────────────────────────────────────────

def _worker(remote, parent_remote, env_fn_pickle):
    """Worker loop that runs in a subprocess."""
    parent_remote.close()
    env_fn = pickle.loads(env_fn_pickle)
    env = env_fn()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                remote.send((obs, reward, done, info))
            elif cmd == "reset":
                obs = env.reset()
                remote.send(obs)
            elif cmd == "get_agent_pos":
                remote.send(env.get_agent_pos())
            elif cmd == "get_box_positions":
                remote.send(env.get_box_positions())
            elif cmd == "get_target_positions":
                remote.send(env.get_target_positions())
            elif cmd == "close":
                remote.close()
                break
            else:
                raise RuntimeError(f"Unknown command: {cmd}")
    except EOFError:
        pass


# ── Vectorised environment ─────────────────────────────────────────────────────

class SubprocVecEnv:
    """
    Runs N environments in separate processes, all stepped in lock-step.

    Interface mirrors OpenAI gym VecEnv:
      obs = env.reset()           → (N, 7, 8, 8) float32
      obs, rews, dones, infos = env.step(actions)  → actions: (N,) int
    """

    def __init__(self, env_fns):
        self.n_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in env_fns])
        self.processes = []
        for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns):
            fn_pickle = cloudpickle.dumps(env_fn)
            p = mp.Process(
                target=_worker,
                args=(work_remote, remote, fn_pickle),
                daemon=True,
            )
            p.start()
            self.processes.append(p)
            work_remote.close()

    def reset(self) -> np.ndarray:
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs, axis=0)

    def step(self, actions: np.ndarray):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", int(action)))
        results = [remote.recv() for remote in self.remotes]
        obs, rewards, dones, infos = zip(*results)
        return (
            np.stack(obs, axis=0),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            list(infos),
        )

    def get_agent_pos(self) -> List:
        for remote in self.remotes:
            remote.send(("get_agent_pos", None))
        return [remote.recv() for remote in self.remotes]

    def get_box_positions(self) -> List:
        for remote in self.remotes:
            remote.send(("get_box_positions", None))
        return [remote.recv() for remote in self.remotes]

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except BrokenPipeError:
                pass
        for p in self.processes:
            p.join()

    def __len__(self):
        return self.n_envs


class DummyVecEnv:
    """
    Sequential (single-process) vectorised environment.
    Slower than SubprocVecEnv but easier to debug and works without pickling.
    """

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.n_envs = len(self.envs)

    def reset(self) -> np.ndarray:
        return np.stack([env.reset() for env in self.envs], axis=0)

    def step(self, actions: np.ndarray):
        obs_list, rew_list, done_list, info_list = [], [], [], []
        for env, action in zip(self.envs, actions):
            obs, rew, done, info = env.step(int(action))
            if done:
                obs = env.reset()
            obs_list.append(obs)
            rew_list.append(rew)
            done_list.append(done)
            info_list.append(info)
        return (
            np.stack(obs_list, axis=0),
            np.array(rew_list, dtype=np.float32),
            np.array(done_list, dtype=bool),
            info_list,
        )

    def get_agent_pos(self) -> List:
        return [env.get_agent_pos() for env in self.envs]

    def get_box_positions(self) -> List:
        return [env.get_box_positions() for env in self.envs]

    def get_target_positions(self) -> List:
        return [env.get_target_positions() for env in self.envs]

    def close(self):
        pass

    def __len__(self):
        return self.n_envs


# ── Factory function ───────────────────────────────────────────────────────────

def make_env(
    n_envs: int = 32,
    data_dir: Optional[str] = None,
    split: str = "train",
    difficulty: str = "unfiltered",
    max_steps: int = 120,
    max_steps_range: int = 5,
    step_penalty: float = -0.01,
    grid_size: int = 8,
    seed: Optional[int] = None,
    use_subproc: bool = True,
    level_generator=None,
):
    """Create N parallel Boxoban environments."""
    from drc_sokoban.envs.boxoban_env import BoxobanEnv

    def make_single(i):
        def _fn():
            return BoxobanEnv(
                data_dir=data_dir,
                split=split,
                difficulty=difficulty,
                max_steps=max_steps,
                max_steps_range=max_steps_range,
                step_penalty=step_penalty,
                grid_size=grid_size,
                seed=(seed + i) if seed is not None else None,
                level_generator=level_generator,
            )
        return _fn

    env_fns = [make_single(i) for i in range(n_envs)]
    VecEnvCls = SubprocVecEnv if use_subproc else DummyVecEnv
    return VecEnvCls(env_fns)
