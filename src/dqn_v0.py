import os
import json
import time
import math
import argparse
import random
from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# --- ENV IMPORT ---
# This expects env_supplychain.py to be next to this file or on PYTHONPATH
from env_supplychain import SupplyChainSimEnv  # :contentReference[oaicite:2]{index=2}


# ------------------------
# Utils: action/observation
# ------------------------
def get_action_space() -> List[Tuple[int, int, int]]:
    """
    Flatten Dict(action) -> discrete index
    order_qty in [0..20], expedite in {0,1}, mitigate in {0,1}
    We match your baseline's granularity: order_qty step = 1 (21 actions)
    -> 21 * 2 * 2 = 84 actions
    """
    return [(q, e, m) for q in range(0, 21, 1) for e in [0, 1] for m in [0, 1]]


def idx_to_env_action(idx: int, action_space: List[Tuple[int, int, int]]):
    q, e, m = action_space[idx]
    return {"order_qty": int(q), "expedite": int(e), "mitigate": int(m)}


def norm_obs(obs: np.ndarray) -> np.ndarray:
    """
    Normalize by the env's stated high bounds:
      [100, 100, 30, 2, 1]
    """
    highs = np.array([100.0, 100.0, 30.0, 2.0, 1.0], dtype=np.float32)
    return (obs.astype(np.float32) / highs).clip(0.0, 1.0)


# ------------------------
# Replay Buffer
# ------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.ptr = 0
        self.full = False

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity,), dtype=np.int64)
        self.rew = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)

    def add(self, o, a, r, o2, d):
        self.obs[self.ptr] = o
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.next_obs[self.ptr] = o2
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.ptr

    def sample(self, batch_size: int):
        n = len(self)
        idx = np.random.randint(0, n, size=batch_size)
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.act[idx]),
            torch.from_numpy(self.rew[idx]),
            torch.from_numpy(self.next_obs[idx]),
            torch.from_numpy(self.done[idx]),
        )


# ------------------------
# MLP Q-network
# ------------------------
class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------
# Config dataclass
# ------------------------
@dataclass
class DQNConfig:
    # env
    max_steps: int = 30

    # training
    episodes: int = 500
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    start_training_after: int = 1_000
    train_freq: int = 1  # train every step
    target_update_freq: int = 200  # in gradient steps
    grad_clip_norm: float = 10.0

    # exploration
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 400  # linear decay over N episodes

    # network
    hidden_size: int = 128

    # seeds
    seeds: List[int] = None

    # others
    device: str = "cpu"
    csv_dir: str = "csv_results"
    config_dir: str = "configs"
    run_name: str = "dqn_v0"


# ------------------------
# Training loop per seed
# ------------------------
def train_one_seed(cfg: DQNConfig, seed: int) -> pd.DataFrame:
    # Repro
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Env (your custom Sim env)  :contentReference[oaicite:3]{index=3}
    env = SupplyChainSimEnv(config={"max_steps": cfg.max_steps}, seed=seed)
    action_space = get_action_space()
    n_actions = len(action_space)

    # Observations
    obs = env.reset()
    obs_dim = int(np.array(obs).shape[0])

    # DQN bits
    q = QNet(obs_dim, n_actions, hidden=cfg.hidden_size).to(cfg.device)
    q_tgt = QNet(obs_dim, n_actions, hidden=cfg.hidden_size).to(cfg.device)
    q_tgt.load_state_dict(q.state_dict())
    q_tgt.eval()

    optim_q = optim.Adam(q.parameters(), lr=cfg.lr)
    huber = nn.SmoothL1Loss(reduction="none")

    # Replay
    rb = ReplayBuffer(cfg.buffer_size, obs_dim)

    # Logging
    rows = []
    start_time = time.perf_counter()
    grad_steps = 0
    recent_returns: List[float] = []
    recent_losses: List[float] = []

    # Epsilon schedule (linear decay over episodes)
    def epsilon_for_ep(ep_idx):
        frac = min(1.0, ep_idx / max(1, cfg.eps_decay_episodes))
        return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * frac

    for ep in range(cfg.episodes):
        ep_reward = 0.0
        ep_losses = []

        obs = env.reset()
        done = False

        eps = epsilon_for_ep(ep)

        while not done:
            o_norm = norm_obs(np.array(obs, dtype=np.float32))
            if np.random.rand() < eps:
                a_idx = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    qs = q(torch.from_numpy(o_norm).unsqueeze(0).to(cfg.device))
                    a_idx = int(torch.argmax(qs, dim=1).item())

            action = idx_to_env_action(a_idx, action_space)
            next_obs, reward, done, _info = env.step(action)

            o2_norm = norm_obs(np.array(next_obs, dtype=np.float32))

            rb.add(o_norm, a_idx, reward, o2_norm, float(done))
            ep_reward += reward
            obs = next_obs

            # Train
            if len(rb) >= cfg.start_training_after and ((grad_steps % cfg.train_freq) == 0):
                ob, ac, rw, ob2, dn = rb.sample(cfg.batch_size)
                ob = ob.to(cfg.device).float()
                ac = ac.to(cfg.device).long()
                rw = rw.to(cfg.device).float()
                ob2 = ob2.to(cfg.device).float()
                dn = dn.to(cfg.device).float()

                qvals = q(ob).gather(1, ac.view(-1, 1)).squeeze(1)
                with torch.no_grad():
                    # Double DQN-ish action selection with current net:
                    next_act = q(ob2).argmax(dim=1)
                    next_q = q_tgt(ob2).gather(1, next_act.view(-1, 1)).squeeze(1)
                    target = rw + cfg.gamma * (1.0 - dn) * next_q

                loss = huber(qvals, target).mean()
                optim_q.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip_norm)
                optim_q.step()

                ep_losses.append(float(loss.item()))
                recent_losses.append(float(loss.item()))
                grad_steps += 1

                # Target hard update
                if grad_steps % cfg.target_update_freq == 0:
                    q_tgt.load_state_dict(q.state_dict())

        # Episode end: stability metrics
        recent_returns.append(ep_reward)
        if len(recent_returns) > 50:
            recent_returns.pop(0)

        ret_ma50 = float(np.mean(recent_returns)) if recent_returns else float("nan")
        ret_std50 = float(np.std(recent_returns)) if len(recent_returns) >= 2 else float("nan")

        mean_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
        std_loss = float(np.std(ep_losses)) if len(ep_losses) >= 2 else float("nan")

        wall_clock_s = time.perf_counter() - start_time

        rows.append(
            dict(
                seed=seed,
                episode=ep + 1,
                total_reward=float(ep_reward),
                epsilon=float(eps),
                mean_loss=mean_loss,
                std_loss=std_loss,
                return_ma50=ret_ma50,
                return_std50=ret_std50,
                wall_clock_s=float(wall_clock_s),
            )
        )

    env.close()
    return pd.DataFrame(rows)


# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="DQN v0 (lightweight)")
    parser.add_argument("--episodes", type=int, default=500, help="episodes per seed")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="list of seeds")
    parser.add_argument("--max_steps", type=int, default=30, help="env steps per episode")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="torch device")
    parser.add_argument("--run_name", type=str, default="dqn_v0", help="base name for outputs")
    args = parser.parse_args()

    cfg = DQNConfig(
        episodes=args.episodes,
        seeds=args.seeds,
        max_steps=args.max_steps,
        device=args.device,
        run_name=args.run_name,
    )

    # Create folders relative to script
    here = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(here, cfg.csv_dir)
    config_dir = os.path.join(here, cfg.config_dir)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)

    # Save config JSON
    config_path = os.path.join(config_dir, f"{cfg.run_name}_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"[INFO] Saved config -> {config_path}")


    all_frames = []
    for seed in cfg.seeds:
        print(f"\n[RUN] seed={seed} episodes={cfg.episodes}")
        df = train_one_seed(cfg, seed)
        seed_csv = os.path.join(csv_dir, f"{cfg.run_name}_seed{seed}.csv")
        df.to_csv(seed_csv, index=False)
        print(f"[INFO] seed={seed} -> saved CSV to {seed_csv}")
        all_frames.append(df)

    agg = pd.concat(all_frames, axis=0, ignore_index=True)
    agg_csv = os.path.join(csv_dir, f"{cfg.run_name}_aggregate.csv")
    agg.to_csv(agg_csv, index=False)
    print(f"[INFO] aggregate -> {agg_csv}")

    summary = (
        agg.groupby("episode")["total_reward"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "reward_mean", "std": "reward_std"})
        .reset_index()
    )
    summary_csv = os.path.join(csv_dir, f"{cfg.run_name}_stability_summary.csv")
    summary.to_csv(summary_csv, index=False)
    print(f"[INFO] stability summary -> {summary_csv}")

    print("\n[DONE] DQN v0 complete.")


if __name__ == "__main__":
    main()
