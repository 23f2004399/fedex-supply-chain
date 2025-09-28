#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Risk-aware Q-learning & DQN (v2)
- Adds SCRI penalty and optional CVaR-style tail-cost penalty to the reward.
- Trains Q-learning and DQN agents and saves CSVs + figures.

Expected repo layout:
  ./src/q_dqn_v2.py           <-- this file
  ./src/env_supplychain.py    <-- your environment
  ./csv_results/              <-- outputs (created if missing)
  ./figures/                  <-- figures (created if missing)
  ./scri_results.csv          <-- (optional) in repo root
  ./scri_results_labeled.csv  <-- (optional) in repo root

Run:
  cd src
  python q_dqn_v2.py --episodes_q 200 --episodes_dqn 500 --device cpu

Outputs:
  ../csv_results/qlearning_v2_*.csv
  ../csv_results/dqn_v2_*.csv
  ../figures/*v2*.png
"""

import os
import json
import math
import time
import argparse
import random
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

# --- Torch is only needed for DQN ---
import torch
import torch.nn as nn
import torch.optim as optim

# --------- ENV IMPORT (script is in ./src) ----------
try:
    # if we are inside ./src (recommended)
    from env_supplychain import SupplyChainSimEnv
except ImportError:
    # fallback if PYTHONPATH makes src a package
    from src.env_supplychain import SupplyChainSimEnv  # type: ignore


# ------------------------
# Paths (relative to ./src)
# ------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CSV_DIR = os.path.join(ROOT, "csv_results")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

SCRI_FILES = [
    os.path.join(ROOT, "scri_results_labeled.csv"),
    os.path.join(ROOT, "scri_results.csv"),
]


# ------------------------
# Risk shaping utilities
# ------------------------
@dataclass
class RiskConfig:
    lambda_scri: float = 10.0     # penalty multiplier for SCRI above threshold
    scri_threshold: float = 0.7   # threshold beyond which penalty applies

    lambda_cvar: float = 0.0      # 0 disables CVaR-like penalty
    cvar_alpha: float = 0.95      # 95% VaR
    cvar_bucket_len: int = 7      # "weekly" bucket (7 steps)
    cvar_window_buckets: int = 50 # rolling window of buckets for VaR estimate


class ExternalSCRI:
    """
    Optional external SCRI source from a CSV; robust to unknown schemas.
    Falls back to None if file missing or column not found.

    Supported patterns:
      - a column named 'scri'
      - (step-indexed) columns like 'step' + 'scri'
    If multiple rows, we map by running step index modulo dataset length.
    """
    def __init__(self, files: List[str]):
        self.data = None
        for f in files:
            if os.path.isfile(f):
                try:
                    df = pd.read_csv(f)
                    if "scri" in df.columns:
                        self.data = df["scri"].astype(float).values
                        break
                    # heuristic: look for a probability-like column
                    for c in df.columns:
                        if c.lower().strip() in ("scri", "risk", "risk_score"):
                            self.data = df[c].astype(float).values
                            break
                    if self.data is not None:
                        break
                except Exception:
                    continue

    def get(self, step_idx: int) -> Optional[float]:
        if self.data is None or len(self.data) == 0:
            return None
        return float(self.data[step_idx % len(self.data)])


def risk_adjust_reward(
    base_reward: float,
    info: dict,
    step_cost: float,
    rcfg: RiskConfig,
    # CVaR bookkeeping:
    weekly_sum: float,
    var_estimate: Optional[float],
    external_scri: Optional[float] = None,
) -> float:
    """
    Shape reward with:
      r' = r - λ_scri * max(0, scri - τ) - λ_cvar * max(0, weekly_cost - VaRα) / bucket_len
    """
    shaped = base_reward

    # SCRI (prefer env info; optional override from external CSV)
    scri_val = external_scri
    if scri_val is None:
        scri_val = float(info.get("scri", 0.0))

    scri_excess = max(0.0, scri_val - rcfg.scri_threshold)
    shaped -= rcfg.lambda_scri * scri_excess

    # CVaR-like tail penalty (on bucketed running cost)
    if rcfg.lambda_cvar > 0.0 and var_estimate is not None:
        # penalty only for current (partial) weekly bucket if it exceeds VaR
        excess = max(0.0, weekly_sum - var_estimate)
        shaped -= rcfg.lambda_cvar * (excess / max(1, rcfg.cvar_bucket_len))

    return shaped


# ------------------------
# Action / Observation utils
# ------------------------
def get_action_space_q_learning() -> List[Tuple[int, int, int]]:
    # Coarser action space like your Q-learning baseline (0..20 step 5)
    return [(q, e, m) for q in range(0, 21, 5) for e in [0, 1] for m in [0, 1]]

def get_action_space_dqn() -> List[Tuple[int, int, int]]:
    # Finer action space as in your DQN v0 (0..20 step 1)
    return [(q, e, m) for q in range(0, 21, 1) for e in [0, 1] for m in [0, 1]]

def idx_to_env_action(idx: int, action_space: List[Tuple[int, int, int]]):
    q, e, m = action_space[idx]
    return {"order_qty": int(q), "expedite": int(e), "mitigate": int(m)}

def norm_obs(obs: np.ndarray) -> np.ndarray:
    highs = np.array([100.0, 100.0, 30.0, 2.0, 1.0], dtype=np.float32)
    return (np.array(obs, dtype=np.float32) / highs).clip(0.0, 1.0)


# ------------------------
# Q-Learning (tabular, discretized)
# ------------------------
def discretize_obs(obs, bins):
    return tuple(np.digitize(obs[i], bins[i]) for i in range(len(obs)))

def get_bins():
    return [
        np.linspace(0, 100, 11),
        np.linspace(0, 100, 11),
        np.linspace(0, 30, 6),
        np.array([0, 1, 2]),
        np.linspace(0, 1, 6),
    ]

class QLearningAgent:
    def __init__(self, obs_bins, action_space, alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        self.q_table = defaultdict(lambda: np.zeros(len(action_space)))
        self.obs_bins = obs_bins
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

    def select_action(self, obs):
        state = discretize_obs(obs, self.obs_bins)
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.action_space))
        return int(np.argmax(self.q_table[state]))

    def update(self, obs, action_idx, reward, next_obs, done):
        s = discretize_obs(obs, self.obs_bins)
        s2 = discretize_obs(next_obs, self.obs_bins)
        best_next = np.max(self.q_table[s2])
        td_target = reward + self.gamma * best_next * (not done)
        td_error = td_target - self.q_table[s][action_idx]
        self.q_table[s][action_idx] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)


def run_qlearning_v2(episodes: int, max_steps: int, rcfg: RiskConfig,
                     run_name="qlearning_v2", seed: int = 42,
                     ext_scri: Optional[ExternalSCRI] = None) -> pd.DataFrame:
    np.random.seed(seed)
    random.seed(seed)

    env = SupplyChainSimEnv(seed=seed, config={"max_steps": max_steps})
    obs_bins = get_bins()
    action_space = get_action_space_q_learning()
    agent = QLearningAgent(obs_bins, action_space)

    # CVaR bookkeeping
    weekly_sum = 0.0
    step_in_bucket = 0
    bucket_len = rcfg.cvar_bucket_len
    bucket_window = deque(maxlen=rcfg.cvar_window_buckets)

    rows = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0

        # reset bucket at episode start
        weekly_sum = 0.0
        step_in_bucket = 0

        while not done:
            a_idx = agent.select_action(obs)
            action = idx_to_env_action(a_idx, action_space)

            next_obs, base_reward, done, info = env.step(action)

            # costs are negative reward in your setup
            step_cost = -float(base_reward)

            # Update weekly bucket
            weekly_sum += step_cost
            step_in_bucket += 1

            # current VaR estimate from past buckets
            var_estimate = None
            if len(bucket_window) > 0:
                var_estimate = float(np.percentile(bucket_window, rcfg.cvar_alpha * 100.0))

            # external SCRI if available
            ext_val = ext_scri.get(ep * max_steps + step_in_bucket - 1) if ext_scri else None

            # risk shaping
            shaped_reward = risk_adjust_reward(
                base_reward=base_reward,
                info=info,
                step_cost=step_cost,
                rcfg=rcfg,
                weekly_sum=weekly_sum,
                var_estimate=var_estimate,
                external_scri=ext_val,
            )

            agent.update(obs, a_idx, shaped_reward, next_obs, done)
            ep_reward += shaped_reward
            obs = next_obs

            # close bucket
            if step_in_bucket >= bucket_len:
                bucket_window.append(weekly_sum)
                weekly_sum = 0.0
                step_in_bucket = 0

        agent.decay_epsilon()
        rows.append(dict(episode=ep + 1, total_reward=ep_reward, epsilon=agent.epsilon))

    env.close()
    return pd.DataFrame(rows)


# ------------------------
# DQN (v2)
# ------------------------
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.ptr = 0
        self.full = False
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.nobs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity,), dtype=np.int64)
        self.rew = np.zeros((capacity,), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)

    def add(self, o, a, r, o2, d):
        self.obs[self.ptr] = o
        self.act[self.ptr] = a
        self.rew[self.ptr] = r
        self.nobs[self.ptr] = o2
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
            torch.from_numpy(self.nobs[idx]),
            torch.from_numpy(self.done[idx]),
        )

class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )
    def forward(self, x): return self.net(x)

@dataclass
class DQNConfig:
    max_steps: int = 30
    episodes: int = 500
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    start_training_after: int = 1_000
    train_freq: int = 1
    target_update_freq: int = 200
    grad_clip_norm: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_episodes: int = 400
    hidden_size: int = 128
    device: str = "cpu"
    seeds: List[int] = None
    run_name: str = "dqn_v2"

def epsilon_for_ep(ep_idx: int, cfg: DQNConfig) -> float:
    frac = min(1.0, ep_idx / max(1, cfg.eps_decay_episodes))
    return cfg.eps_start + (cfg.eps_end - cfg.eps_start) * frac

def train_dqn_one_seed_v2(cfg: DQNConfig, rcfg: RiskConfig, seed: int,
                          ext_scri: Optional[ExternalSCRI]) -> pd.DataFrame:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    env = SupplyChainSimEnv(config={"max_steps": cfg.max_steps}, seed=seed)
    action_space = get_action_space_dqn()
    n_actions = len(action_space)

    obs = env.reset()
    obs_dim = int(np.array(obs).shape[0])

    q = QNet(obs_dim, n_actions, hidden=cfg.hidden_size).to(cfg.device)
    q_t = QNet(obs_dim, n_actions, hidden=cfg.hidden_size).to(cfg.device)
    q_t.load_state_dict(q.state_dict()); q_t.eval()

    opt = optim.Adam(q.parameters(), lr=cfg.lr)
    huber = nn.SmoothL1Loss(reduction="none")
    rb = ReplayBuffer(cfg.buffer_size, obs_dim)

    rows = []
    grad_steps = 0
    recent_returns: List[float] = []
    start_time = time.perf_counter()

    # CVaR bookkeeping
    weekly_sum = 0.0
    step_in_bucket = 0
    bucket_len = rcfg.cvar_bucket_len
    bucket_window = deque(maxlen=rcfg.cvar_window_buckets)

    for ep in range(cfg.episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0
        eps = epsilon_for_ep(ep, cfg)

        # reset bucket at episode start
        weekly_sum = 0.0
        step_in_bucket = 0

        while not done:
            o_norm = norm_obs(np.array(obs, dtype=np.float32))
            if np.random.rand() < eps:
                a_idx = np.random.randint(n_actions)
            else:
                with torch.no_grad():
                    qs = q(torch.from_numpy(o_norm).unsqueeze(0).to(cfg.device))
                    a_idx = int(torch.argmax(qs, dim=1).item())

            action = idx_to_env_action(a_idx, action_space)
            next_obs, base_reward, done, info = env.step(action)
            step_cost = -float(base_reward)

            # update bucket
            weekly_sum += step_cost
            step_in_bucket += 1

            var_estimate = None
            if len(bucket_window) > 0:
                var_estimate = float(np.percentile(bucket_window, rcfg.cvar_alpha * 100.0))

            ext_val = ext_scri.get(ep * cfg.max_steps + step_in_bucket - 1) if ext_scri else None
            shaped_reward = risk_adjust_reward(
                base_reward=base_reward,
                info=info,
                step_cost=step_cost,
                rcfg=rcfg,
                weekly_sum=weekly_sum,
                var_estimate=var_estimate,
                external_scri=ext_val,
            )

            o2_norm = norm_obs(np.array(next_obs, dtype=np.float32))
            rb.add(o_norm, a_idx, shaped_reward, o2_norm, float(done))
            ep_reward += shaped_reward
            obs = next_obs

            # train
            if len(rb) >= cfg.start_training_after and ((grad_steps % cfg.train_freq) == 0):
                ob, ac, rw, ob2, dn = rb.sample(cfg.batch_size)
                ob = ob.to(cfg.device).float()
                ac = ac.to(cfg.device).long()
                rw = rw.to(cfg.device).float()
                ob2 = ob2.to(cfg.device).float()
                dn = dn.to(cfg.device).float()

                qvals = q(ob).gather(1, ac.view(-1, 1)).squeeze(1)
                with torch.no_grad():
                    next_act = q(ob2).argmax(dim=1)
                    next_q = q_t(ob2).gather(1, next_act.view(-1, 1)).squeeze(1)
                    target = rw + cfg.gamma * (1.0 - dn) * next_q

                loss = huber(qvals, target).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), cfg.grad_clip_norm)
                opt.step()

                grad_steps += 1
                if grad_steps % cfg.target_update_freq == 0:
                    q_t.load_state_dict(q.state_dict())

            if step_in_bucket >= bucket_len:
                bucket_window.append(weekly_sum)
                weekly_sum = 0.0
                step_in_bucket = 0

        recent_returns.append(ep_reward)
        if len(recent_returns) > 50:
            recent_returns.pop(0)
        rows.append(dict(
            seed=seed,
            episode=ep + 1,
            total_reward=float(ep_reward),
            epsilon=float(eps),
            return_ma50=float(np.mean(recent_returns)) if recent_returns else float("nan"),
            wall_clock_s=float(time.perf_counter() - start_time),
        ))

    env.close()
    return pd.DataFrame(rows)


# ------------------------
# Plot helpers
# ------------------------
import matplotlib.pyplot as plt

def plot_reward_curves(dfs: List[pd.DataFrame], labels: List[str], title: str, save_as: str):
    plt.figure()
    for df, lab in zip(dfs, labels):
        plt.plot(df["episode"], df["total_reward"], label=lab)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (risk-shaped)")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, save_as))
    plt.close()


# ------------------------
# Main
# ------------------------
def main():
    parser = argparse.ArgumentParser(description="Risk-aware Q-learning & DQN (v2)")
    # Q-learning
    parser.add_argument("--episodes_q", type=int, default=200)
    # DQN
    parser.add_argument("--episodes_dqn", type=int, default=500)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0,1,2])
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    parser.add_argument("--max_steps", type=int, default=30)

    # Risk config
    parser.add_argument("--lambda_scri", type=float, default=10.0)
    parser.add_argument("--scri_threshold", type=float, default=0.7)
    parser.add_argument("--lambda_cvar", type=float, default=0.0)
    parser.add_argument("--cvar_alpha", type=float, default=0.95)
    parser.add_argument("--cvar_bucket_len", type=int, default=7)
    parser.add_argument("--cvar_window_buckets", type=int, default=50)

    args = parser.parse_args()
    rcfg = RiskConfig(
        lambda_scri=args.lambda_scri,
        scri_threshold=args.scri_threshold,
        lambda_cvar=args.lambda_cvar,
        cvar_alpha=args.cvar_alpha,
        cvar_bucket_len=args.cvar_bucket_len,
        cvar_window_buckets=args.cvar_window_buckets,
    )

    # Try to load optional external SCRI
    ext = ExternalSCRI(SCRI_FILES)

    # ---------- Q-learning v2 ----------
    print("[Q-learning v2] Training...")
    q_df = run_qlearning_v2(
        episodes=args.episodes_q,
        max_steps=args.max_steps,
        rcfg=rcfg,
        run_name="qlearning_v2",
        seed=42,
        ext_scri=ext,
    )
    q_csv = os.path.join(CSV_DIR, "qlearning_v2_curve.csv")
    q_df.to_csv(q_csv, index=False)
    print(f"[Q-learning v2] Saved -> {q_csv}")

    # ---------- DQN v2 (multi-seed) ----------
    print("[DQN v2] Training...")
    dqn_frames = []
    for sd in args.seeds:
        cfg = DQNConfig(
            episodes=args.episodes_dqn,
            max_steps=args.max_steps,
            device=args.device,
            seeds=[sd],
            run_name="dqn_v2",
        )
        df = train_dqn_one_seed_v2(cfg, rcfg, sd, ext)
        seed_csv = os.path.join(CSV_DIR, f"dqn_v2_seed{sd}.csv")
        df.to_csv(seed_csv, index=False)
        print(f"[DQN v2] Saved -> {seed_csv}")
        dqn_frames.append(df)

    dqn_all = pd.concat(dqn_frames, axis=0, ignore_index=True)
    dqn_agg = (
        dqn_all.groupby("episode")["total_reward"]
        .agg(["mean", "std"])
        .rename(columns={"mean":"reward_mean", "std":"reward_std"})
        .reset_index()
    )
    dqn_agg_csv = os.path.join(CSV_DIR, "dqn_v2_stability_summary.csv")
    dqn_agg.to_csv(dqn_agg_csv, index=False)
    print(f"[DQN v2] Stability summary -> {dqn_agg_csv}")

    # ---------- Plots ----------
    plot_reward_curves(
        dfs=[q_df],
        labels=["Q-learning v2"],
        title="Q-learning v2 (Risk-shaped) Reward",
        save_as="qlearning_v2_reward_curve.png",
    )

    # For DQN, plot mean over seeds if multiple
    if len(args.seeds) > 1:
        plt.figure()
        plt.plot(dqn_agg["episode"], dqn_agg["reward_mean"], label="DQN v2 (mean)")
        plt.fill_between(
            dqn_agg["episode"],
            dqn_agg["reward_mean"] - dqn_agg["reward_std"],
            dqn_agg["reward_mean"] + dqn_agg["reward_std"],
            alpha=0.2, label="±1 std"
        )
        plt.xlabel("Episode")
        plt.ylabel("Total Reward (risk-shaped)")
        plt.title("DQN v2 (Risk-shaped) Reward (mean ± std across seeds)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "dqn_v2_reward_curve.png"))
        plt.close()
    else:
        plot_reward_curves(
            dfs=dqn_frames,
            labels=[f"DQN v2 (seed {args.seeds[0]})"],
            title="DQN v2 (Risk-shaped) Reward",
            save_as="dqn_v2_reward_curve.png",
        )

    # Side-by-side comparison plot (first DQN seed vs QL)
    plot_reward_curves(
        dfs=[q_df, dqn_frames[0]],
        labels=["Q-learning v2", f"DQN v2 (seed {args.seeds[0]})"],
        title="Q-learning v2 vs DQN v2 (Risk-shaped) Reward",
        save_as="ql_vs_dqn_v2_reward_curve.png",
    )

    # Save a tiny config snapshot
    cfg_dump = {
        "risk_config": asdict(rcfg),
        "episodes_q": args.episodes_q,
        "episodes_dqn": args.episodes_dqn,
        "seeds": args.seeds,
        "max_steps": args.max_steps,
        "device": args.device,
    }
    with open(os.path.join(CSV_DIR, "q_dqn_v2_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2)
    print("[DONE] v2 training complete. CSVs and figures written.")
    

if __name__ == "__main__":
    main()
