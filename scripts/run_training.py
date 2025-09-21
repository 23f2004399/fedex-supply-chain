# scripts/run_training.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import trange

from src.env_supplychain import SupplyChainSimEnv
from src.agent_qlearning import QLearningAgent, get_bins, get_action_space
from src.baselines.policy_sS import SsPolicy, SsParams
from src.baselines.policy_myopic import MyopicPolicy, MyopicParams
from scripts.run_baselines import run_policy, compute_kpis  # reuse functions

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "csv_results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

def run_qlearning(env_cfg, episodes=1000, seeds=[0, 1, 2]):
    curves = []
    results = []
    for seed in seeds:
        env = SupplyChainSimEnv(config=env_cfg, seed=seed)
        obs_bins = get_bins()
        action_space = get_action_space()
        agent = QLearningAgent(obs_bins, action_space)

        rewards = []
        for ep in trange(episodes, desc=f"QL seed {seed}"):
            obs = env.reset()
            done = False
            ep_rew = 0
            ep_log = []
            while not done:
                # action = agent.select_action(obs)
                # next_obs, reward, done, info = env.step(action)
                # agent.learn(obs, action, reward, next_obs, done)

                action_idx = agent.select_action(obs)
                q, e, m = action_space[action_idx]
                action = {"order_qty": q, "expedite": e, "mitigate": m}

                next_obs, reward, done, info = env.step(action)
                agent.update(obs, action_idx, reward, next_obs, done)

                # log
                demand = getattr(env, "demand_forecast", 10)
                inv = int(round(float(obs[0])))
                fulfilled = min(inv, demand)
                ep_log.append({
                    "cost": -float(reward),
                    "scri": float(info.get("scri", 0.0)),
                    "demand": int(demand),
                    "fulfilled": int(fulfilled)
                })
                ep_rew += reward
                obs = next_obs

            rewards.append(ep_rew)
            kpis = compute_kpis(ep_log)
            kpis["method"] = "qlearning"
            kpis["seed"] = seed
            kpis["episode"] = ep
            results.append(kpis)

        curves.append({"seed": seed, "rewards": rewards})

    # Save learning curves
    for c in curves:
        plt.plot(c["rewards"], alpha=0.6, label=f"seed {c['seed']}")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title("Q-learning Learning Curves")
    plt.legend()
    plt.savefig(os.path.join(FIG_DIR, "qlearning_learning_curves.png"))
    plt.close()

    pd.DataFrame(results).to_csv(os.path.join(CSV_DIR, "qlearning_kpis.csv"), index=False)
    print("Saved Q-learning results to csv_results/qlearning_kpis.csv")

def run_all(env_cfg):
    # 1. Train Q-learning (3 seeds × 1000 episodes)
    run_qlearning(env_cfg, episodes=1000, seeds=[0, 1, 2])

    # 2. Baselines (reuse run_baselines)
    from scripts.run_baselines import grid_sS, run_myopic
    sS_rows = grid_sS(env_cfg, episodes=50, seed=42)
    myopic_rows = run_myopic(env_cfg, episodes=50, seed=42)
    df_base = pd.DataFrame(sS_rows + myopic_rows)
    df_base.to_csv(os.path.join(CSV_DIR, "baseline_kpis.csv"), index=False)

    # 3. Benchmark table
    qdf = pd.read_csv(os.path.join(CSV_DIR, "qlearning_kpis.csv"))
    bl = pd.read_csv(os.path.join(CSV_DIR, "baseline_kpis.csv"))

    # mean KPIs
    q_mean = qdf.groupby("method")[["service_level","total_cost","scri_viol","VaR95","TVaR95"]].mean()
    bl_mean = bl.groupby("method")[["service_level","total_cost","scri_viol","VaR95","TVaR95"]].mean()
    bench = pd.concat([bl_mean, q_mean])
    bench.to_csv(os.path.join(CSV_DIR, "benchmark_table.csv"))

    print("Wrote benchmark_table.csv")

    # 4. Plot comparison
    bench[["total_cost","scri_viol"]].plot(kind="bar")
    plt.title("Baseline vs Q-learning KPIs")
    plt.ylabel("Value")
    plt.savefig(os.path.join(FIG_DIR, "baseline_vs_qlearning.png"))
    plt.close()
    print("Saved plots to /figures/")

if __name__ == "__main__":
    env_cfg = {"max_steps": 30}  # same config as before
    run_all(env_cfg)
