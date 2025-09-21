# scripts/run_baselines.py
import os
import json
import numpy as np
import pandas as pd
from tqdm import trange

# Your env (with Student-t copula etc.) and dynamics:
from src.env_supplychain import SupplyChainSimEnv   # same module name/path you uploaded
# KPI definitions aligned with your spec:
# service level, total cost, SCRI-violation count, weekly VaR/TVaR.  :contentReference[oaicite:6]{index=6}

from src.baselines.policy_sS import SsPolicy, SsParams
from src.baselines.policy_myopic import MyopicPolicy, MyopicParams

CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "csv_results")
os.makedirs(CSV_DIR, exist_ok=True)

def compute_kpis(ep_log, week=7):
    total_demand = sum(x["demand"] for x in ep_log)
    fulfilled = sum(x["fulfilled"] for x in ep_log)
    service_level = fulfilled / total_demand if total_demand > 0 else 1.0

    scri_viol = sum(1 for x in ep_log if x["scri"] > 0.7)

    costs = [x["cost"] for x in ep_log]
    weekly_costs = [sum(costs[i:i+week]) for i in range(0, len(costs), week)]
    if weekly_costs:
        var95 = float(np.percentile(weekly_costs, 95))
        tail = [c for c in weekly_costs if c > var95]
        tvar95 = float(np.mean(tail)) if len(tail) else var95
    else:
        var95 = tvar95 = 0.0

    return {
        "service_level": float(service_level),
        "total_cost": float(sum(costs)),
        "scri_viol": int(scri_viol),
        "VaR95": var95,
        "TVaR95": tvar95
    }

def run_policy(env, policy, episodes=50, seed=0, method_name="baseline"):
    rng = np.random.default_rng(seed)
    out = []
    for ep in trange(episodes, desc=method_name, leave=False):
        _ = env.seed(int(seed + ep))
        obs = env.reset()
        done = False
        ep_log = []
        while not done:
            # expose demand_forecast to myopic when available
            if hasattr(policy, "set_forecast"):
                policy.set_forecast(getattr(env, "demand_forecast", 10))
            action = policy.act(obs)
            next_obs, reward, done, info = env.step(action)

            # log per-step for KPIs (align with your env + KPI spec)  :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}
            demand = getattr(env, "demand_forecast", 10)
            inv = int(round(float(obs[0])))
            fulfilled = min(inv, demand)
            ep_log.append({
                "cost": -float(reward),    # env reward = -cost
                "scri": float(info.get("scri", 0.0)),
                "demand": int(demand),
                "fulfilled": int(fulfilled)
            })

            obs = next_obs

        k = compute_kpis(ep_log)
        k["method"] = method_name
        k["seed"] = int(seed)
        k["episode"] = int(ep)
        out.append(k)

    return out

def grid_sS(env_cfg, s_grid=range(0, 95, 5), S_grid=range(10, 105, 5), episodes=50, seed=0):
    results = []
    for s in s_grid:
        for S in S_grid:
            if S <= s:
                continue
            params = SsParams(s=s, S=S)
            policy = SsPolicy(params)
            env = SupplyChainSimEnv(config=env_cfg, seed=seed)
            out = run_policy(env, policy, episodes=episodes, seed=seed, method_name=f"sS(s={s},S={S})")
            for row in out:
                row["s"] = s; row["S"] = S
                results.append(row)
    return results

def run_myopic(env_cfg, episodes=50, seed=0):
    params = MyopicParams(safety_factor=0.0, expedite_threshold=0.9, mitigate_on_disruption=2)
    policy = MyopicPolicy(params)
    env = SupplyChainSimEnv(config=env_cfg, seed=seed)
    out = run_policy(env, policy, episodes=episodes, seed=seed, method_name="myopic")
    for row in out:
        row["s"] = np.nan; row["S"] = np.nan
    return out

def select_best_baselines(df):
    # best by total_cost (min) and by scri_viol (min); keep their rows
    idx_cost = df.groupby("method")["total_cost"].mean().idxmin()
    idx_scri = df.groupby("method")["scri_viol"].mean().idxmin()
    best_cost = df[df["method"] == idx_cost].copy()
    best_scri = df[df["method"] == idx_scri].copy()
    return best_cost, best_scri

def main():
    env_cfg = {"max_steps": 30}  # aligned with your training setup  :contentReference[oaicite:9]{index=9}

    # 1) Grid-search (s,S)
    sS_rows = grid_sS(env_cfg, episodes=50, seed=42)

    # 2) Myopic baseline
    myopic_rows = run_myopic(env_cfg, episodes=50, seed=42)

    # 3) Consolidate baselines and write 'baseline_kpis.csv'
    df_base = pd.DataFrame(sS_rows + myopic_rows)
    base_path = os.path.join(CSV_DIR, "baseline_kpis.csv")
    df_base.to_csv(base_path, index=False)

    # 4) If qlearning_kpis.csv exists, create rl_baseline_vs_ql.csv
    q_path = os.path.join(CSV_DIR, "qlearning_kpis.csv")
    if not os.path.exists(q_path):
        # fall back: try project root uploads
        q_path_alt = os.path.join(os.getcwd(), "qlearning_kpis.csv")
        if os.path.exists(q_path_alt):
            q_path = q_path_alt

    if os.path.exists(q_path):
        qdf = pd.read_csv(q_path)
        q_means = qdf[["total_cost","scri_viol"]].mean()

        # pick best baseline(s)
        best_cost, best_scri = select_best_baselines(df_base)

        rows = []
        if not best_cost.empty:
            b_mean = best_cost[["total_cost"]].mean().iloc[0]
            pct = 100.0*(b_mean - q_means["total_cost"])/b_mean
            rows.append({
                "baseline": best_cost["method"].iloc[0],
                "metric": "total_cost",
                "baseline_mean": b_mean,
                "qlearning_mean": q_means["total_cost"],
                "percent_change_q_vs_baseline": pct,
                "acceptance_pass": bool(pct >= 5.0)
            })
        if not best_scri.empty:
            b_mean = best_scri[["scri_viol"]].mean().iloc[0]
            pct = 100.0*(b_mean - q_means["scri_viol"])/b_mean
            rows.append({
                "baseline": best_scri["method"].iloc[0],
                "metric": "scri_viol",
                "baseline_mean": b_mean,
                "qlearning_mean": q_means["scri_viol"],
                "percent_change_q_vs_baseline": pct,
                "acceptance_pass": bool(pct >= 15.0)
            })

        comp = pd.DataFrame(rows)
        comp_path = os.path.join(CSV_DIR, "rl_baseline_vs_ql.csv")
        comp.to_csv(comp_path, index=False)
        print(f"Wrote comparison to {comp_path}")
    else:
        print("qlearning_kpis.csv not found; skipped RL vs baseline comparison.")

if __name__ == "__main__":
    main()
