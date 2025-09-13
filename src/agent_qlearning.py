import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from src.env_supplychain import SupplyChainSimEnv

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "csv_results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

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

def get_action_space():
    return [(q, e, m) for q in range(0, 21, 5) for e in [0, 1] for m in [0, 1]]

class QLearningAgent:
    def __init__(self, obs_bins, action_space, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
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
        return np.argmax(self.q_table[state])

    def update(self, obs, action_idx, reward, next_obs, done):
        state = discretize_obs(obs, self.obs_bins)
        next_state = discretize_obs(next_obs, self.obs_bins)
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next * (not done)
        td_error = td_target - self.q_table[state][action_idx]
        self.q_table[state][action_idx] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

def compute_kpis(episode_log):
    total_demand = sum(x['demand'] for x in episode_log)
    fulfilled = sum(x['fulfilled'] for x in episode_log)
    service_level = fulfilled / total_demand if total_demand > 0 else 1.0

    scri_viol = sum(x['scri'] > 0.7 for x in episode_log)

    costs = [x['cost'] for x in episode_log]
    weekly_costs = [sum(costs[i:i+7]) for i in range(0, len(costs), 7)]
    if weekly_costs:
        var95 = np.percentile(weekly_costs, 95)
        tvar95 = np.mean([c for c in weekly_costs if c > var95]) if any(c > var95 for c in weekly_costs) else var95
    else:
        var95 = tvar95 = 0.0
    return {
        "service_level": service_level,
        "total_cost": sum(costs),
        "scri_viol": scri_viol,
        "VaR95": var95,
        "TVaR95": tvar95
    }

def run_baseline(env, episodes=10, order_qty=10):
    rewards, kpis = [], []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_log = []
        while not done:
            action = {"order_qty": order_qty, "expedite": 0, "mitigate": 0}
            next_obs, reward, done, info = env.step(action)
            ep_reward += reward
            ep_log.append({
                "cost": -reward, "scri": info["scri"],
                "demand": env.demand_forecast, "fulfilled": min(obs[0], env.demand_forecast)
            })
            obs = next_obs
        rewards.append(ep_reward)
        kpis.append(compute_kpis(ep_log))
    return rewards, kpis

def main():
    env = SupplyChainSimEnv(seed=42, config={"max_steps": 30})
    obs_bins = get_bins()
    action_space = get_action_space()
    agent = QLearningAgent(obs_bins, action_space)
    n_episodes = 200
    reward_curve = []
    kpi_list = []

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        ep_reward = 0
        ep_log = []
        while not done:
            action_idx = agent.select_action(obs)
            q, e, m = action_space[action_idx]
            action = {"order_qty": q, "expedite": e, "mitigate": m}
            next_obs, reward, done, info = env.step(action)
            agent.update(obs, action_idx, reward, next_obs, done)
            ep_reward += reward
            ep_log.append({
                "cost": -reward, "scri": info["scri"],
                "demand": env.demand_forecast, "fulfilled": min(obs[0], env.demand_forecast)
            })
            obs = next_obs
        agent.decay_epsilon()
        reward_curve.append(ep_reward)
        kpi_list.append(compute_kpis(ep_log))
        if (ep+1) % 20 == 0:
            print(f"Episode {ep+1}/{n_episodes} | Reward: {ep_reward:.1f} | Epsilon: {agent.epsilon:.2f}")

    reward_curve = np.array(reward_curve)
    plt.figure()
    plt.plot(reward_curve, label="Q-learning")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Episode Reward Curve (Q-learning)")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, "qlearning_reward_curve.png"))
    plt.close()

    np.savetxt(os.path.join(CSV_DIR, "qlearning_reward_curve.csv"), reward_curve, delimiter=",")

    print("\nRunning myopic baseline...")
    base_rewards, base_kpis = run_baseline(env, episodes=20)
    plt.figure()
    plt.plot(reward_curve, label="Q-learning")
    plt.plot(np.arange(0, len(base_rewards))*10, base_rewards, label="Myopic baseline")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("RL vs Baseline Reward Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(FIG_DIR, "rl_vs_baseline_reward_curve.png"))
    plt.close()

    def kpi_mean_std(kpis, name):
        arr = np.array([list(k.values()) for k in kpis])
        print(f"\n{name} KPI (mean ± std):")
        for i, key in enumerate(kpis[0].keys()):
            print(f"  {key}: {arr[:,i].mean():.2f} ± {arr[:,i].std():.2f}")

    print("\n=== KPI Report ===")
    kpi_mean_std(kpi_list, "Q-learning")
    kpi_mean_std(base_kpis, "Myopic baseline")

    import pandas as pd
    pd.DataFrame(kpi_list).to_csv(os.path.join(CSV_DIR, "qlearning_kpis.csv"), index=False)
    pd.DataFrame(base_kpis).to_csv(os.path.join(CSV_DIR, "baseline_kpis.csv"), index=False)

if __name__ == "__main__":
    main()