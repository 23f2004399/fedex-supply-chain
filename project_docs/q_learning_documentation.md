# Q-Learning in FedEx Supply Chain Project

---

## 1. Overview of Q-Learning

Q-Learning is a **model-free reinforcement learning algorithm** that enables an agent to learn how to act optimally in a stochastic environment by interacting with it and receiving rewards or penalties.

In simple terms — Q-Learning helps an agent answer this question:
> “Given the current state, what action should I take to maximize my total future reward?”

The algorithm does not require prior knowledge of the environment’s dynamics — it learns by **trial and error**.

---

## 2. Core Concepts of Q-Learning

### 2.1 Markov Decision Process (MDP)

Q-Learning operates on the foundation of the **Markov Decision Process (MDP)**, which consists of:

- **S** → Set of states (e.g., inventory, disruption level, etc.)
- **A** → Set of possible actions (e.g., order quantity, expedite, mitigate)
- **P(s'|s,a)** → Transition probability to next state `s'` given current state `s` and action `a`
- **R(s,a)** → Reward obtained after performing action `a` in state `s`
- **γ (gamma)** → Discount factor for future rewards

The goal is to find an optimal policy π*(s) that maximizes the expected cumulative reward.

---

### 2.2 Q-Value Function

The **Q-value** (or **action-value**) function represents the expected cumulative reward of taking an action `a` in a given state `s`, and then following the optimal policy thereafter:

$$
Q^*(s, a) = \mathbb{E} \left[ \sum_{t=0}^{\infty} \gamma^t r_{t+1} \,\middle|\, s_0 = s, a_0 = a, \pi^* \right]
$$

The optimal policy is derived as:

$$
\pi^*(s) = \arg\max_a Q^*(s, a)
$$

---

### 2.3 Bellman Optimality Equation

Q-Learning is grounded on the **Bellman Equation**:

$$
Q^*(s, a) = \mathbb{E} \left[ r + \gamma \max_{a'} Q^*(s', a') \right]
$$

This recursive relationship enables iterative updates of Q-values.

---

### 2.4 Q-Learning Update Rule

At each step, after observing a transition `(s, a, r, s')`, Q-Learning updates its estimate as:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

Where:
- **α** → Learning rate (how strongly new info overrides old info)
- **γ** → Discount factor (importance of future rewards)
- **r** → Immediate reward
- **max Q(s', a')** → Maximum future reward achievable from next state

Over many episodes, Q-values converge to the optimal policy.

---

## 3. Relevance of Q-Learning in the FedEx Supply Chain Project

### 3.1 Real-World Context

In FedEx’s supply chain, several uncertain factors affect operations:
- Customer demand varies unpredictably.
- Disruptions can occur due to logistics issues or supply chain risks.
- Lead times can fluctuate due to global events.

Traditional rule-based policies fail to adapt optimally to such uncertainty.  
**Q-Learning** provides a data-driven way to **learn adaptive policies** that minimize costs and maintain resilience.

---

### 3.2 What Q-Learning Learns Here

In this project, Q-Learning learns **how to manage inventory and logistics decisions** optimally. Specifically, it learns to:

- Decide **how much to order** at each step (`order_qty`).
- Decide **whether to expedite** shipments (`expedite`).
- Decide **whether to mitigate** disruptions (`mitigate`).

The agent explores and discovers **policies that reduce total cost and risk**.

---

## 4. Mathematical Foundation for This Project

### 4.1 State Representation

The **state vector** in the simulation environment is:

$$
s_t = [\text{inventory}, \text{outstanding}, \text{leadtime}, \text{disruption}, \text{SCRI}]
$$

This captures the full operational status of the supply chain at time `t`.

---

### 4.2 Action Representation

Each action $ a_t $ is a combination of three discrete choices:

$$
a_t = (\text{order\_qty}, \text{expedite}, \text{mitigate})
$$

where:
- order_qty ∈ {0, 5, 10, 15, 20}
- expedite ∈ {0, 1}
- mitigate ∈ {0, 1}

Total action combinations = 5 × 2 × 2 = **20**.

---

### 4.3 Reward Function

The environment returns a **negative cost** as the reward:

$$
r_t = -(\text{holding\_cost} + \text{stockout\_cost} + \text{order\_cost} + \text{disruption\_cost} + \text{SCRI\_penalty})
$$

The costs are computed as:

$$
\begin{align}
\text{holding\_cost} &= 0.1 \times \text{inventory} \\\\
\text{stockout\_cost} &= 2.0 \times \text{stockout} \\\\
\text{order\_cost} &= 1.0 \times \text{order\_qty} \\\\
\text{disruption\_cost} &= 5.0 \times (\text{disruption} > 0) \\\\
\text{SCRI\_penalty} &= 10.0 \times (\text{SCRI} > 0.7)
\end{align}
$$

The agent thus aims to **maximize rewards → minimize costs.**

---

### 4.4 Transition Dynamics

The environment evolves based on stochastic processes driven by a **Student-t Copula**, modeling correlated variables:

- **Lead time (L)** ~ t-distribution (heavy-tailed)
- **Demand severity (D)** ~ correlated with disruptions
- **Inter-arrival times** ~ randomly sampled

This gives realistic, dependent variations similar to FedEx’s global logistics.

---

### 4.5 Bellman Update in This Project

The update used in `agent_qlearning.py` is:

```python
td_target = reward + gamma * np.max(q_table[next_state]) * (not done)
td_error = td_target - q_table[state][action_idx]
q_table[state][action_idx] += alpha * td_error
```

Mathematically:

$$
Q_{new}(s_t, a_t) = Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$

This allows the agent to incrementally refine its understanding of the long-term impact of each decision.

---

## 5. Implementation in Code

### 5.1 Environment: `env_supplychain.py`

Key classes & methods:

| Component | Purpose |
|------------|----------|
| `SupplyChainSimEnv` | Inherits from `gym.Env`; defines state, action, reward, and transitions. |
| `_demand_process()` | SimPy process generating stochastic demand via Student-t Copula. |
| `_disruption_process()` | Introduces random disruptions every few steps. |
| `step(action)` | Executes order, updates inventory, computes costs, and returns next state & reward. |
| `render()` | Prints stepwise environment metrics. |

---

### 5.2 Agent: `agent_qlearning.py`

Core class:

```python
class QLearningAgent:
    def __init__(self, obs_bins, action_space, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.05):
        ...
```

#### Key Parameters:
- `alpha` → learning rate
- `gamma` → discount factor
- `epsilon` → exploration probability
- `epsilon_decay` → gradual decrease of ε per episode

#### Core Methods:
- `select_action(obs)` → ε-greedy policy for exploration/exploitation
- `update(obs, action_idx, reward, next_obs, done)` → Q-table update
- `decay_epsilon()` → reduces ε after each episode

---

### 5.3 Discretization of Observations

Since state values are continuous (inventory, SCRI, etc.), they are **binned** into discrete categories for tabular Q-learning:

```python
def discretize_obs(obs, bins):
    return tuple(np.digitize(obs[i], bins[i]) for i in range(len(obs)))
```

This transforms each observation into a tuple key used in the Q-table.

---

### 5.4 Training Loop

```python
for episode in range(n_episodes):
    obs = env.reset()
    done = False
    while not done:
        action_idx = agent.select_action(obs)
        action = action_space[action_idx]
        next_obs, reward, done, info = env.step(action_dict)
        agent.update(obs, action_idx, reward, next_obs, done)
        obs = next_obs
    agent.decay_epsilon()
```

During training:
- The agent interacts with the simulated environment.
- Receives feedback (reward).
- Updates its Q-table.
- Gradually learns optimal supply chain management policies.

---

## 6. Evaluation and KPIs

The project compares the **Q-Learning agent** with a **myopic baseline** (fixed-order quantity policy).

Metrics include:

| KPI | Description |
|------|--------------|
| **Service Level** | Percentage of customer demand fulfilled. |
| **Total Cost** | Sum of all operational costs. |
| **SCRI Violations** | Number of times Supply Chain Risk Index > 0.7. |
| **VaR95 / TVaR95** | Financial risk metrics capturing extreme cost scenarios. |

Plots generated:
- `qlearning_reward_curve.png`
- `rl_vs_baseline_reward_curve.png`

---

## 7. Mathematical Intuition — Why Q-Learning Works Here

The supply chain system can be abstracted as a **sequential decision process** under uncertainty.

Let’s denote:

$$
\mathcal{S} = 	\text{state space}, \quad \mathcal{A} = 	\text{action space}
$$
$$
P(s'|s,a) = 	\text{unknown stochastic transition}, \quad R(s,a) = 	\text{reward function.}
$$

Then, the optimal policy satisfies:

$$
Q^*(s,a) = \mathbb{E} [ R(s,a) + \gamma \max_{a'} Q^*(s',a') ]
$$

Q-Learning approximates this via iterative sampling.

Thus, over multiple episodes:
$$
\lim_{t {->} \infty} Q_t(s,a) = Q^*(s,a)
$$

which means the agent will eventually converge to optimal decisions for ordering and risk mitigation.

---

## 8. Practical Insight for FedEx

| Challenge | Q-Learning Contribution |
|------------|--------------------------|
| Uncertain demand | Learns adaptive reorder quantities |
| Random disruptions | Learns when to mitigate proactively |
| Cost vs. Service trade-off | Learns optimal balance minimizing total cost |
| Risk management | Reduces SCRI violations dynamically |

Over time, the trained Q-agent can serve as a **decision-support system** for operational managers.

---

## 9. Example Output Interpretation

**Training phase logs:**
```
Episode 180/200 | Reward: -295.2 | Epsilon: 0.09
```
- Reward improves (less negative) → lower total cost.
- Epsilon decreases → agent relies more on learned policy.

**Final KPIs:**
```
Q-learning KPI (mean ± std):
  service_level: 0.92 ± 0.03
  total_cost: 145.3 ± 22.1
  scri_viol: 2.1 ± 0.4
  VaR95: 190.5 ± 10.3
```
→ Q-learning achieves better cost-efficiency and lower risk than the static baseline.

---

## 10. Summary

| Aspect | Description |
|--------|--------------|
| **Algorithm** | Q-Learning (tabular, off-policy RL) |
| **Environment** | FedEx supply chain simulator (Gym + SimPy) |
| **State** | Inventory, lead time, disruption, SCRI |
| **Action** | Order, expedite, mitigate |
| **Reward** | Negative operational cost |
| **Goal** | Learn optimal logistics and inventory strategy |
| **Output** | Q-table representing best decisions for every situation |

---
