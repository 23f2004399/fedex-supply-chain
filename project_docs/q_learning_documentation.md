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