# Deep Q-Network (DQN) in FedEx Supply Chain Project

---

## 1. Overview of DQN

The **Deep Q-Network (DQN)** is an advanced **model-free reinforcement learning algorithm** that extends Q-Learning using **deep neural networks** to approximate the Q-value function.  

While classical Q-learning relies on a tabular Q-table (feasible only for small, discrete environments), DQN uses a neural network to handle **large or continuous state spaces**, making it ideal for complex real-world systems like supply chains.

In the FedEx Supply Chain simulation, DQN enables an agent to learn **optimal ordering, expediting, and mitigation strategies** directly from interaction data—without explicit knowledge of the system’s transition probabilities.

---

## 2. Core Concepts of DQN

### 2.1 Markov Decision Process (MDP)

DQN operates over the same MDP foundation as Q-Learning, defined by:

- **S** → Set of all possible states (e.g., inventory, backlog, lead time, disruption index, etc.)
- **A** → Set of actions available to the agent (order, expedite, mitigate)
- **R(s,a)** → Reward (negative of total cost incurred)
- **P(s'|s,a)** → Transition probability to next state `s'` given current `(s,a)`
- **γ (gamma)** → Discount factor for future rewards

The objective remains to find an **optimal policy**:

$$
\pi^*(s) = \arg\max_a Q^*(s,a)
$$

---

### 2.2 Neural Approximation of Q-Function

Instead of maintaining a lookup table, DQN approximates the optimal action-value function using a deep neural network parameterized by θ:

$$
Q(s, a; \theta) \approx Q^*(s, a)
$$

The network takes the **state vector** as input and outputs a Q-value for each possible action.  
The best action in a given state is simply:

$$
a_t = \arg\max_a Q(s_t, a; \theta)
$$

---

### 2.3 Experience Replay

To break the strong correlations between consecutive samples, DQN employs a **Replay Buffer** that stores past transitions:

$$
(s_t, a_t, r_t, s_{t+1}, \text{done})
$$

During training, random mini-batches are sampled from this memory, ensuring:
- Better data efficiency
- More stable learning
- Reduced variance in gradient updates

---

### 2.4 Target Network

To stabilize training, DQN maintains two networks:
- **Q-Network (online network)** — updated every step.
- **Target Network** — updated periodically (every few hundred gradient steps).

The target network’s parameters (θ⁻) are frozen and only synced occasionally:

$$
\theta^- \leftarrow \theta
$$

This prevents oscillations and divergence during bootstrapped updates.

---

## 3. Mathematical Foundation of DQN

### 3.1 Bellman Equation for DQN

The target for the neural network regression is based on the **Bellman Optimality Equation**:

$$
y_t = r_t + \gamma (1 - \text{done}) \max_{a'} Q(s_{t+1}, a'; \theta^-)
$$

The DQN minimizes the **temporal difference (TD) loss**:

$$
L(\theta) = \mathbb{E}_{(s,a,r,s')} \Big[ \big( y_t - Q(s, a; \theta) \big)^2 \Big]
$$

---

### 3.2 Double DQN Adjustment

To reduce the overestimation bias of Q-values, the project uses the **Double DQN** variant:

1. Select next action using the online network:
   $$
   a' = \arg\max_a Q(s_{t+1}, a; \theta)
   $$

2. Evaluate it using the target network:
   $$
   y_t = r_t + \gamma Q(s_{t+1}, a'; \theta^-)
   $$

This separation stabilizes learning by preventing inflated Q-estimates.

---

### 3.3 Gradient Update Rule

The network parameters are updated via gradient descent on the Huber loss:

$$
\nabla_\theta L(\theta) = \mathbb{E} \Big[ (y_t - Q(s,a;\theta)) \nabla_\theta Q(s,a;\theta) \Big]
$$

The project uses:
- **Optimizer:** Adam (learning rate 1e-3)
- **Loss:** SmoothL1Loss (Huber)
- **Gradient Clipping:** 10.0

---

## 4. Implementation in FedEx Supply Chain Project

### 4.1 State Representation

Each state vector encodes the real-time operational snapshot:

$$
s_t = [\text{inventory}, \text{backlog}, \text{leadtime}, \text{disruption\_flag}, \text{SCRI}]
$$

This is normalized by dividing with a fixed upper bound `[100, 100, 30, 2, 1]` to stabilize neural network training.

---

### 4.2 Action Representation

Each action consists of three decision components:

$$
a_t = (\text{order\_qty}, \text{expedite}, \text{mitigate})
$$

where:
- order_qty ∈ {0, 1, 2, …, 20}
- expedite ∈ {0, 1}
- mitigate ∈ {0, 1}

Total actions = 21 × 2 × 2 = **84 discrete actions**  
These are flattened into an integer index (0–83) and later decoded back when interacting with the environment.

---

### 4.3 Reward Function

The reward returned by the environment is a **negative cost function**:

$$
r_t = -(\text{holding\_cost} + \text{stockout\_cost} + \text{order\_cost} + \text{disruption\_cost} + \text{SCRI\_penalty})
$$

This motivates the agent to **minimize supply-chain costs** and **maximize efficiency**.

---

### 4.4 Environment Dynamics

The FedEx environment models stochastic disruptions using correlated heavy-tailed variables (e.g., via a Student-t copula).  
Hence, the next state distribution \( P(s'|s,a) \) is **unknown and noisy**, mimicking real-world uncertainty in lead times, demand, and disruptions.

The DQN agent interacts with this environment in episodic cycles of fixed length (e.g., 30 steps).

---

### 4.5 Training Pipeline

1. **Initialize** replay buffer, online Q-network, and target Q-network.
2. For each episode:
   - Reset environment → get `state`
   - For each step:
     1. Choose action via ε-greedy strategy:
        $$
        a_t = 
        \begin{cases}
        \text{random action,} & \text{with prob } \varepsilon_t \\
        \arg\max_a Q(s_t,a;\theta), & \text{otherwise}
        \end{cases}
        $$
     2. Execute action, observe `(r_t, s_{t+1}, done)`
     3. Store transition in replay buffer
     4. Sample mini-batch and compute loss
     5. Update network parameters via gradient descent
     6. Every `target_update_freq` steps: synchronize target network
3. **Decay ε** linearly from 1.0 → 0.05 over 400 episodes.

---

### 4.6 Hyperparameters

| Parameter | Description | Value |
|------------|-------------|--------|
| `γ` | Discount factor | 0.99 |
| `α` | Learning rate (Adam) | 1e-3 |
| `ε_start → ε_end` | Exploration range | 1.0 → 0.05 |
| `ε_decay_episodes` | Decay schedule | 400 |
| `batch_size` | Mini-batch size | 64 |
| `buffer_size` | Replay buffer capacity | 50,000 |
| `target_update_freq` | Target network sync frequency | 200 steps |
| `max_steps` | Max steps per episode | 30 |

---

## 5. Mathematical Formulation in This Project

Each training step performs the following:

1. **Target calculation:**

$$
y_i = r_i + \gamma (1 - d_i) Q(s_{i+1}, \arg\max_a Q(s_{i+1}, a; \theta); \theta^-)
$$

2. **Loss computation:**

$$
L = \frac{1}{N} \sum_i \text{Huber}(y_i - Q(s_i, a_i; \theta))
$$

3. **Parameter update:**

$$
\theta \leftarrow \theta - \alpha \nabla_\theta L
$$

---

## 6. Role of DQN in FedEx Supply Chain Optimization

The DQN agent learns **adaptive policies** that balance:
- **Inventory holding cost** vs **stockout risk**
- **Ordering cost** vs **mitigation investment**
- **Expediting cost** vs **service-level gains**

By continuously exploring and learning, the agent autonomously discovers efficient decision thresholds such as:
- When backlog exceeds a threshold, expedite orders.
- When disruption risk rises, activate mitigation early.
- Maintain optimal inventory buffer levels.

This enables a **data-driven operational policy** instead of a fixed rule-based system.

---

## 7. Why DQN Works for This Problem

Traditional Q-learning fails when:
- The state space is large or continuous.
- Dynamics are stochastic and nonlinear.

The DQN overcomes this by:
- **Function approximation:** Generalizing from limited samples.
- **Replay memory:** Reducing correlation in updates.
- **Target network:** Stabilizing training dynamics.
- **Double DQN:** Preventing Q-value overestimation.

Together, these innovations let the DQN agent **simulate and learn FedEx-style supply-chain resilience** with minimal supervision.

---

## 8. References

- Mnih, V. et al. (2015). *Human-level control through deep reinforcement learning.* **Nature.**
- Hessel, M. et al. (2018). *Rainbow: Combining Improvements in Deep Reinforcement Learning.* **AAAI.**
- Sutton & Barto (2018). *Reinforcement Learning: An Introduction.*

---