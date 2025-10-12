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