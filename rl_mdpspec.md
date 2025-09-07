# Supply Chain MDP Specification and KPIs

## 1. Markov Decision Process (MDP) Formalisation

### 1.1. States

Let the state at time *t*, `s_t`, be defined as a tuple capturing the relevant supply chain information:

- **Inventory levels**: `I_t` (vector for each SKU or aggregate)
- **Outstanding orders**: `O_t` (orders placed but not yet received)
- **Demand forecast**: `D_t` (predicted demand for next period)
- **Lead times**: `L_t` (time until replenishment arrives)
- **Disruption status**: `Δ_t` (binary or categorical, e.g., 0 = normal, 1 = delayed, 2 = disrupted)
- **SCRI status**: `S_t^SCRI` (current value of the Supply Chain Risk Index)
- **Other features**: e.g., supplier reliability, shipment in transit, etc.

Formally:

```
s_t = (I_t, O_t, D_t, L_t, Δ_t, S_t^SCRI, …)
```

---

### 1.2. Actions

At each time step, the agent chooses an action `a_t`:

- **Order quantity**: `Q_t` (vector for each SKU or aggregate)
- **Replenishment decision**: e.g., expedite, delay, cancel, or split orders
- **Mitigation actions**: e.g., switch supplier, increase safety stock, etc.

Formally:

```
a_t = (Q_t, replenishment_action_t, mitigation_action_t, …)
```

---

### 1.3. Transition Dynamics

The environment transitions from `s_t` to `s_{t+1}` according to stochastic dynamics:

- **Inventory update**:  
  `I_{t+1} = I_t + received_orders_t - demand_t`
- **Order pipeline update**: Outstanding orders are updated based on lead times and disruptions.
- **Demand realization**: Actual demand is sampled from a distribution (empirical or parametric, e.g., from DataCoSupplyChainDataset).
- **Disruption process**: Disruptions may occur stochastically, affecting lead times and SCRI.
- **SCRI update**: SCRI is recalculated based on new state and risk factors.

Formally:

```
P(s_{t+1} | s_t, a_t) = SupplyChainTransition(s_t, a_t)
```

---

### 1.4. Reward/Cost Function

The reward (or cost) at each step is defined as:

- **Holding cost**: `c_h * I_{t+1}`
- **Stockout/penalty cost**: `c_p * max(0, D_t - I_t)`
- **Ordering cost**: `c_o * Q_t`
- **Disruption cost**: `c_d * 1[Δ_t > 0]`
- **SCRI violation penalty**: `c_SCRI * 1[S_t^SCRI > threshold]`

Total cost at time *t*:

```
C_t = c_h * I_{t+1} 
    + c_p * max(0, D_t - I_t) 
    + c_o * Q_t 
    + c_d * 1[Δ_t > 0] 
    + c_SCRI * 1[S_t^SCRI > threshold]
```

The agent seeks to **minimise the expected cumulative cost** over the planning horizon.

---

## 2. Key Performance Indicators (KPIs)

### 2.1. Service Level

- **Definition**: Fraction of demand fulfilled from available inventory, over a given period.
- **Formula**:

```
Service Level = (Σ_t min(I_t, D_t)) / (Σ_t D_t)
```

---

### 2.2. Total Cost

- **Definition**: Sum of all incurred costs (holding, stockout, ordering, disruption, SCRI penalties) over the evaluation period.
- **Formula**:

```
Total Cost = Σ_t C_t
```

---

### 2.3. SCRI-Violation Count

- **Definition**: Number of time steps where the Supply Chain Risk Index (SCRI) exceeds a predefined threshold.
- **Formula**:

```
SCRI-Violation Count = Σ_t 1[S_t^SCRI > threshold]
```

---

### 2.4. VaR/TVaR at Weekly Horizon

- **Value-at-Risk (VaR)**: The α-quantile of the weekly cost distribution (e.g., 95th percentile).
- **Tail Value-at-Risk (TVaR)**: The expected cost, conditional on exceeding the VaR threshold.

Formulas:

```
VaR_α   = inf { x : P(Weekly Cost ≤ x) ≥ α }
TVaR_α  = E[Weekly Cost | Weekly Cost > VaR_α]
```

---

## 3. Assumptions

- Demand, lead times, and disruptions are empirically estimated from `DataCoSupplyChainDataset`.
- SCRI is computed as per the method in `scri_method.md` and results in `scri_results.csv`.
- All costs and thresholds are to be calibrated using historical data and business requirements.
- The MDP is episodic, with each episode corresponding to a fixed planning horizon (e.g., several weeks).
- Actions and states can be aggregated or disaggregated by SKU, location, or other relevant dimensions as needed.

---

## References

- `DataCoSupplyChainDataset.csv`
- `scri_method.md`, `scri_results.csv`
- Standard MDP and RL literature for supply chain management
