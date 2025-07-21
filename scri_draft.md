# Supply Chain Resilience Index (SCRI) – Draft Proposal

## Objective

To develop a **Supply Chain Resilience Index (SCRI)** that quantifies the resilience of each node in the supply chain by combining the impact of **Frequency**, **Severity (Cost)**, and **Lead Time** disruptions.

---

## 1. SCRI Formula

We define the Supply Chain Resilience Index (SCRI) for a node as:

```text
SCRI = w_f * F_norm + w_s * S_norm + w_l * L_norm
```

Where:

- `F_norm`: Normalized disruption **Frequency**
- `S_norm`: Normalized **Severity** (cost or revenue impact)
- `L_norm`: Normalized **Lead Time** delay
- `w_f`, `w_s`, `w_l`: Weights assigned to Frequency, Severity, and Lead Time, respectively

---

## 2. Proposed Weighting Scheme

Weights are chosen based on the relative perceived impact of each component on resilience.

| Component | Symbol | Proposed Weight | Rationale |
|----------|--------|------------------|-----------|
| Frequency | `w_f` | 0.4 | Frequent disruptions indicate instability |
| Severity (Cost) | `w_s` | 0.35 | High cost impact reduces adaptability |
| Lead Time | `w_l` | 0.25 | Longer delays reduce responsiveness |

Total weight = 1.0

---

## 3. Variable Mapping

| Field Name                        | Risk Type   | SCRI Component |
|----------------------------------|-------------|----------------|
| Type                             | Frequency   | F_norm         |
| Days for shipping (real)         | Lead Time   | L_norm         |
| Days for shipment (scheduled)    | Lead Time   | L_norm         |
| Benefit per order                | Other       | (Excluded)     |
| Sales per customer               | Cost        | S_norm         |
| Quantity per order               | Cost        | S_norm         |
| Orders per customer              | Frequency   | F_norm         |
| Late delivery rate               | Frequency   | F_norm         |
| Delivery gap (days)              | Lead Time   | L_norm         |
| Shipping cost per order          | Cost        | S_norm         |

---

## 4. Normalization Method

Each component will be normalized to bring the values into the `[0, 1]` range.

```text
X_norm = (X - min(X)) / (max(X) - min(X))
```

Where `X` is the raw value of the metric (e.g., frequency, cost, lead time).

---

## 5. Preliminary SCRI Computation (per Node)

Let:

- `F_norm_i`, `S_norm_i`, `L_norm_i` be the normalized values for node _i_.

Then:

```text
SCRI_i = 0.4 * F_norm_i + 0.35 * S_norm_i + 0.25 * L_norm_i
```

Higher SCRI indicates **greater vulnerability**; a lower SCRI indicates **higher resilience**.

---

## 6. Next Steps

- Apply normalization to real dataset values.
- Calculate SCRI for each node using the formula.
- Rank nodes by resilience (low to high SCRI).
