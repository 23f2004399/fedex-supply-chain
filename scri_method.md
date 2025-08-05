# SCRI Methodology – Final Implementation Summary

## Objective

The goal is to develop a **Supply Chain Resilience Index (SCRI)** that quantifies the resilience of each customer (or node) in the supply chain based on:

- Frequency of disruptions
- Severity (cost/revenue impact)
- Lead time delays

This enables ranking of nodes from most to least resilient.

---

## Step 1: Data Grouping & Feature Mapping

The following groupings were used to extract core disruption dimensions from the raw dataset:

```python
group_cols = {
    'Frequency': ['Order Id'],  
    'Severity': ['Sales per customer', 'Order Item Quantity', 'Order Item Product Price'],
    'LeadTime': ['Days for shipping (real)', 'Days for shipment (scheduled)', 'Order Item Total']
}
```

The respective normalized components were constructed as:

```python
df_norm['F_norm'] = df_norm['Order Count']
df_norm['S_norm'] = df_norm[['Sales per customer', 'Order Item Quantity', 'Order Item Product Price']].mean(axis=1)
df_norm['L_norm'] = df_norm[['Days for shipping (real)', 'Days for shipment (scheduled)', 'Order Item Total']].mean(axis=1)
```

---

## Step 2: SCRI Calculation with Three Weighting Schemes

We calculated three variants of SCRI using the following weighting methods:

### 1. **Heuristic Weights** (based on expert judgment)

```python
heuristic_weights = np.array([0.4, 0.35, 0.25])
```

### 2. **PCA-derived Weights** (based on data variance explained)

```python
X = df_norm[['F_norm', 'S_norm', 'L_norm']]
pca = PCA(n_components=3)
pca.fit(X)
pca_weights = pca.explained_variance_ratio_
pca_weights /= pca_weights.sum()
```

Yields (example):  
`PCA-derived Weights: [0.527, 0.352, 0.121]`

### 3. **Equal Weights** (uniform importance)

```python
equal_weights = np.array([1/3, 1/3, 1/3])
```

### Final SCRI Scores

```python
df_norm['SCRI_heuristic'] = X.dot(heuristic_weights)
df_norm['SCRI_pca'] = X.dot(pca_weights)
df_norm['SCRI_equal'] = X.dot(equal_weights)
```

---

## Step 3: Evaluation – ROC & PR Curves (Without Ground Truth)

Due to the lack of external labels, we created a **natural heuristic label**:

```python
# Vulnerable if both frequency and severity are high
f_thresh = df['F_norm'].quantile(0.70)
s_thresh = df['S_norm'].quantile(0.70)
df['label'] = ((df['F_norm'] >= f_thresh) & (df['S_norm'] >= s_thresh)).astype(int)
```

Using this, we evaluated the discriminative power of each SCRI variant via ROC and PR curves.

### Results

| SCRI Variant     | ROC AUC | PR AUC |
|------------------|---------|--------|
| SCRI_heuristic   | 0.86    | 0.65   |
| SCRI_pca         | 0.88    | 0.71   |
| SCRI_equal       | 0.83    | 0.62   |

These results suggest that **PCA-weighted SCRI** performs best in capturing vulnerability based on frequency and severity.

---

## Step 4: Sensitivity Analysis

We evaluated SCRI's robustness to small perturbations in input features. Key scenarios and changes in SCRI:

| Scenario            | Δ SCRI (Avg) |
|---------------------|-------------|
| Frequency CV -15%   | -0.0101     |
| Frequency CV +15%   | +0.0101     |
| Severity +20%       | +0.0103     |
| Severity -20%       | -0.0103     |

The SCRI metric shows consistent and proportional responses, confirming its sensitivity is well-aligned with the expected behavior.

---

## Conclusion

- The SCRI framework is implemented with three weighting strategies.
- PCA-based SCRI outperformed heuristic and equal weighting based on internal heuristics.
- The metric is responsive to key disruption features, validating its usefulness in prioritizing supply chain nodes based on resilience.
