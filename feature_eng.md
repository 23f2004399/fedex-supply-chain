# Feature Engineering - Derived Variables

Below are the proposed new features to derive, along with their formulas and a short rationale for inclusion in modeling.

---

## Inter-Arrival Time

**Definition**: Time difference between consecutive orders from the same customer.

**Formula**:

```text
Inter_Arrival_Time = Order_Date (n) - Order_Date (n-1) for same Customer_Id
```

**Rationale**:
> Captures purchasing frequency. Customers with shorter inter-arrival times are more engaged and predictable, helping demand forecasting and churn prediction.

---

## Disruption Severity Proxy

**Definition**: Product of late delivery risk and order cost as a proxy for impact of disruption.

**Formula**:

```text
Disruption_Severity = Late_delivery_risk × Order Item Total
```

**Rationale**:
> Models how costly a late delivery would be. Even if risk is low, a high-value order justifies prioritizing shipping. Useful for prioritization and service level planning.

---

## Lead-Time Variance

**Definition**: Variance between scheduled and actual shipping times.

**Formula**:

```text
Lead_Time_Variance = Days for shipping (real) - Days for shipment (scheduled)
```

**Rationale**:
> Measures fulfillment reliability. High variance indicates process inefficiency or supply chain risk. Important for inventory planning and SLA adherence.

---
