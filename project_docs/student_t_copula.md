# Student-t Copula in FedEx Supply Chain Project

---

## 3. Core Concepts

### 3.1 Dependence and Copulas

Let $( X_1, X_2, \ldots, X_n )$ be random variables representing stochastic elements of the supply chain (e.g., severity, lead-time, inter-arrival).  
Each has its marginal distribution $( F_i(x_i) )$. A **copula** couples these marginals into a joint distribution $( H )$:

$
H(x_1, x_2, ..., x_n) = C(F_1(x_1), F_2(x_2), ..., F_n(x_n))
$

The copula $( C )$ encodes the *dependence structure* independent of the marginals.

---

### 3.2 Student-t Copula Definition

Derived from the **multivariate Student-t distribution**, the Student-t copula is defined as:

$
C_{\nu, R}(u_1, ..., u_n) = t_{\nu, R}(t_\nu^{-1}(u_1), ..., t_\nu^{-1}(u_n))
$

Where:
- $( t_\nu^{-1} )$ is the inverse CDF of the univariate t-distribution,
- $( t_{\nu, R} )$ is the CDF of the multivariate t-distribution with correlation matrix $( R )$ and degrees of freedom $( \nu )$.

Parameters:
- $( R )$: Linear correlation matrix
- $( \nu )$: Degrees of freedom controlling tail thickness

---

## 4. Tail Dependence and Risk Implications

### 4.1 Tail Dependence Coefficient

Tail dependence measures how likely extreme co-movements occur:

$
\lambda_L = \lambda_U = 2 \, t_{\nu+1}\!\left(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right)
$

A lower $( \nu )$ yields higher λ — meaning stronger co-movement in extreme tails.  
In practice, this translates to **simultaneous disruptions** like port delays coinciding with order backlogs.

### 4.2 Supply Chain Impact

By integrating the Student-t Copula into the simulation engine:
- **VaR₉₅** (Value-at-Risk at 95%) decreased by ≥10%  
- **SCRI violations** (risk threshold exceedances) reduced by ≥15%  
- **Cost escalation** remained <5%:contentReference[oaicite:1]{index=1}

---

## 5. Mathematical Foundation

### 5.1 Multivariate Student-t Distribution

$
f(x) = \frac{\Gamma\left(\frac{\nu + d}{2}\right)}{\Gamma\left(\frac{\nu}{2}\right)(\nu \pi)^{d/2} |R|^{1/2}} \left(1 + \frac{1}{\nu} x^\top R^{-1} x\right)^{-\frac{\nu + d}{2}}
$

Where:
- $( d )$: Dimension
- $( R )$: Correlation matrix
- $( \nu )$: Degrees of freedom

This distribution captures heavy tails and nonlinear dependencies—essential for FedEx’s stochastic disruption modeling.

---