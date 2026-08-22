# Practical Session: KKT Conditions

## The "Water-Filling" Algorithm in Telecommunications

This practical session focuses on solving a constrained optimization problem analytically using KKT conditions and numerically using Python. We will derive and implement the famous "Water-Filling" algorithm for power allocation.

:::info Objective
To implement a constrained optimization problem, solve it analytically via KKT conditions, and compare the custom algorithm against a generic numerical solver (scipy.optimize).
:::

## 1. Problem Statement: Power Allocation

Consider a transmitter with a total power budget $P_{tot}$ to be distributed among $N$ parallel communication channels. Each channel $i$ has a specific gain $g_i > 0$.

**Goal**: Maximize the system capacity (data rate) subject to power constraints.

### The Objective Function

Based on the Shannon-Hartley theorem, the capacity is:

$$
C(p) = \sum_{i=1}^{N} \log_2(1 + g_i p_i)
$$

Where $p = [p_1, \dots, p_N]^T$ is the power allocation vector.

### The Constraints

- **Budget Constraint**: The total power is fixed.
  $$
  \sum_{i=1}^{N} p_i = P_{tot}
  $$
- **Positivity Constraint**: Power cannot be negative.
  $$
  p_i \ge 0, \quad \forall i
  $$

## 2. Mathematical Analysis (Part 1)

Before coding, we derive the optimal structure using KKT conditions.

### Standard Form

First, convert the maximization problem to a minimization problem:

$$
\min_p - \sum_{i=1}^{N} \log_2(1 + g_i p_i)
$$

Subject to:

$$
h(p) = \sum p_i - P_{tot} = 0
$$

$$
g_i(p) = -p_i \le 0, \quad \forall i
$$

### Lagrangian

We define the Lagrangian $\mathcal{L}(p, \nu, \lambda)$ with multipliers $\nu$ (for equality) and $\lambda$ (for inequality):

$$
\mathcal{L}(p, \nu, \lambda) = - \sum_{i=1}^{N} \ln(1 + g_i p_i) + \nu \left( \sum_{i=1}^{N} p_i - P_{tot} \right) - \sum_{i=1}^{N} \lambda_i p_i
$$

(Note: We use $\ln$ instead of $\log_2$ for easier derivation, scaling by a constant).

### KKT Derivation

From the Stationarity condition ($\nabla_p \mathcal{L} = 0$):

$$
\frac{-g_i}{1 + g_i p_i} + \nu - \lambda_i = 0 \implies \nu - \lambda_i = \frac{g_i}{1 + g_i p_i}
$$

From Complementary Slackness ($\lambda_i p_i = 0$):

If $p_i > 0$, then $\lambda_i = 0$. The equation becomes $\nu = \frac{g_i}{1 + g_i p_i}$, which implies $p_i = \frac{1}{\nu} - \frac{1}{g_i}$.

If optimal $p_i = 0$, the condition holds directly.

Combining these, we get the Water-Filling Solution:

$$
\boxed{p_i^* = \max \left( 0, \frac{1}{\nu} - \frac{1}{g_i} \right)}
$$

Here, $1/\nu$ acts as the "water level" and $1/g_i$ is the "noise level" (ground). We pour power (water) into channels with low noise until the budget is used.

## 3. Python Implementation (Part 2)

We will implement two solvers: a generic one (scipy) and our custom KKT-based one.

### Step 1: Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# 1. Setup

N = 10
P_tot = 10.0
np.random.seed(42)
g = np.random.uniform(0.1, 2.0, N) # Channel gains
noise_levels = 1.0 / g
```

### Step 2: Generic Solver (SciPy)

We use SLSQP to handle the constraints numerically.

```python
# Objective: Minimize negative capacity

def objective(p):
return -np.sum(np.log2(1 + g \* p))

# Constraints

constraints = ({'type': 'eq', 'fun': lambda p: np.sum(p) - P*tot})
bounds = [(0, None) for * in range(N)] # p_i >= 0

# Solve

result = minimize(objective, x0=np.ones(N) \* P_tot/N, method='SLSQP', bounds=bounds, constraints=constraints)
p_scipy = result.x

print(f"SciPy Allocation: {p_scipy}")
```

### Step 3: Custom KKT Solver (Water-Filling)

We need to find the specific water level $\nu$ (or rather, the level $L = 1/\nu$) such that the sum of powers equals $P_{tot}$.

$$ \sum*{i=1}^N \max(0, L - \frac{1}{g_i}) = P*{tot} $$

Since the total power is strictly increasing with $L$, we can use Binary Search (Bisection).

```python
def water_filling_solver(g, P_tot):
noise = 1.0 / g

    # Bisection search for water level L (1/nu)
    low, high = 0.0, np.max(noise) + P_tot
    tolerance = 1e-6

    for _ in range(100): # Max iterations
        level = (low + high) / 2
        p_temp = np.maximum(0, level - noise)

        if np.sum(p_temp) > P_tot:
            high = level # Too much power, lower the level
        else:
            low = level  # Too little power, raise the level

        if np.abs(np.sum(p_temp) - P_tot) < tolerance:
            break

    return np.maximum(0, level - noise), level

p_kkt, water_level = water_filling_solver(g, P_tot)
print(f"KKT Allocation: {p_kkt}")
```

## Step 4: Verification & Visualization

Compare the results and visualize the "water filling" effect.

```python
# 1. Verification

error = np.linalg.norm(p_scipy - p_kkt)
print(f"Euclidean distance between solutions: {error:.2e}")

# 2. Visualization

plt.figure(figsize=(10, 6))
indices = np.arange(N)
width = 0.6

# Stacked bar chart


plt.bar(indices, noise_levels, width, label='Noise Level (1/g)', color='lightgray')
plt.bar(indices, p_kkt, width, bottom=noise_levels, label='Allocated Power', color='skyblue')

# Water level line

plt.axhline(y=water_level, color='blue', linestyle='--', label='Water Level')

plt.xlabel('Channel Index')
plt.ylabel('Power / Level')
plt.title('Water-Filling Algorithm Visualization')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
```

:::success Conclusion
You should observe that the allocated power "fills up" the valleys created by the noise levels up to a constant water level. Channels with very high noise ($1/g_i > \text{level}$) receive zero power, matching the KKT derivation.
:::

$$
$$
