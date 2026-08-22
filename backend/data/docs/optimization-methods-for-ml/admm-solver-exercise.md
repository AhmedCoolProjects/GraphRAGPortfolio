# Practical Session: ADMM

## Decentralized Energy Market Clearing

This session explores how to solve a decentralized optimization problem using the Alternating Direction Method of Multipliers (ADMM). We will simulate a smart grid where households trade energy while keeping the grid balanced, without a central authority knowing everyone's preferences.

:::info Objective
To implement a decentralized market clearing algorithm using ADMM, separating the problem into local agent updates and a global clearing step.
:::

## 1. Problem Introduction

Consider a smart grid with $N$ households (agents). Each household $i$ has a preferred energy target $d_i$:

- $d_i > 0$: Wants to buy energy.
- $d_i < 0$: Wants to sell energy (e.g., has solar panels).

**The Constraint**: The grid must be balanced. The sum of all trades must be zero ($\sum x_i = 0$).

**The Goal**: Minimize the total "discomfort" (deviation from the target) for all agents.

### The Global Problem

$$
\begin{aligned}
\text{minimize} \quad & \sum_{i=1}^{N} \frac{1}{2}(x_i - d_i)^2 \\
\text{subject to} \quad & \sum_{i=1}^{N} x_i = 0
\end{aligned}
$$

To solve this in a decentralized way (where no single entity knows all $d_i$), we reformulate it using ADMM by introducing an auxiliary variable $z$.

### ADMM Formulation (The Sharing Problem)

$$
\begin{aligned}
\text{minimize} \quad & \sum_{i=1}^{N} \frac{1}{2}(x_i - d_i)^2 + I_{\mathcal{C}}(z) \\
\text{subject to} \quad & x_i - z_i = 0, \quad \forall i
\end{aligned}
$$

- $z$: Auxiliary variables representing the "consensus" state.
- $I_{\mathcal{C}}(z)$: Indicator function for the set $\mathcal{C} = \{z \in \mathbb{R}^N \mid \sum z_i = 0\}$. It is 0 if valid, $+\infty$ if invalid.

## 2. Mathematical Derivation (Exercises)

### 1. The Augmented Lagrangian

We add the dual variable $y$ and the quadratic penalty $\rho$:

$$\mathcal{L}_{\rho}(x, z, y) = \sum_{i=1}^{N} \frac{1}{2}(x_i - d_i)^2 + I_{\mathcal{C}}(z) + \sum_{i=1}^{N} y_i(x_i - z_i) + \frac{\rho}{2} \sum_{i=1}^{N} \|x_i - z_i\|^2$$

### 2. The x-update (The Agent's Problem)

Each agent updates $x_i$ to minimize $\mathcal{L}_{\rho}$ with respect to $x_i$, treating $z$ and $y$ as constants.

$$
x_i^{k+1} = \arg \min_{x_i} \left( \frac{1}{2}(x_i - d_i)^2 + y_i^k x_i + \frac{\rho}{2}(x_i - z_i^k)^2 \right)
$$

Taking the derivative and setting to 0:

$$
(x_i - d_i) + y_i^k + \rho(x_i - z_i^k) = 0
$$

$$
x_i^{k+1} = \frac{d_i + \rho z_i^k - y_i^k}{1 + \rho}
$$

**Interpretation**: The agent balances their preference ($d_i$) against the market constraint ($z_i$) and the price signal ($y_i$).

### 3. The z-update (The Clearinghouse Problem)

The central entity updates $z$ to minimize $\mathcal{L}_{\rho}$ subject to $\sum z_i = 0$.

$$
z^{k+1} = \arg \min_{z} \left( I_{\mathcal{C}}(z) + \sum_{i=1}^{N} -y_i^k z_i + \frac{\rho}{2} \sum_{i=1}^{N} (x_i^{k+1} - z_i)^2 \right)
$$

This is equivalent to projecting the vector $v = x^{k+1} + \frac{1}{\rho} y^k$ onto the zero-sum hyperplane.
The projection of a vector $v$ onto the zero-mean set is simply removing its mean:

$$
z_i^{k+1} = v_i - \bar{v}
$$

where $\bar{v} = \frac{1}{N} \sum v_j$.

### 4. The Dual Update

The standard ADMM dual update:

$$
y_i^{k+1} = y_i^k + \rho(x_i^{k+1} - z_i^{k+1})
$$

## 3. Python Implementation

We simulate $N=50$ households and verify that the market clears (sum of trades approaches 0).

```python
import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Simulation

np.random.seed(42)
N = 50
rho = 1.0
max_iter = 100

# Random targets between -10 and 10

d = np.random.uniform(-10, 10, N)

# Initialize variables

x = np.zeros(N)
z = np.zeros(N)
y = np.zeros(N)

residuals = []
market_imbalance = []

# 2. ADMM Loop

for k in range(max_iter): # --- Step 1: x-update (Agents) --- # This happens in parallel for each agent
x = (d + rho \* z - y) / (1 + rho)

    # --- Step 2: z-update (Clearinghouse) ---
    # Create the vector v to project
    v = x + (1/rho) * y
    # Project onto zero-sum set (subtract mean)
    z = v - np.mean(v)

    # --- Step 3: Dual update (Prices) ---
    y = y + rho * (x - z)

    # --- Monitoring ---
    # Primal residual: how far are x and z apart?
    r_prim = np.linalg.norm(x - z)
    residuals.append(r_prim)

    # Market imbalance: does sum(x) = 0?
    imbalance = np.abs(np.sum(x))
    market_imbalance.append(imbalance)

# 3. Visualization

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(residuals)
plt.yscale('log')
plt.title('Primal Residual Convergence')
plt.xlabel('Iteration')
plt.ylabel('||x - z||')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(market_imbalance, color='orange')
plt.yscale('log')
plt.title('Market Imbalance (Sum of Trades)')
plt.xlabel('Iteration')
plt.ylabel('|Sum(x)|')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Final Imbalance: {np.sum(x):.4e}")
print(f"First 5 Agent Targets: {d[:5]}")
print(f"First 5 Agent Trades: {x[:5]}")
```

:::success Result
You should see the Market Imbalance drop exponentially. This confirms that even though agents only optimized for themselves locally, the ADMM coordination forced the global grid to balance perfectly.
:::
