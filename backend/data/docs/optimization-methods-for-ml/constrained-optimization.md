# Constrained Optimization


**From Theory to Algorithms**

This page covers the formulation of constrained optimization problems, the theory of duality, the Karush-Kuhn-Tucker (KKT) conditions, and practical algorithms like Linear Programming and ADMM.

:::info Source
Based on the course "Constrained optimization" by Pr. Pierre Hellier (Université de Rennes / Inria).
:::

## 1. Problem Formulation

Constrained optimization involves minimizing a loss function while satisfying specific equality and inequality requirements.

#### General Form

The standard problem is phrased as finding $x^*$ such that:

$$
\begin{aligned}
x^* = & \arg \min_{x \in \mathbb{R}^n} L(x) \\
\text{subject to } \quad & h_j(x) = 0, \quad \forall j=1,\dots,p \\
& g_i(x) < 0, \quad \forall i=1,\dots,q
\end{aligned}
$$

- $L(x)$: The objective (loss) function.
- $h_j(x)$: Equality constraints (e.g., physical laws, conservation of mass).
- $g_i(x)$: Inequality constraints (e.g., resource limits, non-negativity).

#### Common Applications

| Domain | Problem Type | Constraint Example |
| --- | --- | --- |
| Machine Learning | Support Vector Machines (SVM) | $y_i(w^T x_i + b) \ge 1$ (margin constraint) |
| Statistics | Sparse Regression (Lasso) | $\|\theta\|_1 \le$ (sparsity constraint) |
| Logistics | Optimal Transport | Supply $\le$ Factory limit; Demand $\ge$ Store need |


## 2. Duality and KKT Conditions

To solve constrained problems, we often convert them into unconstrained "dual" problems using Lagrange multipliers.

### The Lagrangian

We define the Lagrangian function $\mathcal{L}$ by adding weighted constraints to the objective:

$$
\mathcal{L}(x, u, v) = F(x) + \sum_{i=1}^{q} u_i g_i(x) + \sum_{j=1}^{p} v_j h_j(x)
$$

- $u_i \ge 0$: Dual variables (multipliers) for inequality constraints.
- $v_j$: Dual variables for equality constraints (no sign restriction).

### The Dual Problem

The dual function $D(u, v)$ is the minimum of the Lagrangian with respect to $x$:

$$
D(u, v) = \inf_x \mathcal{L}(x, u, v)
$$

:::tip Weak vs. Strong Duality
Weak Duality: The dual solution is always a lower bound for the primal solution ($D^* \le F^*$).
Strong Duality: The duality gap is zero ($D^* = F^*$). This holds for convex problems that satisfy Slater's condition (existence of a strictly feasible point).
:::

### Karush-Kuhn-Tucker (KKT) Conditions

For a solution to be optimal (under strong duality), it must satisfy the KKT conditions. These are necessary (and for convex problems, sufficient).

:::important The 4 KKT Conditions

**Stationarity**: The forces must balance.

$$
\nabla F(x^*) + \sum u_i^* \nabla g_i(x^*) + \sum v_j^* \nabla h_j(x^*) = 0
$$

**Primal Feasibility**: The solution must be valid.

$$
g_i(x^*) \le 0 \quad \text{and} \quad h_j(x^*) = 0
$$

**Dual Feasibility**: Inequality multipliers must be non-negative.


$$
u_i^* \ge 0
$$

**Complementary Slackness**: You only "pay" for active constraints.


$$
u_i^* g_i(x^*) = 0
$$

(Either the constraint is active $g_i=0$, or the multiplier is $u_i=0$.)
:::

## 3. Linear Programming (LP)

Linear programming is a specific class of problems where the objective and all constraints are linear.

### LP Formulation

$$
\begin{aligned}
x^* = & \arg \max_{x} c^T x \\
\text{subject to } \quad & Ax \le b \\
& x \ge 0
\end{aligned}
$$

The feasible region defines a polytope (a multi-dimensional polygon).

:::note Fundamental Theorem of LP
If an LP problem has a unique optimal solution, that solution is always a vertex of the polytope.
:::

<details>
<summary>View Solvers for LP</summary>

Simplex Method: Moves along the edges of the polytope from vertex to vertex until the optimum is found. Efficient in practice but exponential worst-case complexity.

Interior Point Methods: Traverses the interior of the feasible set using a "barrier function" (e.g., Log-barrier) that penalizes getting too close to the boundaries.

</details>

## 4. ADMM

_Alternating Direction Method of Multipliers_

For large-scale or distributed problems, we use ADMM. It combines the decomposability of dual ascent with the robustness of multipliers.

### Problem Form

Minimize a separable objective:

$$
\text{minimize } f(x) + g(z)
$$

$$
\text{subject to } Ax + Bz = c
$$

### Augmented Lagrangian

ADMM adds a quadratic penalty term ($\rho > 0$) to the standard Lagrangian to ensure stability:

$$\mathcal{L}_{\rho}(x, z, y) = f(x) + g(z) + y^T(Ax + Bz - c) + \frac{\rho}{2} \|Ax + Bz - c\|_2^2$$

### The Algorithm

ADMM solves the problem by updating $x$, $z$, and $y$ sequentially:

1. **x-update**: Minimize w.r.t $x$ (often done in parallel).

    $$
    x^{k+1} := \arg \min_x \mathcal{L}_{\rho}(x, z^k, y^k)
    $$

2. **z-update**: Minimize w.r.t $z$.

    $$
    z^{k+1} := \arg \min_z \mathcal{L}_{\rho}(x^{k+1}, z, y^k)
    $$

3. **Dual-update**: Update the price/dual variable based on the residual.

    $$
    y^{k+1} := y^k + \rho(Ax^{k+1} + Bz^{k+1} - c)
    $$

:::tip Why use ADMM?
It is extremely robust and allows for decentralized optimization. For example, in a market exchange problem, agents can optimize their own objectives locally ($x$-update), while a central authority updates prices based on supply and demand ($y$-update).
:::