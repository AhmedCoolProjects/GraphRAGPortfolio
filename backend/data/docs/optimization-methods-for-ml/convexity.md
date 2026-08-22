# Convexity

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

**General Optimization and Convex Analysis**

This page summarizes the core concepts of optimization, focusing on the formulation of problems, the properties of convex sets and functions, and the role of gradients in finding optimal solutions.

:::info Source
Based on the course "General optimization and convex analysis" by Pr. Pierre Hellier (Université de Rennes / Inria).
:::

## 1. General Formulation

Optimization problems generally aim to find a variable $x^*$ that minimizes a specific loss function.

### Unconstrained Optimization

The goal is to solve for:

$$x^* = \arg \min_{x \in C} L(x)$$

$L(x)$: Loss function (or objective function).

$x \in \mathbb{R}^n$: Vector of $n$ variables.

$C$: Set of admissible solutions.

Objective: Find a vector $x^*$ such that $\forall x \in C, L(x^*) < L(x)$.

### Constrained Optimization

Often, the solution must satisfy specific rules (constraints):

$$
\begin{aligned}
& \text{minimize } && L(x) \\
& \text{subject to } && h_j(x) = 0, \quad \forall j=1,\dots,p \\
& && g_i(x) < 0, \quad \forall i=1,\dots,q
\end{aligned}
$$

This is equivalent to unconstrained optimization where the admissible set $C$ is defined by the intersection of these equality ($h_j$) and inequality ($g_i$) constraints.

Key Definitions

| Term | Definition |
| --- | --- |
| Feasible Point | Any point $x \in C$ that satisfies all constraints. |
| Optimal Value | The minimal function value $L^* = L(x^*)$. |
| Optimal Solution | $x^*$ is optimal if $\forall x \in C, L(x^*) < L(x)$. |
| Sub-optimal (Local) | $x^*$ is a local optimum if it is optimal only within a small ball around $x^*$. |

## 2. Convexity

Convexity is a powerful property because it guarantees that local minima are global minima.

### Convex Sets

A set $C$ is convex if the line segment connecting any two points in the set lies entirely within the set.

$$
x, y \in C, \quad 0 < \alpha < 1 \implies \alpha x + (1-\alpha)y \in C
$$

Examples of Convex Sets:

- $\mathbb{R}^n$ and Positive Orthant $\mathbb{R}^n_+$
- Hyperplane: $\{x \in \mathbb{R}^n \mid a^T x = b\}$
- Half-space: $\{x \in \mathbb{R}^n \mid a^T x < b\}$
- Polyhedra: $\{x \in \mathbb{R}^n \mid Ax < b\}$

:::tip Operations Preserving Convexity
If $C_k$ are convex sets, the following remain convex:
- Intersection: $\cap_k C_k$
- Cartesian Product: $C_1 \times C_2 \dots$
- Affine Transformation: If $C$ is convex, then $\{Ax + b \mid x \in C\}$ is convex.
:::

### Convex Functions

A function is convex if its graph lies below the line segment joining any two points on the graph (the chord).

$$
f(\alpha x + (1-\alpha)y) \le \alpha f(x) + (1-\alpha)f(y)
$$

- Strictly Convex: The inequality is strict ($<$).
- Concave: If $f$ is convex, then $-f$ is concave.
- Second Order Condition: If $f$ is twice differentiable, $f$ is convex $\iff f'' \ge 0$.

Common Convex Functions:

- Affine: $ax + b$
- Exponential: $e^x$
- Power: $|x|^p$ for $p \ge 1$
- Neg-entropy: $x \log(x)$ for $x > 0$

## 3. Smoothness and Gradients

To solve optimization problems efficiently, we often rely on differentiability.

### Differentiability Classes

- $C^0$: Continuous functions.
- $C^1$: Continuous first derivatives.
- $C^\infty$: Smooth functions (infinitely differentiable).

### The Gradient

For a function $F: \mathbb{R}^n \to \mathbb{R}$, the gradient is the vector of partial derivatives:

$$
\nabla_x F(x) = \left\{ \frac{\partial F(x)}{\partial x_1}, \dots, \frac{\partial F(x)}{\partial x_m} \right\}^T
$$

The gradient points in the direction of steepest ascent (where $F$ increases the most).

:::note Convexity First-Order Condition
A differentiable function $f$ is convex if and only if:

$$
f(x) \ge f(x') + \langle \nabla f(x'), x - x' \rangle
$$

:::

## 4. Optimality Conditions

#### Necessary Condition

If $x^*$ is a local minimum, the gradient must vanish:

$$
\nabla F(x^*) = 0
$$

#### Sufficient Condition (The "Convex Advantage")

For convex functions, the zero gradient condition is not just necessary, it is sufficient.

:::important The Main Theorem
If $f$ is convex:

Any local minimum is a global minimum.

$\nabla f(x^*) = 0 \iff x^*$ is a global minimum.
:::

<details>
<summary>View Proof Sketch</summary>

1. Local is Global:
Suppose $x^*$ is a local minimum but not global. There exists some $x$ where $f(x) < f(x^*)$. By convexity, for small $t$, the point $tx^* + (1-t)x$ is close to $x^*$.

$$
f(tx^* + (1-t)x) \le t f(x^*) + (1-t)f(x)
$$

Since $f(x) < f(x^*)$, the right side is strictly less than $f(x^*)$. This contradicts that $x^*$ is a local minimum.

2. Gradient Condition:
If $\nabla f(x^*) = 0$, by the first-order convexity condition:

$$
f(x) \ge f(x^*) + \langle \nabla f(x^*), x - x^* \rangle
$$

Since the gradient is 0, $f(x) \ge f(x^*)$ for all $x$. Thus, $x^*$ is global.

</details>




