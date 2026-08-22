# Unconstrained Optimization

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

**Formulation, Algorithms, and Non-Smooth Methods**

This page covers the foundations of unconstrained optimization, ranging from basic optimality conditions and least squares to advanced descent algorithms (Newton, SGD, Adam) and non-smooth techniques like Proximal Gradient Descent.

:::info Source
Based on the course "Unconstrained optimization" by Pr. Pierre Hellier (Université de Rennes / Inria).
:::

## 1. Formulation and Optimality

The goal is to find a vector $x^*$ that minimizes a loss function $L(x)$ over the entire domain $\mathbb{R}^n$.

$$
x^* = \arg \min_{x \in \mathbb{R}^n} L(x)
$$

### Optimality Conditions

To identify potential minima, we analyze derivatives.

1. First-Order Necessary Condition
   If $x^*$ is a local minimum, the gradient must be zero (stationary point):

   $$
   {
   \nabla F(x^*) = 0
   }
   $$

2. Second-Order Sufficient Condition
   If $\nabla F(x^*) = 0$ and the Hessian matrix $H(x^*)$ is positive definite, then $x^*$ is a strict local minimum.

   $$
   H(x) = \nabla^2 F(x) = \begin{pmatrix} \frac{\partial^2 F}{\partial x_1^2} & \dots & \frac{\partial^2 F}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 F}{\partial x_n \partial x_1} & \dots & \frac{\partial^2 F}{\partial x_n^2} \end{pmatrix}
   $$

:::note Convexity
If the function $F$ is convex, any local minimum is automatically a global minimum.
:::

## 2. Least Squares Problems

A classic optimization problem involves predicting a target $y$ from features $A$ via a linear model $Ax \approx y$.

### Ordinary Least Squares (OLS)

Minimize the sum of squared errors:

$$
F(x) = \|Ax - y\|^2
$$

**Analytic Solution:** The gradient is $\nabla F(x) = 2A^T A x - 2A^T y$. Setting it to zero yields the Normal Equations:

$$
x^* = (A^T A)^{-1} A^T y
$$

### Regularized Least Squares

To prevent overfitting or handle ill-posed problems ($m < n$), we add a penalty term $R(x)$.

<Tabs>
<TabItem value="ridge" label="Ridge (L2)" default>
Minimizes $\|Ax - y\|^2 + \lambda \|x\|_2^2$.

**Solution:** $(A^T A + \lambda I)^{-1} A^T y$

Shrinks coefficients towards zero, handling multicollinearity and invertibility issues.

</TabItem>
<TabItem value="lasso" label="Lasso (L1)">
Minimizes $\|Ax - y\|^2 + \lambda \|x\|_1$.

**Effect:** Enforces **sparsity** (sets many coefficients exactly to 0), performing feature selection.

_Note: The L1 norm is non-differentiable at 0, requiring specific algorithms._

</TabItem>
</Tabs>

## 3. Descent Algorithms

Iterative methods update the solution step-by-step:

$$
x_{k+1} = x_k + \alpha_k d_k
$$

where $d_k$ is the descent direction and $\alpha_k$ is the step size (learning rate).

### Gradient & Newton Methods

| Method                  | Direction $d_k$              | Complexity         | Convergence |
| :---------------------- | :--------------------------- | :----------------- | :---------- |
| **Gradient Descent**    | $-\nabla L(x_k)$             | $\mathcal{O}(n)$   | Linear      |
| **Newton's Method**     | $-H(x_k)^{-1} \nabla L(x_k)$ | $\mathcal{O}(n^3)$ | Quadratic   |
| **Quasi-Newton (BFGS)** | $-B_k^{-1} \nabla L(x_k)$    | $\mathcal{O}(n^2)$ | Superlinear |

:::tip Step Size Strategies

- **Fixed**: Simple but sensitive (too small = slow, too large = diverge).
- **Lipschitz**: $\alpha = 1/K$ if gradient is $K$-Lipschitz.
- **Line Search**: Backtracking to satisfy Wolfe conditions (ensure sufficient decrease).

:::

### Stochastic Gradient Descent (SGD)

When data size $m$ is huge, computing the full gradient is expensive. SGD approximates it using a single sample (or mini-batch).

$$
x_{k+1} = x_k - \alpha \nabla \mathcal{L}(f(X_i), Y_i)
$$

### Advanced Optimizers

- **Momentum**: Adds a "velocity" term to smooth oscillations and accelerate convergence.
- **RMSProp**: Adapts learning rates based on moving average of squared gradients.
- **Adam**: Combines Momentum and RMSProp. State-of-the-art for Deep Learning.

## 4. Non-Smooth Optimization

When the objective includes non-differentiable terms (like Lasso's L1 norm), standard gradient descent fails. We use Proximal methods.

### Proximal Operator

The proximal operator maps a point to a nearby location that minimizes the non-smooth function $g$:

$$
\text{prox}_{\lambda g}(v) = \arg \min_x \left( g(x) + \frac{1}{2\lambda} \|x - v\|^2 \right)
$$

For Lasso ($g(x) = \|x\|_1$), this is the **Soft Thresholding** operator:

$$
S_\lambda(v) = \text{sign}(v) \max(|v| - \lambda, 0)
$$

### Algorithms

**ISTA (Iterative Soft-Thresholding Algorithm)**

Combines a gradient step on the smooth part ($f$) with a proximal step on the non-smooth part ($g$).

$$
x_{k+1} = \text{prox}_{\alpha g}(x_k - \alpha \nabla f(x_k))
$$

**FISTA (Fast ISTA)**

Applies Nesterov's acceleration to ISTA, achieving significantly faster convergence ($\mathcal{O}(1/k^2)$ vs $\mathcal{O}(1/k)$).

$$
\begin{aligned}
y_{k+1} &= x_k + \frac{t_k - 1}{t_{k+1}}(x_k - x_{k-1}) \\
x_{k+1} &= \text{prox}_{\alpha g}(y_{k+1} - \alpha \nabla f(y_{k+1}))
\end{aligned}
$$
