# ADMM

**Alternating Direction Method of Multipliers**

The Alternating Direction Method of Multipliers _(ADMM)_ is presented as a powerful algorithm designed for **large-scale** and **decentralized** optimization.

## Motivation & Benefits

The primary motivation for ADMM is to solve a specific conflict between two older optimization methods. It aims to give us the "best of both worlds": the **robustness** of the *Method of Multipliers* and the **parallelization** of *Dual Decomposition*.

The **Splitting** Problem:
- **Dual Decomposition** allows you to split a large problem into smaller pieces _(parallelizable)_, but it can be **slow** or **unstable**.
- **Method of Multipliers** is very **stable** because it adds a penalty term _(Augmented Lagrangian $$\mathcal{L}_{\rho}$$ )_, but this penalty _links all variables together_, making parallel processing **impossible**.

> **ADMM** fixes this by updating variables in an alternating sequence, restoring the ability to decompose the problem while keeping the stability

## Precursors

## ADMM Core Concept

## Augmented Lagrangian

**($\mathcal{L}_{\rho}$)**


## ADMM Algorithm Steps

## Example