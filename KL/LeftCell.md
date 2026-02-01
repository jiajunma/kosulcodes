# LeftCell HeckeModule Implementation Guide

This document defines the `LeftCellModule`, which implements the Hecke algebra $H_n$ as a module over itself via left multiplication.

## 0. File
**important**  
  all code here should be in LeftCellModule.py

## 1. Mathematical Definition
The module $M$ has a standard basis $\{T_w\}_{w \in S_n}$. The left action of the Hecke algebra generator $T_s$ (for a simple reflection $s$) is given by:
$$ T_s \cdot T_w = \begin{cases} T_{sw} & \text{if } sw > w \\ q T_{sw} + (q-1) T_w & \text{if } sw < w \end{cases} $$

## 2. HeckeModule Configuration

### Basis and Keys
- **Basis B**: The set of permutations $S_n$.
- **Keys**: Permutations represented as tuples (e.g., `(1, 2, 3)`).

### Action Parameters (Left Action)
For a simple reflection $s$ and basis element $w$, the action $T_s \cdot T_w$ is determined by:

- **Case $sw > w$**:
  - **Type**: `U-` 
  - **Companions $C$**: $\{sw, w\}$
  - *Mathematical Check*: $T_s T_w = -T_w + 1 \cdot (T_{sw} + T_w) = T_{sw}$.
- **Case $sw < w$**:
  - **Type**: `U+`
  - **Companions $C$**: $\{sw, w\}$
  - *Mathematical Check*: $T_s T_w = -T_w + v^2 \cdot (T_{sw} + T_w) = v^2 T_{sw} + (v^2 - 1) T_w$.

Note: In the `HeckeModule` base class, the action is defined as:
$T_s \cdot T_x = -T_x + \text{coeff} \sum_{y \in C} T_y$. 
This matches the standard Hecke algebra multiplication rules where $q = v^2$.

### Essential Functions
- **Length Function**: $\ell(w)$ is the inversion number of the permutation.
- **Bruhat Order**: Standard Bruhat order on $S_n$.
- **Basic Elements $BB$**: $\{id\}$.
- **Bar on Basic Elements**: $bar(T_{id}) = T_{id}$.

## 3. Implementation Plan

### Step 1: Initialize the Basis
- Generate all permutations of $\{1, \dots, n\}$.
- Group permutations by length into a dictionary `{length: [permutations]}` to facilitate inductive computation.

### Step 2: Implement Action Logic
- Provide a method `get_type_and_companions(s, w)` that:
  1. Computes $sw$.
  2. Compares $\ell(sw)$ and $\ell(w)$.
  3. Returns the corresponding Type and the set $\{sw, w\}$.

### Step 3: Integrate with HeckeModule Base
- **!!Important!!** The `LeftCellModule` class should inherit from `HeckeModule`.
- **!!Important!!** You are not allowed to implement the general algorithm for computing the bar involution (R polynomial) and KL polynomial
- Your task is to provide basic functions to define the HeckeModule (basis B, basic basis BB, bar on BB, lenght, action by simple reflection etc. )
- Implement abstract methods:
  - `ell(x)`: Return the inversion number of the permutation $x$.
  - `is_bruhat_leq(y, x)`: Return True if $y \le x$ in the Bruhat order.

### Step 4: Verification
- Verify the bar involution on $S_n$ matches the theoretical expectation: $bar(T_w) = (T_{w^{-1}})^{-1} $.
- Verify that the canonical basis elements computed match the standard Kazhdan-Lusztig polynomials for $S_n$.
- In the test, your program should allow n be arbitrary integer. 
