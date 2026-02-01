# HeckeModule Base Class Development Guideline

This document outlines the design and implementation of the `HeckeModule` base class, which serves as a foundation for various modules with Hecke action.

## 1. Overview
The `HeckeModule` class provides a unified interface for basis management, Hecke algebra actions, and the computation of canonical (Kazhdan-Lusztig) bases.

We always use $v$ as the indeterminate and $q = v^2$. 

## 2. Core Attributes
The base class maintains:
- **Basis B**: An interface or list of the standard basis elements.
- **Elements**: Linear combinations of basis elements with coefficients in Laurent polynomials (implemented as a dictionary: `{basis_element: coefficient}`).
- **Simple Reflections S**: Generators of the Coxeter group.
- **Coxeter Multiplication**: A method to multiply Coxeter group elements.
- **Basic Elements BB**: A subset of the basis (usually generators) for which the bar involution is explicitly known.
- **Left/Right Action**: Action of Coxeter generators on the basis.
- **Length Functions**: Length functions for both Coxeter group elements and basis elements (maintain a table of basis elements grouped by length).

## 3. Types and Actions
For each pair $(s, x)$ where $s \in S$ and $x \in B$, the action is determined by a **Type** and a set of **Companions** $C \subseteq B$. The action is given by:

- **Type G**: 
  $T_s \cdot T_x = - T_x + \sum_{y \in C} (v^2+1) T_y$
- **Type U+**:
  $T_s \cdot T_x = - T_x + \sum_{y \in C} v^2 T_y$
- **Type T+**:
  $T_s \cdot T_x = - T_x + \sum_{y \in C} (v^2-1) T_y$
- **Type T- or U-**:
  $T_s \cdot T_x = - T_x + \sum_{y \in C} T_y$

### Caching
- The program should maintain a cache for the action of simple reflections on the standard basis via a function called `compute_types`.
- Optional: Cache $T_w \cdot T_x$ results to accelerate repeated computations.



## 4. Bar Involution
The bar involution is an involution on the Hecke module that maps $v \mapsto v^{-1}$. For an element $\sum_{y \in B} p_y T_y$, it is defined as:
$$bar\left(\sum_{y \in B} p_y T_y\right) = \sum_{y \in B} bar(p_y) \cdot bar(T_y)$$

### Computation Table
The bar involution on the standard basis $\{T_x\}_{x \in B}$ is precomputed during initialization:

1. **Step 0**: For basic elements $y \in BB$, the value of $bar(T_y)$ is provided directly.
2. **Induction**: Compute $bar(T_x)$ by induction on the length $\ell(x)$.
   - Suppose $bar(T_x)$ is known for all $\ell(x) \le k$.
   - For an element $y_0$ of length $k+1$, find a simple reflection $s$ and an element $x$ of length $k$ such that $T_s \cdot T_x$ contains $T_{y_0}$ as the unique "higher" term with coefficient 1.
   - Using the relation $T_s \cdot T_x = \sum c_y T_y$, we deduce:
     $$bar(T_{y_0}) = bar(T_s) \cdot bar(T_x) - \sum_{y \neq y_0} bar(c_y) \cdot bar(T_y)$$
   - Where $bar(T_s) = q^{-1} T_s + (q^{-1}-1) = v^{-2} T_s + (v^{-2}-1)$.

### Verification
- After computation, verify the bar involution property: $bar(bar(T_x)) = T_x$ for all $x \in B$.

## 5. R-polynomials
The R-polynomials $R_{y,x}$ represent the bar involution in the standard basis:
$$bar(T_x) = \sum_{y \in B} R_{y,x} T_y$$

- **Properties**:
  - $R_{x,x} = 1$
  - $R_{y,x} = 0$ unless $x \ge y$ in the appropriate Bruhat-like order (The default one will be the order by length).
  - $\sum_{y \in B} R_{z,y} bar(R_{y,x}) = \delta_{z,x}$ (In matrix form: $R \cdot bar(R) = I$).

## 6. Kazhdan-Lusztig (Canonical) Basis
The canonical basis elements $C_x$ are uniquely determined by:
- $bar(C_x) = C_x$ (Self-duality)
- $C_x = \sum_{y \le x} P_{y,x}(v) T_y$ where $P_{x,x} = 1$ and $P_{y,x} \in v^{-1}\mathbb{Z}[v^{-1}]$ for $y < x$.

### Inductive Computation of KL Polynomials
Based on Lusztig's book Lemma 24.2.1, the polynomials $P_{y,x}$ are computed by induction on the length difference $\ell(x) - \ell(y)$. 

In the implementation, compute KL polynomials in one shot and save $P_{y,x}$ as `KL[x][y]`. First, set $KL[x][x] = 1$.

In our convention, the transition matrix $R$ acts from the left ($P = R \cdot bar(P)$), which is opposite to the row-wise action in the lemma ($p = bar(p) \cdot r$).

For a fixed $x$, we compute $P_{y,x}$ for all $y$ with $\ell(x) - \ell(y) = k+1$ as follows:

1. **Inductive Step**: Suppose $P_{z,x}$ is known for all $z$ with $\ell(x) - \ell(z) \le k$.
   - Compute the auxiliary polynomial $q_{y,x}$:
     $$q_{y,x} = \sum_{y < z \le x} R_{y,z} \cdot bar(P_{z,x})$$
   - $P_{y,x}$ is the unique element in $v^{-1}\mathbb{Z}[v^{-1}]$ satisfying the equation $P_{y,x} - bar(P_{y,x}) = q_{y,x}$.
   - Practically, $P_{y,x}$ is obtained by collecting all terms of $q_{y,x}$ with negative powers of $v$:
     $$P_{y,x} = \sum_{i < 0} \text{coeff}(q_{y,x}, v^i) v^i$$

### Testing
After computing the KL polynomials, verify that everything works correctly for the `LeftCell` case.
In the testing also print the canonical basis



