# HeckeModule Base Class Development Guideline

This document outlines the design and implementation of the `HeckeModule` base class, which serves as a foundation for various module with Hecke action. 


## 1. Overview
The `HeckeModule` class should provide a unified interface for basis management, Hecke algebra actions, and the computation of canonical (Kazhdan-Lusztig) bases. 

## 2. Core Attributes
The base class should maintain:
- **Basis B**: An interface or list of the standard basis elements. 
- **Elements**: Linear combinations of basis elements with coefficients in Laurent polynomials (implemented as a dictionary: `{basis_element: coefficient}`).
- **Simple Reflections S**: Generators of the Coxeter group.
- **Coxeter Multiplication**: A method to multiply Coxeter group elements.
- **Basic Elements BB**: A subset of the basis (usually generators) for which the bar involution is explicitly known.
- **Left/Right Action**: Action of Coxeter generators on the basis.
- **Length Functions**: Length functions for both Coxeter group elements and basis elements (recommended: maintain a table of basis elements grouped by length).

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
- The program should maintain a cache for the action of simple reflections on the standard basis. 
 The function called "compute_types"
- Optional: Cache $T_w \cdot T_x$ results to accelerate repeated computations.



## 4. Bar Involution
The bar involution is a bar involution on the Hecke module that maps $v \mapsto v^{-1}$. For an element $\sum_{y \in B} p_y T_y$, it is defined as:
$$bar(\sum_{y \in B} p_y T_y) = \sum_{y \in B} bar(p_y) \cdot bar(T_y)$$

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
- $C_x = \sum_{y \le x} P_{y,x}(v) T_y$ where $P_{x,x} = 1$ and $P_{y,x} \in v\mathbb{Z}[v]$ for $y < x$.

The implementation will include an inductive algorithm to compute $P_{y,x}$ using the R-polynomials.





