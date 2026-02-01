# HeckeModule Base Class Development Guideline

This document outlines the design and implementation of the `HeckeModule` base class, which serves as a foundation for various module with Hecke action. 


## 1. Overview
The `HeckeModule` class should provide a unified interface for basis management, Hecke algebra actions, and the computation of canonical (Kazhdan-Lusztig) bases. 

## 2. Core Attributes
The base class should maintain (unrelized ):
- assume interface of basis B which list the standard basis of the element. 
- Elements in HeckeModule are linear combinations of basis element with coefficients in Laurent polynomials, you can realize it as dictionary with keys in B and value in Laurent polynomial
- the set S of simple roots, which is the generator of the coxeter group
- muliplication of elements in the coxeter group
- the set of basic_element BB, a set of generator of the element
- left multiplication of coxeter element on the basis
- length function on coxeter group element 
- length function on basis elements. (you may maintain a table of the basis according to the length)

## 3. Type of roots, 
- For each pair (s, x) where s in S and x in B there are types and companions 
    - types can be in the list G, U+, U-, T+, T-.
    - the set **C** of companions is a subset of B. 
- The actions are give by the following formula: y running over **C** 
   - Type G: 
     T_s * T_x = - T_x + sum_y (v^2+1) * T_y 
   - Type U+:
     T_s * T_x = - T_x + sum_y v^2 * T_y 
   - Type T+:
     T_s * T_x = - T_x + sum_y (v^2-1) * T_y 
   - Type T- or U-:
     T_s * T_x = - T_x + sum_y  T_y 

- The program should maintain a cache for the action of simple elements on the standard basis. 
- It also should have an option to cache T_w * T_x, if T_w * T_x is already calculated to speed up. 



## 4. Bar Involution
The bar involution is an involution on elements in the Hecke module
- it sends the coefficient p(v) to p(v^(-1)) 
The bar involution on the standard basis T_x is calculated in the following way:
- it sends sum_y p_y T_y to sum_y bar(p_y) bar(T_y)
- For basic element, (i.e. elements in BB), the bar(T_y) is given directly
- In the _init_ stage should compute bar(T_x) for all x in B and put it in a table.  
- For other element, it is calculated in the following way inductively. 
   - **STEP 0** add the formula for basic element in the table
   
   - **induction** is on the length of x. 
   Suppose we computed the bar involution for all element with length <= k 
   * x loop over all elements of length k
     * s loop over all simple reflections 
        compute T_s * T_x  = sum c_y T_y
        we have following cases: 
        -- for all y with c_y, length(y) <= length(x), then do nothing.
        -- Otherwise, we assert there is an uniqe y with c_y none zero and length(y) > length(x). 
        In this case, we assert  
         --- length(y) = length(x)+1
         --- c_y = 1
        we call this element y_0
        Now we deduce that 
       
        bar(T_{y_0}) = bar(T_s) * bar(T_x) - sum_{y \neq y_0} bar(c_y) bar(T_y)
        
        -- Here bar(T_s) = q^{-1} T_s + (q^{-1}-1)
    * when finish the looping over (x,s), the bar involution of all length k+1 element should be calculated already. 
    * increase k by 1 and continue looping until k reach the maximal length. 

- After computed the bar on standard basis, you should write a test to check bar is indeed an involution, i.e. bar(bar(T_x)) = T_x for all x in B. 

## 5. R polynomail

- The R-polynomials $R_{x,y}$ are defined via the bar involution on the standard basis:
  bar T_x = sum_{y \in B} R_{y,x} T_y $$
  where $R_{y,x}$ is a Laurent polynomial in $v$.
- Specifically, the R-polynomials should satisfy:
    - $R_{x,x} = 1$
    - $R_{y,x} = 0$ if $x$ is not "greater than or equal to" $y$ in the appropriate Bruhat-like order (often defined by the length function).
- Use the R-polynomials to facilitate the computation of the Kazhdan-Lusztig basis (canonical basis).

- The R-polynomials satisfy the following property (a consequence of the bar involution being an involution):
  $\sum_{y \in B} R_{z,y} bar(R_{y,x}) = \delta_{z,x}$
  In matrix form, if $R$ is the matrix where $R_{y,x}$ is the entry at row $y$ and column $x$, then $R \cdot bar(R) = I$.


## 6. Kazhdan-Lusztig (Canonical) Basis
The canonical basis elements $C_x$ are uniquely determined by:
- $bar(C_x) = C_x$ (Self-duality)
- $C_x = \sum_{y \le x} P_{y,x}(v) T_y$ where $P_{x,x} = 1$ and $P_{y,x} \in v\mathbb{Z}[v]$ for $y < x$.
- The implementation should include an inductive algorithm to compute $P_{y,x}$ using the R-polynomials.
- Leave this part empty first. 

          

     




