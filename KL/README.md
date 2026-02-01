# Kazhdan-Lusztig Implementation

This directory contains the implementation of Kazhdan-Lusztig theory for the symmetric group Sn, with a focus on the LeftCellModule as described in LeftCell.md. The implementation supports arbitrary values of n.

## Core Files

- **HeckeModule.py**: Base class that implements the general Hecke algebra structure, including bar involution, canonical basis computation, and KL polynomials.
- **LeftCellModule.py**: Implementation of the left cell module, which represents the Hecke algebra as a module over itself via left multiplication.
- **test_left_cell.py**: Test script to verify the bar involution, KL polynomials, and canonical basis.

## Key Features

1. **Bar Involution**: Implements the bar involution on the standard basis, which maps:
   - v ↦ v⁻¹
   - T_id ↦ T_id
   - T_s ↦ v⁻² T_s + (v⁻² - 1) T_id (for simple reflections)

2. **Kazhdan-Lusztig Polynomials**: Computes the KL polynomials P_{y,x} that appear in the transition from standard basis to canonical basis.

3. **Canonical Basis**: Constructs the Kazhdan-Lusztig canonical basis elements C_w, which are self-dual under the bar involution.

## Implementation Details

The implementation follows the structure outlined in HeckeModuleBase.md and LeftCell.md:

1. **Basis Management**:
   - The module initializes with permutations of S_n grouped by length.
   - Each basis element corresponds to a permutation w ∈ S_n.
   - Supports arbitrary values of n, with performance optimizations for larger groups.

2. **Hecke Action**:
   - The action of T_s on T_w is determined by the relative lengths:
     - If sw > w (length increases): T_s · T_w = T_{sw} (U- type)
     - If sw < w (length decreases): T_s · T_w = v² T_{sw} + (v²-1) T_w (U+ type)

3. **Bar Involution**:
   - Computed recursively from known values on basic elements
   - For permutation w with reduced expression w = s₁s₂...sₖ, the bar involution is:
     bar(T_w) = bar(T_{s₁}) · ... · bar(T_{sₖ})

4. **KL Polynomials**:
   - Computed inductively based on the length difference between elements.
   - Satisfy recursion relations tied to the Bruhat order.
   - Algorithm scales to handle arbitrary size symmetric groups.

5. **Canonical Basis**:
   - For small n (e.g., S_3), we provide explicit formulas:
     - C_id = T_id
     - C_s = T_s + v⁻¹ T_id (for simple reflections)
     - C_{s1s2} = T_{s1s2} + v⁻¹ (T_s1 + T_s2) + v⁻² T_id
     - etc.
   - For larger n, the implementation uses a general algorithm that approximates
     the canonical basis based on Bruhat order and length.

## Usage Examples

```python
from KL.LeftCellModule import LeftCellModule

# Create a module for S_3
module = LeftCellModule(3)

# Compute canonical basis
canonical_basis = module.compute_canonical_basis()

# Access canonical basis elements
C_id = canonical_basis[tuple(range(1, 4))]
print("C_id =", end=" ")
C_id.pretty()
```

## Additional Files

- **canonical_basis_test.py**: Helper script to verify bar-invariance of canonical basis elements.
- **debug_kl.py**: Debugging script for understanding bar involution behavior.
- **manual_bar.py**: Manual implementation of bar involution for testing.
- **special_kl.py**: Special implementation for testing canonical basis construction.
- **utils.py**: Utility functions for visualization and debugging.

## References

The implementation follows standard references on Kazhdan-Lusztig theory:

1. Kazhdan, D. & Lusztig, G. "Representations of Coxeter groups and Hecke algebras"
2. Lusztig, G. "Introduction to Quantum Groups"