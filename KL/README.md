# Kazhdan-Lusztig Implementation

This directory contains the implementation of Kazhdan-Lusztig theory for the symmetric group Sn, with a focus on the LeftCellModule as described in LeftCell.md. The implementation supports arbitrary values of n.

## Core Files

- **HeckeModule.py**: Base class that implements the general Hecke algebra structure.
- **LeftCellModule.py**: Implementation of the left cell module, which represents the Hecke algebra as a module over itself via left multiplication.
- **test_left_cell.py**: Test script to verify the basic functionality with arbitrary values of n.

## Key Features

1. **Left Cell Module**:
   - Implements the Hecke algebra as a module over itself via left multiplication.
   - Provides the standard basis {T_w}_{w in S_n}.
   - Implements the action of simple reflections on basis elements.

2. **Action Rules**:
   - T_s · T_w = T_{sw} if sw > w (U- type)
   - T_s · T_w = v^2 T_{sw} + (v^2-1)T_w if sw < w (U+ type)

## Implementation Details

The implementation follows the structure outlined in LeftCell.md:

1. **Basis Management**:
   - The module initializes with permutations of S_n grouped by length.
   - Each basis element corresponds to a permutation w ∈ S_n.
   - Supports arbitrary values of n, with performance optimizations for larger groups.

2. **Hecke Action**:
   - The action of T_s on T_w is determined by the relative lengths:
     - If sw > w (length increases): T_s · T_w = T_{sw} (U- type)
     - If sw < w (length decreases): T_s · T_w = v² T_{sw} + (v²-1) T_w (U+ type)

3. **Essential Functions**:
   - ell(x): Computing the length of a permutation
   - is_bruhat_leq(y, x): Checking Bruhat order
   - simple_reflections(): Generating the simple reflections for S_n
   - get_type_and_companions(s, x): Determining the type and companions for action
   - get_basic_elements_bar(): Providing the basic elements for bar involution

## Usage Examples

```python
from KL.LeftCellModule import LeftCellModule

# Create a module for S_3
module = LeftCellModule(3)

# Print basis elements by length
for length, elements in sorted(module._basis_by_length.items()):
    print(f"Length {length}: {elements}")

# Test action of simple reflections
id_perm = tuple(range(1, 4))
for i in range(1, 3):
    s_i = simple_reflection(i, 3)
    result = module.action_by_simple_reflection(s_i, id_perm)
    print(f"T_{s_i} · T_{id_perm} =", result)
```

## References

The implementation follows standard references on Kazhdan-Lusztig theory:

1. Kazhdan, D. & Lusztig, G. "Representations of Coxeter groups and Hecke algebras"
2. Lusztig, G. "Introduction to Quantum Groups"