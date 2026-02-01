import sympy as sp
import sys
import os

# Add parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KL.LeftCellModule import LeftCellModule, v
from perm import (
    generate_permutations,
    inverse_permutation,
    permutation_prod,
    simple_reflection,
    length_of_permutation
)

def manual_bar(x, coeffs):
    """Apply manual bar involution to an element."""
    result = {}
    for perm, coeff in coeffs.items():
        # Bar on coefficient (v -> v^-1)
        if isinstance(coeff, int) or coeff.is_constant():
            coeff_bar = coeff
        else:
            coeff_bar = sp.expand(coeff.subs(v, v**-1))

        # Bar on basis element
        if perm == (1,2,3):  # identity
            bar_T = {(1,2,3): 1}
        elif perm == (2,1,3) or perm == (1,3,2):  # simple reflection
            bar_T = {perm: v**-2, (1,2,3): v**-2 - 1}
        else:
            # For non-simple elements, we don't have direct formulas
            # This is just a placeholder
            bar_T = {perm: 1}

        # Multiply and add to result
        for y, c_y in bar_T.items():
            result[y] = result.get(y, 0) + coeff_bar * c_y

    return {y: sp.expand(c) for y, c in result.items() if c != 0}

def main():
    # Initialize module
    module = LeftCellModule(3)

    # Identity and simple reflections
    id_perm = (1, 2, 3)
    s1 = (2, 1, 3)
    s2 = (1, 3, 2)

    # Test canonical basis for simple reflections
    C_s1 = {s1: 1, id_perm: v**-1}  # KL normalization

    print("Canonical basis element:")
    print(f"C_{s1} = {C_s1}")

    # Compute bar manually
    bar_C_s1_manual = manual_bar(s1, C_s1)
    print("\nManual bar involution:")
    print(f"bar(C_{s1}) = {bar_C_s1_manual}")

    # Compute bar using module
    bar_T_s1 = module.bar_basis_element(s1)
    print("\nModule bar on basis element:")
    print(f"bar(T_{s1}) = {bar_T_s1}")

    # Compute module's bar on C_s1
    element = module.T(s1) + v**-1 * module.T(id_perm)
    bar_element = element.bar()
    print("\nModule bar on C_s1:")
    print(f"bar(C_{s1}) = {bar_element.coeffs}")

    # Check if bar invariant
    is_equal = bar_C_s1_manual == C_s1
    print(f"\nManual bar invariant: {is_equal}")

    # Element using the HeckeElement class
    print("\nUsing HeckeElement:")
    C_s1_element = module.compute_canonical_basis()[s1]
    print(f"C_{s1} = {C_s1_element.coeffs}")
    bar_C_s1_element = C_s1_element.bar()
    print(f"bar(C_{s1}) = {bar_C_s1_element.coeffs}")
    is_bar_invariant = C_s1_element.is_bar_invariant()
    print(f"is_bar_invariant: {is_bar_invariant}")

    # Show the difference for debugging
    diff = {}
    for y in set(C_s1_element.coeffs.keys()) | set(bar_C_s1_element.coeffs.keys()):
        coeff1 = C_s1_element.coeffs.get(y, 0)
        coeff2 = bar_C_s1_element.coeffs.get(y, 0)
        diff_coeff = sp.expand(coeff1 - coeff2)
        if diff_coeff != 0:
            diff[y] = diff_coeff

    print("\nDifference between C_s1 and bar(C_s1):")
    print(diff)

    # Verify the bar involution on standard basis elements
    print("\nBar involution on standard basis:")
    T_id = module.T(id_perm)
    bar_T_id = T_id.bar()
    print(f"bar(T_{id_perm}) = {bar_T_id.coeffs}")

    T_s1 = module.T(s1)
    bar_T_s1_elem = T_s1.bar()
    print(f"bar(T_{s1}) = {bar_T_s1_elem.coeffs}")

if __name__ == "__main__":
    main()