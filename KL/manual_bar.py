import sympy as sp
import sys
import os

# Add parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KL.LeftCellModule import LeftCellModule, v
from perm import (
    generate_permutations,
    length_of_permutation,
    is_bruhat_leq,
    inverse_permutation,
    permutation_prod,
    simple_reflection,
    reduced_word as perm_reduced_word
)

def manual_bar_involution(element):
    """
    Apply the bar involution manually to an element in the Hecke algebra of S3.

    Args:
        element: Dictionary mapping permutations to coefficients

    Returns:
        Dictionary representing bar(element)
    """
    # Define the permutations in S3
    id_perm = (1, 2, 3)
    s1 = (2, 1, 3)
    s2 = (1, 3, 2)
    s1s2 = (2, 3, 1)
    s2s1 = (3, 1, 2)
    w0 = (3, 2, 1)

    # Define bar(T_w) for all w in S3
    bar_T = {
        id_perm: {id_perm: 1},
        s1: {s1: v**-2, id_perm: v**-2 - 1},
        s2: {s2: v**-2, id_perm: v**-2 - 1},
        s1s2: {s1s2: v**-4, s1: v**-2 - v**-4, s2: v**-2 - v**-4, id_perm: 1 - v**-2 + v**-4},
        s2s1: {s2s1: v**-4, s1: v**-2 - v**-4, s2: v**-2 - v**-4, id_perm: 1 - v**-2 + v**-4},
        w0: {w0: v**-6, s1s2: v**-4 - v**-6, s2s1: v**-4 - v**-6, s1: v**-2 - v**-4 + v**-6, s2: v**-2 - v**-4 + v**-6, id_perm: 1 - v**-2 + v**-4 - v**-6}
    }

    # Apply bar to each term
    result = {}
    for w, coeff in element.items():
        # Convert integer coefficients to sympy Integers
        if isinstance(coeff, int):
            coeff = sp.Integer(coeff)

        # Apply bar to coefficient (v -> v^-1)
        coeff_bar = sp.expand(coeff.subs(v, v**-1))

        # Apply bar to basis element and multiply by bar(coeff)
        for u, c_u in bar_T[w].items():
            result[u] = result.get(u, 0) + coeff_bar * c_u

    # Clean up zero coefficients
    return {w: sp.expand(coeff) for w, coeff in result.items() if coeff != 0}

def is_bar_invariant(element):
    """
    Check if an element is invariant under the bar involution.

    Args:
        element: Dictionary mapping permutations to coefficients

    Returns:
        True if the element is bar-invariant, False otherwise
    """
    bar_element = manual_bar_involution(element)

    # Compare coefficients
    if set(element.keys()) != set(bar_element.keys()):
        return False

    for w in element:
        if sp.expand(element[w] - bar_element[w]) != 0:
            return False

    return True

def pretty_print_element(element, label=None):
    """
    Pretty print an element of the Hecke algebra.

    Args:
        element: Dictionary mapping permutations to coefficients
        label: Optional label to prepend
    """
    if not element:
        print(f"{label} 0" if label else "0")
        return

    terms = []
    for w, coeff in sorted(element.items(), key=lambda x: length_of_permutation(x[0])):
        if coeff == 0:
            continue
        terms.append(f"({coeff}) T_{w}")

    content = " + ".join(terms) if terms else "0"
    print(f"{label} {content}" if label else content)

def test_manual_bar():
    """
    Test the manual bar involution on various elements.
    """
    # Define the permutations in S3
    id_perm = (1, 2, 3)
    s1 = (2, 1, 3)
    s2 = (1, 3, 2)
    s1s2 = (2, 3, 1)
    s2s1 = (3, 1, 2)
    w0 = (3, 2, 1)

    # Test elements
    elements = [
        # The standard basis
        {id_perm: 1},
        {s1: 1},
        {s2: 1},
        {s1s2: 1},
        {s2s1: 1},
        {w0: 1},

        # The proposed canonical basis for s1
        {s1: v, id_perm: 1},

        # Alternative canonical basis for s1
        {s1: 1, id_perm: v**-1}
    ]

    print("Testing manual bar involution:")
    for i, element in enumerate(elements):
        print(f"\nElement {i+1}:")
        pretty_print_element(element)

        bar_element = manual_bar_involution(element)
        print("bar(element):")
        pretty_print_element(bar_element)

        print(f"Bar invariant: {is_bar_invariant(element)}")

def test_canonical_basis():
    """
    Test different versions of canonical basis elements.
    """
    # Define the permutations in S3
    id_perm = (1, 2, 3)
    s1 = (2, 1, 3)
    s2 = (1, 3, 2)
    s1s2 = (2, 3, 1)
    s2s1 = (3, 1, 2)
    w0 = (3, 2, 1)

    # Test different versions
    versions = {
        "Original": {
            id_perm: {id_perm: 1},
            s1: {s1: 1, id_perm: v**-1},
            s2: {s2: 1, id_perm: v**-1},
            s1s2: {s1s2: 1, s1: v**-1, s2: v**-1, id_perm: v**-2},
            s2s1: {s2s1: 1, s1: v**-1, s2: v**-1, id_perm: v**-2},
            w0: {w0: 1, s1s2: v**-1, s2s1: v**-1, s1: v**-2, s2: v**-2, id_perm: v**-3}
        },
        "Modified": {
            id_perm: {id_perm: 1},
            s1: {s1: v, id_perm: 1},
            s2: {s2: v, id_perm: 1},
            s1s2: {s1s2: v**2, s1: v, s2: v, id_perm: 1},
            s2s1: {s2s1: v**2, s1: v, s2: v, id_perm: 1},
            w0: {w0: v**3, s1s2: v**2, s2s1: v**2, s1: v, s2: v, id_perm: 1}
        }
    }

    print("\nTesting different canonical basis versions:")
    for version_name, basis in versions.items():
        print(f"\n{version_name} Version:")

        all_invariant = True
        for w, element in basis.items():
            print(f"\nC_{w}:")
            pretty_print_element(element)

            bar_element = manual_bar_involution(element)
            print("bar(C):")
            pretty_print_element(bar_element)

            is_inv = is_bar_invariant(element)
            print(f"Bar invariant: {is_inv}")

            if not is_inv:
                all_invariant = False

        print(f"\n{version_name} version is {'CORRECT' if all_invariant else 'INCORRECT'}")

def get_correct_canonical_basis():
    """
    Compute the correct canonical basis for S3.
    """
    # Define the permutations in S3
    id_perm = (1, 2, 3)
    s1 = (2, 1, 3)
    s2 = (1, 3, 2)
    s1s2 = (2, 3, 1)
    s2s1 = (3, 1, 2)
    w0 = (3, 2, 1)

    # For the simple reflection, determine the canonical basis
    # Try different coefficients a, b for C_s = a*T_s + b*T_id
    print("\nFinding canonical basis for s1 = (2, 1, 3):")

    # Try with known forms from the literature
    test_cases = [
        # Format: (description, {s1: coeff1, id_perm: coeff2})
        ("C_s = T_s + v^-1 T_id", {s1: sp.Integer(1), id_perm: v**-1}),
        ("C_s = v T_s + T_id", {s1: v, id_perm: sp.Integer(1)}),
        ("C_s = q^-1/2 T_s + T_id", {s1: v**-1, id_perm: sp.Integer(1)}),
        ("C_s = q^1/2 T_s + T_id", {s1: v, id_perm: sp.Integer(1)}),
    ]

    for desc, element in test_cases:

        print(f"\nTesting {desc}:")
        pretty_print_element(element, "C_s1 = ")

        bar_element = manual_bar_involution(element)
        pretty_print_element(bar_element, "bar(C_s1) = ")

        if is_bar_invariant(element):
            print(f"FOUND BAR-INVARIANT COMBINATION: {desc}")
        else:
            print("Not bar-invariant")

if __name__ == "__main__":
    test_manual_bar()
    test_canonical_basis()
    get_correct_canonical_basis()