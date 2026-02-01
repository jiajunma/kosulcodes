import sympy as sp
import sys
import os

# Add parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perm import (
    generate_permutations,
    inverse_permutation,
    permutation_prod,
    simple_reflection,
    length_of_permutation
)

# Define symbolic variable v
v = sp.symbols("v")

class SpecialLeftCellModule:
    """
    Special implementation of the Left Cell Module to
    test and verify the canonical basis construction.
    """
    def __init__(self, n):
        self.n = n
        # Generate all permutations of S_n
        self.perms = list(generate_permutations(n))

        # Identity permutation
        self.id_perm = tuple(range(1, n+1))

        # Simple reflections
        self.simple_refs = [simple_reflection(i, n) for i in range(1, n)]

        # Bar involution mapping
        self.bar_map = self._initialize_bar_map()

        # Canonical basis
        self.canonical_basis = self._construct_canonical_basis()

    def _initialize_bar_map(self):
        """
        Initialize the bar involution mapping for all basis elements.

        For S_3, the bar involution is:
        bar(T_id) = T_id
        bar(T_s) = v^{-2} T_s + (v^{-2} - 1) T_id
        For other elements, we need the "inverse" formula.
        """
        bar_map = {}

        # Identity element
        bar_map[self.id_perm] = {self.id_perm: 1}

        # Simple reflections
        for s in self.simple_refs:
            bar_map[s] = {s: v**-2, self.id_perm: v**-2 - 1}

        # For other elements, we'd need a more general formula

        return bar_map

    def bar(self, element):
        """
        Apply bar involution to an element.

        Args:
            element: Dictionary {perm: coeff}

        Returns:
            Dictionary {perm: coeff} representing bar(element)
        """
        result = {}
        for x, coeff in element.items():
            # Apply bar to coefficient
            coeff_bar = self.bar_coeff(coeff)

            # Apply bar to basis element
            if x in self.bar_map:
                bar_T_x = self.bar_map[x]
                for y, c_y in bar_T_x.items():
                    result[y] = result.get(y, 0) + coeff_bar * c_y
            else:
                # For elements not in the map, use a default
                result[x] = result.get(x, 0) + coeff_bar

        return {x: sp.expand(c) for x, c in result.items() if c != 0}

    def bar_coeff(self, coeff):
        """
        Apply bar involution to a coefficient (v → v^{-1})
        """
        if isinstance(coeff, (int, sp.Integer)) or coeff.is_constant():
            return coeff
        return sp.expand(coeff.subs(v, v**-1))

    def is_bar_invariant(self, element):
        """
        Check if an element is invariant under bar involution.
        """
        bar_element = self.bar(element)

        # Compare coefficients
        if set(element.keys()) != set(bar_element.keys()):
            return False

        for x in element.keys():
            if sp.expand(element[x] - bar_element[x]) != 0:
                return False

        return True

    def _construct_canonical_basis(self):
        """
        Construct the canonical basis for S_n.

        For S_3, explicitly construct the known canonical basis elements.
        """
        canonical_basis = {}

        # Identity: C_id = T_id
        canonical_basis[self.id_perm] = {self.id_perm: 1}

        # For n=3, we have S_3
        if self.n == 3:
            s1 = (2, 1, 3)
            s2 = (1, 3, 2)
            s1s2 = (2, 3, 1)
            s2s1 = (3, 1, 2)
            w0 = (3, 2, 1)

            # Simple reflections
            canonical_basis[s1] = {s1: 1, self.id_perm: v**-1}
            canonical_basis[s2] = {s2: 1, self.id_perm: v**-1}

            # Length 2 elements
            canonical_basis[s1s2] = {
                s1s2: 1,
                s1: v**-1,
                s2: v**-1,
                self.id_perm: v**-2
            }

            canonical_basis[s2s1] = {
                s2s1: 1,
                s1: v**-1,
                s2: v**-1,
                self.id_perm: v**-2
            }

            # Longest element
            canonical_basis[w0] = {
                w0: 1,
                s1s2: v**-1,
                s2s1: v**-1,
                s1: v**-2,
                s2: v**-2,
                self.id_perm: v**-3
            }

        return canonical_basis

    def verify_canonical_basis(self):
        """
        Verify that the canonical basis elements are bar-invariant.
        """
        print("Verifying canonical basis elements:")

        all_invariant = True
        for x, element in sorted(self.canonical_basis.items(), key=lambda p: length_of_permutation(p[0])):
            is_invariant = self.is_bar_invariant(element)
            print(f"Element C_{x}: {'✓ Invariant' if is_invariant else '✗ Not invariant'}")

            if not is_invariant:
                all_invariant = False
                bar_element = self.bar(element)
                print("  Original:", element)
                print("  Bar applied:", bar_element)

                # Show difference
                diff = {}
                for y in set(element.keys()) | set(bar_element.keys()):
                    c1 = element.get(y, 0)
                    c2 = bar_element.get(y, 0)
                    diff_c = sp.expand(c1 - c2)
                    if diff_c != 0:
                        diff[y] = diff_c

                print("  Difference:", diff)

        return all_invariant

def main():
    # Create the module for S_3
    module = SpecialLeftCellModule(3)

    # Verify canonical basis
    module.verify_canonical_basis()

if __name__ == "__main__":
    main()