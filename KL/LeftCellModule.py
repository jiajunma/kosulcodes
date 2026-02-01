import sympy as sp
from functools import lru_cache
import sys
import os

# Add parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KL.HeckeModule import HeckeModule, HeckeElement, v
from perm import (
    generate_permutations,
    length_of_permutation,
    is_bruhat_leq,
    inverse_permutation,
    permutation_prod,
    simple_reflection,
    reduced_word as perm_reduced_word
)

class LeftCellModule(HeckeModule):
    """
    Implements the Hecke algebra as a module over itself via left multiplication.

    The module has a standard basis {T_w}_{w in S_n}, with the left action of
    the Hecke algebra generator T_s (for a simple reflection s) given by:

    T_s · T_w =
      T_{sw}                 if sw > w  (U- type)
      v^2 T_{sw} + (v^2-1)T_w  if sw < w  (U+ type)
    """

    def __init__(self, n):
        """
        Initialize the LeftCellModule for the symmetric group S_n.

        Args:
            n: Size of the permutation group S_n
        """
        self.n = n
        super().__init__()

    def initialize_basis(self):
        """
        Initialize the basis of permutations in S_n and group them by length.
        """
        # Generate all permutations
        perms = list(generate_permutations(self.n))

        # Group permutations by length
        self._basis_by_length = {}
        for perm in perms:
            length = self.ell(perm)
            if length not in self._basis_by_length:
                self._basis_by_length[length] = []
            self._basis_by_length[length].append(perm)

    @lru_cache(maxsize=10000)
    def ell(self, x):
        """
        Return the length (inversion number) of permutation x.

        Args:
            x: A permutation tuple

        Returns:
            The number of inversions in x
        """
        return length_of_permutation(x)

    def is_bruhat_leq(self, y, x):
        """
        Check if permutation y is less than or equal to x in the Bruhat order.

        Args:
            y, x: Permutation tuples

        Returns:
            True if y ≤ x in Bruhat order, False otherwise
        """
        return is_bruhat_leq(y, x)

    def simple_reflections(self):
        """
        Return the list of simple reflections for S_n.

        Returns:
            List of permutation tuples representing simple reflections
        """
        return [simple_reflection(i, self.n) for i in range(1, self.n)]

    def reduced_word(self, x):
        """
        Return a reduced word for x as a list of simple reflections.

        Args:
            x: A permutation tuple

        Returns:
            List of simple reflections whose product equals x
        """
        return perm_reduced_word(x)

    def get_type_and_companions(self, s, x):
        """
        Determine the type and companions for the action of T_s on T_x.

        Args:
            s: A simple reflection (permutation tuple)
            x: A basis element (permutation tuple)

        Returns:
            (type_str, companions) where:
                type_str: 'U-' if sx > x, 'U+' if sx < x
                companions: Set containing {sx, x} or just {sx} depending on type
        """
        sx = permutation_prod(s, x)

        if self.ell(sx) > self.ell(x):
            # Case sx > x: T_s · T_x = T_{sx}
            return 'U-', {sx}
        else:
            # Case sx < x: T_s · T_x = v^2 T_{sx} + (v^2-1)T_x
            return 'U+', {sx, x}

    def get_basic_elements_bar(self):
        """
        Return basic elements with known bar involution.

        For the LeftCellModule, we know bar(T_{id}) = T_{id}.

        Returns:
            Dictionary {id_perm: {id_perm: 1}}
        """
        id_perm = tuple(range(1, self.n + 1))
        return {id_perm: {id_perm: sp.Integer(1)}}

    def T_s(self, i):
        """
        Return the i-th simple reflection T_{s_i} as a HeckeElement.

        Args:
            i: Index of the simple reflection (1-based)

        Returns:
            HeckeElement representing T_{s_i}
        """
        if i < 1 or i >= self.n:
            raise ValueError(f"Invalid index {i} for simple reflection in S_{self.n}")
        s_i = simple_reflection(i, self.n)
        return self.T(s_i)

    def T_w(self, w):
        """
        Return the standard basis element T_w.

        Args:
            w: A permutation tuple

        Returns:
            HeckeElement representing T_w
        """
        return self.T(w)

    def action_by_simple_reflection(self, s, w):
        """
        Compute the action of T_s on T_w.

        Args:
            s: A simple reflection (permutation tuple)
            w: A basis element (permutation tuple)

        Returns:
            Dictionary representing the result of T_s · T_w
        """
        sx = permutation_prod(s, w)

        if self.ell(sx) > self.ell(w):
            # Case sw > w: T_s · T_w = T_{sw}
            return {sx: 1}
        else:
            # Case sw < w: T_s · T_w = v^2 T_{sw} + (v^2-1)T_w
            return {sx: v**2, w: v**2 - 1}

    def get_standard_canonical_basis(self):
        """
        Get the standard canonical basis from HeckeA implementation.

        This is used for verification purposes, not as an implementation
        of the canonical basis algorithm in this module.

        Returns:
            Dictionary mapping permutations to their canonical basis elements
        """
        # Import HeckeA here to avoid circular imports
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from HeckeA import HeckeA

        # Create a HeckeA instance with the same n
        hecke_a = HeckeA(self.n)

        # Get the canonical basis
        standard_basis = hecke_a.canonical_basis()

        # Convert to our format
        result = {}
        for w, element in standard_basis.items():
            result[w] = HeckeElement(self, element.coeffs)

        return result


# Test function to demonstrate usage
def test_left_cell_module(n=3):
    """
    Test the LeftCellModule implementation with S_n.

    Args:
        n: Size of the permutation group S_n
    """
    module = LeftCellModule(n)

    # Print basis elements by length
    print(f"Basis elements of S_{n} grouped by length:")
    for length, elements in sorted(module._basis_by_length.items()):
        print(f"Length {length}: {elements}")

    # Test action of simple reflections
    print("\nAction of simple reflections:")
    id_perm = tuple(range(1, n + 1))
    for i in range(1, n):
        s_i = simple_reflection(i, n)
        result = module.act_simple(s_i, {id_perm: sp.Integer(1)})
        print(f"T_{s_i} · T_{id_perm} =", end=" ")
        module.pretty_print_element(result)

    # Test action through function
    print("\nAction using action_by_simple_reflection:")
    for i in range(1, n):
        s_i = simple_reflection(i, n)
        result = module.action_by_simple_reflection(s_i, id_perm)
        print(f"T_{s_i} · T_{id_perm} =", end=" ")
        module.pretty_print_element(result)


if __name__ == "__main__":
    test_left_cell_module(3)