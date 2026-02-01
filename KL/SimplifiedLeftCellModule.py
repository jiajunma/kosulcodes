import sympy as sp
from functools import lru_cache
import sys
import os

# Add parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only the basic functionality needed for the LeftCellModule
from perm import (
    generate_permutations,
    length_of_permutation,
    is_bruhat_leq,
    permutation_prod,
    simple_reflection
)

# Define symbolic variable v
v = sp.symbols("v")

class SimplifiedLeftCellModule:
    """
    Simplified implementation of the Hecke algebra as a module over itself via left multiplication.

    Based on updated requirements in LeftCell.md, this implementation focuses on the basic
    functions to define the HeckeModule without implementing the general algorithms for
    R polynomials and KL polynomials.

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
        self._initialize()

    def _initialize(self):
        """Initialize the module with the basic data structures."""
        # Initialize the basis and group by length
        self.initialize_basis()

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

    def basic_elements(self):
        """
        Return the set of basic elements BB = {id}.

        These are elements where the bar involution is known directly.

        Returns:
            Set containing the identity permutation
        """
        return {tuple(range(1, self.n + 1))}

    def bar_on_basic_elements(self):
        """
        Return the bar involution on basic elements.

        For the LeftCellModule, bar(T_{id}) = T_{id}.

        Returns:
            Dictionary mapping the identity to its bar involution
        """
        id_perm = tuple(range(1, self.n + 1))
        return {id_perm: {id_perm: 1}}

    def action_by_simple_reflection(self, s, w):
        """
        Compute the action of T_s on T_w.

        Args:
            s: A simple reflection (permutation tuple)
            w: A basis element (permutation tuple)

        Returns:
            Dictionary representing the result of T_s · T_w
        """
        sw = permutation_prod(s, w)

        if self.ell(sw) > self.ell(w):
            # Case sw > w: T_s · T_w = T_{sw}
            return {sw: 1}
        else:
            # Case sw < w: T_s · T_w = v^2 T_{sw} + (v^2-1)T_w
            return {sw: v**2, w: v**2 - 1}

    def pretty_print_element(self, element_dict):
        """
        Pretty print a linear combination of basis elements.

        Args:
            element_dict: Dictionary mapping permutations to coefficients
        """
        if not element_dict:
            print("0")
            return

        terms = []
        for perm, coeff in sorted(element_dict.items(), key=lambda x: (self.ell(x[0]), x[0])):
            if coeff == 0:
                continue

            if coeff == 1:
                coeff_str = ""
            elif coeff == -1:
                coeff_str = "-"
            else:
                coeff_str = f"{coeff} * "

            term = f"{coeff_str}T_{perm}"
            terms.append(term)

        print(" + ".join(terms).replace("+ -", "- "))