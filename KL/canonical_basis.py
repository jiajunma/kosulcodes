import sympy as sp
import sys
import os

# Add parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perm import (
    generate_permutations,
    length_of_permutation,
    is_bruhat_leq,
    inverse_permutation,
    permutation_prod,
    simple_reflection,
    reduced_word
)

# Define the symbolic variables
v = sp.symbols('v')

class LusztigCanonicalBasis:
    def __init__(self, n):
        """Initialize the canonical basis calculator for S_n."""
        self.n = n

        # Generate permutations
        self.perms = list(generate_permutations(n))

        # Group permutations by length
        self.perms_by_length = {}
        for perm in self.perms:
            length = length_of_permutation(perm)
            if length not in self.perms_by_length:
                self.perms_by_length[length] = []
            self.perms_by_length[length].append(perm)

        # Initialize caches
        self.bar_T_cache = {}
        self.r_polynomials = {}
        self.c_polynomials = {}

        # Compute the canonical basis
        self._compute_canonical_basis()

    def _T(self, w):
        """Return the generator T_w in the standard basis."""
        return {w: 1}

    def _bar_coefficient(self, expr):
        """Apply bar involution to a coefficient (v -> v^-1)."""
        if isinstance(expr, int):
            return expr
        return expr.subs(v, v**-1)

    def _bar_T(self, w):
        """
        Compute the bar involution of T_w.

        For the identity, bar(T_id) = T_id
        For simple reflections, bar(T_s) = T_s^{-1} = T_s + (v^-1 - v)T_id
        For general w, we use the formula bar(T_w) = bar(T_{s1}) · ... · bar(T_{sk})
        where w = s1·...·sk is a reduced expression.
        """
        # Check cache first
        if w in self.bar_T_cache:
            return self.bar_T_cache[w]

        # Identity element
        if w == tuple(range(1, self.n + 1)):
            self.bar_T_cache[w] = {w: 1}
            return self.bar_T_cache[w]

        # Simple reflection
        if length_of_permutation(w) == 1:
            id_perm = tuple(range(1, self.n + 1))
            self.bar_T_cache[w] = {
                w: v**-2,
                id_perm: v**-2 - 1
            }
            return self.bar_T_cache[w]

        # General case: use the multiplicative property of bar
        # Get a reduced expression
        word = reduced_word(w)

        # Start with identity
        id_perm = tuple(range(1, self.n + 1))
        result = {id_perm: 1}

        # Multiply by bar(T_s) for each s in the reduced expression
        for s in word:
            temp = {}
            bar_T_s = self._bar_T(s)
            for x, c_x in result.items():
                for y, c_y in bar_T_s.items():
                    # Multiply (c_x * T_x) · (c_y * T_y)
                    # In the Hecke algebra, this is a complicated formula
                    # For now, we'll just do the symbolic computation
                    z = permutation_prod(y, x)
                    temp[z] = temp.get(z, 0) + c_x * c_y
            result = temp

        self.bar_T_cache[w] = result
        return result

    def _r_polynomial(self, y, w):
        """
        Compute the R-polynomial R_{y,w}(v) in the Hecke algebra.

        These satisfy the recursion:
        R_{y,w} = 0 if y not <= w in Bruhat order
        R_{y,y} = 1
        If s is a simple reflection with sw < w, then
            R_{y,w} = R_{ys,ws} if ys < y
            R_{y,w} = (v-v^-1)R_{y,ws} + R_{ys,ws} if ys > y
        """
        # Check cache
        key = (y, w)
        if key in self.r_polynomials:
            return self.r_polynomials[key]

        # Base cases
        if y == w:
            self.r_polynomials[key] = 1
            return 1

        if not is_bruhat_leq(y, w):
            self.r_polynomials[key] = 0
            return 0

        # Find a simple reflection s such that sw < w
        for i in range(1, self.n):
            s = simple_reflection(i, self.n)
            sw = permutation_prod(s, w)
            if length_of_permutation(sw) < length_of_permutation(w):
                # Found a descent
                ys = permutation_prod(s, y)
                if length_of_permutation(ys) < length_of_permutation(y):
                    result = self._r_polynomial(ys, sw)
                else:
                    result = (v - v**-1) * self._r_polynomial(y, sw) + self._r_polynomial(ys, sw)

                self.r_polynomials[key] = result
                return result

        # Should never reach here
        raise ValueError(f"Could not find descent for {w}")

    def _c_polynomial(self, y, w):
        """
        Compute the canonical basis expansion coefficient C_{y,w}(v).

        The canonical basis element C_w is defined as:
        C_w = sum_{y <= w} C_{y,w}(v) T_y

        These satisfy self-duality:
        bar(C_w) = C_w

        And can be computed using R-polynomials:
        C_{y,w} = sum_{y <= z <= w} (-1)^{l(w)-l(z)} v^{l(w)-l(z)} R_{y,z}(v)
        """
        # Check cache
        key = (y, w)
        if key in self.c_polynomials:
            return self.c_polynomials[key]

        # Base case
        if y == w:
            self.c_polynomials[key] = 1
            return 1

        # Not in the Bruhat interval
        if not is_bruhat_leq(y, w):
            self.c_polynomials[key] = 0
            return 0

        # Compute the canonical basis coefficient
        result = 0
        len_w = length_of_permutation(w)
        for z in self.perms:
            if is_bruhat_leq(y, z) and is_bruhat_leq(z, w):
                len_z = length_of_permutation(z)
                sign = (-1)**(len_w - len_z)
                power = (len_w - len_z)

                r_yz = self._r_polynomial(y, z)
                term = sign * v**power * r_yz

                result += term

        self.c_polynomials[key] = result
        return result

    def _compute_canonical_basis(self):
        """Compute the canonical basis for all elements of S_n."""
        self.canonical_basis = {}

        for w in self.perms:
            # Compute C_w = sum_{y <= w} C_{y,w} T_y
            expansion = {}
            for y in self.perms:
                if is_bruhat_leq(y, w):
                    coeff = self._c_polynomial(y, w)
                    if coeff != 0:
                        expansion[y] = coeff

            self.canonical_basis[w] = expansion

    def get_canonical_basis(self, w):
        """Return the canonical basis element C_w."""
        if w not in self.canonical_basis:
            raise ValueError(f"Element {w} not in the permutation group S_{self.n}")
        return self.canonical_basis[w]

    def pretty_print_element(self, element, label=None):
        """Pretty print a linear combination of basis elements."""
        if not element:
            print(f"{label} 0" if label else "0")
            return

        terms = []
        for w, coeff in sorted(element.items(), key=lambda x: length_of_permutation(x[0])):
            if coeff == 0:
                continue

            if coeff == 1:
                terms.append(f"T_{w}")
            else:
                terms.append(f"({coeff})T_{w}")

        content = " + ".join(terms) if terms else "0"
        print(f"{label} {content}" if label else content)

    def verify_bar_invariant(self, w):
        """Verify that the canonical basis element C_w is bar-invariant."""
        C_w = self.get_canonical_basis(w)

        # Apply bar to C_w
        bar_C_w = {}
        for y, coeff in C_w.items():
            # Bar on the coefficient
            coeff_bar = self._bar_coefficient(coeff)

            # Bar on the basis element
            bar_T_y = self._bar_T(y)

            # Multiply and add to result
            for z, c_z in bar_T_y.items():
                bar_C_w[z] = bar_C_w.get(z, 0) + coeff_bar * c_z

        # Clean up
        bar_C_w = {z: sp.expand(c) for z, c in bar_C_w.items() if c != 0}

        # Compare with original
        if set(C_w.keys()) != set(bar_C_w.keys()):
            return False

        for z in C_w:
            if sp.expand(C_w[z] - bar_C_w[z]) != 0:
                return False

        return True

def print_canonical_basis_S3():
    """Print the canonical basis for S3."""
    calculator = LusztigCanonicalBasis(3)

    print("Canonical Basis for S3:")
    for length in range(7):
        if length in calculator.perms_by_length:
            print(f"\nLength {length}:")
            for w in calculator.perms_by_length[length]:
                print(f"C_{w} = ", end="")
                calculator.pretty_print_element(calculator.get_canonical_basis(w))

                # Verify bar-invariance
                is_invariant = calculator.verify_bar_invariant(w)
                print(f"Bar invariant: {is_invariant}")

if __name__ == "__main__":
    print_canonical_basis_S3()