import sympy as sp
from abc import ABC, abstractmethod
from functools import lru_cache

v = sp.symbols("v")
q = v * v

class HeckeElement:
    """
    An element in the Hecke algebra or module.
    Represents a linear combination of basis elements with coefficients in Z[v, v^{-1}].
    """
    def __init__(self, module, coeffs=None):
        """
        Initialize a Hecke element.

        Args:
            module: The parent HeckeModule
            coeffs: Dictionary mapping basis elements to coefficients
        """
        self.module = module
        self.coeffs = coeffs or {}
        # Clean up zero coefficients
        self.coeffs = {k: sp.expand(v) for k, v in self.coeffs.items() if v != 0}

    def __add__(self, other):
        """
        Addition of two Hecke elements.
        """
        if not isinstance(other, HeckeElement):
            return NotImplemented
        if self.module is not other.module:
            raise ValueError("Cannot add elements from different HeckeModule instances")
        return HeckeElement(self.module, self.module.add_elements(self.coeffs, other.coeffs))

    def __neg__(self):
        """
        Negation of a Hecke element.
        """
        return HeckeElement(self.module, self.module.scalar_multiply(self.coeffs, -1))

    def __sub__(self, other):
        """
        Subtraction of two Hecke elements.
        """
        if not isinstance(other, HeckeElement):
            return NotImplemented
        return self + (-other)

    def __mul__(self, other):
        """
        Action of a Hecke element on another.
        """
        if isinstance(other, HeckeElement):
            if self.module is not other.module:
                raise ValueError("Cannot multiply elements from different HeckeModule instances")
            coeffs = self.module.act(self.coeffs, other.coeffs)
            return HeckeElement(self.module, coeffs)
        return NotImplemented

    def __rmul__(self, scalar):
        """
        Scalar multiplication: scalar * element
        """
        return HeckeElement(self.module, self.module.scalar_multiply(self.coeffs, scalar))

    def bar(self):
        """
        Apply the bar involution to this element.
        """
        coeffs = self.module.bar_element(self.coeffs)
        return HeckeElement(self.module, coeffs)

    def is_bar_invariant(self):
        """
        Check if this element is invariant under bar involution.
        """
        return self.module.is_bar_invariant(self.coeffs)

    def pretty(self, label=None):
        """
        Pretty print this element.
        """
        return self.module.pretty_print_element(self.coeffs, label=label)


class HeckeModule(ABC):
    """
    Base class for Hecke modules with Kazhdan-Lusztig theory.
    """
    def __init__(self):
        """
        Initialize a HeckeModule.
        """
        self._basis_by_length = {}
        self._bar_cache = {}
        self._type_cache = {}
        self._r_polys = {}
        self._kl_polys = {}
        self._canonical_basis = {}

        # Initialize basis
        self.initialize_basis()

        # Initialize basic elements bar
        self._basic_elements_bar = self.get_basic_elements_bar()

        # Cache types for optimization
        self.compute_types()

    @abstractmethod
    def initialize_basis(self):
        """
        Initialize the basis and group elements by length.
        """
        pass

    @abstractmethod
    def ell(self, x):
        """
        Return the length of basis element x.
        """
        pass

    @abstractmethod
    def is_bruhat_leq(self, y, x):
        """
        Check if y <= x in the Bruhat order.
        """
        pass

    @abstractmethod
    def get_type_and_companions(self, s, x):
        """
        Return the type and companions for the action of T_s on T_x.

        Args:
            s: A simple reflection
            x: A basis element

        Returns:
            A tuple (type_str, companions) where:
                type_str: One of 'U-', 'U+', 'T-', 'T+', 'G'
                companions: A set of basis elements that appear in T_s * T_x
        """
        pass

    @abstractmethod
    def get_basic_elements_bar(self):
        """
        Return a dictionary {basic_element: bar_value} for elements where
        the bar involution is known directly.
        """
        pass

    def compute_types(self):
        """
        Pre-compute and cache types and companions for all combinations of
        simple reflections and basis elements.
        """
        for s in self.simple_reflections():
            for length, elements in self._basis_by_length.items():
                for x in elements:
                    key = (s, x)
                    if key not in self._type_cache:
                        type_str, companions = self.get_type_and_companions(s, x)
                        self._type_cache[key] = (type_str, companions)

    def T(self, x):
        """
        Return the standard basis element T_x.
        """
        return HeckeElement(self, {x: sp.Integer(1)})

    def simple_reflections(self):
        """
        Return the list of simple reflections.
        """
        pass  # Should be implemented by subclasses

    def add_elements(self, element1, element2):
        """
        Add two elements represented as coefficient dictionaries.
        """
        result = dict(element1)
        for x, coeff in element2.items():
            result[x] = result.get(x, 0) + coeff
        return {x: sp.expand(coeff) for x, coeff in result.items() if coeff != 0}

    def scalar_multiply(self, element, poly):
        """
        Multiply an element by a scalar polynomial.
        """
        if poly == 0:
            return {}
        return {x: sp.expand(poly * coeff) for x, coeff in element.items() if coeff != 0}

    def act(self, left, right):
        """
        Action of left on right, as coefficient dictionaries.
        """
        result = {}
        for x, coeff_x in left.items():
            for y, coeff_y in right.items():
                prod = self.act_basis(x, y)
                for z, coeff_z in prod.items():
                    result[z] = result.get(z, 0) + coeff_x * coeff_y * coeff_z
        return {z: sp.expand(coeff) for z, coeff in result.items() if coeff != 0}

    def act_basis(self, x, y):
        """
        Action of T_x on T_y.
        """
        result = {y: sp.Integer(1)}  # Start with T_y
        for s in self.reduced_word(x):
            result = self.act_simple(s, result)
        return result

    def act_simple(self, s, element):
        """
        Action of T_s on an element.
        """
        result = {}
        for x, coeff in element.items():
            key = (s, x)
            if key in self._type_cache:
                type_str, companions = self._type_cache[key]
            else:
                type_str, companions = self.get_type_and_companions(s, x)
                self._type_cache[key] = (type_str, companions)

            # Apply the action based on type
            result[x] = result.get(x, 0) - coeff  # -T_x term

            # Add companion terms with appropriate coefficients
            if type_str == 'G':
                for y in companions:
                    result[y] = result.get(y, 0) + coeff * (q + 1)
            elif type_str == 'U+':
                for y in companions:
                    result[y] = result.get(y, 0) + coeff * q
            elif type_str == 'T+':
                for y in companions:
                    result[y] = result.get(y, 0) + coeff * (q - 1)
            elif type_str in ['T-', 'U-']:
                for y in companions:
                    result[y] = result.get(y, 0) + coeff

        return {x: sp.expand(coeff) for x, coeff in result.items() if coeff != 0}

    def reduced_word(self, x):
        """
        Return a reduced word for x as a list of simple reflections.
        """
        # This should be implemented by subclasses
        pass

    def bar_coeff(self, expr):
        """
        Apply the bar involution to a coefficient.
        Involution maps v ↦ v^{-1}.
        """
        # If expr is a number, it's invariant under bar
        if isinstance(expr, (int, sp.Integer)) or expr.is_constant():
            return expr

        # Otherwise, substitute v with v^{-1}
        return sp.expand(expr.subs({v: v ** -1}))

    def bar_element(self, element):
        """
        Apply the bar involution to an element.
        """
        if not element:
            return {}

        # Compute bar(T_x) for each basis element inductively
        result = {}
        for x, coeff in element.items():
            coeff_bar = self.bar_coeff(coeff)
            bar_T_x = self.bar_basis_element(x)
            for y, c_y in bar_T_x.items():
                result[y] = result.get(y, 0) + coeff_bar * c_y

        return {x: sp.expand(coeff) for x, coeff in result.items() if coeff != 0}

    def bar_basis_element(self, x):
        """
        Compute the bar involution of a basis element T_x.
        Uses a precomputed cache for performance.
        """
        # Check if already in cache
        if x in self._bar_cache:
            return self._bar_cache[x]

        # Check if it's a basic element with known bar value
        if x in self._basic_elements_bar:
            self._bar_cache[x] = self._basic_elements_bar[x]
            return self._basic_elements_bar[x]

        # Find the length of x
        length_x = self.ell(x)

        # Find all elements of length one less than x
        lower_elements = self._basis_by_length.get(length_x - 1, [])

        # Find a simple reflection s and an element y such that T_s * T_y contains T_x with coefficient 1
        for s in self.simple_reflections():
            for y in lower_elements:
                # Compute T_s * T_y
                action = self.act_simple(s, {y: sp.Integer(1)})

                # Check if T_x appears with coefficient 1
                if x in action and action[x] == 1:
                    # Found suitable s and y
                    # Compute bar(T_s) * bar(T_y)
                    bar_T_s = self.bar_T_s()
                    bar_T_y = self.bar_basis_element(y)

                    # Compute the product bar(T_s) * bar(T_y)
                    product = {}
                    for u, c_u in bar_T_s.items():
                        for v, c_v in bar_T_y.items():
                            prod_uv = self.act_basis(u, v)
                            for w, c_w in prod_uv.items():
                                product[w] = product.get(w, 0) + c_u * c_v * c_w

                    # Remove the T_x term and any other terms in T_s * T_y
                    for z, c_z in action.items():
                        if z != x:  # Skip T_x itself
                            bar_T_z = self.bar_basis_element(z)
                            for w, c_w in bar_T_z.items():
                                product[w] = product.get(w, 0) - self.bar_coeff(c_z) * c_w

                    # The result is bar(T_x)
                    self._bar_cache[x] = {w: sp.expand(c) for w, c in product.items() if c != 0}
                    return self._bar_cache[x]

        raise ValueError(f"Could not compute bar(T_{x})")

    def bar_T_s(self):
        """
        Return bar(T_s) for a simple reflection s.

        The bar involution on the standard basis has:
        bar(T_s) = v^{-2} T_s + (v^{-2} - 1) T_id

        For the Hecke algebra, this corresponds to:
        bar(T_s) = T_s^{-1} = v^{-2}*T_s + (v^{-2} - 1)

        Note: Sometimes the literature uses the normalization:
        bar(T_s) = T_s + (q-1)*e = T_s + (v^2-1)*e

        but we are using the v-normalization with q = v^2.
        """
        # All simple reflections have the same bar value in the form
        # bar(T_s) = v^{-2} T_s + (v^{-2} - 1)T_id

        # Get the first simple reflection
        s = next(iter(self.simple_reflections()))

        # The identity permutation
        id_perm = tuple(range(1, len(s) + 1))

        # Return the dictionary representation
        return {s: v ** -2, id_perm: v ** -2 - 1}

    def is_bar_invariant(self, element):
        """
        Check if an element is invariant under the bar involution.
        """
        bar_element = self.bar_element(element)
        return self.is_equal(element, bar_element)

    def is_equal(self, element1, element2):
        """
        Check if two elements are equal.
        """
        diff = self.add_elements(element1, self.scalar_multiply(element2, -1))
        return not diff

    def compute_r_polynomials(self):
        """
        Compute R-polynomials for all pairs x, y in the basis.
        R_{y,x} are defined as coefficients in the expansion:
        bar(T_x) = \sum_y R_{y,x} T_y
        """
        if self._r_polys:
            return self._r_polys

        # Initialize R-polynomials
        self._r_polys = {}

        # Compute for each basis element
        for length, elements in sorted(self._basis_by_length.items()):
            for x in elements:
                bar_T_x = self.bar_basis_element(x)
                for y, coeff in bar_T_x.items():
                    self._r_polys[(y, x)] = coeff

                # Ensure R_{x,x} = 1
                self._r_polys[(x, x)] = self._r_polys.get((x, x), sp.Integer(1))

        return self._r_polys

    def compute_kl_polynomials(self):
        """
        Compute Kazhdan-Lusztig polynomials for all pairs in the basis.
        """
        if self._kl_polys:
            return self._kl_polys

        # Ensure R-polynomials are computed
        self.compute_r_polynomials()

        # Initialize KL polynomials
        self._kl_polys = {}

        # Set P_{x,x} = 1 for all x
        for length, elements in self._basis_by_length.items():
            for x in elements:
                self._kl_polys[(x, x)] = sp.Integer(1)

        # Compute P_{y,x} inductively on length difference
        for k in range(1, max(self._basis_by_length.keys()) + 1):
            # For each pair (y, x) with ell(x) - ell(y) = k
            for x_length, x_elements in sorted(self._basis_by_length.items()):
                y_length = x_length - k
                if y_length < 0:
                    continue

                y_elements = self._basis_by_length.get(y_length, [])

                for x in x_elements:
                    for y in y_elements:
                        if not self.is_bruhat_leq(y, x):
                            continue

                        # Compute q_{y,x}
                        q_yx = sp.Integer(0)

                        for z_length in range(y_length + 1, x_length + 1):
                            z_elements = self._basis_by_length.get(z_length, [])
                            for z in z_elements:
                                if not (self.is_bruhat_leq(y, z) and self.is_bruhat_leq(z, x)):
                                    continue

                                R_yz = self._r_polys.get((y, z), sp.Integer(0))
                                P_zx = self._kl_polys.get((z, x), sp.Integer(0))

                                if R_yz != 0 and P_zx != 0:
                                    q_yx += R_yz * self.bar_coeff(P_zx)

                        # Extract negative powers of v
                        P_yx = self._extract_negative_powers(q_yx)
                        self._kl_polys[(y, x)] = P_yx

        return self._kl_polys

    def _extract_negative_powers(self, poly):
        """
        Extract terms with negative powers of v from a polynomial.
        """
        if poly == 0:
            return sp.Integer(0)

        # Expand the polynomial
        poly = sp.expand(poly)

        # If it's already a number, return 0
        if poly.is_constant():
            return sp.Integer(0)

        result = sp.Integer(0)

        # Extract individual terms
        for term in sp.Add.make_args(poly):
            # Extract the power of v
            powers = term.as_powers_dict()
            v_power = powers.get(v, 0)

            # Keep only terms with negative powers of v
            if isinstance(v_power, int):
                if v_power < 0:
                    result += term
            else:
                # Handle symbolic powers
                is_neg = getattr(v_power, "is_negative", None)
                if is_neg is True:
                    result += term

        return result

    def compute_canonical_basis(self):
        """
        Compute the Kazhdan-Lusztig canonical basis.

        The canonical basis C_w is defined to be the unique basis such that:
        1. bar(C_w) = C_w (self-dual under bar involution)
        2. C_w = T_w + sum_{y<w} q_{y,w} T_y where q_{y,w} ∈ v^{-1}Z[v^{-1}]

        This is not a direct computation but requires solving equations to
        find bar-invariant combinations of the standard basis.
        """
        if self._canonical_basis:
            return self._canonical_basis

        # Ensure R-polynomials are computed
        self.compute_r_polynomials()

        # Initialize canonical basis
        self._canonical_basis = {}

        # For each element, construct the canonical basis element
        for length, elements in sorted(self._basis_by_length.items()):
            for w in elements:
                # The canonical basis element for the identity is T_id
                if length == 0:  # Identity
                    self._canonical_basis[w] = HeckeElement(self, {w: sp.Integer(1)})
                    continue

                # Start with T_w
                coeffs = {w: sp.Integer(1)}

                # For each y < w (in the Bruhat order), compute the coefficient
                # We need to ensure bar(C_w) = C_w
                for y_len in range(length):
                    for y in self._basis_by_length.get(y_len, []):
                        if not self.is_bruhat_leq(y, w):
                            continue

                        # Compute coefficient of T_y in C_w
                        # These should be elements of v^{-1}Z[v^{-1}]
                        # The formula is derived from solving bar(C_w) = C_w

                        # Compute bar(T_w) as a linear combination of standard basis elements
                        bar_T_w = self.bar_basis_element(w)

                        # For simple elements, special cases
                        if length == 1:  # Simple reflection
                            # For a simple reflection s, C_s = T_s + v^{-1} T_id
                            id_perm = tuple(range(1, len(w) + 1))
                            coeffs[id_perm] = v ** -1
                            continue

                        # For higher-length elements, we need a more complex calculation
                        # This implementation depends on the specific structure of the module
                        # For LeftCellModule, the canonical basis has a nice form

                        # Simplified approach for testing - compute KL polynomials first
                        self.compute_kl_polynomials()

                        # Use KL polynomials to define C_w
                        P_yw = self._kl_polys.get((y, w), sp.Integer(0))
                        if P_yw != 0:
                            # According to Lusztig's formula:
                            # C_w = sum_{y≤w} (-1)^{l(w)-l(y)} v^{l(w)-2l(y)} P_{y,w}(v^{-1}) T_y
                            d = length - y_len
                            coeff = (-1)**d * (v**(length - 2*y_len)) * self.bar_coeff(P_yw)
                            coeffs[y] = sp.expand(coeff)

                # Create the canonical basis element
                self._canonical_basis[w] = HeckeElement(self, coeffs)

        return self._canonical_basis

    def C(self, x):
        """
        Return the canonical basis element C_x.
        """
        if not self._canonical_basis:
            self.compute_canonical_basis()
        return self._canonical_basis.get(x, None)

    def format_laurent(self, expr, var=v):
        """
        Format a Laurent polynomial in a readable way.
        """
        expr = sp.expand(expr)
        terms = {}
        for term in sp.Add.make_args(expr):
            powers = term.as_powers_dict()
            exp = powers.get(var, 0)
            if isinstance(exp, int):
                exp = int(exp)
            else:
                is_int = getattr(exp, "is_integer", None)
                if is_int is False:
                    raise ValueError(f"Non-integer power in {term}")
                if is_int is None:
                    raise ValueError(f"Non-integer power in {term}")
                exp = int(exp)
            coeff = sp.simplify(term / (var ** exp))
            terms[exp] = terms.get(exp, 0) + coeff
        terms = {e: sp.simplify(c) for e, c in terms.items() if c != 0}
        if not terms:
            return "0"
        pieces = []
        for exp in sorted(terms.keys(), reverse=True):
            coeff = terms[exp]
            coeff_str = sp.sstr(coeff)
            if exp == 0:
                pieces.append(f"{coeff_str}")
                continue
            if coeff == 1:
                base = f"{var}" if exp == 1 else f"{var}^{{{exp}}}"
            elif coeff == -1:
                base = f"-{var}" if exp == 1 else f"-{var}^{{{exp}}}"
            else:
                base = f"{coeff_str}*{var}" if exp == 1 else f"{coeff_str}*{var}^{{{exp}}}"
            pieces.append(base)
        return " + ".join(pieces).replace("+ -", "- ")

    def pretty_print_element(self, element, label=None):
        """
        Pretty print an element.
        """
        if not element:
            print(f"{label} 0" if label else "0")
            return
        terms = []
        for x in sorted(element, key=lambda x: (self.ell(x), x)):
            coeff = sp.expand(element[x])
            if coeff == 0:
                continue
            terms.append(f"({self.format_laurent(coeff)}) T_{str(x)}")
        content = " + ".join(terms) if terms else "0"
        print(f"{label} {content}" if label else content)