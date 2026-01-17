import sympy as sp

from perm import (
    generate_permutations,
    is_bruhat_leq,
    inverse_permutation,
    length_of_permutation,
    permutation_prod,
    reduced_word as perm_reduced_word,
    simple_reflection,
)

v = sp.symbols("v")
q = v * v

class HeckeElement:
    def __init__(self, algebra, coeffs=None):
        self.algebra = algebra
        self.coeffs = coeffs or {}

    def __add__(self, other):
        if not isinstance(other, HeckeElement):
            return NotImplemented
        if self.algebra is not other.algebra:
            raise ValueError("Cannot add elements from different HeckeA instances")
        return HeckeElement(self.algebra, self.algebra.add_elements(self.coeffs, other.coeffs))

    def __mul__(self, other):
        if isinstance(other, HeckeElement):
            if self.algebra is not other.algebra:
                raise ValueError("Cannot multiply elements from different HeckeA instances")
            coeffs = self.algebra.hecke_multiply(self.coeffs, other.coeffs)
            return HeckeElement(self.algebra, coeffs)
        return NotImplemented

    def __rmul__(self, scalar):
        """
        Scalar multiplication: scalar * element
        """
        return HeckeElement(self.algebra, self.algebra.scalar_multiply(self.coeffs, scalar))

    def bar(self):
        coeffs = self.algebra.bar_element(self.coeffs)
        return HeckeElement(self.algebra, coeffs)

    def is_bar_invariant(self):
        return self.algebra.is_bar_invariant(self.coeffs)

    def pretty(self, label=None):
        self.algebra.pretty_print_element(self.coeffs, label=label)


class HeckeA:
    def __init__(self, n):
        self.n = n
        self._length_cache = {}
        self._mul_simple_cache = {}
        self._tw_tu_cache = {}
        self._inverse_cache = {}

    def length(self, w):
        if w not in self._length_cache:
            self._length_cache[w] = length_of_permutation(w)
        return self._length_cache[w]

    def simple_reflections(self):
        return [simple_reflection(i, self.n) for i in range(1, self.n)]


    def hecke_multiply_by_simple(self, element, s):
        key = (s, tuple(sorted(element.items())))
        if key in self._mul_simple_cache:
            return self._mul_simple_cache[key]
        result = {}
        for w, coeff in element.items():
            sw = permutation_prod(s, w)
            if self.length(sw) > self.length(w):
                result[sw] = result.get(sw, 0) + coeff
            else:
                result[w] = result.get(w, 0) + coeff * (q - 1)
                result[sw] = result.get(sw, 0) + coeff * q
        self._mul_simple_cache[key] = result
        return result

    def scalar_multiply(self, element, poly):
        if poly == 0:
            return {}
        return {w: sp.expand(poly * coeff) for w, coeff in element.items() if coeff != 0}

    def add_elements(self, element1, element2):
        result = dict(element1)
        for w, coeff in element2.items():
            result[w] = result.get(w, 0) + coeff
        return {w: sp.expand(coeff) for w, coeff in result.items() if coeff != 0}

    def hecke_multiply_by_simple_inverse(self, element, s):
        return self.add_elements(
            self.scalar_multiply(self.hecke_multiply_by_simple(element, s), v ** -2),
            self.scalar_multiply(element, v ** -2 - 1),
        )

    def multiply_Tw_Tu(self, w, u):
        key = (w, u)
        if key in self._tw_tu_cache:
            return self._tw_tu_cache[key]
        product = {u: sp.Integer(1)}
        for s in perm_reduced_word(w):
            product = self.hecke_multiply_by_simple(product, s)
        self._tw_tu_cache[key] = product
        return product

    def hecke_multiply(self, element1, element2):
        result = {}
        for w, coeff_w in element1.items():
            for u, coeff_u in element2.items():
                product = self.multiply_Tw_Tu(w, u)
                for k, vcoeff in product.items():
                    result[k] = result.get(k, 0) + coeff_w * coeff_u * vcoeff
        return result

    def format_laurent(self, expr, var=v):
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
                base = f"{var}" if exp == 1 else f"{var}**{exp}"
            elif coeff == -1:
                base = f"-{var}" if exp == 1 else f"-{var}**{exp}"
            else:
                base = f"{coeff_str}*{var}" if exp == 1 else f"{coeff_str}*{var}**{exp}"
            pieces.append(base)
        return " + ".join(pieces).replace("+ -", "- ")

    def bar_coeff(self, expr):
        return sp.expand(expr.subs({v: v ** -1}))
    
    def regular_coefficient(self, coeffs):
        return {w: sp.expand(coeff) for w, coeff in coeffs.items() if sp.expand(coeff) != 0}

    def is_equal(self, element1, element2):
        return self.regular_coefficient(element1) == self.regular_coefficient(element2)

    def inverse_T_w(self, w):
        """
        Compute T_w^{-1} in the T-basis.

        Examples:
            >>> w = (4, 2, 1, 3)
            >>> H = HeckeA(len(w))
            >>> inv = H.inverse_T_w(w)
            >>> prod = H.regular_coefficient(H.hecke_multiply(inv, H.T(w).coeffs))
            >>> prod == {tuple(range(1, len(w) + 1)): sp.Integer(1)}
            True
        """
        key = tuple(w)
        if key in self._inverse_cache:
            return self._inverse_cache[key]
        product = {tuple(range(1, self.n + 1)): sp.Integer(1)}
        for s in reversed(perm_reduced_word(w)):
            product = self.hecke_multiply_by_simple_inverse(product, s)
        self._inverse_cache[key] = product
        return product

    def bar_element(self, element):
        result = {}
        for w, coeff in element.items():
            coeff_bar = self.bar_coeff(coeff)
            w_inv = inverse_permutation(w)
            inv_tw = self.inverse_T_w(w_inv)
            for u, c_u in inv_tw.items():
                result[u] = result.get(u, 0) + coeff_bar * c_u
        return {k: sp.expand(vv) for k, vv in result.items() if vv != 0}

    def is_bar_invariant(self, element):
        return self.bar_element(element) == {
            k: sp.expand(vv) for k, vv in element.items() if vv != 0
        }

    def pretty_print_element(self, element, label=None):
        if not element:
            print(f"{label} 0" if label else "0")
            return
        terms = []
        for w in sorted(element, key=tuple):
            coeff = sp.expand(element[w])
            if coeff == 0:
                continue
            terms.append(f"({self.format_laurent(coeff)}) T_{w}")
        content = " + ".join(terms) if terms else "0"
        print(f"{label} {content}" if label else content)

    def kl_polynomial(self, x, y, cache):
        if x == y:
            return sp.Integer(1)
        if not is_bruhat_leq(x, y):
            return sp.Integer(0)
        key = (x, y)
        if key in cache:
            return cache[key]
        for s in self.simple_reflections():
            sy = permutation_prod(s, y)
            if self.length(sy) < self.length(y):
                sx = permutation_prod(s, x)
                if self.length(sx) < self.length(x):
                    value = self.kl_polynomial(sx, sy, cache)
                else:
                    value = q * self.kl_polynomial(sx, sy, cache)
                    value += (q - 1) * self.kl_polynomial(x, sy, cache)
                    for z in cache["perms"]:
                        if z == x or z == sy:
                            continue
                        if not is_bruhat_leq(x, z) or not is_bruhat_leq(z, sy):
                            continue
                        sz = permutation_prod(s, z)
                        if self.length(sz) >= self.length(z):
                            continue
                        diff = self.length(sy) - self.length(z) - 1
                        if diff < 0 or diff % 2 != 0:
                            continue
                        deg = diff // 2
                        p_z_sy = self.kl_polynomial(z, sy, cache)
                        mu = sp.Poly(p_z_sy, v).coeff_monomial(v ** (2 * deg))
                        if mu != 0:
                            value -= mu * self.kl_polynomial(x, z, cache)
                cache[key] = sp.simplify(value)
                return cache[key]
        raise ValueError("No descent found to apply KL recursion")

    def kl_polynomials(self):
        perms = list(generate_permutations(self.n))
        cache = {"perms": perms}
        kl = {}
        for y in sorted(perms, key=self.length):
            for x in perms:
                if is_bruhat_leq(x, y):
                    kl[(x, y)] = self.kl_polynomial(x, y, cache)
        return kl

    def canonical_basis(self):
        perms = list(generate_permutations(self.n))
        cache = {"perms": perms}
        basis = {}
        for w in perms:
            coeffs = {}
            for x in perms:
                if not is_bruhat_leq(x, w):
                    continue
                p_xw = self.kl_polynomial(x, w, cache)
                d = self.length(w) - self.length(x)
                coeff = (-1) ** d * v ** d * p_xw.subs(v, v ** -1)
                coeffs[x] = sp.expand(coeff)
            basis[w] = HeckeElement(self, coeffs)
        return basis

    def T(self, w):
        return HeckeElement(self, {tuple(w): sp.Integer(1)})

if __name__ == "__main__":
    import sys
    import time
    import doctest

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python HeckeA.py <n>")
    n = int(sys.argv[1])
    doctest.testmod()
    algebra = HeckeA(n)
    start = time.perf_counter()
    kl = algebra.kl_polynomials()
    basis = algebra.canonical_basis()
    elapsed = time.perf_counter() - start
    print(f"Computed KL polynomials for S_{n} (total {len(kl)} pairs).")
    for w in sorted(basis, key=tuple):
        basis[w].pretty(label=f"C_{w} =")
    perms = list(generate_permutations(n))
    all_bar = all(basis[w].is_bar_invariant() for w in perms)
    print(f"Canonical basis bar-invariant: {all_bar}")
    print(f"Total computation time: {elapsed:.3f}s")