import sympy as sp

from perm import (
    generate_permutations,
    is_bruhat_leq,
    length_of_permutation,
    inverse_permutation,
    permutation_prod,
    simple_reflection,
)

v = sp.symbols("v")
q = v * v


_length_cache = {}


def length(w):
    if w not in _length_cache:
        _length_cache[w] = length_of_permutation(w)
    return _length_cache[w]


def simple_reflections(n):
    return [simple_reflection(i, n) for i in range(1, n)]


_mul_simple_cache = {}


def hecke_multiply_by_simple(element, s, lengths=length):
    """
    Left-multiply a Hecke algebra element by a simple reflection s.

    element: dict {w: coeff} representing sum coeff * T_w.
    """
    key = (s, tuple(sorted(element.items())))
    if key in _mul_simple_cache:
        return _mul_simple_cache[key]
    result = {}
    for w, coeff in element.items():
        sw = permutation_prod(s, w)
        if lengths(sw) > lengths(w):
            result[sw] = result.get(sw, 0) + coeff
        else:
            # When the simple reflection s acts on T_w and lengths(sw) < lengths(w), the Hecke relation is
            # T_s T_w = (q - 1) T_w + q T_{sw}, which reflects the quadratic Hecke algebra relation
            # (T_s + 1)(T_s - q) = 0 and its action on the standard basis.
            # So T_s^2 = (q - 1) T_s + q.
            result[w] = result.get(w, 0) + coeff * (q - 1)
            result[sw] = result.get(sw, 0) + coeff * q
    _mul_simple_cache[key] = result
    return result


def hecke_multiply_by_simple_inverse(element, s, lengths=length):
    """
    Left-multiply by T_s^{-1} where T_s^{-1} = q^{-1} T_s + (q^{-1} - 1).
    """
    result = {}
    for w, coeff in element.items():
        sw = permutation_prod(s, w)
        result[sw] = result.get(sw, 0) + coeff * v ** -2
        result[w] = result.get(w, 0) + coeff * (v ** -2 - 1)
    return result


_tw_tu_cache = {}


def multiply_Tw_Tu(w, u, n, lengths=length):
    """
    Multiply T_w * T_u in the Hecke algebra.
    """
    key = (w, u)
    if key in _tw_tu_cache:
        return _tw_tu_cache[key]
    product = {u: sp.Integer(1)}
    for s in reduced_word(w, n):
        product = hecke_multiply_by_simple(product, s, lengths)
    _tw_tu_cache[key] = product
    return product


def hecke_multiply(element1, element2, n, lengths=length):
    """
    Multiply two Hecke algebra elements in the T-basis.
    """
    result = {}
    for w, coeff_w in element1.items():
        product = {u: coeff_w * coeff_u for u, coeff_u in element2.items()}
        for s in reduced_word(w, n):
            product = hecke_multiply_by_simple(product, s, lengths)
        for k, vcoeff in product.items():
            result[k] = result.get(k, 0) + vcoeff
    return result


def format_laurent(expr, var=v):
    """
    Format expr as a Laurent polynomial in var.
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
            base = f"{var}" if exp == 1 else f"{var}**{exp}"
        elif coeff == -1:
            base = f"-{var}" if exp == 1 else f"-{var}**{exp}"
        else:
            base = f"{coeff_str}*{var}" if exp == 1 else f"{coeff_str}*{var}**{exp}"
        pieces.append(base)
    return " + ".join(pieces).replace("+ -", "- ")


def bar_coeff(expr):
    return sp.expand(expr.subs({v: v ** -1}))


def inverse_T_w(w, n, lengths):
    """
    Compute T_w^{-1} in the T-basis via a reduced word.
    """
    product = {tuple(range(1, n + 1)): sp.Integer(1)}
    for s in reduced_word(w, n):
        product = hecke_multiply_by_simple_inverse(product, s, lengths)
    return product


def bar_element(element, n, lengths):
    """
    Bar involution: q -> q^{-1}, v -> v^{-1}, T_w -> T_{w^{-1}}^{-1}.
    """
    result = {}
    for w, coeff in element.items():
        coeff_bar = bar_coeff(coeff)
        w_inv = inverse_permutation(w)
        inv_tw = inverse_T_w(w_inv, n, lengths)
        for u, c_u in inv_tw.items():
            result[u] = result.get(u, 0) + coeff_bar * c_u
    return {k: sp.expand(vv) for k, vv in result.items() if vv != 0}


def is_bar_invariant(element, n, lengths):
    return bar_element(element, n, lengths) == {k: sp.expand(vv) for k, vv in element.items() if vv != 0}


def pretty_print_element(element, label=None):
    """
    Pretty print a Hecke algebra element in the T-basis.
    """
    if not element:
        print(f"{label} 0" if label else "0")
        return
    terms = []
    for w in sorted(element, key=tuple):
        coeff = sp.expand(element[w])
        if coeff == 0:
            continue
        terms.append(f"({format_laurent(coeff)}) T_{w}")
    content = " + ".join(terms) if terms else "0"
    print(f"{label} {content}" if label else content)


def reduced_word(w, n):
    """
    Return a reduced word (as simple reflections) for w using bubble sort.
    """
    w = list(w)
    word = []
    for i in range(n):
        for j in range(n - 1):
            if w[j] > w[j + 1]:
                w[j], w[j + 1] = w[j + 1], w[j]
                word.append(simple_reflection(j + 1, n))
    return word


def kl_polynomial(x, y, n, lengths, cache):
    """
    Compute KL polynomial P_{x,y}(q) for type A_n.
    """
    if x == y:
        return sp.Integer(1)
    if not is_bruhat_leq(x, y):
        return sp.Integer(0)
    key = (x, y)
    if key in cache:
        return cache[key]

    for s in simple_reflections(n):
        sy = permutation_prod(s, y)
        if lengths(sy) < lengths(y):
            sx = permutation_prod(s, x)
            if lengths(sx) < lengths(x):
                value = kl_polynomial(sx, sy, n, lengths, cache)
            else:
                value = q * kl_polynomial(sx, sy, n, lengths, cache)
                value += (q - 1) * kl_polynomial(x, sy, n, lengths, cache)
                for z in cache["perms"]:
                    if z == x or z == sy:
                        continue
                    if not is_bruhat_leq(x, z) or not is_bruhat_leq(z, sy):
                        continue
                    sz = permutation_prod(s, z)
                    if lengths(sz) >= lengths(z):
                        continue
                    diff = lengths(sy) - lengths(z) - 1
                    if diff < 0 or diff % 2 != 0:
                        continue
                    deg = diff // 2
                    p_z_sy = kl_polynomial(z, sy, n, lengths, cache)
                    mu = sp.Poly(p_z_sy, v).coeff_monomial(v ** (2 * deg))
                    if mu != 0:
                        value -= mu * kl_polynomial(x, z, n, lengths, cache)
            cache[key] = sp.simplify(value)
            return cache[key]

    raise ValueError("No descent found to apply KL recursion")


def kl_polynomials(n):
    perms = list(generate_permutations(n))
    cache = {"perms": perms}
    kl = {}
    for y in sorted(perms, key=length):
        for x in perms:
            if is_bruhat_leq(x, y):
                kl[(x, y)] = kl_polynomial(x, y, n, length, cache)
    return kl


def canonical_basis(n):
    """
    Compute the canonical (Kazhdan-Lusztig) basis in the T-basis.

    Returns:
        dict: w -> {x: coefficient in v} giving C_w in the T-basis.
    """
    perms = list(generate_permutations(n))
    cache = {"perms": perms}
    basis = {}
    for w in perms:
        coeffs = {}
        for x in perms:
            if not is_bruhat_leq(x, w):
                continue
            p_xw = kl_polynomial(x, w, n, length, cache)
            d = length(w) - length(x)
            coeff = (-1) ** d * v ** d * p_xw.subs(v, v ** -1)
            coeffs[x] = sp.expand(coeff)
        basis[w] = coeffs
    return basis


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python HeckeA.py <n>")
    n = int(sys.argv[1])
    start = time.perf_counter()
    kl = kl_polynomials(n)
    basis = canonical_basis(n)
    elapsed = time.perf_counter() - start
    print(f"Computed KL polynomials for S_{n} (total {len(kl)} pairs).")
    for w in sorted(basis, key=tuple):
        pretty_print_element(basis[w], label=f"C_{w} =")
    perms = list(generate_permutations(n))
    all_bar = all(is_bar_invariant(basis[w], n, length) for w in perms)
    print(f"Canonical basis bar-invariant: {all_bar}")
    print(f"Total computation time: {elapsed:.3f}s")