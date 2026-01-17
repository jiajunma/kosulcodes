import sympy as sp

from RB import beta_to_sigma
from perm import inverse_permutation, length_of_permutation, permutation_prod, simple_reflection

q = sp.symbols("q")


def apply_perm_to_positions(p, subset):
    return {p[i - 1] for i in subset}


def tilde_inverse(w, beta):
    w_inv = inverse_permutation(w)
    beta_img = apply_perm_to_positions(w, beta)
    return w_inv, beta_img


def _toggle(beta, idx):
    return set(beta) ^ {idx}


def right_action(w, beta, i):
    """
    Compute T_{(w,beta)} * T_{s_i}.

    Returns:
        dict mapping (w,beta) -> coefficient (in q).
    """
    n = len(w)
    s = simple_reflection(i, n)
    ws = permutation_prod(w, s)
    l_w = length_of_permutation(w)
    l_ws = length_of_permutation(ws)

    beta = set(beta)
    s_beta = apply_perm_to_positions(s, beta)

    sigma = beta_to_sigma(w, beta)
    sigma_prime = beta_to_sigma(ws, s_beta)

    l = {i, i + 1}
    result = {}

    def add(term, coeff):
        result[term] = result.get(term, 0) + coeff

    wtilde_s = (ws, s_beta)
    wtilde_s_prime = (ws, _toggle(s_beta, i + 1))
    w_prime = (w, _toggle(beta, i + 1))
    w_prime_s = (ws, apply_perm_to_positions(s, _toggle(beta, i + 1)))

    if l_ws > l_w:
        if (i + 1) not in sigma_prime:
            add(wtilde_s, sp.Integer(1))
        else:
            add(wtilde_s, sp.Integer(1))
            add(wtilde_s_prime, sp.Integer(1))
        return result

    # l_ws < l_w
    if beta & l == {i}:
        add(w_prime, sp.Integer(1))
        add(w_prime_s, sp.Integer(1))
        return result

    if (i not in sigma) or ((i + 1) in (beta - sigma)):
        add((w, beta), q - 1)
        add(wtilde_s, q)
        return result

    if l.issubset(sigma):
        add((w, beta), q - 2)
        add(w_prime, q - 1)
        add(wtilde_s, q - 1)
        return result

    raise ValueError("Right action case not covered by the formula.")


def left_action(w, beta, i):
    """
    Compute T_{s_i} * T_{(w,beta)} using the anti-automorphism.
    """
    w_inv, beta_inv = tilde_inverse(w, beta)
    right = right_action(w_inv, beta_inv, i)
    result = {}
    for (w2, beta2), coeff in right.items():
        w_back, beta_back = tilde_inverse(w2, beta2)
        result[(w_back, beta_back)] = result.get((w_back, beta_back), 0) + coeff
    return result


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        raise SystemExit("Usage: python HeckeRB.py <n> <i> <w as tuple> <beta as set>")
    n = int(sys.argv[1])
    i = int(sys.argv[2])
    w = eval(sys.argv[3], {"__builtins__": {}})
    beta = eval(sys.argv[4], {"__builtins__": {}})
    if not isinstance(w, tuple) or not isinstance(beta, set) or len(w) != n:
        raise SystemExit("Input format: w is tuple, beta is set, len(w)=n")
    print("Right action:", right_action(w, beta, i))
    print("Left action:", left_action(w, beta, i))
