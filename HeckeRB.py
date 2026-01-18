import sympy as sp

from HeckeA import v, q
from RB import beta_to_sigma, generate_all_beta, right_action_rb, tilde_inverse 
from perm import (
    generate_permutations,
    inverse_permutation,
    length_of_permutation,
    permutation_prod,
    reduced_word,
    simple_reflection,
    right_descending_element,
)


def normalize_key(w, beta):
    return (tuple(w), tuple(sorted(beta)))


def denormalize_key(key):
    w, beta = key
    return tuple(w), set(beta)


class HeckeRB:
    def __init__(self, n):
        self.n = n
        self._basis = []
        self._right_action_cache = {}
        for w in generate_permutations(n):
            for beta in generate_all_beta(w):
                self._basis.append(normalize_key(w, beta))

    def basis(self):
        for key in self._basis:
            yield key

    def _add_monomial(self, result, term, coeff):
        key = normalize_key(*term)
        result[key] = result.get(key, 0) + coeff

    def _simple_index(self, s):
        diffs = [idx for idx, val in enumerate(s, start=1) if val != idx]
        if len(diffs) != 2:
            raise ValueError(f"Not a simple reflection: {s}")
        return diffs[0]

    def right_action_basis_simple(self, wtilde, i):
        """
        Compute T_(w,beta) * T_{s_i}.

        Returns:
            dict mapping (w,beta) keys to coefficients in q.
        """
        key = (normalize_key(*wtilde), i)
        if key in self._right_action_cache:
            return dict(self._right_action_cache[key])
        w, beta = wtilde
        n = len(w)
        s = simple_reflection(i, n)
        ws = permutation_prod(w, s)
        l_w = length_of_permutation(w)
        l_ws = length_of_permutation(ws)

        beta = set(beta)
        wtilde_s = right_action_rb(wtilde, s)
        ws, ws_beta = wtilde_s

        sigma = beta_to_sigma(w, beta)
        sigma_prime = beta_to_sigma(ws, ws_beta)
        def wprime(wtilde):
            return (wtilde[0], set(wtilde[1]).symmetric_difference({i + 1}))

        iota = {i, i + 1}
        result = {}

        wtilde_s_prime = wprime(wtilde_s)
        wtilde_prime = wprime(wtilde)
        wtilde_prime_s = right_action_rb(wtilde_prime, s)
        
        if l_ws > l_w:
            if (i + 1) not in sigma_prime:
                self._add_monomial(result, wtilde_s, sp.Integer(1))
            else:
                self._add_monomial(result, wtilde_s, sp.Integer(1))
                self._add_monomial(result, wtilde_s_prime, sp.Integer(1))
        else:  # l_ws < l_w
            if (beta & iota) == {i}:
                self._add_monomial(result, wtilde_prime, sp.Integer(1))
                self._add_monomial(result, wtilde_prime_s, sp.Integer(1))
            elif i not in sigma or (i + 1) in (beta - sigma):
                self._add_monomial(result, wtilde, q - 1)
                self._add_monomial(result, wtilde_s, q)
            elif iota.issubset(sigma):
                self._add_monomial(result, wtilde, q - 2)
                self._add_monomial(result, wtilde_prime, q - 1)
                self._add_monomial(result, wtilde_s, q - 1)
            else:
                raise Exception("Right action case not covered by the formula.")
        self._right_action_cache[key] = dict(result)
        return result


    def right_action_simple(self, element, i):
        result = {}
        for key, coeff in element.items():
            w, beta = denormalize_key(key)
            action = self.right_action_basis_simple((w, beta), i)
            for k, c in action.items():
                result[k] = result.get(k, 0) + coeff * c
        return result



    def right_action_basis_by_w(self, wtilde, w):
        key = (normalize_key(*wtilde), tuple(w))
        if key in self._right_action_cache:
            return dict(self._right_action_cache[key])
        i  = right_descending_element(w)
        if i is None:
            element = {normalize_key(*wtilde): sp.Integer(1)}
            self._right_action_cache[key] = dict(element)
            return element
        s = simple_reflection(i, self.n)
        w_prev = permutation_prod(w, s)
        element = self.right_action_basis_by_w(wtilde, w_prev)
        result = self.right_action_simple(element, i)
        self._right_action_cache[key] = dict(result)
        return result

    def right_action_T_w(self, element, w):
        """
        Compute the right action of T_w on an arbitrary element.
        """
        result = {}
        for key, coeff in element.items():
            wtilde = denormalize_key(key)
            action = self.right_action_basis_by_w(wtilde, w)
            for k, c in action.items():
                result[k] = result.get(k, 0) + coeff * c
        return result


    def basis_element(self, w, beta):
        return {normalize_key(w, beta): sp.Integer(1)}

    def ell_wtilde(self, w, beta):
        """
        Length of w~ = (w, beta), using ell(w) + |beta|.
        """
        return length_of_permutation(w) + len(beta)

    def T_wtilde(self, w, beta):
        """
        Standard basis element T_{w~}.
        """
        return {normalize_key(w, beta): sp.Integer(1)}

    def H_wtilde(self, w, beta):
        """
        New basis element H_{w~} = (-v)^{-ell(w~)} T_{w~}.
        """
        coeff = (-v) ** (-self.ell_wtilde(w, beta))
        return {normalize_key(w, beta): sp.expand(coeff)}

    def tilde_inverse_element(self, element):
        """
        Apply (w,beta) -> (w^{-1}, w(beta)) on each basis term.
        """
        result = {}
        for key, coeff in element.items():
            w, beta = denormalize_key(key)
            w_inv, beta_img = tilde_inverse(w, beta)
            new_key = normalize_key(w_inv, beta_img)
            result[new_key] =  coeff
        return result

    def left_action_T_w(self, element, w):
        """
        Left action by T_w using anti-automorphism:
        left = tilde_inverse( tilde_inverse(element) * T_{w^{-1}} ).
        """
        w_inv = inverse_permutation(w)
        temp = self.right_action_T_w(self.tilde_inverse_element(element), w_inv)
        return self.tilde_inverse_element(temp)

    def right_action_hecke(self, element, hecke_element):
        """
        Right action by a HeckeA element (linear extension of T_w actions).
        hecke_element: dict {w: coeff} in the T-basis.
        """
        result = {}
        for w, coeff in hecke_element.items():
            action = self.right_action_T_w(element, w)
            for k, c in action.items():
                result[k] = result.get(k, 0) + coeff * c
        return result

    def left_action_hecke(self, element, hecke_element):
        """
        Left action by a HeckeA element (linear extension of T_w actions).
        hecke_element: dict {w: coeff} in the T-basis.
        """
        result = {}
        for w, coeff in hecke_element.items():
            action = self.left_action_T_w(element, w)
            for k, c in action.items():
                result[k] = result.get(k, 0) + coeff * c
        return result
