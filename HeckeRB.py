import sympy as sp
from sympy import collect
from HeckeA import v, q, HeckeA
from RB import beta_to_sigma, sigma_to_beta, generate_all_sigma, is_decreasing_on_subset, right_action_rb, root_type_right, tilde_inverse_sigma, str_colored_partition, normalize_key, denormalize_key 

from perm import (
    generate_permutations,
    inverse_permutation,
    length_of_permutation,
    permutation_prod,
    reduced_word,
    simple_reflection,
    right_descending_element,
)
from RB_Bruhat import (
    node_key,
    get_bruhat_order,
    is_bruhat_leq as rb_is_bruhat_leq,
    bruhat_lower_elements as rb_bruhat_lower_elements,
    bruhat_covers as rb_bruhat_covers,
)




class HeckeRB:
    def __init__(self, n):
        self.n = n
        self._basis = dict() 
        self._right_action_cache = {}
        for w in generate_permutations(n):
            for sigma in generate_all_sigma(w):
                self._basis[normalize_key(w, sigma)] = root_type_right(w, sigma) 
        
        # Group elements by length
        self.elements_by_length = {}
        self.max_length = 0
        for key in self._basis:
            ell = self.ell_wtilde(key)
            if ell not in self.elements_by_length:
                self.elements_by_length[ell] = []
            self.elements_by_length[ell].append(key)
            self.max_length = max(self.max_length, ell)

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
        We assume wtilde is normalized
        Compute T_(w,beta) * T_{s_i}.

        Returns:
            dict mapping (w,beta) keys to coefficients in q.
        """
        s = simple_reflection(i, self.n)
        key = (wtilde, s)
        if key in self._right_action_cache:
            return dict(self._right_action_cache[key])
        # T for type and C for companion
        T, C = self._basis[wtilde][i]
        result = {wtilde: - sp.Integer(1)}

        if T == "G":
            result[wtilde] = result.get(wtilde, 0) + (q + 1)
        elif T == "U+":
            for c in C:
                result[c] = result.get(c, 0) + q
        elif T == "T+":
            for c in C:
                result[c] = result.get(c, 0) + (q - 1)
        elif T == "U-" or T == "T-":
            for c in C:
                result[c] = result.get(c, 0) + sp.Integer(1)
        result = {k: sp.expand(c) for k, c in result.items() if sp.expand(c) != 0}
        self._right_action_cache[key] = result
        return result



    def right_action_simple(self, element, i):
        """Right action by T_{s_i} on an element in the T-basis."""
        result = {}
        for key, coeff in element.items():
            w, beta = denormalize_key(key)
            action = self.right_action_basis_simple((w, beta), i)
            for k, c in action.items():
                result[k] = result.get(k, 0) + coeff * c
        return result

    def right_action_basis_by_w(self, wtilde, w):
        """
        Compute the right action of T_w on a basis element T_{wtilde}.

        Args:
            wtilde: tuple (w, beta) representing the basis element
            w: permutation to act by

        Returns:
            Dictionary mapping basis elements to their coefficients
        """
        key = (normalize_key(*wtilde), tuple(w))
        # Check if result is already in cache
        if key in self._right_action_cache:
            return dict(self._right_action_cache[key])

        # Base case: w is identity
        i = right_descending_element(w)
        if i is None:
            element = {normalize_key(*wtilde): sp.Integer(1)}
            return element

        # Recursive case: factor w = w_prev * s_i
        s = simple_reflection(i, self.n)
        w_prev = permutation_prod(w, s)
        assert permutation_prod(w_prev, s) == tuple(w), f"permutation_prod(w_prev, s) = {permutation_prod(w_prev, s)} != {w}"


        # Compute T_{wtilde} * T_{w_prev} recursively
        element = self.right_action_basis_by_w(wtilde, w_prev)

        # Then right action by T_{s_i} to get T_{wtilde} * T_{w_prev} * T_{s_i} = T_{wtilde} * T_w
        result = self.right_action_simple(element, i)

        # Cache and return the result
        self._right_action_cache[key] = result
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


    def ell_wtilde(self, wtilde):
        """
        Length of w~ = (w, beta), using ell(w) + |beta|.
        """
        # this is already calculated in basis
        return self._basis[wtilde][0]




    def T_wtilde(self, w, sigma):
        """
        Standard basis element T_{w~}.
        """
        return {normalize_key(w, sigma): sp.Integer(1)}

    def basis_element(self, w, sigma):
        return self.T_wtilde(w, sigma)



    def H_wtilde(self, w, sigma):
        """
        New basis element H_{w~} = (-v)^{-ell(w~)} T_{w~}.
        """
        coeff = (-v) ** (-self.ell_wtilde(normalize_key(w, sigma)))
        return {normalize_key(w, sigma): sp.expand(coeff)}

    def T_to_H(self, element):
        """
        Convert an element from T-basis to H-basis.
        
        Given element = ∑ c_k T_{w̃_k} in T-basis,
        returns the same element expressed in H-basis.
        
        Since H_{w̃} = (-v)^{-ℓ(w̃)} T_{w̃}, we have T_{w̃} = (-v)^{ℓ(w̃)} H_{w̃}.
        So ∑ c_k T_{w̃_k} = ∑ c_k (-v)^{ℓ(w̃_k)} H_{w̃_k}
        
        Args:
            element: dict mapping (w, beta) keys to coefficients (in T-basis)
        
        Returns:
            dict mapping (w, beta) keys to coefficients (in H-basis)
        """
        result = {}
        for key, coeff in element.items():
            ell = self.ell_wtilde(key)
            # T_{w̃} = (-v)^{ℓ(w̃)} H_{w̃}
            h_coeff = coeff * ((-v) ** ell)
            result[key] = result.get(key, sp.Integer(0)) + h_coeff
        return {k: sp.expand(c) for k, c in result.items() if sp.expand(c) != 0}

    def H_to_T(self, element):
        """
        Convert an element from H-basis to T-basis.
        
        Given element = ∑ d_k H_{w̃_k} in H-basis,
        returns the same element expressed in T-basis.
        
        Since H_{w̃} = (-v)^{-ℓ(w̃)} T_{w̃}, we have:
        ∑ d_k H_{w̃_k} = ∑ d_k (-v)^{-ℓ(w̃_k)} T_{w̃_k}
        
        Args:
            element: dict mapping (w, beta) keys to coefficients (in H-basis)
        
        Returns:
            dict mapping (w, beta) keys to coefficients (in T-basis)
        """
        result = {}
        for key, coeff in element.items():
            ell = self.ell_wtilde(key)
            # H_{w̃} = (-v)^{-ℓ(w̃)} T_{w̃}
            t_coeff = coeff * ((-v) ** (-ell))
            result[key] = result.get(key, sp.Integer(0)) + t_coeff
        return {k: sp.expand(c) for k, c in result.items() if sp.expand(c) != 0}

    def tilde_inverse_element(self, element):
        """
        Apply (w,beta) -> (w^{-1}, w(beta)) on each basis term.
        """
        result = {}
        for key, coeff in element.items():
            w, beta = denormalize_key(key)
            w_inv, beta_img = tilde_inverse_sigma(w, beta)
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

    # =========================================================================
    # Bruhat order on RB (using RB_Bruhat module)
    # =========================================================================
    def _normalize_to_key(self, wtilde):
        """Convert wtilde to normalized key format for Bruhat order.
        
        RB_Bruhat expects keys in format (tuple(w), tuple(sorted(sigma))).
        Our _basis uses keys in format (tuple(w), frozenset(sigma)).
        """
        if isinstance(wtilde, tuple) and len(wtilde) == 2:
            w, sigma = wtilde
            # Convert frozenset to tuple(sorted(...))
            if isinstance(sigma, frozenset):
                return (tuple(w), tuple(sorted(sigma)))
            elif isinstance(sigma, (set, list)):
                return (tuple(w), tuple(sorted(sigma)))
            else:
                # Already in correct format
                return wtilde
        return wtilde

    def is_bruhat_leq(self, wtilde1, wtilde2):
        """
        Check if wtilde1 <= wtilde2 in the Bruhat order on RB.
        
        Args:
            wtilde1, wtilde2: pairs (w, beta) or normalized keys
        """
        key1 = self._normalize_to_key(wtilde1)
        key2 = self._normalize_to_key(wtilde2)
        return rb_is_bruhat_leq(key1, key2, self.n)

    def bruhat_lower_elements(self, wtilde):
        """Return all elements < wtilde in Bruhat order."""
        key = self._normalize_to_key(wtilde)
        return rb_bruhat_lower_elements(key, self.n)

    def bruhat_covers(self, wtilde):
        """Return elements covered by wtilde (immediate predecessors)."""
        key = self._normalize_to_key(wtilde)
        return rb_bruhat_covers(key, self.n)

    # =========================================================================
    # Special elements w_i = (id, {1,...,i})
    # =========================================================================
    def special_element_key(self, i):
        f"""
        Return the key for special element w_i = (id, {1,...,i}).
        This is the same as (id, {i}) in (w,sigma) notation.
        These are fundamental for the bar involution.
        """
        identity = tuple(range(1, self.n + 1))
        if i == 0: 
            sigma_i = frozenset() 
        else:
            sigma_i = frozenset({i})
        return normalize_key(identity, sigma_i)

    # =========================================================================
    # Bar involution
    # =========================================================================
    def bar_coeff(self, expr):
        """Apply bar involution to coefficient: v -> v^{-1} and normalize."""
        return collect(expr.subs({v: v ** -1}).expand(), v)

    def compute_bar_involution(self, verbose=False):
        """
        Compute bar involution on every standard basis element T_{w̃}.
        
        Algorithm:
        1. Create a list indexed by length of all basis elements
        2. Set bar(T_{w_i}) for special elements using:
           bar(T_{w_i}) = v^{-2i} T_{w_i} + ∑_{j<i} (v^{-2i} - v^{-2i+2}) T_{w_j}
        3. Iterate over length from 0 to maximal:
           - For each w̃ of length ℓ, compute T_{w̃} · T_s for all simple s
           - Find w̃'' with ℓ(w̃'') > ℓ(w̃) and coefficient 1
           - Compute: bar(T_{w̃''}) = bar(T_{w̃}) · bar(T_s) - ∑_{w̃'≠w̃''} bar(a) bar(T_{w̃'})
        4. Verify all elements are reached
        
        Returns:
            dict: mapping key -> bar(T_{key}) as element dict
        """
        bar_table = {}
        
        # Step 1: Create list indexed by length
        elements_by_length = self.elements_by_length
        max_length = self.max_length
        
        if verbose:
            print("Step 1: Elements by length")
            for ell in sorted(elements_by_length.keys()):
                print(f"  Length {ell}: {len(elements_by_length[ell])} elements")
        
        # Step 2: Set bar(T_{w_i}) for special elements
        # bar(T_{w_i}) = v^{-2i} T_{w_i} + ∑_{j<i} (v^{-2i} - v^{-2(i-1)}) T_{w_j}
        #             = v^{-2i} T_{w_i} + ∑_{j<i} (v^{-2i} - v^{-2i+2}) T_{w_j}
        if verbose:
            print("\nStep 2: Bar involution for special elements T_{w_i}")
        
        special_keys = [self.special_element_key(i) for i in range(self.n + 1)]
        
        for i in range(self.n + 1):
            key_i = special_keys[i]
            bar_i = {}
            
            # Coefficient of T_{w_i}: v^{-2i}
            bar_i[key_i] = v ** (-2 * i)
            
            # Coefficient of T_{w_j} for j < i: (v^{-2i} - v^{-2i+2})
            for j in range(i):
                key_j = special_keys[j]
                bar_i[key_j] = v ** (-2 * i) - v ** (-2 * i + 2)
            
            #bar_i = {k: sp.expand(c) for k, c in bar_i.items() if sp.expand(c) != 0}
            bar_table[key_i] = bar_i
            
            if verbose:
                w, beta = denormalize_key(key_i)
                print(f"  bar(T_{{w_{i}}}) = {self._format_T_element(bar_i)}")
        
        # Step 3: Iterate by length from 0 to max_length
        if verbose:
            print("\nStep 3: Iterating by length")
        
        # bar(T_s) in Hecke algebra = q^{-1} T_s + (q^{-1}-1) = v^(-2) T_s + (v^(-2) - 1)
        
        for ell in range(max_length):
                # Process all elements of length ell with known bar
                for key in elements_by_length.get(ell, []):
                    assert key in bar_table, f"bar_table missing key {key}"
                    
                    
                    # Try each simple reflection on right action
                    for s_idx in range(1, self.n):
                        # Compute T_{w̃} · T_s
                        action = self.right_action_basis_simple(key, s_idx)
                        
                        # Find terms with ℓ(w̃'') > ℓ(w̃)
                        higher_terms = []
                        
                        for k, c in action.items():
                            ell_k = self.ell_wtilde(k)
                            if ell_k > ell:
                                higher_terms.append((k, c))
                        
                        if len(higher_terms) == 0:
                            continue

                        if len(higher_terms) > 1:
                            perm_str = self._format_wtilde(w, beta)
                            print(f"More than one higher term found in T_[{perm_str}] · T_s{s_idx}: {higher_terms}")
                            assert False
                        # Check the claim
                        
                        key_higher, coeff_higher = higher_terms[0]
                        c_expanded = sp.expand(coeff_higher)
                            
                        # Verify coefficient is 1
                        assert c_expanded == 1, f"Expected coefficient 1 for higher term, got {c_expanded} in T_[{self._format_wtilde(w, beta)}] · T_s{s_idx}"

                        if key_higher in bar_table:
                            continue
                        
                        # Check all correction terms are available
                        for k_other, a_other in action.items():
                            if k_other != key_higher and k_other not in bar_table:
                                raise ValueError(f"bar_table missing required key {k_other} for computation of bar({key_higher})")
                        
                        # Compute bar(T_{w̃''})
                        bar_w = bar_table[key]
                        
                        # bar(T_{w̃}) · bar(T_s) = bar(T_{w̃}) · (q^{-1} T_s + (q^{-1}-1))
                        bar_w_times_Ts = self.right_action_simple(bar_w, s_idx)
                        
                        result = {}
                        for k, c in bar_w_times_Ts.items():
                            result[k] = v**(-2) * c
                        for k, c in bar_w.items():
                            result[k] = result.get(k, sp.Integer(0)) + (v**(-2) - 1) * c
                        
                        # Subtract bar(a) * bar(T_{w̃'}) for w̃' ≠ w̃''
                        for k_other, a_other in action.items():
                            if k_other == key_higher:
                                continue
                            bar_other = bar_table[k_other]
                            a_bar = self.bar_coeff(a_other)
                            for k, c in bar_other.items():
                                result[k] = result.get(k, sp.Integer(0)) - a_bar * c
                        
                        result = {k: sp.expand(c) for k, c in result.items() if sp.expand(c) != 0}
                        bar_table[key_higher] = result
                        
                        if verbose:
                            w_h, beta_h = denormalize_key(key_higher)
                            print(f"    bar(T_[{w_h},{beta_h}]) from T_[{w},{beta}] · T_s{s_idx}")
                    
                    # Also try left action T_s · T_{w̃} for this element
                    for s_idx in range(1, self.n):
                        # T_s · T_{w̃} = ι(ι(T_{w̃}) · T_s)
                        action = self.left_action_T_w({key: sp.Integer(1)}, simple_reflection(s_idx, self.n))

                        # Find higher term
                        key_higher = []
                        coeff_higher = [] 
                        for k, c in action.items():
                            if self.ell_wtilde(k) > ell:
                                key_higher.append(k)
                                coeff_higher.append(c)

                        if not key_higher:
                            continue

                        assert len(key_higher) == 1, f"Expected one higher term, got {len(key_higher)} in T_s{s_idx} · T_[{w},{beta}]"

                        key_higher, coeff_higher = key_higher[0], coeff_higher[0]
                        c_expanded = sp.expand(coeff_higher)
                        assert c_expanded == 1, f"Expected coefficient 1 for higher term, got {c_expanded} in T_s{s_idx} · T_[{w},{beta}]"

                        if key_higher in bar_table:
                            continue

                        # Check correction terms available
                        for k_other in action:
                            if k_other != key_higher and k_other not in bar_table:
                                raise ValueError(f"bar_table missing required key {k_other} for computation of bar({key_higher})")

                        # bar(T_s · T_{w̃}) = bar(T_s) · bar(T_{w̃})
                        bar_w = bar_table[key]

                        # T_s · bar(T_{w̃}) via anti-automorphism
                        Ts_left_bar_w = self.left_action_T_w(bar_w, simple_reflection(s_idx, self.n))

                        # Apply the same formula as in the right action case: bar(T_s) = q^{-1} T_s + (q^{-1}-1)
                        result = {}
                        for k, c in Ts_left_bar_w.items():
                            result[k] = v**(-2) * c
                        for k, c in bar_w.items():
                            result[k] = result.get(k, sp.Integer(0)) + (v**(-2) - 1) * c

                        # Subtract bar(a) * bar(T_{w̃'}) for w̃' ≠ w̃''
                        for k_other, a_other in action.items():
                            if k_other == key_higher:
                                continue
                            bar_other = bar_table[k_other]
                            a_bar = self.bar_coeff(a_other)
                            for k, c in bar_other.items():
                                result[k] = result.get(k, sp.Integer(0)) - a_bar * c


                        result = {k: sp.expand(c) for k, c in result.items() if sp.expand(c) != 0}
                        bar_table[key_higher] = result
                        
                        if verbose:
                            w_h, beta_h = denormalize_key(key_higher)
                            print(f"    bar(T_[{w_h},{beta_h}]) from T_s{s_idx} · T_[{w},{beta}] (left)")
        
        # Step 4: Verify all elements are reached
        missing = []
        for key in self._basis:
            if key not in bar_table:
                missing.append(key)

        if len(bar_table) < len(self._basis):
            # Compose a readable missing list (at most 10 elements shown)
            missing_info = []
            for key in missing:
                w, beta = denormalize_key(key)
                missing_info.append(f"T_[{w},{beta}]")
            raise RuntimeError(
                f"bar_table is missing {len(missing)} out of {len(self._basis)} basis elements.\n"
                f"First missing: {', '.join(missing_info)}{more}"
            )

        if verbose:
            print(f"\nStep 4: Verification")
            print(f"  Total basis elements: {len(self._basis)}")
            print(f"  Elements with bar computed: {len(bar_table)}")
            print("  All elements reached!")
        return bar_table

    def compute_R(self):
        """
        Compute the R-matrix for the Hecke module.
        This is the coefficient in the expansion:
        bar(H_x) = ∑_y R_{y,x} H_y

        self.R[x][y] := R_{y,x}
        """
        R = {}
        if not hasattr(self, 'bar_table'):
            self.bar_table = self.compute_bar_involution()
        for x in self._basis:
            R[x] = {}
            for y in self.bar_table[x]:
                R[x][y] = sp.expand(self.bar_table[x][y] * ((-v) ** (self.ell_wtilde(x) + self.ell_wtilde(y))))
        self.R = R
        for x in self._basis:
            assert R[x][x] == sp.Integer(1)
        return R


    def bar_T(self, element):
        """
        Compute the bar involution of an element in the T-basis.

        Uses the pre-computed bar_table from compute_bar_involution.
        The bar involution on the T-basis is:
            bar(∑ c_k T_{w̃_k}) = ∑ bar(c_k) bar(T_{w̃_k})

        where:
        - bar(c) = c|_{v → v^{-1}} (coefficient bar involution)
        - bar(T_{w̃_k}) is the pre-computed value from bar_table

        This satisfies:
        - bar(bar(x)) = x (involution property)
        - bar(v · x) = v^{-1} · bar(x) (semilinearity)

        Args:
            element: dict mapping (w, beta) keys to coefficients (in T-basis)

        Returns:
            dict mapping (w, beta) keys to coefficients (in T-basis)
        """
        # Ensure bar_table is computed
        if not hasattr(self, 'bar_table'):
            self.bar_table = self.compute_bar_involution()

        result = {}
        for key, coeff in element.items():
            # Apply coefficient bar: v -> v^{-1}
            coeff_bar = self.bar_coeff(coeff)

            # Use pre-computed bar(T_{w̃}) from bar_table
            if key in self.bar_table:
                bar_basis = self.bar_table[key]
                for k, c in bar_basis.items():
                    result[k] = result.get(k, sp.Integer(0)) + coeff_bar * c
            else:
                raise ValueError(f"Key {key} not found in bar_table. Run compute_bar_involution() first.")

        return {k: sp.expand(c) for k, c in result.items() if sp.expand(c) != 0}

    def bar_H(self, element):
        """
        Compute the bar involution of an element in the H-basis.
        
        Uses the T-basis bar involution with basis conversion:
            bar_H(x) = T_to_H(bar_T(H_to_T(x)))
        
        Args:
            element: dict mapping (w, beta) keys to coefficients (in H-basis)
        
        Returns:
            dict mapping (w, beta) keys to coefficients (in H-basis)
        """
        t_element = self.H_to_T(element)
        t_bar = self.bar_T(t_element)
        return self.T_to_H(t_bar)

    def bar_element(self, element):
        """
        Compute the bar involution of an element (alias for bar_T).
        
        Args:
            element: dict mapping (w, beta) keys to coefficients (in T-basis)
        
        Returns:
            dict mapping (w, beta) keys to coefficients (in T-basis)
        """
        return self.bar_T(element)

    def regular_coefficient(self, element):
        """Remove zero coefficients from element and normalize expressions."""
        return {k: collect(c.expand(), v) for k, c in element.items() if sp.expand(c) != 0}

    def is_bar_invariant(self, element):
        """Check if an element is bar-invariant."""
        bar_elem = self.bar_element(element)
        return self.is_equal(element, bar_elem)

    def is_equal(self, e1, e2):
        """
        Check if two elements are equal by verifying that the difference of
        their coefficients is 0 for all basis elements ~w.
        """
        
        # Get all keys (basis elements ~w) from both elements
        all_keys = set(e1.keys()) | set(e2.keys())

        # For each basis element ~w, check if c_1(~w) - c_2(~w) = 0
        for key in all_keys:
            coeff1 = e1.get(key, sp.Integer(0))
            coeff2 = e2.get(key, sp.Integer(0))
            diff = sp.expand(coeff1 - coeff2)

            # If any coefficient difference is not 0, elements are not equal
            if diff != 0:
                return False

        # All coefficient differences are 0, elements are equal
        return True

    # =========================================================================
    # W-graph structure (Proposition 9 from the paper)
    # =========================================================================
    def is_in_Phi_i(self, key, i):
        """
        Check if w̃ = (w, sigma) is in Φ_i.
        
        Using the definition provided:
        tilde w in the decent set of s_i if the root type for s_i 
        under the right action is G, U+, or T+.
        """
        if i < 1 or i >= self.n:
            return False
        
        T, C = self._basis[key][i]
        return T in ["G", "U+", "T+"]

    def wtilde_star_si(self, key, i):
        """
        Compute w̃ * s_i = (w, sigma) * s_i.
        
        This is the right action: (ws, s(β)) where s(β) = {s(j) : j ∈ β}.
        We convert sigma to beta, apply the action, and convert back to sigma.
        """
        w, sigma = denormalize_key(key)
        beta = sigma_to_beta(w, sigma)
        n = len(w)
        s = simple_reflection(i, n)
        ws = permutation_prod(w, s)
        # s acts on β by swapping indices i and i+1
        s_beta = set()
        for j in beta:
            if j == i:
                s_beta.add(i + 1)
            elif j == i + 1:
                s_beta.add(i)
            else:
                s_beta.add(j)
        
        # Convert back to sigma
        s_sigma = beta_to_sigma(ws, s_beta)
        return normalize_key(ws, s_sigma)

    # =========================================================================
    # Kazhdan-Lusztig basis computation
    # =========================================================================
    def compute_kl_polynomials(self, verbose=False):
        """
        Compute all KL polynomials P_{y,x} using the inductive algorithm.
        
        Algorithm (following Lusztig's Lemma 24.2.1):
        1. Ensure bar_table is computed (R_{x,y} = bar_table[y][x])
        2. Set P_{x,x} = 1 for all x
        3. For each length gap k = 1, 2, 3, ...:
           For each pair (y, x) with ℓ(x) - ℓ(y) = k:
             a) Compute q_{y,x} = ∑_{y < z ≤ x} R_{y,z} · bar(P_{z,x})
             b) P_{y,x} = negative-power part of q_{y,x}
        
        Returns:
            dict: KL[x][y] = P_{y,x} as a polynomial in v^{-1}
        """
        # Ensure bar_table is computed
        if not hasattr(self, 'bar_table'):
            if verbose:
                print("Computing bar_table first...")
            self.bar_table = self.compute_bar_involution(verbose=False)
        self.compute_R() 
        # Initialize KL table
        KL = {}
        
        # Step 1: Group elements by length
        elements_by_length = self.elements_by_length
        max_length = self.max_length
        
        if verbose:
            print(f"\nComputing KL polynomials for {len(self._basis)} basis elements")
            print(f"Max length: {max_length}")
        
        # Initialize all KL[x][x] dicts
        for x in self._basis:
            KL[x] = {}
            # Step 2: Base case P_{x,x} = 1
            #KL[x][x] = sp.Integer(1)
            KL[x][x] = sp.Integer(1)

        
        # Step 3: Induction on length gap k = ℓ(x) - ℓ(y)
        for k in range(1, max_length + 1):
            if verbose:
                print(f"\nStep 3: Processing length gap k = {k}")
            
            count = 0
            for ell_x in range(k, max_length + 1):
                ell_y = ell_x - k
                if ell_y not in elements_by_length:
                    continue
                
                for x in elements_by_length.get(ell_x, []):
                    for y in elements_by_length.get(ell_y, []):
                        # Compute q_{y,x} = ∑_{y < z ≤ x} R'_{y,z} · bar(P_{z,x})
                        q_yx = sp.Integer(0)
                        
                        # Use already computed KL[x] which contains P_{z,x} for ell_z > ell_y
                        for z, P_zx in KL[x].items():
                            # y < z implies length(y) < length(z)
                            ell_z = self.ell_wtilde(z)
                            if ell_z <= ell_y:
                                continue
                            
                            # R'_{y,z} = coefficient of H_y in bar(H_z)
                            R_prime_yz = self.R[z].get(y, sp.Integer(0))
                            if R_prime_yz == 0 or R_prime_yz == sp.Integer(0):
                                continue
                            
                            # bar(P_{z,x}): v -> v^{-1}
                            bar_P_zx = self.bar_coeff(P_zx)
                            
                            # Add to sum
                            q_yx = q_yx + R_prime_yz * bar_P_zx
                        
                        # P_{y,x} = negative-power part of q_{y,x}
                        if q_yx != 0:
                            q_yx = sp.expand(q_yx)
                            P_yx = self._extract_negative_powers(q_yx)
                            if P_yx != 0:
                                KL[x][y] = P_yx
                                count += 1
            
            if verbose and count > 0:
                print(f"  Computed {count} polynomials at gap k = {k}")
        
        # Store in cache
        self._kl_table = KL
        
        if verbose:
            print(f"\nKL polynomial computation complete!")
        
        return KL
    
    def _extract_negative_powers(self, poly):
        """
        Extract terms with negative powers of v from a polynomial.
        
        Returns the sum of all terms c_i * v^i where i < 0.
        """
        poly = sp.expand(poly)
        
        if poly == 0 or poly == sp.Integer(0):
            return sp.Integer(0)
        
        result = sp.Integer(0)
        
        # Manually extract terms
        for term in sp.Add.make_args(poly):
            # Get the power of v in this term
            powers = term.as_powers_dict()
            v_power = powers.get(v, 0)
            
            if v_power < 0:
                result += term
        
        return sp.expand(result)
    
    def kl_polynomial(self, y_key, w_key):
        """
        Get the KL polynomial P_{y,x} from the pre-computed table.
        
        The canonical basis element C_x is defined as:
        C_x = H_x + ∑_{y < x} P_{y,x} H_y
        
        where:
        - P_{y,x} ∈ v^{-1}Z[v^{-1}] for y < x
        - C_x is bar-invariant: bar(C_x) = C_x
        
        Note: This returns KL[w_key][y_key] = P_{y_key, w_key}
        """
        # Compute table if not already done
        if not hasattr(self, '_kl_table'):
            self.compute_kl_polynomials(verbose=False)
        
        # Return P_{y,x}
        return self._kl_table.get(w_key, {}).get(y_key, sp.Integer(0))

    def compute_inverse_kl_polynomials(self, verbose=False):
        """
        Compute the inverse KL polynomials Q_{z,y} such that
        sum_y Q_{z,y} P_{y,x} = delta_{z,x}.
        
        Since P is unipotent upper triangular (with respect to length),
        we can compute Q inductively:
        Q_{x,x} = 1
        Q_{z,x} = - sum_{z <= y < x} Q_{z,y} P_{y,x} for z < x
        """
        if not hasattr(self, '_kl_table'):
            self.compute_kl_polynomials(verbose=verbose)
            
        Q = {}
        # Initialize diagonal
        for x in self._basis:
            Q[x] = {x: sp.Integer(1)}
            
        # Group by length to compute by induction on length gap
        elements_by_length = self.elements_by_length
        max_length = self.max_length
        
        # Induction on length gap k = ell(x) - ell(z)
        for k in range(1, max_length + 1):
            if verbose:
                print(f"Computing inverse KL polynomials for length gap k={k}...")
            for ell_x in range(k, max_length + 1):
                ell_z = ell_x - k
                if ell_z not in elements_by_length:
                    continue
                for x in elements_by_length.get(ell_x, []):
                    for z in elements_by_length.get(ell_z, []):
                        # We want Q_{z,x} = - sum_{z <= y < x} Q_{z,y} P_{y,x}
                        # Since we iterate by length gap k, all Q_{z,y} for ell(y) < ell(x) are known.
                        
                        s = sp.Integer(0)
                        # Iterate over y such that P_{y,x} is non-zero
                        for y, p_yx in self._kl_table[x].items():
                            if y == x:
                                continue
                            # Check if Q_{z,y} is already computed and non-zero
                            q_zy = Q[y].get(z, sp.Integer(0))
                            if q_zy != 0:
                                s += q_zy * p_yx
                                
                        if s != 0:
                            Q[x][z] = sp.expand(-s)
                            
        self._inverse_kl_table = Q
        return Q

    def inverse_kl_polynomial(self, z_key, y_key):
        """
        Get the inverse KL polynomial Q_{z,y} from the pre-computed table.
        Satisfies sum_y Q_{z,y} P_{y,x} = delta_{z,x}.
        """
        if not hasattr(self, '_inverse_kl_table'):
            self.compute_inverse_kl_polynomials()
        
        # Return Q_{z,y} = self._inverse_kl_table[y][z]
        return self._inverse_kl_table.get(y_key, {}).get(z_key, sp.Integer(0))

    def compute_mu_coefficients(self):
        """
        Compute the mu coefficient μ(ỹ, w̃).
        
        The mu coefficient is defined as the coefficient of v^{-1} 
        in the KL polynomial P_{ỹ,w̃}.
        
        For the W-graph, μ(ỹ, w̃) ≠ 0 implies an edge from ỹ to w̃.
        
        In the standard basis action:
        H̃_{w̃} · H̃_s =  ∑_{ỹ: s ∈ τ(ỹ), s ∉ τ(w̃)} μ(ỹ, w̃) H̃_{ỹ} 
        
        Returns:
            the dictionary of mu coefficients
            self.mu[w̃][ỹ] := μ(ỹ, w̃)
            here all coefficient in the dict must be non-zero
        """
        self.mu = {}
        for w_key in self.basis():
            self.mu[w_key] = {}
            ell_w = self.ell_wtilde(w_key)
            
            for y_key in self.basis():
                ell_y = self.ell_wtilde(y_key)
                diff = ell_w - ell_y
                
                p = self.kl_polynomial(y_key, w_key)
                if p == 0 or p == sp.Integer(0):
                    continue
                
                # Extract coefficient of v^{-1}
                # This is the standard definition: μ(y,w) = coeff of v^{-1} in P_{y,w}
                #target_exp = -diff
                target_exp = -1
                # Extract the coefficient of v**target_exp in the polynomial p
                poly = sp.expand(p)
                coeff = poly.coeff(v, target_exp)
                if coeff != sp.Integer(0):
                    self.mu[w_key][y_key] = abs(coeff)


    def mu_coefficient(self, y_key, w_key):
        """
        Compute the mu coefficient μ(ỹ, w̃).
        
        The mu coefficient is defined as the coefficient of v^{-1} 
        in the KL polynomial P_{ỹ,w̃}.
        
        For the W-graph, μ(ỹ, w̃) ≠ 0 implies an edge from ỹ to w̃.
        
        Returns:
            The mu coefficient (integer or 0)
        """
        # Ensure mu coefficients are computed
        if not hasattr(self, 'mu'):
            self.compute_mu_coefficients()
        
        # Return the mu coefficient from the precomputed table
        return self.mu.get(w_key, {}).get(y_key, sp.Integer(0))



    def right_action_H_underline_simple(self, element, i):
        """
        Right action by H_underline_{s_i} = v^{-1} T_{s_i} + v^{-1} on an element in H-basis.
        H_underline_{s_i} is the canonical basis element C'_{s_i}.
        
        Args:
            element: dict mapping (w, beta) keys to coefficients (in H-basis)
            i: index of simple reflection s_i
            
        Returns:
            dict mapping (w, beta) keys to coefficients (in H-basis)
        """
        # Create H_underline_{s_i} = v^{-1} T_{s_i} + v^{-1} in T-basis
        s = simple_reflection(i, self.n)
        h_underline_si = {tuple(s): -v**(-1), tuple(range(1, self.n + 1)): -v**(-1)}
        
        # Apply right action by H_underline_{s_i} using right_action_hecke
        result = self.right_action_hecke(element, h_underline_si)
        
        return result

    def verify_proposition_9(self, w_key, i):
        """
        Verify Proposition 9 for a given basis element and simple reflection s_i.
        
        Proposition 9 states:
        H_underline_w * H_underline_si = 
            -(v^-1 + v) H_underline_w if w in Phi_i
            H_underline_{w * si} + sum_{w' < w, w' in Phi_i} mu(w', w) H_underline_w' if w not in Phi_i
        """
        # 1. Left side: H_underline_w * H_underline_si

        h_w = self.canonical_basis_element(w_key) # in H-basis
        h_w_H = self.H_to_T(h_w)
        #print(f"h_w_H: {h_w_H}")
        lhs = self.right_action_H_underline_simple(h_w_H, i)
        lhs = self.T_to_H(lhs)
        
        # 2. Right side
        rhs = {}
        if self.is_in_Phi_i(w_key, i):
            # -(v^-1 + v) H_underline_w
            factor = -(v**-1 + v)
            for k, c in h_w.items():
                rhs[k] = sp.expand(factor * c)
        else:
            # H_underline_{w * si} + sum_{w' < w, w' in Phi_i} mu(w', w) H_underline_w'
            w_star_si = self.wtilde_star_si(w_key, i)
            h_w_star = self.canonical_basis_element(w_star_si)
            for k, c in h_w_star.items():
                rhs[k] = rhs.get(k, sp.Integer(0)) + c
            
            # Mu terms
            if not hasattr(self, 'mu'):
                self.compute_mu_coefficients()
            
            # Sum over w' < w
            for w_prime_key, mu_val in self.mu.get(w_key, {}).items():
                if self.is_in_Phi_i(w_prime_key, i):
                    h_w_prime = self.canonical_basis_element(w_prime_key)
                    for k, c in h_w_prime.items():
                        rhs[k] = rhs.get(k, sp.Integer(0)) + mu_val * c
        
        rhs = {k: sp.expand(c) for k, c in rhs.items() if sp.expand(c) != 0}
        
        # 3. Compare
        equal = self.is_equal(lhs, rhs)
        if not equal:
            # Optional: print detailed difference if needed during debug
            pass
        return equal
    



    def print_mu_coefficients(self, max_pairs=None):
        """
        Print all non-zero mu coefficients.
        
        Args:
            max_pairs: Maximum number of pairs to print (None for all)
        """
        # Ensure mu coefficients are computed
        if not hasattr(self, 'mu'):
            self.compute_mu_coefficients()
        
        print(f"\nMu coefficients μ(ỹ, w̃) for the W-graph (non-zero only):")
        print("="*70)
        
        # Collect all non-zero mu coefficients from the table
        mu_pairs = []
        for x_key in self.mu:
            w_x, beta_x = denormalize_key(x_key)
            for y_key in self.mu[x_key]:
                mu_val = self.mu[x_key][y_key]
                w_y, beta_y = denormalize_key(y_key)
                mu_pairs.append((y_key, x_key, w_y, beta_y, w_x, beta_x, mu_val))
        
        if len(mu_pairs) == 0:
            print("  (all mu coefficients are 0)")
            return
        
        print(f"Total non-zero entries: {len(mu_pairs)}")
        print()
        
        pairs_to_show = mu_pairs if max_pairs is None else mu_pairs[:max_pairs]
        
        for _, _, w_y, beta_y, w_x, beta_x, mu_val in pairs_to_show:
            y_str = self._format_wtilde(w_y, beta_y)
            x_str = self._format_wtilde(w_x, beta_x)
            print(f"  μ({y_str}, {x_str}) = {mu_val}")
        
        if max_pairs is not None and len(mu_pairs) > max_pairs:
            print(f"  ... ({len(mu_pairs) - max_pairs} more pairs)")

    def canonical_basis_element(self, wtilde):
        """
        Compute the canonical (KL) basis element H̃_{w̃}.
        
        C_{w̃} = ∑_{ỹ <=  w̃} P_{ỹ,w̃} H_{ỹ}
        
        The element C_{w̃} is bar-invariant.
        """

        # Ensure KL polynomials are computed
        if not hasattr(self, '_kl_table'):
            self.compute_kl_polynomials()
        
        # C_{w̃} = ∑_{ỹ <= w̃} P_{ỹ,w̃} H_{ỹ}
        result = {}
        
        # Sum over all elements in KL[x]
        x_key = normalize_key(*wtilde)
        for y_key, p in self._kl_table[x_key].items():
            if p != 0 and p != sp.Integer(0):
                result[y_key] = p
        
        return result 


    def format_element(self, element, use_H_basis=True):
        """
        Format an element for pretty printing with colored partitions.

        Args:
            element: dict mapping keys to coefficients
            use_H_basis: if True, show as H_{w̃}, else as T_{w̃} (prefix is omitted)
        """
        if not element:
            return "0"

        terms = []
        for key in sorted(element.keys(), key=lambda k: (self.ell_wtilde(k), k)):
            w, beta = denormalize_key(key)
            coeff = sp.expand(element[key])
            if coeff == 0:
                continue

            # Normalize and format coefficient
            coeff = collect(coeff.expand(), v)

            if coeff == 1:
                coeff_str = ""
            elif coeff == -1:
                coeff_str = "-"
            else:
                # Convert to string and improve formatting
                coeff_str = str(coeff).replace("**", "^")
                # Add parentheses only if needed (multiple terms)
                if "+" in coeff_str or " - " in coeff_str:
                    coeff_str = f"({coeff_str})"

            # Format basis element with colored partition
            wtilde_str = self._format_wtilde(w, beta)
            if use_H_basis:
                basis_str = f"H[{wtilde_str}]"
            else:
                basis_str = f"[{wtilde_str}]"

            if coeff_str == "":
                terms.append(basis_str)
            elif coeff_str == "-":
                terms.append(f"-{basis_str}")
            else:
                terms.append(f"{coeff_str}{basis_str}")

        if not terms:
            return "0"

        result = terms[0]
        for t in terms[1:]:
            if t.startswith("-"):
                result += f" {t}"
            else:
                result += f" + {t}"
        return result



    def print_basis_info(self):
        """Print information about the basis elements."""
        print(f"HeckeRB bimodule for n={self.n}")
        print(f"Number of basis elements: {len(self._basis)}")
        
        # Count by length
        elements_by_length = self.elements_by_length
        
        print("Elements by length:")
        for ell in sorted(elements_by_length.keys()):
            print(f"  ℓ={ell}: {len(elements_by_length[ell])} elements")

    def print_hecke_action(self, i, max_elements=10):
        """
        Print the right action of T_{s_i} on basis elements.

        Args:
            i: index of simple reflection s_i
            max_elements: maximum number of elements to display
        """
        print(f"\nRight action of s_{i} on basis:")
        print("-" * 60)

        count = 0
        for key in self._basis:
            if count >= max_elements:
                print(f"  ... ({len(self._basis) - count} more elements)")
                break

            w, beta = denormalize_key(key)
            element = {key: sp.Integer(1)}
            result = self.right_action_simple(element, i)

            # Format the result
            lhs = f"[{self._format_wtilde(w, beta)}]"
            rhs = self._format_T_element(result)
            print(f"  {lhs} · s_{i} = {rhs}")
            count += 1

    def _format_wtilde(self, w, beta):
        """Format (w, β) for display with colored partition using str_colored_partition from RB.py."""
        # Use the str_colored_partition function from RB.py
        colored_w = str_colored_partition(w, beta)
        # Remove parentheses and commas
        colored_w = colored_w.replace("(", "").replace(")", "").replace(", ", "")
        
        # Add a representation of the beta set if needed, or just length
        ell = self.ell_wtilde(normalize_key(w, beta))
        return f"{colored_w}_{{{ell}}}"

    def _format_T_element(self, element):
        """Format an element in the T-basis, omitting the T_ prefix."""
        if not element:
            return "0"

        terms = []
        for key in sorted(element.keys()):
            w, beta = denormalize_key(key)
            coeff = sp.expand(element[key])
            if coeff == 0:
                continue

            wtilde_str = self._format_wtilde(w, beta)
            if coeff == 1:
                terms.append(f"[{wtilde_str}]")
            elif coeff == -1:
                terms.append(f"-[{wtilde_str}]")
            else:
                coeff_str = str(coeff).replace("**", "^")
                terms.append(f"({coeff_str})[{wtilde_str}]")

        if not terms:
            return "0"

        result = terms[0]
        for t in terms[1:]:
            if t.startswith("-"):
                result += f" {t}"
            else:
                result += f" + {t}"
        return result

    def verify_bar_involution(self, max_elements=10):
        """
        Verify the bar involution: bar(bar(x)) = x for basis elements.

        Returns:
            bool: True if all verified elements satisfy bar(bar(x)) = x
        """
        print(f"\nVerifying bar involution (bar ∘ bar = id):")
        print("-" * 60)

        all_ok = True
        count = 0
        unstable_count = 0
        total_checked = 0

        for key in self._basis:
            total_checked += 1

            # Only print the first max_elements elements
            show_detail = count < max_elements

            w, beta = denormalize_key(key)

            # Compute bar(T_{w̃})
            T_wtilde = {key: sp.Integer(1)}
            bar_T = self.bar_T(T_wtilde)

            # Compute bar(bar(T_{w̃}))
            bar_bar_T = self.bar_T(bar_T)

            # Check if bar(bar(T_{w̃})) = T_{w̃}
            is_ok = self.is_equal(bar_bar_T, T_wtilde)

            if not is_ok:
                unstable_count += 1
                all_ok = False

            if show_detail:
                status = "✓" if is_ok else "✗"
                wtilde_str = self._format_wtilde(w, beta)
                print(f"  bar(bar([{wtilde_str}])) = [{wtilde_str}]: {status}")

                if  not is_ok:
                    print(f"    bar([{wtilde_str}]) = {self._format_T_element(bar_T)}")
                    print(f"    bar(bar([{wtilde_str}])) = {self._format_T_element(bar_bar_T)}")
                count += 1

        if count < len(self._basis):
            print(f"  ... ({len(self._basis) - count} more elements)")

        # Print summary statistics
        print(f"\nSummary: {unstable_count} out of {total_checked} elements are not bar-bar stable ({unstable_count/total_checked:.2%})")

        return all_ok

    def compute_bar_on_basis(self):
        """
        Compute bar(T_{w̃}) for all basis elements.
        
        Returns:
            dict: mapping key -> bar(T_{key}) as element dict
        """
        bar_table = {}
        for key in self._basis:
            T_wtilde = {key: sp.Integer(1)}
            bar_T = self.bar_T(T_wtilde)
            bar_table[key] = bar_T
        
        return bar_table

    def print_bar_involution_table(self, max_elements=10):
        """Print the bar involution on basis elements (omitting T_ prefix)."""
        print(f"\nBar involution on basis:")
        print("-" * 60)

        count = 0
        for key in self._basis:
            if count >= max_elements:
                print(f"  ... ({len(self._basis) - count} more elements)")
                break

            w, beta = denormalize_key(key)
            T_wtilde = {key: sp.Integer(1)}
            bar_T = self.bar_T(T_wtilde)

            wtilde_str = self._format_wtilde(w, beta)
            bar_str = self._format_T_element(bar_T)
            print(f"  bar([{wtilde_str}]) = {bar_str}")
            count += 1

    def print_canonical_basis(self, max_elements=None):
        """Print the canonical basis elements H̃_{w̃} in the H basis."""
        if not hasattr(self, '_C_tilde') or not self._C_tilde:
            self.compute_canonical_basis_iterative()

        basis = self._C_tilde

        print(f"\nCanonical basis elements (H̃_w̃ in H basis):")
        print("-" * 60)

        # Sort by length for display
        sorted_keys = sorted(basis.keys(),
                            key=lambda k: (self.ell_wtilde(k), k))

        if max_elements is None:
            max_elements = len(sorted_keys)

        for key in sorted_keys[:max_elements]:
            ell = self.ell_wtilde(key)
            element_str = self.format_element(basis[key])
            wtilde_str = self._format_wtilde(w, beta)
            print(f"  H̃[{wtilde_str}] (ℓ={ell}) = {element_str}")

        if len(sorted_keys) > max_elements:
            print(f"  ... ({len(sorted_keys) - max_elements} more elements)")

    def print_kl_polynomials(self, max_pairs=20):
        """Print non-trivial KL polynomials P_{y,w}."""
        print(f"\nKazhdan-Lusztig polynomials P_{{ỹ,w̃}} (non-trivial only):")
        print("-" * 60)
        
        if not hasattr(self, '_kl_table'):
            self.compute_kl_polynomials()

        count = 0
        # Sort basis by length
        sorted_keys = sorted(self._basis, 
                            key=lambda k: (self.ell_wtilde(k), k))
        
        for w_key in sorted_keys:
            if w_key not in self._kl_table:
                continue
            
            for y_key, p in self._kl_table[w_key].items():
                # Only show non-trivial polynomials (not 0 or 1 for diagonal)
                if y_key == w_key:
                    continue
                
                if p != 0 and p != sp.Integer(0):
                    if count >= max_pairs:
                        print(f"  ... (more pairs)")
                        return
                    
                    y_w, y_beta = denormalize_key(y_key)
                    w_w, w_beta = denormalize_key(w_key)
                    
                    y_str = self._format_wtilde(y_w, y_beta)
                    w_str = self._format_wtilde(w_w, w_beta)
                    p_str = str(p).replace("**", "^")
                    print(f"  P[{y_str}, {w_str}] = {p_str}")
                    count += 1
        
        if count == 0:
            print("  (all KL polynomials are 0 or 1)")

    def print_mu_coefficients(self, max_pairs=20):
        """Print non-zero mu coefficients μ(ỹ, w̃)."""
        print(f"\nMu coefficients μ(ỹ, w̃) for the W-graph (non-zero only):")
        print("-" * 60)
        
        count = 0
        sorted_keys = sorted(self._basis, 
                            key=lambda k: (self.ell_wtilde(k), k))
        
        for w_key in sorted_keys:
            w, beta_w = denormalize_key(w_key)
            
            lower = self.bruhat_lower_elements(w_key)
            for y_key in lower:
                y, beta_y = denormalize_key(y_key)
                mu = self.mu_coefficient(y_key, w_key)
                
                if mu != 0 and mu != sp.Integer(0):
                    if count >= max_pairs:
                        print(f"  ... (more pairs)")
                        return
                    
                    y_str = self._format_wtilde(y, beta_y)
                    w_str = self._format_wtilde(w, beta_w)
                    print(f"  μ([{y_str}], [{w_str}]) = {mu}")
                    count += 1
        
        if count == 0:
            print("  (all mu coefficients are 0)")

    def print_w_graph_edges(self, max_edges=30):
        """Print the W-graph edges (where μ ≠ 0)."""
        print(f"\nW-graph edges (ỹ → w̃ where μ(ỹ, w̃) ≠ 0):")
        print("-" * 60)

        edges = []
        for w_key in self._basis:
            w, beta_w = denormalize_key(w_key)

            lower = self.bruhat_lower_elements(w_key)
            for y_key in lower:
                y, beta_y = denormalize_key(y_key)
                mu = self.mu_coefficient(y_key, w_key)

                if mu != 0 and mu != sp.Integer(0):
                    edges.append((y_key, w_key, mu))

        # Sort by length difference
        edges.sort(key=lambda e: (
            self.ell_wtilde(*denormalize_key(e[1])) - self.ell_wtilde(*denormalize_key(e[0])),
            e[0], e[1]
        ))

        for i, (y_key, w_key, mu) in enumerate(edges[:max_edges]):
            y, beta_y = denormalize_key(y_key)
            w, beta_w = denormalize_key(w_key)
            y_str = self._format_wtilde(y, beta_y)
            w_str = self._format_wtilde(w, beta_w)
            ell_diff = self.ell_wtilde(w, beta_w) - self.ell_wtilde(y, beta_y)
            print(f"  [{y_str}] --({mu})--> [{w_str}]  (Δℓ={ell_diff})")

        if len(edges) > max_edges:
            print(f"  ... ({len(edges) - max_edges} more edges)")

        print(f"\nTotal W-graph edges: {len(edges)}")

    def print_special_elements(self):
        """Print information about special elements w_i = (id, {1,...,i})."""
        print(f"\nSpecial elements w_i = (id, {{1,...,i}}):")
        print("-" * 60)

        identity = tuple(range(1, self.n + 1))

        for i in range(self.n + 1):
            key = self.special_element_key(i)
            w, beta = denormalize_key(key)
            ell = self.ell_wtilde(w, beta)

            wtilde_str = self._format_wtilde(w, beta)
            print(f"  w_{i} = [{wtilde_str}], ℓ = {ell}")

            # Show H̃_{w_i} in H basis
            if hasattr(self, '_C_tilde') and key in self._C_tilde:
                h_tilde = self._C_tilde[key]
                h_str = self.format_element(h_tilde)
                print(f"       H̃_{i} = {h_str}")

    def print_all_results(self, max_elements=15):
        """Print all computed results in a comprehensive format."""
        print(f"\n{'='*70}")
        print(f"  COMPLETE RESULTS FOR HeckeRB BIMODULE (n={self.n})")
        print(f"{'='*70}")
        
        # Basic info
        self.print_basis_info()
        
        # Special elements
        self.print_special_elements()
        
        # Hecke action
        for i in range(1, self.n):
            self.print_hecke_action(i, max_elements=max_elements)
        
        # Canonical basis
        self.print_canonical_basis(max_elements=max_elements)
        
        # KL polynomials
        self.print_kl_polynomials(max_pairs=max_elements)
        
        # Mu coefficients
        self.print_mu_coefficients(max_pairs=max_elements)
        
        # W-graph edges
        self.print_w_graph_edges(max_edges=max_elements)
        
        # Bar involution
        self.print_bar_involution_table(max_elements=max_elements)
        
        # Verify bar involution
        self.verify_bar_involution(max_elements=max_elements)
        
        print(f"\n{'='*70}")
        print(f"  END OF RESULTS")
        print(f"{'='*70}")


if __name__ == "__main__":
    import sys
    import time
    
    if len(sys.argv) < 2:
        print("Usage: python HeckeRB.py <n> [options]")
        print("  n: size of the symmetric group")
        print("  options:")
        print("    --all       : print all results (comprehensive)")
        print("    --action    : show Hecke algebra action on basis")
        print("    --bar       : show bar involution table")
        print("    --verify    : verify bar involution (bar ∘ bar = id)")
        print("    --kl        : show KL polynomials")
        print("    --mu        : show mu coefficients")
        print("    --wgraph    : show W-graph edges")
        print("    --special   : show special elements info")
        sys.exit(1)
    
    n = int(sys.argv[1])
    show_all = "--all" in sys.argv
    show_action = "--action" in sys.argv
    show_bar = "--bar" in sys.argv
    verify_bar = "--verify" in sys.argv
    show_kl = "--kl" in sys.argv
    show_mu = "--mu" in sys.argv
    show_wgraph = "--wgraph" in sys.argv
    show_special = "--special" in sys.argv
    
    print(f"\n=== Computing HeckeRB bimodule for n={n} ===\n")
    
    start = time.perf_counter()
    
    # Create the HeckeRB bimodule
    R = HeckeRB(n)
    
    # If --all, print everything
    if show_all:
        R.compute_canonical_basis_iterative()
        elapsed = time.perf_counter() - start
        R.print_all_results(max_elements=20)
        print(f"\nTotal computation time: {elapsed:.3f}s")
        sys.exit(0)
    
    R.print_basis_info()
    
    # Show special elements if requested
    if show_special:
        R.compute_canonical_basis_iterative()
        R.print_special_elements()
    
    # Show Hecke action if requested
    if show_action:
        for i in range(1, n):
            R.print_hecke_action(i, max_elements=15)
    
    # Compute the canonical basis
    print("\nComputing canonical basis...")
    basis = R.compute_canonical_basis_iterative()
    
    elapsed = time.perf_counter() - start
    
    # Print the canonical basis elements
    R.print_canonical_basis(max_elements=20)
    
    print("-" * 60)
    print(f"Total computation time: {elapsed:.3f}s")
    
    # Show KL polynomials if requested
    if show_kl:
        R.print_kl_polynomials(max_pairs=20)
    
    # Show mu coefficients if requested
    if show_mu:
        R.print_mu_coefficients(max_pairs=20)
    
    # Show W-graph if requested
    if show_wgraph:
        R.print_w_graph_edges(max_edges=30)
    
    # Show bar involution table if requested
    if show_bar:
        R.print_bar_involution_table(max_elements=15)
    
    # Verify bar involution if requested
    if verify_bar:
        R.verify_bar_involution(max_elements=15)
    
    # Verify canonical basis elements for special elements are bar-invariant
    print("\nVerifying bar-invariance of canonical basis elements for special elements...")
    identity = tuple(range(1, n + 1))
    all_ok = True

    # Create canonical basis elements for special elements if not already computed
    if not hasattr(R, '_C_tilde') or not R._C_tilde:
        R.compute_canonical_basis_iterative()

    for i in range(n + 1):
        key = R.special_element_key(i)
        if key in R._C_tilde:
            # Get the canonical basis element H̃_w_i
            canonical_elem = R._C_tilde[key]

            # Apply bar involution (use bar_H for H-basis elements)
            bar_canonical = R.bar_H(canonical_elem)

            # Check if bar(H̃_w_i) = H̃_w_i
            is_bar_invariant = R.is_equal(canonical_elem, bar_canonical)

            # Print the result
            status = "✓" if is_bar_invariant else "✗"
            print(f"  H̃_{i} is bar-invariant: {status}")

            # For educational purposes, also show the explicit form
            h_tilde_str = R.format_element(canonical_elem)
            print(f"    H̃_{i} = {h_tilde_str}")

            if not is_bar_invariant:
                all_ok = False
                bar_h_tilde_str = R.format_element(bar_canonical)
                print(f"    bar(H̃_{i}) = {bar_h_tilde_str}")

    if all_ok:
        print("\nAll special canonical basis elements are bar-invariant ✓")
    else:
        print("\nSome special canonical basis elements are NOT bar-invariant ✗")
    
    if all_ok:
        print("\nAll special elements are bar-invariant ✓")
