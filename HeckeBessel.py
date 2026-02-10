import sympy as sp

from HeckeA import v, q
from HeckeRB import HeckeRB
from RB import denormalize_key, normalize_key, root_type_left
from perm import simple_reflection, reduced_word


def _image_of_subset(w, subset):
    return {w[i - 1] for i in subset}


def is_bessel_sigma(w, sigma, n):
    """
    Bessel subset condition (RB'_{n-1,n}): n in w(sigma).
    Here sigma is a subset of positions (1-indexed).
    """
    return n in _image_of_subset(w, sigma)


class HeckeBessel(HeckeRB):
    """
    Hecke module for the Bessel GGP case.

    This is a restriction of the FJ case to the subset:
        RB'_{n-1,n} = {(w, sigma) in RB' | n in w(sigma)}

    Left action is only defined for simple reflections s_1,...,s_{n-2}.
    Right action follows the full FJ case (s_1,...,s_{n-1}).
    """

    def __init__(self, n, strict=True):
        super().__init__(n)
        if not hasattr(self, "_left_basis"):
            self._left_basis = {}
            for key in self._basis:
                w, sigma = denormalize_key(key)
                self._left_basis[key] = root_type_left(w, sigma)
        if not hasattr(self, "_left_action_cache"):
            self._left_action_cache = {}
        self._bessel_strict = strict
        self._filter_to_bessel_subset()
        self._bessel_dims = self._compute_bessel_dims()
        self._rebuild_length_data()

    def _is_bessel_key(self, key):
        w, sigma = denormalize_key(key)
        return is_bessel_sigma(w, sigma, self.n)

    def _filter_companions(self, data, keep_keys, max_i=None):
        """
        Filter companion lists to keep only keys inside keep_keys.
        If strict, raise if any companions are dropped.

        Each entry is a list indexed by i, with entry[0] storing length.
        """
        filtered = {}
        for key, entry in data.items():
            if not isinstance(entry, list):
                raise TypeError(f"Unexpected root-type entry for {key}: {type(entry)}")
            new_entry = list(entry)
            upper = len(entry) - 1
            if max_i is not None:
                upper = min(upper, max_i)
            for i in range(1, upper + 1):
                if entry[i] is None:
                    continue
                T, C = entry[i]
                C_keep = [c for c in C if c in keep_keys]
                if self._bessel_strict and len(C_keep) != len(C):
                    dropped = [c for c in C if c not in keep_keys]
                    raise ValueError(
                        "Bessel restriction dropped companions for "
                        f"{key} at i={i}: {dropped}"
                    )
                new_entry[i] = (T, C_keep)
            filtered[key] = new_entry
        return filtered

    def _filter_to_bessel_subset(self):
        keep_keys = {k for k in self._basis if self._is_bessel_key(k)}
        self._basis = {k: v for k, v in self._basis.items() if k in keep_keys}
        self._left_basis = {
            k: v for k, v in self._left_basis.items() if k in keep_keys
        }

        # Filter companions to enforce restriction (Lemma 4.4).
        self._basis = self._filter_companions(self._basis, keep_keys)
        # Left action only uses s_1..s_{n-2}
        self._left_basis = self._filter_companions(
            self._left_basis, keep_keys, max_i=self.n - 2
        )

        # Reset caches and length data
        self._right_action_cache = {}
        self._left_action_cache = {}
        self._rebuild_length_data()

    def _rebuild_length_data(self):
        self.elements_by_length = {}
        self.max_length = 0
        for key in self._basis:
            ell = self.ell_wtilde(key)
            self.elements_by_length.setdefault(ell, []).append(key)
            self.max_length = max(self.max_length, ell)

    def ell_wtilde(self, wtilde):
        if hasattr(self, "_bessel_dims") and wtilde in self._bessel_dims:
            return self._bessel_dims[wtilde]
        return super().ell_wtilde(wtilde)

    def _compute_bessel_dims(self):
        """
        Compute Bessel orbit dimensions using the same constraints
        as compute_bessel_dims.py, but based on this instance.
        """
        def iter_simple_reflections():
            for i in range(1, self.n - 1):
                yield ("s1", i)
            for i in range(1, self.n):
                yield ("s2", i)

        def get_entry(key, side, i):
            if side == "s1":
                entry = self._left_basis[key]
                return entry[i] if i <= self.n - 2 else None
            entry = self._basis[key]
            return entry[i]

        def apply_constraint(dims, a, b, delta):
            if a in dims and b in dims:
                if dims[b] != dims[a] + delta:
                    raise ValueError(
                        f"Inconsistent dimensions: dim({b})={dims[b]} "
                        f"but dim({a})={dims[a]} implies {dims[a] + delta}"
                    )
                return False
            if a in dims and b not in dims:
                dims[b] = dims[a] + delta
                return True
            if b in dims and a not in dims:
                dims[a] = dims[b] - delta
                return True
            return False

        dims = {}
        for i in range(1, self.n + 1):
            dims[self.special_element_key(i)] = 0

        changed = True
        while changed:
            changed = False
            for key in self.basis():
                for side, i in iter_simple_reflections():
                    entry = get_entry(key, side, i)
                    if entry is None:
                        continue
                    typ_a, comps = entry
                    for comp in comps:
                        entry_b = get_entry(comp, side, i)
                        if entry_b is None:
                            continue
                        typ_b, _ = entry_b

                        if typ_a in ("U-", "T-") and typ_b in ("U+", "T+"):
                            if apply_constraint(dims, key, comp, 1):
                                changed = True
                            continue
                        if typ_a in ("U+", "T+") and typ_b in ("U-", "T-"):
                            if apply_constraint(dims, key, comp, -1):
                                changed = True
                            continue

                        if typ_a == "T-" and typ_b == "T-":
                            if apply_constraint(dims, key, comp, 0):
                                changed = True

        return dims

    def special_element_key(self, i):
        """
        Return the key for Bessel special elements sigma_i (1 <= i <= n):
            sigma_i = (w_i, {i})
            w_i = (1, ..., i-1, n, i, ..., n-1)  (one-line notation)
        """
        if i < 1 or i > self.n:
            raise ValueError(f"i must be in [1, {self.n}], got {i}")
        w = list(range(1, self.n + 1))
        if i != self.n:
            w.pop()
            w.insert(i - 1, self.n)
        return normalize_key(tuple(w), {i})

    # Restrict left action to s_1,...,s_{n-2}.
    def left_action_simple(self, element, i):
        if i >= self.n - 1:
            return dict(element)
        result = {}
        for key, coeff in element.items():
            w, beta = denormalize_key(key)
            action = self.left_action_basis_simple((w, beta), i)
            for k, c in action.items():
                result[k] = result.get(k, 0) + coeff * c
        return result

    def left_action_basis_simple(self, wtilde, i):
        if i >= self.n - 1:
            return {normalize_key(*wtilde): sp.Integer(1)}
        s = simple_reflection(i, self.n)
        key = (wtilde, s)
        if key in self._left_action_cache:
            return dict(self._left_action_cache[key])
        T, C = self._left_basis[wtilde][i]
        result = {wtilde: -sp.Integer(1)}
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
        self._left_action_cache[key] = result
        return result

    def left_action_T_w(self, element, w):
        """
        Left action by T_w using simple reflections (restricted to s_1..s_{n-2}).
        """
        result = dict(element)
        for s in reduced_word(w):
            i = self._simple_index(s)
            if i >= self.n - 1:
                raise ValueError(
                    f"Left action not defined for s_{i} in Bessel case (only s_1..s_{self.n-2})."
                )
            result = self.left_action_simple(result, i)
        return result

    # Bruhat helpers: restrict to Bessel subset.
    def bruhat_lower_elements(self, wtilde):
        lower = super().bruhat_lower_elements(wtilde)
        return [k for k in lower if self._is_bessel_key(k)]

    def bruhat_covers(self, wtilde):
        covers = super().bruhat_covers(wtilde)
        return [k for k in covers if self._is_bessel_key(k)]

    def format_element(self, element, use_H_basis=True, use_C_basis=False):
        """
        Format an element for Bessel case, using H_b / C_b labels.
        """
        s = super().format_element(element, use_H_basis=use_H_basis, use_C_basis=use_C_basis)
        if use_C_basis:
            return s.replace("C[", "C_b[")
        if use_H_basis:
            return s.replace("H[", "H_b[")
        return s

    # =========================================================================
    # Bar involution (Bessel case)
    # =========================================================================
    def compute_bar_involution(self, verbose=False):
        """
        Compute bar involution on every standard basis element T_{w̃}
        in the Bessel subset, following the FJ case algorithm.

        Special elements are the Bessel chain sigma_i (1 <= i <= n):
            sigma_i = (w_i, {i})
            w_i = (1, ..., i-1, n, i, ..., n-1)

        Initialization (Bessel normalization):
            bar(T_{sigma_i}) = T_{sigma_i}
        """
        bar_table = {}

        elements_by_length = self.elements_by_length
        max_length = self.max_length

        if verbose:
            print("Step 1: Elements by length")
            for ell in sorted(elements_by_length.keys()):
                print(f"  Length {ell}: {len(elements_by_length[ell])} elements")

        # Step 2: Bar involution for Bessel special elements (sigma_i, 1..n)
        if verbose:
            print("\nStep 2: Bar involution for Bessel special elements T_{w_i}")

        for i in range(1, self.n + 1):
            key_i = self.special_element_key(i)
            bar_table[key_i] = {key_i: sp.Integer(1)}

            if verbose:
                w, beta = denormalize_key(key_i)
                print(f"  bar(T_{{w_{i}}}) = {self._format_T_element(bar_table[key_i])}")

        # Step 3: Iterate by length
        if verbose:
            print("\nStep 3: Iterating by length")

        for ell in range(max_length):
            for key in elements_by_length.get(ell, []):
                assert key in bar_table, f"bar_table missing key {key}"

                # Try each simple reflection on right action (s_1..s_{n-1})
                for s_idx in range(1, self.n):
                    action = self.right_action_basis_simple(key, s_idx)

                    # Find higher term
                    higher_terms = []
                    for k, c in action.items():
                        if self.ell_wtilde(k) > ell:
                            higher_terms.append((k, c))

                    if len(higher_terms) == 0:
                        continue
                    if len(higher_terms) > 1:
                        raise AssertionError(
                            f"More than one higher term in right action for {key}: {higher_terms}"
                        )

                    key_higher, coeff_higher = higher_terms[0]
                    c_expanded = sp.expand(coeff_higher)
                    assert c_expanded == 1, f"Expected coefficient 1, got {c_expanded}"

                    if key_higher in bar_table:
                        continue

                    # Check correction terms available
                    for k_other, _ in action.items():
                        if k_other != key_higher and k_other not in bar_table:
                            raise ValueError(
                                f"bar_table missing required key {k_other} for computation of bar({key_higher})"
                            )

                    bar_w = bar_table[key]

                    # bar(T_w) * bar(T_s) = bar(T_w) * (v^{-2} T_s + (v^{-2} - 1))
                    bar_w_times_Ts = self.right_action_simple(bar_w, s_idx)

                    result = {}
                    for k, c in bar_w_times_Ts.items():
                        result[k] = v ** (-2) * c
                    for k, c in bar_w.items():
                        result[k] = result.get(k, sp.Integer(0)) + (v ** (-2) - 1) * c

                    # Subtract correction terms
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
                        print(f"    bar(T_[{w_h},{beta_h}]) from right action by s_{s_idx}")

                # Also try left action T_s · T_{w̃} (s_1..s_{n-2})
                for s_idx in range(1, self.n - 1):
                    action = self.left_action_T_w(
                        {key: sp.Integer(1)},
                        simple_reflection(s_idx, self.n),
                    )

                    key_higher = []
                    coeff_higher = []
                    for k, c in action.items():
                        if self.ell_wtilde(k) > ell:
                            key_higher.append(k)
                            coeff_higher.append(c)

                    if not key_higher:
                        continue

                    assert len(key_higher) == 1, f"Expected one higher term, got {len(key_higher)}"
                    key_higher, coeff_higher = key_higher[0], coeff_higher[0]
                    c_expanded = sp.expand(coeff_higher)
                    assert c_expanded == 1, f"Expected coefficient 1, got {c_expanded}"

                    if key_higher in bar_table:
                        continue

                    for k_other in action:
                        if k_other != key_higher and k_other not in bar_table:
                            raise ValueError(
                                f"bar_table missing required key {k_other} for computation of bar({key_higher})"
                            )

                    bar_w = bar_table[key]

                    Ts_left_bar_w = self.left_action_T_w(
                        bar_w, simple_reflection(s_idx, self.n)
                    )

                    result = {}
                    for k, c in Ts_left_bar_w.items():
                        result[k] = v ** (-2) * c
                    for k, c in bar_w.items():
                        result[k] = result.get(k, sp.Integer(0)) + (v ** (-2) - 1) * c

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
                        print(f"    bar(T_[{w_h},{beta_h}]) from left action by s_{s_idx}")

        # Step 4: Verify all elements are reached
        missing = [key for key in self._basis if key not in bar_table]
        if missing:
            missing_info = []
            for key in missing[:10]:
                w, beta = denormalize_key(key)
                missing_info.append(f"T_[{w},{beta}]")
            more = "" if len(missing) <= 10 else f" ... ({len(missing) - 10} more)"
            raise RuntimeError(
                f"bar_table is missing {len(missing)} out of {len(self._basis)} basis elements.\n"
                f"First missing: {', '.join(missing_info)}{more}"
            )

        if verbose:
            print("\nStep 4: Verification")
            print(f"  Total basis elements: {len(self._basis)}")
            print(f"  Elements with bar computed: {len(bar_table)}")
            print("  All elements reached!")

        return bar_table
