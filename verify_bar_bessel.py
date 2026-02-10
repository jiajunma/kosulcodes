#!/usr/bin/env python3

import sympy as sp

from HeckeA import v
from HeckeBessel import HeckeBessel
from RB import denormalize_key
from perm import simple_reflection


def compute_bar_involution_bessel_id(R, verbose=False):
    """
    Compute bar involution on the Bessel subset with initialization:
        bar(T_{sigma_i}) = T_{sigma_i} for 1 <= i <= n.
    """
    bar_table = {}

    elements_by_length = R.elements_by_length
    max_length = R.max_length

    if verbose:
        print("Step 1: Elements by length")
        for ell in sorted(elements_by_length.keys()):
            print(f"  Length {ell}: {len(elements_by_length[ell])} elements")

    # Step 2: Initialize special elements by identity
    if verbose:
        print("\nStep 2: Bar involution for Bessel special elements T_{sigma_i}")

    for i in range(1, R.n + 1):
        key_i = R.special_element_key(i)
        bar_table[key_i] = {key_i: sp.Integer(1)}
        if verbose:
            w, beta = denormalize_key(key_i)
            print(f"  bar(T_[{w},{beta}]) = T_[{w},{beta}]")

    # Step 3: Iterate by length
    if verbose:
        print("\nStep 3: Iterating by length")

    for ell in range(max_length):
        for key in elements_by_length.get(ell, []):
            assert key in bar_table, f"bar_table missing key {key}"

            # Right action (s_1..s_{n-1})
            for s_idx in range(1, R.n):
                action = R.right_action_basis_simple(key, s_idx)

                higher_terms = []
                for k, c in action.items():
                    if R.ell_wtilde(k) > ell:
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

                for k_other, _ in action.items():
                    if k_other != key_higher and k_other not in bar_table:
                        raise ValueError(
                            f"bar_table missing required key {k_other} for computation of bar({key_higher})"
                        )

                bar_w = bar_table[key]

                # bar(T_w) * bar(T_s) = bar(T_w) * (v^{-2} T_s + (v^{-2} - 1))
                bar_w_times_Ts = R.right_action_simple(bar_w, s_idx)

                result = {}
                for k, c in bar_w_times_Ts.items():
                    result[k] = v ** (-2) * c
                for k, c in bar_w.items():
                    result[k] = result.get(k, sp.Integer(0)) + (v ** (-2) - 1) * c

                for k_other, a_other in action.items():
                    if k_other == key_higher:
                        continue
                    bar_other = bar_table[k_other]
                    a_bar = R.bar_coeff(a_other)
                    for k, c in bar_other.items():
                        result[k] = result.get(k, sp.Integer(0)) - a_bar * c

                result = {k: sp.expand(c) for k, c in result.items() if sp.expand(c) != 0}
                bar_table[key_higher] = result

                if verbose:
                    w_h, beta_h = denormalize_key(key_higher)
                    print(f"    bar(T_[{w_h},{beta_h}]) from right action by s_{s_idx}")

            # Left action (s_1..s_{n-2})
            for s_idx in range(1, R.n - 1):
                action = R.left_action_T_w(
                    {key: sp.Integer(1)},
                    simple_reflection(s_idx, R.n),
                )

                key_higher = []
                coeff_higher = []
                for k, c in action.items():
                    if R.ell_wtilde(k) > ell:
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

                Ts_left_bar_w = R.left_action_T_w(
                    bar_w, simple_reflection(s_idx, R.n)
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
                    a_bar = R.bar_coeff(a_other)
                    for k, c in bar_other.items():
                        result[k] = result.get(k, sp.Integer(0)) - a_bar * c

                result = {k: sp.expand(c) for k, c in result.items() if sp.expand(c) != 0}
                bar_table[key_higher] = result

                if verbose:
                    w_h, beta_h = denormalize_key(key_higher)
                    print(f"    bar(T_[{w_h},{beta_h}]) from left action by s_{s_idx}")

    # Step 4: Verify all elements are reached
    missing = [key for key in R._basis if key not in bar_table]
    if missing:
        missing_info = []
        for key in missing[:10]:
            w, beta = denormalize_key(key)
            missing_info.append(f"T_[{w},{beta}]")
        more = "" if len(missing) <= 10 else f" ... ({len(missing) - 10} more)"
        raise RuntimeError(
            f"bar_table is missing {len(missing)} out of {len(R._basis)} basis elements.\n"
            f"First missing: {', '.join(missing_info)}{more}"
        )

    if verbose:
        print("\nStep 4: Verification")
        print(f"  Total basis elements: {len(R._basis)}")
        print(f"  Elements with bar computed: {len(bar_table)}")
        print("  All elements reached!")

    return bar_table


def verify_bar_bessel(n, verbose=False, max_elements=20):
    R = HeckeBessel(n, strict=True)
    R.bar_table = compute_bar_involution_bessel_id(R, verbose=verbose)
    return R.verify_bar_involution(max_elements=max_elements)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Verify bar involution for Bessel with bar(T_sigma_i)=T_sigma_i")
    parser.add_argument("-n", type=int, default=3, help="Size parameter for HeckeBessel")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose computation output")
    parser.add_argument("--max-verify", type=int, default=20, help="Maximum number of elements to verify")
    args = parser.parse_args()

    print(f"Computing bar involution for HeckeBessel(n={args.n}) with identity init...")
    ok = verify_bar_bessel(args.n, verbose=args.verbose, max_elements=args.max_verify)
    print("\nOverall result:", "PASS" if ok else "FAIL")
