#!/usr/bin/env python3

import sys
import sympy as sp

from HeckeBessel import HeckeBessel
from HeckeA import v
from RB import denormalize_key
from verify_bar_bessel import compute_bar_involution_bessel_id


def verify_canonical_bases_bessel(n, verbose=False):
    print(f"Initializing HeckeBessel for n={n}...")
    R = HeckeBessel(n, strict=True)

    print("Computing bar involution (Bessel init: bar(T_sigma_i)=T_sigma_i)...")
    R.bar_table = compute_bar_involution_bessel_id(R, verbose=False)

    print("Computing KL polynomials...")
    R.compute_kl_polynomials(verbose=True)

    if verbose:
        print("\nCanonical basis elements C[w] in H-basis:")
        print("=" * 70)

        elements_by_length = R.elements_by_length
        for ell in sorted(elements_by_length.keys()):
            print(f"\nLength {ell}:")
            for key in elements_by_length[ell]:
                w, beta = denormalize_key(key)
                wtilde_str = R._format_wtilde(w, beta)
                c_w = R.canonical_basis_element((w, beta))
                c_str = R.format_element(c_w, use_H_basis=True)
                print(f"  C_b[{wtilde_str}] = {c_str}")

    print("\nVerifying bar-invariance of canonical basis elements...")
    all_ok = True
    count = 0
    total = len(list(R.basis()))

    for key in R.basis():
        count += 1
        w, beta = denormalize_key(key)
        wtilde_str = R._format_wtilde(w, beta)

        c_w = R.canonical_basis_element((w, beta))
        bar_c_w = R.bar_H(c_w)

        if R.is_equal(c_w, bar_c_w):
            print(f"[{count}/{total}] C_b[{wtilde_str}] is bar-invariant \u2713")
        else:
            all_ok = False
            print(f"[{count}/{total}] C_b[{wtilde_str}] is NOT bar-invariant \u2717")
            print(f"  C_w: {R.format_element(c_w, use_H_basis=True)}")
            print(f"  bar(C_w): {R.format_element(bar_c_w, use_H_basis=True)}")

            diff = {}
            all_keys = set(c_w.keys()) | set(bar_c_w.keys())
            for k in all_keys:
                d = sp.expand(c_w.get(k, 0) - bar_c_w.get(k, 0))
                if d != 0:
                    diff[k] = d
            if diff:
                print(f"  Difference: {R._format_T_element(diff)}")

    if all_ok:
        print("\nSUCCESS: All canonical basis elements are bar-invariant.")
    else:
        print("\nFAILURE: Some canonical basis elements are not bar-invariant.")


if __name__ == "__main__":
    n = 2
    verbose = False

    for arg in sys.argv[1:]:
        if arg == "-v" or arg == "--verbose":
            verbose = True
        else:
            try:
                n = int(arg)
            except ValueError:
                print(f"Invalid argument: {arg}. Using default n=2.")

    verify_canonical_bases_bessel(n, verbose=verbose)
