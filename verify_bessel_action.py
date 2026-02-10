import sys
import sympy as sp

from HeckeBessel import HeckeBessel
from HeckeA import simple_reflection
from RB import denormalize_key


def verify_bessel_closure(n, strict=True):
    print(f"Initializing HeckeBessel for n={n} (strict={strict})...")
    R = HeckeBessel(n, strict=strict)
    basis_set = set(R.basis())

    print("Checking right action closure (s_1..s_{n-1})...")
    right_bad = []
    for key in basis_set:
        element = {key: sp.Integer(1)}
        for i in range(1, n):
            s = simple_reflection(i, n)
            out = R.right_action_T_w(element, s)
            for out_key in out.keys():
                if out_key not in basis_set:
                    right_bad.append((key, i, out_key))

    print("Checking left action closure (s_1..s_{n-2})...")
    left_bad = []
    for key in basis_set:
        element = {key: sp.Integer(1)}
        for i in range(1, n - 1):
            out = R.left_action_simple(element, i)
            for out_key in out.keys():
                if out_key not in basis_set:
                    left_bad.append((key, i, out_key))

    if not right_bad and not left_bad:
        print("SUCCESS: Bessel subset is closed under the defined Hecke actions.")
        return True

    if right_bad:
        print(f"Right action violations: {len(right_bad)}")
        for key, i, out_key in right_bad[:20]:
            print(f"  right: {denormalize_key(key)} * s_{i} -> {denormalize_key(out_key)}")
        if len(right_bad) > 20:
            print(f"  ... ({len(right_bad) - 20} more)")

    if left_bad:
        print(f"Left action violations: {len(left_bad)}")
        for key, i, out_key in left_bad[:20]:
            print(f"  left: s_{i} * {denormalize_key(key)} -> {denormalize_key(out_key)}")
        if len(left_bad) > 20:
            print(f"  ... ({len(left_bad) - 20} more)")

    return False


if __name__ == "__main__":
    n = 2
    strict = True
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print(f"Invalid n: {sys.argv[1]}. Using default n=2.")
    if "--non-strict" in sys.argv:
        strict = False
    verify_bessel_closure(n, strict=strict)
