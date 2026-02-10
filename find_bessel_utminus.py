import sys

from HeckeBessel import HeckeBessel
from RB import denormalize_key


def is_ut_minus(entry, i):
    if entry[i] is None:
        return False
    typ, _ = entry[i]
    return typ in {"U-", "T-"}


def find_utminus(n):
    R = HeckeBessel(n, strict=True)
    results = []
    for key in R.basis():
        left_ok = True
        right_ok = True

        left_entry = R._left_basis[key]
        for i in range(1, n - 1):  # s_1..s_{n-2}
            if not is_ut_minus(left_entry, i):
                left_ok = False
                break

        if left_ok:
            right_entry = R._basis[key]
            for i in range(1, n):  # s_1..s_{n-1}
                if not is_ut_minus(right_entry, i):
                    right_ok = False
                    break

        if left_ok and right_ok:
            results.append(key)

    return results


if __name__ == "__main__":
    n = 2
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print(f"Invalid n: {sys.argv[1]}. Using default n=2.")

    matches = find_utminus(n)
    print(f"n={n}, matches={len(matches)}")
    for key in matches:
        print(denormalize_key(key))
