import sys

from HeckeBessel import HeckeBessel


def _iter_simple_reflections(n):
    # left: s_1..s_{n-2}, right: s_1..s_{n-1}
    for i in range(1, n - 1):
        yield ("s1", i)
    for i in range(1, n):
        yield ("s2", i)


def _get_entry(R, key, side, i):
    if side == "s1":
        entry = R._left_basis[key]
        return entry[i] if i <= R.n - 2 else None
    entry = R._basis[key]
    return entry[i]


def _apply_constraint(dims, a, b, delta):
    """
    Enforce dim(b) = dim(a) + delta.
    Returns True if dims updated, False otherwise.
    """
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


def compute_bessel_dims(n):
    R = HeckeBessel(n, strict=True)
    dims = {}

    # Base chain sigma_i has dimension 0
    for i in range(1, n + 1):
        dims[R.special_element_key(i)] = 0

    changed = True
    while changed:
        changed = False
        for key in R.basis():
            for side, i in _iter_simple_reflections(n):
                entry = _get_entry(R, key, side, i)
                if entry is None:
                    continue
                typ_a, comps = entry
                for comp in comps:
                    entry_b = _get_entry(R, comp, side, i)
                    if entry_b is None:
                        continue
                    typ_b, _ = entry_b

                    # Rule 1: (+) vs (-) differ by 1
                    if typ_a in ("U-", "T-") and typ_b in ("U+", "T+"):
                        if _apply_constraint(dims, key, comp, 1):
                            changed = True
                        continue
                    if typ_a in ("U+", "T+") and typ_b in ("U-", "T-"):
                        if _apply_constraint(dims, key, comp, -1):
                            changed = True
                        continue

                    # Rule 2: both T- have equal dimension
                    if typ_a == "T-" and typ_b == "T-":
                        if _apply_constraint(dims, key, comp, 0):
                            changed = True

    return R, dims


if __name__ == "__main__":
    n = 2
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print(f"Invalid n: {sys.argv[1]}. Using default n=2.")

    R, dims = compute_bessel_dims(n)
    missing = [k for k in R.basis() if k not in dims]
    print(f"n={n}, computed dims={len(dims)}, missing={len(missing)}")
    for key in sorted(dims.keys(), key=lambda k: (R.ell_wtilde(k), k)):
        print(key, dims[key])
    if missing:
        print("Missing keys:")
        for k in missing:
            print(k)
