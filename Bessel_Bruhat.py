"""
Bruhat order for the Bessel case.

This mirrors RB_Bruhat.py but restricts to the Bessel subset and the Bessel
special chain sigma_i (1 <= i <= n).
"""

from graphviz import Digraph

from HeckeBessel import HeckeBessel
from compute_bessel_dims import compute_bessel_dims


# Cache for Bruhat order data
_bruhat_cache = {}


def _precompute_root_types(R, orbit_set):
    """
    Precompute types and companions for left/right actions.

    Keys are normalized (w, sigma) with sigma as frozenset.
    s1 corresponds to left action, s2 to right action.
    """
    types = {}
    companions = {}
    n = R.n

    for key in orbit_set:
        left_entry = R._left_basis[key]
        right_entry = R._basis[key]

        for i in range(1, n):  # s_1..s_{n-1}
            s1 = ("s1", i)
            s2 = ("s2", i)

            if i <= n - 2 and left_entry[i] is not None:
                t_left, c_left = left_entry[i]
                types[(key, s1)] = t_left
                companions[(key, s1)] = [c for c in c_left if c in orbit_set]

            if right_entry[i] is not None:
                t_right, c_right = right_entry[i]
                types[(key, s2)] = t_right
                companions[(key, s2)] = [c for c in c_right if c in orbit_set]

    return types, companions


def bruhat_downsets_bessel(n):
    """
    Compute Bruhat downsets for the Bessel subset using Lemma 2.5-style recursion.
    """
    R = HeckeBessel(n, strict=True)
    orbit_set = set(R.basis())

    downset = {key: {key} for key in orbit_set}
    types, companions = _precompute_root_types(R, orbit_set)

    # Bessel base elements (sigma_i) are minimal and pairwise incomparable
    # under the normalized dimension convention, so keep each downset as itself.

    simple_reflections = []
    for i in range(1, n):
        if i <= n - 2:
            simple_reflections.append(("s1", i))  # left action
        simple_reflections.append(("s2", i))  # right action

    changed = True
    while changed:
        changed = False
        for alpha in list(downset.keys()):
            for s in simple_reflections:
                if types.get((alpha, s)) not in ("U-", "T-"):
                    continue
                for beta in companions.get((alpha, s), []):
                    if types.get((beta, s)) not in ("U+", "T+"):
                        continue
                    candidate = set()
                    for alpha_prime in downset[alpha]:
                        for comp in [alpha_prime] + companions.get((alpha_prime, s), []):
                            if comp not in downset:
                                raise KeyError(f"Missing companion {comp} in downset keys")
                            candidate.add(comp)
                    candidate.add(beta)
                    if not candidate.issubset(downset[beta]):
                        downset[beta].update(candidate)
                        changed = True

    return downset


def get_bruhat_order(n):
    """
    Get (or compute and cache) the Bruhat order data for the Bessel subset.

    Returns:
        tuple: (downsets dict, leq_cache dict)
    """
    if n not in _bruhat_cache:
        downsets = bruhat_downsets_bessel(n)
        leq_cache = {}
        for key, downset in downsets.items():
            for y in downset:
                leq_cache[(y, key)] = True
        _bruhat_cache[n] = (downsets, leq_cache)
    return _bruhat_cache[n]


def is_bruhat_leq(key1, key2, n=None):
    """
    Check if key1 <= key2 in the Bruhat order on the Bessel subset.
    """
    if n is None:
        n = len(key1[0])
    downsets, leq_cache = get_bruhat_order(n)
    return leq_cache.get((key1, key2), False)


def bruhat_lower_elements(key, n=None):
    """
    Return all elements strictly less than key in Bruhat order.
    """
    if n is None:
        n = len(key[0])
    downsets, _ = get_bruhat_order(n)
    downset = downsets.get(key, {key})
    return {y for y in downset if y != key}


def bruhat_covers(key, n=None):
    """
    Return elements covered by key (immediate predecessors in Bruhat order).
    """
    if n is None:
        n = len(key[0])
    downsets, _ = get_bruhat_order(n)
    downset = downsets.get(key, {key})
    candidates = {y for y in downset if y != key}
    for z in list(candidates):
        candidates.difference_update(downsets.get(z, {z}) - {z})
    return candidates


def _node_id(w, sigma):
    w_part = "".join(str(x) for x in w)
    s_part = "_".join(str(i) for i in sorted(sigma))
    return f"w{w_part}_s{s_part}"


def _node_label(w, sigma, dim_value):
    entries = []
    for idx, val in enumerate(w, start=1):
        color = "red" if idx in sigma else "blue"
        entries.append(f'<FONT COLOR="{color}">{val}</FONT>')
    w_label = "[" + ",".join(entries) + "]"
    dim_label = f"{dim_value}"
    return (
        "<<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\">"
        f"<TR><TD>{w_label}</TD></TR>"
        f"<TR><TD>{dim_label}</TD></TR>"
        "</TABLE>>"
    )


def bessel_hasse_diagram(n, output_path):
    """
    Build and render the Hasse diagram for the Bessel subset.

    Args:
        n: size of the symmetric group.
        output_path: path without file extension for Graphviz output.
    """
    dot = Digraph(comment=f"Bessel Bruhat diagram for n={n}")
    dot.attr(splines="true")
    dot.attr("node", shape="box", fontsize="10")
    dot.attr("edge", fontsize="9", arrowhead="normal")

    R, dims = compute_bessel_dims(n)
    nodes = {}
    for key in R.basis():
        w, sigma = key
        nodes[key] = (w, sigma)
        dim_value = dims.get(key, "?")
        dot.node(_node_id(w, sigma), label=_node_label(w, sigma, dim_value))

    downsets, _ = get_bruhat_order(n)
    edges = set()
    for x in downsets:
        candidates = set(downsets[x])
        candidates.discard(x)
        for z in list(candidates):
            candidates.difference_update(downsets.get(z, {z}) - {z})
        for y in candidates:
            edges.add((y, x))

    for (w1, s1), (w2, s2) in edges:
        dot.edge(_node_id(w2, s2), _node_id(w1, s1))

    dot.render(output_path, format="svg", cleanup=True)
    return dot


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        raise SystemExit("Usage: python Bessel_Bruhat.py <n> <output_path_without_ext>")

    n = int(sys.argv[1])
    out_path = sys.argv[2]
    bessel_hasse_diagram(n, out_path)
