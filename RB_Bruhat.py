from graphviz import Digraph
from itertools import chain

from RB import (
    generate_all_w_sigma_pairs,
    dim_Omega_w_sigma,
)
from RB import root_type_left as root_type1
from RB import root_type_right as root_type2

from perm import length_of_permutation


def node_key(w, sigma):
    return (tuple(w), tuple(sorted(sigma)))


def node_id(w, sigma):
    w_part = "".join(str(x) for x in w)
    s_part = "_".join(str(i) for i in sorted(sigma))
    return f"w{w_part}_s{s_part}"


def node_label(w, sigma):
    entries = []
    for idx, val in enumerate(w, start=1):
        color = "red" if idx in sigma else "blue"
        entries.append(f'<FONT COLOR="{color}">{val}</FONT>')
    w_label = "[" + ",".join(entries) + "]"
    dim_label = f"{dim_Omega_w_sigma(w, sigma)}"
    return (
        "<<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\">"
        f"<TR><TD>{w_label}</TD></TR>"
        f"<TR><TD>{dim_label}</TD></TR>"
        "</TABLE>>"
    )

def _sigma_i_key(n, i):
    w = tuple(range(1, n + 1))
    if i == 0:
        return node_key(w, set())
    return node_key(w, {i})

def _precompute_root_types(orbit_set):
    types = {}
    companions = {}
    for key in orbit_set:
        w, sigma = key
        n = len(w)
        rt1 = root_type1(w, set(sigma))
        rt2 = root_type2(w, set(sigma))
        for i in range(1, n):
            s1 = ("s1", i)
            s2 = ("s2", i)
            types[(key, s1)] = rt1[i]["type"]
            types[(key, s2)] = rt2[i]["type"]
            companions[(key, s1)] = [
                node_key(ww, ss)
                for (ww, ss) in rt1[i]["companions"]
                if node_key(ww, ss) in orbit_set
            ]
            companions[(key, s2)] = [
                node_key(ww, ss)
                for (ww, ss) in rt2[i]["companions"]
                if node_key(ww, ss) in orbit_set
            ]
    return types, companions

def bruhat_downsets(n):
    """
    Compute Bruhat downsets using Lemma 2.4 with Lemma 3.14/3.15 base cases.
    """
    downset = {
        node_key(w, sigma): {node_key(w, sigma)}
        for (w, sigma) in generate_all_w_sigma_pairs(n)
    }
    types, companions = _precompute_root_types(downset.keys())

    # Lemma 3.14/3.15: I^diamond is the chain sigma_0 <= sigma_1 <= ... <= sigma_n.
    base_chain = [_sigma_i_key(n, i) for i in range(0, n + 1)]
    for i, key in enumerate(base_chain):
        downset[key] = set(base_chain[: i + 1])

    simple_reflections = []
    for i in range(1, n):
        simple_reflections.append(("s1", i))
        simple_reflections.append(("s2", i))

    changed = True
    while changed:
        changed = False
        for alpha in downset:
            for s in simple_reflections:
                if types[(alpha, s)] not in ("U-", "T-"):
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


# Cache for Bruhat order data
_bruhat_cache = {}


def get_bruhat_order(n):
    """
    Get (or compute and cache) the Bruhat order data for S_n.
    
    Returns:
        tuple: (downsets dict, leq_cache dict)
    """
    if n not in _bruhat_cache:
        downsets = bruhat_downsets(n)
        # Build leq cache for fast lookup
        leq_cache = {}
        for key, downset in downsets.items():
            for y in downset:
                leq_cache[(y, key)] = True
        _bruhat_cache[n] = (downsets, leq_cache)
    return _bruhat_cache[n]


def is_bruhat_leq(key1, key2, n=None):
    """
    Check if key1 <= key2 in the Bruhat order on RB.
    
    Args:
        key1, key2: normalized keys (w, sigma) as tuples
        n: size of symmetric group (inferred from keys if not provided)
    
    Returns:
        bool: True if key1 <= key2 in Bruhat order
    """
    if n is None:
        n = len(key1[0])
    downsets, leq_cache = get_bruhat_order(n)
    return leq_cache.get((key1, key2), False)


def bruhat_lower_elements(key, n=None):
    """
    Return all elements strictly less than key in Bruhat order.
    
    Args:
        key: normalized key (w, sigma) as tuple
        n: size of symmetric group (inferred from key if not provided)
    
    Returns:
        set: all keys y such that y < key
    """
    if n is None:
        n = len(key[0])
    downsets, _ = get_bruhat_order(n)
    downset = downsets.get(key, {key})
    return {y for y in downset if y != key}


def bruhat_covers(key, n=None):
    """
    Return elements covered by key (immediate predecessors in Bruhat order).
    
    Args:
        key: normalized key (w, sigma) as tuple
        n: size of symmetric group (inferred from key if not provided)
    
    Returns:
        set: all keys y such that y is covered by key (y < key and no z with y < z < key)
    """
    if n is None:
        n = len(key[0])
    downsets, _ = get_bruhat_order(n)
    downset = downsets.get(key, {key})
    candidates = {y for y in downset if y != key}
    # Remove non-covers: if y < z < key, then y is not a cover
    for z in list(candidates):
        candidates.difference_update(downsets.get(z, {z}) - {z})
    return candidates

def bruhat_hasse_edges(n):
    """
    Compute Hasse edges from Bruhat downsets (cover relations).
    """
    downset = bruhat_downsets(n)
    edges = set()
    for x in downset:
        candidates = set(downset[x])
        candidates.discard(x)
        # Delete transitive edges: if y <= z < x then y is not a cover of x.
        for z in list(candidates):
            candidates.difference_update(downset[z] - {z})
        for y in candidates:
            edges.add((y, x))
    return edges


def rb_hasse_diagram(n, output_path):
    """
    Build and render the Hasse diagram for RB using U- and T- companions.

    Args:
        n: size of the symmetric group.
        output_path: path without file extension for Graphviz output.
    """
    dot = Digraph(comment=f"RB Bruhat diagram for n={n}")
    dot.attr(splines="true")
    #dot.attr(splines="false")
    dot.attr("node", shape="box", fontsize="10")
    #dot.attr("edge", fontsize="9", arrowhead="none")
    dot.attr("edge", fontsize="9", arrowhead="normal")

    nodes = {}
    for w, sigma in generate_all_w_sigma_pairs(n):
        key = node_key(w, sigma)
        nodes[key] = (w, sigma)
        dot.node(node_id(w, sigma), label=node_label(w, sigma))

    for (w1, s1), (w2, s2) in bruhat_hasse_edges(n):
        if node_key(w1, s1) not in nodes:
            nodes[node_key(w1, s1)] = (w1, s1)
            dot.node(node_id(w1, s1), label=node_label(w1, s1))
        dot.edge(node_id(w2, s2), node_id(w1, s1))

    dot.render(output_path, format="svg", cleanup=True)
    return dot


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        raise SystemExit("Usage: python RB_Bruhat.py <n> <output_path_without_ext>")

    n = int(sys.argv[1])
    out_path = sys.argv[2]
    rb_hasse_diagram(n, out_path)
