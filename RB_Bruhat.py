from graphviz import Digraph
from itertools import chain

from RB import (
    generate_all_w_sigma_pairs,
    root_type1,
    root_type2,
)
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
    return (
        "<<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"0\">"
        f"<TR><TD>{w_label}</TD></TR>"
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
