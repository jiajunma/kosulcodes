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


def rb_edges(n):
    """
    Generate edges for the RB Bruhat diagram.

    Nodes are (w, sigma) in RB. Edges connect a node to its U- and T- companions
    (from both root_type1 and root_type2).

    Returns:
        A list of (src, dst, label) where src/dst are (w, sigma) pairs.
    """
    edges = []
    seen = set()
    for w, sigma in generate_all_w_sigma_pairs(n):
        rt1 = root_type1(w, sigma)
        rt2 = root_type2(w, sigma)
        for result in chain(rt1.values(),  rt2.values()):
            if result["type"] in ("U-", "T-"):
                for (w2, sigma2) in result["companions"]:
                    edge_key = (node_key(w, sigma), node_key(w2, sigma2))
                    if edge_key in seen: 
                        continue
                    seen.add(edge_key)
                    reverse_key = (node_key(w2, sigma2), node_key(w, sigma))
                    seen.add(reverse_key)
                    edges.append(((w, sigma), (w2, sigma2), ""))
    return edges


def rb_hasse_diagram(n, output_path):
    """
    Build and render the Hasse diagram for RB using U- and T- companions.

    Args:
        n: size of the symmetric group.
        output_path: path without file extension for Graphviz output.
    """
    dot = Digraph(comment=f"RB Bruhat diagram for n={n}")
    dot.attr(splines="false")
    dot.attr("node", shape="box", fontsize="10")
    dot.attr("edge", fontsize="9", arrowhead="none")

    nodes = {}
    for w, sigma in generate_all_w_sigma_pairs(n):
        key = node_key(w, sigma)
        nodes[key] = (w, sigma)
        dot.node(node_id(w, sigma), label=node_label(w, sigma))

    for (w1, s1), (w2, s2), label in rb_edges(n):
        if node_key(w2, s2) not in nodes:
            nodes[node_key(w2, s2)] = (w2, s2)
            dot.node(node_id(w2, s2), label=node_label(w2, s2))
        dot.edge(node_id(w1, s1), node_id(w2, s2), label=label)

    dot.render(output_path, format="svg", cleanup=True)
    return dot


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        raise SystemExit("Usage: python RB_Bruhat.py <n> <output_path_without_ext>")

    n = int(sys.argv[1])
    out_path = sys.argv[2]
    rb_hasse_diagram(n, out_path)
