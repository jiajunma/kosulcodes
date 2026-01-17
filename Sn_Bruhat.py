from graphviz import Digraph

from perm import generate_permutations, is_bruhat_leq, length_of_permutation


def bruhat_downset(n):
    """
    Compute Bruhat downsets using the rank-matrix criterion.
    """
    perms = list(generate_permutations(n))
    downset = {w: set() for w in perms}
    for u in perms:
        for v in perms:
            if is_bruhat_leq(u, v):
                downset[v].add(u)
    return downset


def bruhat_hasse_edges(n):
    """
    Compute Hasse edges from Bruhat downsets by deleting transitive edges.
    """
    downset = bruhat_downset(n)
    edges = set()
    for x in downset:
        candidates = set(downset[x])
        candidates.discard(x)
        for z in list(candidates):
            candidates.difference_update(downset[z] - {z})
        for y in candidates:
            edges.add((x, y))
    return edges


def permutation_label(w):
    """Readable label for a permutation tuple."""
    return "[" + ",".join(str(x) for x in w) + "]"


def hasse_diagram_bruhat(n, output_path):
    """
    Build and render the Hasse diagram of the Bruhat order on S_n.

    Args:
        n: size of the symmetric group.
        output_path: path without file extension for Graphviz output.

    Returns:
        The Graphviz Digraph object.
    """
    dot = Digraph(comment=f"Bruhat order Hasse diagram for S_{n}")
    dot.attr(splines="false" )
    dot.attr("node", shape="box", fontsize="10")
    dot.attr("edge", fontsize="9", arrowhead="none")

    perms = list(generate_permutations(n))
    for w in perms:
        dot.node(permutation_label(w))

    # Group nodes by length to keep ranks aligned (more symmetric layout).
    by_length = {}
    for w in perms:
        by_length.setdefault(length_of_permutation(w), []).append(w)
    for length, group in sorted(by_length.items()):
        with dot.subgraph() as s:
            s.attr(rank="same")
            for w in group:
                s.node(permutation_label(w))

    for u, v in bruhat_hasse_edges(n):
        dot.edge(permutation_label(u), permutation_label(v))

    dot.render(output_path, format="svg", cleanup=True)
    return dot


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        raise SystemExit("Usage: python bruhat_hasse.py <n> <output_path_without_ext>")

    n = int(sys.argv[1])
    out_path = sys.argv[2]
    hasse_diagram_bruhat(n, out_path)
