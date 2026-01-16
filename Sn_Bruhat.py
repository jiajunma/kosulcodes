from graphviz import Digraph

from perm import generate_permutations, length_of_permutation


def bruhat_covers(n):
    """
    Generate cover relations u -> v in the Bruhat order on S_n.

    A cover is given by v = u * t where t is a reflection (a,b) and
    length(v) = length(u) + 1. In complete notation, this means:
    pick positions a < b with u(a) < u(b), and swap the entries at those
    positions to form v.

    Returns:
        A list of (u, v, (a, b)) tuples, each u and v are permutation tuples.
    """
    covers = []
    for u in generate_permutations(n):
        lu = length_of_permutation(u)
        for a in range(1, n):
            for b in range(a + 1, n + 1):
                if u[a - 1] < u[b - 1]:
                    v_list = list(u)
                    v_list[a - 1], v_list[b - 1] = v_list[b - 1], v_list[a - 1]
                    v = tuple(v_list)
                    if length_of_permutation(v) == lu + 1:
                        covers.append((u, v, (a, b)))
    return covers


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

    for u, v, t in bruhat_covers(n):
        a, b = t
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
