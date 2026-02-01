import sympy as sp
import sys
import os
import matplotlib.pyplot as plt
import networkx as nx

# Add parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perm import is_bruhat_leq, length_of_permutation

def permutation_label(w):
    """Readable label for a permutation tuple."""
    return "".join(str(x) for x in w)

def visualize_bruhat_order(module, max_length=None):
    """
    Visualize the Bruhat order for the basis elements in a HeckeModule.

    Args:
        module: A HeckeModule instance
        max_length: Maximum length of permutations to include (None for all)

    Returns:
        A NetworkX graph representing the Bruhat order
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes for permutations, grouped by length
    elements_by_length = {}
    all_elements = []

    for length, elements in sorted(module._basis_by_length.items()):
        if max_length is not None and length > max_length:
            continue
        elements_by_length[length] = elements
        all_elements.extend(elements)

    # Add edges for Bruhat relations
    for x in all_elements:
        for y in all_elements:
            if x != y and module.is_bruhat_leq(y, x):
                # Check if it's a cover relation (no elements in between)
                is_cover = True
                for z in all_elements:
                    if z != x and z != y and module.is_bruhat_leq(y, z) and module.is_bruhat_leq(z, x):
                        is_cover = False
                        break

                if is_cover:
                    G.add_edge(permutation_label(x), permutation_label(y))

    # Add position attributes for drawing
    pos = {}
    for length, elements in elements_by_length.items():
        width = len(elements)
        for i, x in enumerate(sorted(elements, key=tuple)):
            pos[permutation_label(x)] = (i - width/2, -length)

    nx.set_node_attributes(G, pos, 'pos')

    # Draw the graph
    plt.figure(figsize=(10, 8))

    # Draw nodes
    node_colors = []
    for node in G.nodes():
        length = 0
        for x in all_elements:
            if permutation_label(x) == node:
                length = module.ell(x)
                break
        # Color by length
        node_colors.append(plt.cm.viridis(length / max(module._basis_by_length.keys())))

    nx.draw(G, pos, with_labels=True,
            node_color=node_colors,
            node_size=600,
            font_size=8,
            arrowsize=15,
            font_weight='bold')

    plt.title(f"Bruhat Order for S_{module.n}")
    plt.savefig(f"bruhat_order_S{module.n}.png")
    plt.close()

    return G

def print_kl_matrix(module, max_length=None):
    """
    Print the matrix of KL polynomials P_{y,x} for elements up to max_length.

    Args:
        module: A HeckeModule instance with KL polynomials computed
        max_length: Maximum length of permutations to include (None for all)
    """
    if not module._kl_polys:
        module.compute_kl_polynomials()

    # Collect all elements up to max_length
    all_elements = []
    for length, elements in sorted(module._basis_by_length.items()):
        if max_length is not None and length > max_length:
            continue
        all_elements.extend(sorted(elements, key=tuple))

    # Print header
    print("\nKazhdan-Lusztig Polynomial Matrix P_{y,x}:")
    print("     ", end="")
    for x in all_elements:
        print(f"{permutation_label(x):6}", end="")
    print()

    # Print rows
    for y in all_elements:
        print(f"{permutation_label(y):5}", end="")
        for x in all_elements:
            if (y, x) in module._kl_polys and module._kl_polys[(y, x)] != 0:
                value = module._kl_polys[(y, x)]
                if value == 1:
                    print("     1", end="")
                else:
                    print("     *", end="")
            else:
                print("     ·", end="")
        print()

def print_canonical_basis(module, max_examples=None):
    """
    Print the canonical basis elements C_x in terms of the standard basis.

    Args:
        module: A HeckeModule instance with canonical basis computed
        max_examples: Maximum number of examples to print (None for all)
    """
    if not module._canonical_basis:
        module.compute_canonical_basis()

    count = 0
    for x in sorted(module._canonical_basis.keys(), key=lambda x: (length_of_permutation(x), x)):
        if max_examples is not None and count >= max_examples:
            break

        element = module._canonical_basis[x]

        # Skip elements that are just T_x
        if len(element.coeffs) == 1 and x in element.coeffs and element.coeffs[x] == 1:
            continue

        print(f"C_{x} =", end=" ")
        element.pretty()
        count += 1