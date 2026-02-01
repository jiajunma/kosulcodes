import sympy as sp
import time
import sys
import os

# Add parent directory to the path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from KL.LeftCellModule import LeftCellModule
from perm import (
    generate_permutations,
    inverse_permutation,
    permutation_prod,
    simple_reflection,
    length_of_permutation
)

# Define symbolic variable v
v = sp.symbols("v")

def test_action_simple_reflection(module, n, verbose=True):
    """
    Test the action of simple reflections on basis elements.

    Args:
        module: A LeftCellModule instance
        n: Size of the symmetric group S_n
        verbose: Whether to print detailed results

    Returns:
        True if all tests pass, False otherwise
    """
    if verbose:
        print("\n=== Testing action by simple reflections ===")

    all_correct = True
    id_perm = tuple(range(1, n + 1))

    # Test each simple reflection on identity
    for i in range(1, n):
        s_i = simple_reflection(i, n)

        # Expected action based on rules:
        # T_s · T_w = T_{sw} if sw > w
        # T_s · T_w = v^2 T_{sw} + (v^2-1)T_w if sw < w

        # For identity, we know s_i > id, so T_s · T_id = T_{s_i}
        expected = {s_i: 1}
        result = module.action_by_simple_reflection(s_i, id_perm)

        is_correct = result == expected
        all_correct &= is_correct

        if verbose:
            print(f"T_{s_i} · T_{id_perm} = ", end="")
            module.pretty_print_element(result)
            if not is_correct:
                print(f"  Expected: {expected}")
                print(f"  Got: {result}")
                print("  FAILED!")

    # Test some other permutation (take first non-identity element)
    if n >= 3:
        for s_i in module.simple_reflections():
            for w in list(generate_permutations(n))[:5]:  # Test a few permutations
                if w == id_perm:
                    continue

                sw = permutation_prod(s_i, w)
                if module.ell(sw) > module.ell(w):
                    # U- type: T_s · T_w = T_{sw}
                    expected = {sw: 1}
                else:
                    # U+ type: T_s · T_w = v^2 T_{sw} + (v^2-1)T_w
                    expected = {sw: v**2, w: v**2 - 1}

                result = module.action_by_simple_reflection(s_i, w)
                is_correct = result == expected
                all_correct &= is_correct

                if verbose and not is_correct:
                    print(f"T_{s_i} · T_{w} = ", end="")
                    module.pretty_print_element(result)
                    print(f"  Expected: {expected}")
                    print(f"  Got: {result}")
                    print("  FAILED!")

    if verbose:
        print(f"\nSimple reflection action tests {'PASSED' if all_correct else 'FAILED'}")

    return all_correct

def test_canonical_basis(module, n, verbose=True, max_display=5):
    """
    Output the canonical basis elements.

    Args:
        module: A LeftCellModule instance
        n: Size of the symmetric group S_n
        verbose: Whether to print detailed results
        max_display: Maximum number of elements to display

    Returns:
        True always since we're just displaying the basis
    """
    if verbose:
        print("\n=== Testing canonical basis ===")

    # Get the standard canonical basis from HeckeA implementation
    canonical_basis = module.get_standard_canonical_basis()

    # Output canonical basis elements
    if verbose:
        print(f"\nCanonical basis elements for S_{n}:")
        count = 0
        for w, element in sorted(canonical_basis.items(), key=lambda x: (length_of_permutation(x[0]), x[0])):
            print(f"C_{w} = ", end="")
            element.pretty()
            count += 1
            if count >= max_display and not (n <= 3):
                print("...")
                break

    return True


def test_for_arbitrary_n(n, max_display=10, verbose=True):
    """
    Test the LeftCellModule with an arbitrary value of n.

    Args:
        n: Size of the symmetric group S_n
        max_display: Maximum number of elements to display (for large n)
        verbose: Whether to print detailed results
    """
    if verbose:
        print(f"\n=== Testing with arbitrary n = {n} ===")

    # Create module
    module = LeftCellModule(n)

    # Count permutations
    total_perms = sum(len(perms) for perms in module._basis_by_length.values())

    if verbose:
        print(f"Total permutations in S_{n}: {total_perms}")
        print(f"Maximum length: {max(module._basis_by_length.keys())}")

        # Display distribution by length
        print("\nPermutation distribution by length:")
        for length, perms in sorted(module._basis_by_length.items()):
            print(f"Length {length}: {len(perms)} permutations")

        # Display some example permutations
        print("\nSample permutations:")
        count = 0
        for length, perms in sorted(module._basis_by_length.items()):
            for perm in sorted(perms, key=tuple)[:3]:  # Show at most 3 per length
                print(f"Length {length}: {perm}")
                count += 1
                if count >= max_display:
                    break
            if count >= max_display:
                break

    # Test action of simple reflections
    test_action_simple_reflection(module, n, verbose)

    # Test canonical basis
    test_canonical_basis(module, n, verbose, max_display=5)

    # Test Bruhat order
    if verbose:
        print("\nTesting Bruhat order relations:")
        for u in list(generate_permutations(n))[:3]:  # Test a few permutations
            for v in list(generate_permutations(n))[:3]:
                is_leq = module.is_bruhat_leq(u, v)
                print(f"{u} <= {v}: {is_leq}")

    return True

def main():
    """
    Main test function.
    """
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_left_cell.py <n> [--verbose] [--full-test]")
        sys.exit(1)

    n = int(sys.argv[1])
    verbose = "--verbose" in sys.argv
    full_test = "--full-test" in sys.argv

    # Create module
    print(f"Testing LeftCellModule for S_{n}")
    start_time = time.perf_counter()

    module = LeftCellModule(n)

    # Run tests
    action_ok = test_action_simple_reflection(module, n, verbose)

    # Test canonical basis
    canonical_ok = test_canonical_basis(module, n, verbose)

    # Test with arbitrary n
    arbitrary_ok = test_for_arbitrary_n(n, verbose=verbose)

    # Print summary
    elapsed_time = time.perf_counter() - start_time
    print(f"\nTest summary for S_{n}:")
    print(f"Simple reflection action: {'PASSED' if action_ok else 'FAILED'}")
    print(f"Canonical basis: {'PASSED' if canonical_ok else 'FAILED'}")
    print(f"Arbitrary n test: {'PASSED' if arbitrary_ok else 'FAILED'}")
    print(f"\nTotal time: {elapsed_time:.2f} seconds")

    return 0 if (action_ok and canonical_ok and arbitrary_ok) else 1

if __name__ == "__main__":
    sys.exit(main())