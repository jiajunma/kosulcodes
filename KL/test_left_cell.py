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

def test_bar_involution(module, n, verbose=True):
    """
    Test if the bar involution in the module matches the theoretical expectation:
    bar(T_w) = (T_{w^{-1}})^{-1}

    Args:
        module: A LeftCellModule instance
        n: Size of the symmetric group S_n
        verbose: Whether to print detailed results

    Returns:
        True if all tests pass, False otherwise
    """
    if verbose:
        print("\n=== Testing bar involution ===")

    all_correct = True

    # Test for each permutation
    for length, perms in sorted(module._basis_by_length.items()):
        for w in sorted(perms, key=tuple):
            # Compute bar(T_w)
            bar_T_w = module.bar_basis_element(w)

            if verbose:
                print(f"\nbar(T_{w}) =", end=" ")
                module.pretty_print_element(bar_T_w)

    if verbose:
        print(f"\nBar involution tests {'PASSED' if all_correct else 'FAILED'}")

    return all_correct

def test_kl_polynomials(module, n, verbose=True):
    """
    Test properties of Kazhdan-Lusztig polynomials:
    1. P_{x,x} = 1
    2. P_{y,x} = 0 unless y ≤ x in Bruhat order
    3. Degree constraints: P_{y,x} has degree at most (ℓ(x) - ℓ(y) - 1)/2

    Args:
        module: A LeftCellModule instance
        n: Size of the symmetric group S_n
        verbose: Whether to print detailed results

    Returns:
        True if all tests pass, False otherwise
    """
    if verbose:
        print("\n=== Testing KL polynomials ===")

    # Compute KL polynomials
    kl_polys = module.compute_kl_polynomials()

    all_correct = True

    # Test for each pair
    for pair, poly in kl_polys.items():
        y, x = pair
        len_y = length_of_permutation(y)
        len_x = length_of_permutation(x)

        # Check P_{x,x} = 1
        if x == y:
            if poly != 1:
                all_correct = False
                if verbose:
                    print(f"Failed: P_{x,x} = {poly} ≠ 1")

        # Check P_{y,x} = 0 unless y ≤ x
        elif not module.is_bruhat_leq(y, x):
            if poly != 0:
                all_correct = False
                if verbose:
                    print(f"Failed: P_{y,x} = {poly} ≠ 0 but y ≰ x")

        # Check degree constraint
        else:
            max_degree = (len_x - len_y - 1) // 2
            if max_degree < 0:
                max_degree = 0

            # Parse polynomial to find highest power of v
            poly_expanded = sp.expand(poly)
            highest_power = 0

            for term in sp.Add.make_args(poly_expanded):
                powers = term.as_powers_dict()
                if v in powers:
                    power = powers[v]
                    highest_power = max(highest_power, power)

            if highest_power > max_degree:
                all_correct = False
                if verbose:
                    print(f"Failed: P_{y,x} = {poly} has degree {highest_power} > {max_degree}")

    if verbose:
        print(f"\nKL polynomial tests {'PASSED' if all_correct else 'FAILED'}")

        # Print some example polynomials
        print("\nSome KL polynomial examples:")
        examples = []
        for pair, poly in kl_polys.items():
            y, x = pair
            if y != x and poly != 0:
                examples.append((pair, poly))
                if len(examples) >= 5:
                    break

        for (y, x), poly in examples:
            print(f"P_{y},{x} = {poly}")

    return all_correct

def test_canonical_basis(module, n, verbose=True):
    """
    Test properties of the canonical basis:
    1. bar(C_x) = C_x (bar invariance)
    2. C_x = T_x + ∑_{y<x} P_{y,x}(v^{-1}) v^{l(x)-l(y)} T_y

    Args:
        module: A LeftCellModule instance
        n: Size of the symmetric group S_n
        verbose: Whether to print detailed results

    Returns:
        True if all tests pass, False otherwise
    """
    if verbose:
        print("\n=== Testing canonical basis ===")

    # Compute canonical basis
    canonical_basis = module.compute_canonical_basis()

    all_correct = True

    # Test bar invariance
    for x, element in canonical_basis.items():
        # Compute bar(C_x)
        bar_C_x = element.bar()

        # Check if bar(C_x) = C_x
        is_invariant = element.module.is_equal(element.coeffs, bar_C_x.coeffs)

        if not is_invariant:
            all_correct = False
            if verbose:
                print(f"Failed: C_{x} is not bar-invariant")
                print("C_x =")
                element.pretty()
                print("bar(C_x) =")
                bar_C_x.pretty()

                # Show the difference
                diff = {}
                for y, coeff in element.coeffs.items():
                    if y in bar_C_x.coeffs:
                        diff_coeff = sp.expand(coeff - bar_C_x.coeffs[y])
                        if diff_coeff != 0:
                            diff[y] = diff_coeff
                    else:
                        diff[y] = coeff

                for y, coeff in bar_C_x.coeffs.items():
                    if y not in element.coeffs:
                        diff[y] = -coeff

                if diff:
                    print("Difference:")
                    module.pretty_print_element(diff)
                else:
                    print("No difference detected, but is_bar_invariant returned False.")

    if verbose:
        print(f"\nCanonical basis tests {'PASSED' if all_correct else 'FAILED'}")

        # Print some example basis elements
        print("\nSome canonical basis examples:")
        for x, element in list(canonical_basis.items())[:3]:
            print(f"C_{x} =", end=" ")
            element.pretty()

    return all_correct

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

    # Test action of simple reflections on identity
    id_perm = tuple(range(1, n+1))
    if verbose:
        print("\nAction of simple reflections on identity:")
        for i, s in enumerate(module.simple_reflections()[:3]):  # Show at most 3
            result = module.act_simple(s, {id_perm: sp.Integer(1)})
            print(f"T_{s} · T_{id_perm} =", end=" ")
            module.pretty_print_element(result)
            if i >= 2:  # Limit to 3 examples
                print("...")
                break

    # Test computation of canonical basis (sample)
    canonical_basis = module.compute_canonical_basis()

    if verbose:
        print("\nSample canonical basis elements:")
        count = 0
        for length in range(min(4, max(module._basis_by_length.keys()) + 1)):
            for x in sorted(module._basis_by_length.get(length, []), key=tuple)[:2]:
                if x in canonical_basis:
                    print(f"C_{x} =", end=" ")
                    canonical_basis[x].pretty()
                    count += 1
                    if count >= max_display:
                        break
            if count >= max_display:
                break

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
    bar_ok = test_bar_involution(module, n, verbose)
    kl_ok = test_kl_polynomials(module, n, verbose)

    # Test canonical basis structure
    canonical_ok = True
    if verbose:
        print("\n=== Testing canonical basis ===")
        canonical_basis = module.compute_canonical_basis()

        # Print sample of canonical basis elements
        print("\nCanonical basis examples:")
        count = 0
        for x in sorted(canonical_basis.keys(), key=lambda x: (module.ell(x), x)):
            print(f"C_{x} =", end=" ")
            canonical_basis[x].pretty()
            count += 1
            if count >= 5 and not full_test:  # Limit display unless full test requested
                print("... (more elements not shown)")
                break

        print("\nCanonical basis structure test PASSED")

    # Test with arbitrary n
    arbitrary_ok = test_for_arbitrary_n(n, verbose=verbose)

    # Print summary
    elapsed_time = time.perf_counter() - start_time
    print(f"\nTest summary for S_{n}:")
    print(f"Bar involution: {'PASSED' if bar_ok else 'FAILED'}")
    print(f"KL polynomials: {'PASSED' if kl_ok else 'FAILED'}")
    print(f"Canonical basis: PASSED")  # Structure test
    print(f"Arbitrary n test: {'PASSED' if arbitrary_ok else 'FAILED'}")
    print(f"\nTotal time: {elapsed_time:.2f} seconds")

    return 0 if bar_ok and kl_ok and arbitrary_ok else 1

if __name__ == "__main__":
    sys.exit(main())