#!/usr/bin/env python3


from HeckeRB import HeckeRB
from RB import denormalize_key
import sympy as sp

def print_bar_involution_details(R, limit=30):
    """
    Print detailed information about the bar involution for each basis element.

    Args:
        R: HeckeRB instance
        limit: Maximum number of elements to show details for
    """
    print("\nDetailed Bar Involution Table:")
    print("=" * 110)
    print(f"{'Element':<25} {'Bar Involution':<45} ")
    print("=" * 110)

    # Sort keys by size of sigma for clearer pattern visibility
    sorted_keys = sorted(R._basis, key=lambda k: (len(denormalize_key(k)[1]), k))

    count = 0
    for key in sorted_keys:
        if count >= limit:
            print(f"... ({len(R._basis) - limit} more elements not shown)")
            break

        w, sigma = denormalize_key(key)
        T_wtilde = {key: sp.Integer(1)}
        bar_T = R.bar_T(T_wtilde)

        # Format the element and its bar involution
        wtilde_str = R._format_wtilde(w, sigma)
        bar_str = R._format_T_element(bar_T)


        print(f"[{wtilde_str}]".ljust(25), "→".ljust(5), f"{bar_str}".ljust(45))
        count += 1

    print("=" * 110)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Verify bar involution property in HeckeRB')
    parser.add_argument('-n', type=int, default=3, help='Size parameter for HeckeRB')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print detailed bar involution for each basis element')
    parser.add_argument('--limit', type=int, default=20,
                        help='Maximum number of elements to show in verbose mode (default: 20)')
    parser.add_argument('--max-verify', type=int, default=100,
                        help='Maximum number of elements to verify in standard display (default: 100)')

    args = parser.parse_args()

    # Initialize HeckeRB
    R = HeckeRB(args.n)

    # Compute bar involution table
    print(f"Computing bar involution for HeckeRB(n={args.n})...")
    bar_table = R.compute_bar_involution(verbose=False)

    # Print detailed bar involution information if requested
    if args.verbose:
        print_bar_involution_details(R, args.limit)

    # Always verify all standard basis elements
    print(f"\nVerifying if bar(bar(T_{{~w}})) = T_{{~w}} for all standard basis elements:")

    # Set max_elements for verification display
    result = R.verify_bar_involution(max_elements=args.max_verify)

    # Exit with status based on verification result
    print("\nOverall result:", "PASS" if result else "FAIL")
    exit(0 if result else 1)
