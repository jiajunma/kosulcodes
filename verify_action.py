#!/usr/bin/env python3
"""
Verify that the right action in HeckeRB satisfies the Hecke algebra relations:

1. Quadratic relation: T_s^2 + (1-q)T_s - q = 0 for all simple reflections s
   Equivalently: T_s^2 = q + (q-1)T_s

2. Braid relation (adjacent): T_{s_i}T_{s_j}T_{s_i} = T_{s_j}T_{s_i}T_{s_j} when |i-j| = 1

3. Commutation relation: T_{s_i}T_{s_j} = T_{s_j}T_{s_i} when |i-j| > 1
"""

import sympy as sp
from HeckeRB import HeckeRB
from HeckeA import q


def elements_equal(elem1, elem2):
    """Check if two elements are equal by comparing coefficients."""
    all_keys = set(elem1.keys()) | set(elem2.keys())
    for key in all_keys:
        coeff1 = elem1.get(key, sp.Integer(0))
        coeff2 = elem2.get(key, sp.Integer(0))
        diff = sp.expand(coeff1 - coeff2)
        if diff != 0:
            return False
    return True


def verify_quadratic_relation(R, i, basis_key, verbose=False):
    """
    Verify T_s^2 = q + (q-1)T_s for a given basis element.
    
    This means: T_{w̃} · T_s · T_s = T_{w̃} · q + T_{w̃} · T_s · (q-1)
    """
    # Start with basis element
    elem = {basis_key: sp.Integer(1)}
    
    # Apply T_s once
    elem_Ts = R.right_action_simple(elem, i)
    
    # Apply T_s twice
    elem_Ts_Ts = R.right_action_simple(elem_Ts, i)
    
    # Compute RHS: T_{w̃} · q + T_{w̃} · T_s · (q-1)
    rhs = {}
    # Add q * T_{w̃}
    for key, coeff in elem.items():
        rhs[key] = rhs.get(key, sp.Integer(0)) + q * coeff
    # Add (q-1) * T_{w̃} · T_s
    for key, coeff in elem_Ts.items():
        rhs[key] = rhs.get(key, sp.Integer(0)) + (q - 1) * coeff
    
    # Simplify both sides
    lhs = {k: sp.expand(c) for k, c in elem_Ts_Ts.items() if sp.expand(c) != 0}
    rhs = {k: sp.expand(c) for k, c in rhs.items() if sp.expand(c) != 0}
    
    is_equal = elements_equal(lhs, rhs)
    
    if verbose and not is_equal:
        print(f"    Quadratic relation FAILED for s_{i} on {basis_key}")
        print(f"      LHS (T_s^2): {lhs}")
        print(f"      RHS (q + (q-1)T_s): {rhs}")
    
    return is_equal


def verify_braid_relation(R, i, j, basis_key, verbose=False):
    """
    Verify T_{s_i}T_{s_j}T_{s_i} = T_{s_j}T_{s_i}T_{s_j} for adjacent i, j.
    
    This means: T_{w̃} · T_{s_i} · T_{s_j} · T_{s_i} = T_{w̃} · T_{s_j} · T_{s_i} · T_{s_j}
    """
    assert abs(i - j) == 1, "i and j must be adjacent for braid relation"
    # Start with basis element
    elem = {basis_key: sp.Integer(1)}
    
    # LHS: T_{w̃} · T_{s_i} · T_{s_j} · T_{s_i}
    lhs = R.right_action_simple(elem, i)
    lhs = R.right_action_simple(lhs, j)
    lhs = R.right_action_simple(lhs, i)
    
    # RHS: T_{w̃} · T_{s_j} · T_{s_i} · T_{s_j}
    rhs = R.right_action_simple(elem, j)
    rhs = R.right_action_simple(rhs, i)
    rhs = R.right_action_simple(rhs, j)
    
    # Simplify both sides
    lhs = {k: sp.expand(c) for k, c in lhs.items() if sp.expand(c) != 0}
    rhs = {k: sp.expand(c) for k, c in rhs.items() if sp.expand(c) != 0}
    
    is_equal = elements_equal(lhs, rhs)
    
    if verbose and not is_equal:
        print(f"    Braid relation FAILED for s_{i}, s_{j} on {basis_key}")
        print(f"      LHS (T_{i}T_{j}T_{i}): {lhs}")
        print(f"      RHS (T_{j}T_{i}T_{j}): {rhs}")
    
    return is_equal


def verify_commutation_relation(R, i, j, basis_key, verbose=False):
    """
    Verify T_{s_i}T_{s_j} = T_{s_j}T_{s_i} for non-adjacent i, j.
    
    This means: T_{w̃} · T_{s_i} · T_{s_j} = T_{w̃} · T_{s_j} · T_{s_i}
    """
    # Start with basis element
    elem = {basis_key: sp.Integer(1)}
    
    # LHS: T_{w̃} · T_{s_i} · T_{s_j}
    lhs = R.right_action_simple(elem, i)
    lhs = R.right_action_simple(lhs, j)
    
    # RHS: T_{w̃} · T_{s_j} · T_{s_i}
    rhs = R.right_action_simple(elem, j)
    rhs = R.right_action_simple(rhs, i)
    
    # Simplify both sides
    lhs = {k: sp.expand(c) for k, c in lhs.items() if sp.expand(c) != 0}
    rhs = {k: sp.expand(c) for k, c in rhs.items() if sp.expand(c) != 0}
    
    is_equal = elements_equal(lhs, rhs)
    
    if verbose and not is_equal:
        print(f"    Commutation relation FAILED for s_{i}, s_{j} on {basis_key}")
        print(f"      LHS (T_{i}T_{j}): {lhs}")
        print(f"      RHS (T_{j}T_{i}): {rhs}")
    
    return is_equal


def verify_all_relations(n, verbose=False, max_elements=None):
    """
    Verify all Hecke algebra relations for HeckeRB(n).
    
    Args:
        n: size of symmetric group
        verbose: if True, print details of failures
        max_elements: maximum number of basis elements to check (None = all)
    """
    print(f"{'='*70}")
    print(f"  Verifying Hecke Algebra Relations for HeckeRB(n={n})")
    print(f"{'='*70}\n")
    
    # Create HeckeRB module
    print(f"Initializing HeckeRB({n})...")
    R = HeckeRB(n)
    basis_elements = list(R.basis())
    print(f"Number of basis elements: {len(basis_elements)}\n")
    
    if max_elements is not None:
        basis_elements = basis_elements[:max_elements]
        print(f"Testing on first {len(basis_elements)} elements\n")
    
    # Test 1: Quadratic relation for all simple reflections
    print(f"{'='*70}")
    print(f"Test 1: Quadratic Relation T_s^2 = q + (q-1)T_s")
    print(f"{'='*70}")
    
    quadratic_pass = 0
    quadratic_fail = 0
    
    for i in range(1, n):
        print(f"\nChecking s_{i}:")
        pass_count = 0
        fail_count = 0
        
        for basis_key in basis_elements:
            if verify_quadratic_relation(R, i, basis_key, verbose=verbose):
                pass_count += 1
            else:
                fail_count += 1
                if not verbose:
                    # Print first few failures
                    if fail_count <= 3:
                        print(f"  FAILED on basis element {basis_key}")
        
        print(f"  s_{i}: {pass_count} passed, {fail_count} failed")
        quadratic_pass += pass_count
        quadratic_fail += fail_count
    
    print(f"\nQuadratic Relation Summary: {quadratic_pass} passed, {quadratic_fail} failed")
    
    # Test 2: Braid relation for adjacent simple reflections
    print(f"\n{'='*70}")
    print(f"Test 2: Braid Relation T_i T_j T_i = T_j T_i T_j (adjacent i,j)")
    print(f"{'='*70}")
    
    braid_pass = 0
    braid_fail = 0
    
    for i in range(1, n - 1):
        j = i + 1
        print(f"\nChecking (s_{i}, s_{j}):")
        pass_count = 0
        fail_count = 0
        
        for basis_key in basis_elements:
            if verify_braid_relation(R, i, j, basis_key, verbose=verbose):
                pass_count += 1
            else:
                fail_count += 1
                if not verbose:
                    if fail_count <= 3:
                        print(f"  FAILED on basis element {basis_key}")
        
        print(f"  (s_{i}, s_{j}): {pass_count} passed, {fail_count} failed")
        braid_pass += pass_count
        braid_fail += fail_count
    
    print(f"\nBraid Relation Summary: {braid_pass} passed, {braid_fail} failed")
    
    # Test 3: Commutation relation for non-adjacent simple reflections
    print(f"\n{'='*70}")
    print(f"Test 3: Commutation Relation T_i T_j = T_j T_i (non-adjacent i,j)")
    print(f"{'='*70}")
    
    commute_pass = 0
    commute_fail = 0
    
    # Find all pairs of non-adjacent indices
    non_adjacent_pairs = []
    for i in range(1, n):
        for j in range(i + 2, n):
            non_adjacent_pairs.append((i, j))
    
    if len(non_adjacent_pairs) == 0:
        print("\nNo non-adjacent pairs for n={n}")
    else:
        for i, j in non_adjacent_pairs:
            print(f"\nChecking (s_{i}, s_{j}):")
            pass_count = 0
            fail_count = 0
            
            for basis_key in basis_elements:
                if verify_commutation_relation(R, i, j, basis_key, verbose=verbose):
                    pass_count += 1
                else:
                    fail_count += 1
                    if not verbose:
                        if fail_count <= 3:
                            print(f"  FAILED on basis element {basis_key}")
            
            print(f"  (s_{i}, s_{j}): {pass_count} passed, {fail_count} failed")
            commute_pass += pass_count
            commute_fail += fail_count
        
        print(f"\nCommutation Relation Summary: {commute_pass} passed, {commute_fail} failed")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Quadratic Relation:   {quadratic_pass} passed, {quadratic_fail} failed")
    print(f"Braid Relation:       {braid_pass} passed, {braid_fail} failed")
    print(f"Commutation Relation: {commute_pass} passed, {commute_fail} failed")
    
    total_pass = quadratic_pass + braid_pass + commute_pass
    total_fail = quadratic_fail + braid_fail + commute_fail
    total_tests = total_pass + total_fail
    
    print(f"\nTotal: {total_pass}/{total_tests} tests passed ({100*total_pass/total_tests:.1f}%)")
    
    if total_fail == 0:
        print("\n✓ All Hecke algebra relations verified!")
    else:
        print(f"\n✗ {total_fail} tests failed")
    
    print(f"{'='*70}\n")
    
    return total_fail == 0


if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Verify Hecke algebra relations for HeckeRB module"
    )
    parser.add_argument(
        "n", type=int, help="Size of the symmetric group"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed information about failures"
    )
    parser.add_argument(
        "--max-elements", "-m", type=int, default=None,
        help="Maximum number of basis elements to test (default: all)"
    )
    
    args = parser.parse_args()
    
    if args.n < 2:
        print("Error: n must be at least 2")
        sys.exit(1)
    
    success = verify_all_relations(args.n, verbose=args.verbose, max_elements=args.max_elements)
    sys.exit(0 if success else 1)
