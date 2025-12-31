from itertools import permutations, chain, combinations

def generate_permutations(n):
    """
    Generate all permutations of numbers from 1 to n
    
    Args:
        n: The upper bound of the range (inclusive)
    
    Yields:
        A tuple representing each permutation
    """
    
    # Generate numbers from 1 to n
    numbers = list(range(1, n + 1))
    
    # Generate and yield all permutations
    yield from permutations(numbers)


def powerset(iterable):
    """Generate all subsets of an iterable"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def is_decreasing_on_subset(w, I):
    """
    Test if w satisfies: for all i,j in I where i<j, we have w(i) > w(j)
    (i.e., w is decreasing on the subset of positions I)
    
    Args:
        w: A permutation (tuple or list)
        I: A set of positions (1-indexed)
    
    Returns:
        True if w is decreasing on positions in I, False otherwise
    """
    # Convert I to a sorted list for easy comparison
    positions = sorted(I)
    
    # Check all pairs (i,j) where i < j in I
    for idx in range(len(positions)):
        for jdx in range(idx + 1, len(positions)):
            i = positions[idx]
            j = positions[jdx]
            # i < j by construction, check if w(i) > w(j)
            if w[i-1] <= w[j-1]:  # Convert to 0-indexed
                return False
    
    return True


def is_beta_on_subset(w, I):
    """
    Test if w satisfies the beta condition on subset I:
    For all j not in I and i in I, either i > j or w[i] > w[j]
    
    Args:
        w: A permutation (tuple or list)
        I: A set of positions (1-indexed)
    
    Returns:
        True if w satisfies the beta condition on I, False otherwise
    """
    n = len(w)
    
    # Check all pairs (i, j) where i in I and j not in I

    for j in I:
        for i in range(1, j):
            if (i not in I) and w[i-1] <= w[j-1]:  # Convert to 0-indexed
                return False
    return True




def generate_all_beta(w):
    """
    Given a permutation w, yield all possible subsets I (beta) where w satisfies the beta condition on I.
    
    Args:
        w: A permutation (tuple or list)
    
    Yields:
        Each subset I (as a set) where w satisfies the beta condition on positions in I
    """
    n = len(w)
    # Generate all possible subsets of positions {1, 2, ..., n}
    positions = set(range(1, n + 1))
    
    for subset in powerset(positions):
        I = set(subset)
        if is_beta_on_subset(w, I):
            yield I

def generate_all_sigma(w):
    """
    Given a permutation w, yield all possible subsets I (sigma) where w is decreasing on I.
    
    Args:
        w: A permutation (tuple or list)
    
    Yields:
        Each subset I (as a set) where w is decreasing on positions in I
    """
    n = len(w)
    # Generate all possible subsets of positions {1, 2, ..., n}
    positions = set(range(1, n + 1))
    
    for subset in powerset(positions):
        I = set(subset)
        if is_decreasing_on_subset(w, I):
            yield I


def beta_to_sigma(w, beta):
    """
    Map beta to sigma according to equation (6):
    σ = σ(w̃) = {i ∈ β | ∀j (j > i) & (w(j) > w(i)) ⟹ j ∉ β}
    
    In other words, i is in sigma if:
    - i is in beta, AND
    - for all j > i where w(j) > w(i), j is NOT in beta
    
    Args:
        w: A permutation (tuple or list)
        beta: A set of positions (1-indexed)
    
    Returns:
        A set sigma computed from beta according to the formula
    """
    sigma = set()
    
    for i in beta:
        # Check the condition: ∀j (j > i) & (w(j) > w(i)) ⟹ j ∉ β
        condition_satisfied = True
        
        for j in range(i + 1, len(w) + 1):
            # If j > i and w(j) > w(i)
            if w[j-1] > w[i-1]:
                # Then j must NOT be in beta
                if j in beta:
                    condition_satisfied = False
                    break
        
        if condition_satisfied:
            sigma.add(i)

    # Assert that sigma is decreasing (w is decreasing on sigma)
    assert is_decreasing_on_subset(w, sigma), f"Sigma {sigma} is not decreasing on w={w}"
    return sigma



def str_colored_partition(w, I):
    """
    Generate colored partition string for a permutation w.
    Elements at positions in I are colored blue, others are colored red.
    
    Args:
        w: A permutation (tuple or list)
        I: A set of positions to color blue (1-indexed)
    
    Returns:
        A string with ANSI color codes showing the colored partition
    """
    # ANSI color codes
    BLUE = '\033[94m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    n = len(w)
    result = []
    
    for i in range(1, n + 1):
        if i in I:
            result.append(f"{BLUE}{w[i-1]}{RESET}")
        else:
            result.append(f"{RED}{w[i-1]}{RESET}")
    
    return f"[{', '.join(result)}]"


def generate_all_w_sigma_pairs(n):
    """
    Generate all (w, sigma) pairs where w is a permutation of 1 to n
    and sigma is a subset where w is decreasing.
    
    Args:
        n: The size of the permutation
    
    Yields:
        Tuples of (w, sigma) where w is a permutation and sigma is a valid subset
    """
    for w in generate_permutations(n):
        for sigma in generate_all_sigma(w):
            yield (w, sigma)


def sigma_to_beta(w, sigma):
    """
    Map sigma to beta according to the rule:
    If i is in sigma, then all j such that j < i and w[j] < w[i] are also in beta.
    
    Args:
        w: A permutation (tuple or list)
        sigma: A set of positions (1-indexed)
    
    Returns:
        A set beta computed from sigma according to the formula
    """
    beta = set()
    
    # Start with all elements in sigma
    beta.update(sigma)
    
    # For each i in sigma
    for i in sigma:
        # Add all j where j < i and w[j] < w[i]
        for j in range(1, i):
            if w[j-1] < w[i-1]:
                beta.add(j)

    # Assert that beta satisfies the beta condition
    assert is_beta_on_subset(w, beta), f"Beta {beta} does not satisfy the beta condition on w={w}"
    return beta



def print_all_w_sigma_pairs(n, verbose=1):
    """
    Generate and print all (w, sigma) pairs for permutations of size n.
    Also prints the mapping sigma -> beta -> sigma and verifies if they're equal.
    
    Args:
        n: The size of the permutation
        verbose: Verbosity level (0=minimal, 1=normal, 2=detailed)
    """
    count = 0
    mismatch_count = 0
    
    for w, sigma in generate_all_w_sigma_pairs(n):
        count += 1
        
        # Compute beta from sigma
        beta = sigma_to_beta(w, sigma)
        
        # Compute sigma' from beta
        sigma_prime = beta_to_sigma(w, beta)
        
        # Check if original sigma equals sigma'
        is_equal = (sigma == sigma_prime)
        status = "✓" if is_equal else "✗"
        
        if not is_equal:
            mismatch_count += 1
        
        # Only print sigma and beta details if verbose level is high enough
        if verbose >= 2:
            print(f"{count}: (w,σ)={str_colored_partition(w, sigma)} (w,β)={str_colored_partition(w, beta)}")
            if not is_equal:
                print(f"   WARNING: σ ≠ σ' for permutation {w}")
    
    print(f"Total pairs: {count}")
    print(f"Mismatches: {mismatch_count}")
    
    if mismatch_count == 0:
        print("✓ All mappings σ -> β -> σ preserve the original σ")
    else:
        print(f"✗ Found {mismatch_count} mismatches where σ ≠ σ'")

def verify_sigma_beta_equality(n, verbose=1):
    """
    Verify that for each permutation w of size n, the number of valid sigma
    equals the number of valid beta.
    
    Args:
        n: The size of the permutation
        verbose: Verbosity level (0=minimal, 1=normal, 2=detailed)
    
    Returns:
        True if all permutations satisfy the equality, False otherwise
    """
    all_equal = True
    
    for w in generate_permutations(n):
        # Count all valid sigma for this permutation
        sigma_count = sum(1 for _ in generate_all_sigma(w))
        
        # Count all valid beta for this permutation
        beta_count = sum(1 for _ in generate_all_beta(w))
        
        if sigma_count != beta_count:
            if verbose >= 1:
                print(f"Mismatch found for permutation {w}:")
                print(f"  Sigma count: {sigma_count}")
                print(f"  Beta count: {beta_count}")
            all_equal = False
        else:
            if verbose >= 2:
                print(f"Permutation {w}: σ={sigma_count}, β={beta_count} ✓")
    
    if verbose >= 1:
        if all_equal:
            print(f"\n✓ Verification passed: All permutations of size {n} have equal sigma and beta counts")
        else:
            print(f"\n✗ Verification failed: Some permutations have unequal counts")
    
    return all_equal

# Example usage:
if __name__ == "__main__":

    import sys
    
    # Read n from command line, default to 4 if no parameter provided
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 4
    
    # Read verbose level from command line, default to 1 if no parameter provided
    if len(sys.argv) > 2:
        verbose = int(sys.argv[2])
    else:
        verbose = 1
    
    # Verify that for each permutation w, the number of valid sigma equals the number of valid beta
    verify_sigma_beta_equality(n, verbose)
    
    print_all_w_sigma_pairs(n, verbose)
