import itertools
import argparse
from math import comb, factorial

def generate_all_pm(m, n):
    """
    Generate all partial matchings between {1, ..., m} and {1, ..., n}.
    A partial matching is represented as a frozenset of (i, j) pairs.
    """
    pms = []
    for k in range(min(m, n) + 1):
        for I in itertools.combinations(range(1, m + 1), k):
            for J in itertools.combinations(range(1, n + 1), k):
                for p in itertools.permutations(J):
                    pms.append(frozenset(zip(I, p)))
    return pms


def special_sigma_i(m, n, i):
    """
    Special basic element sigma_i in PM(m,n):
      I = {m-i+1, ..., m}, J = {1, ..., i},
    with the unique order-preserving bijection I -> J.
    """
    if i < 0 or i > min(m, n):
        raise ValueError(f"Require 0 <= i <= min(m,n), got i={i}, m={m}, n={n}")
    if i == 0:
        return frozenset()
    I = list(range(m - i + 1, m + 1))
    J = list(range(1, i + 1))
    return frozenset(zip(I, J))


def all_special_sigmas(m, n):
    """
    Return [sigma_0, sigma_1, ..., sigma_{min(m,n)}].
    """
    return [special_sigma_i(m, n, i) for i in range(0, min(m, n) + 1)]


def count_pm(m, n):
    """
    Count |PM(m,n)| by explicit enumeration.
    """
    return len(generate_all_pm(m, n))


def count_pm_formula(m, n):
    """
    Count |PM(m,n)| via:
      sum_{k=0}^{min(m,n)} C(m,k) C(n,k) k!
    """
    total = 0
    for k in range(0, min(m, n) + 1):
        total += comb(m, k) * comb(n, k) * factorial(k)
    return total


def is_partial_matching(pm):
    """
    Check if a given set of pairs (i, j) is a partial matching.
    A partial matching means each i and each j appears at most once.
    """
    if not isinstance(pm, (frozenset, set)):
        raise TypeError("pm must be a frozenset or a set of pairs (i, j)")
    
    domain = [i for i, _ in pm]
    codomain = [j for _, j in pm]
    
    return len(pm) == len(set(domain)) == len(set(codomain))


def get_type_companion(pm, side, i):
    """
    Determine the type and companion of a partial matching for a simple reflection.
    
    Args:
        pm: frozenset of (i, j) pairs / (maze)
        side: 'left' for s_i (1 <= i < m), 'right' for s_i' (1 <= i < n)
        i: index of the simple reflection
        
    Returns:
        A tuple (type, companions) where:
        - type: 'G', 'U-', or 'U+'
        - companions: list containing pm itself and its companion
    """
    pm_dict = dict(pm)
    inv_pm_dict = {j: k for k, j in pm}
    
    if side == 'left':
        # sigma_dagger(i) = mu(i) if i in I else 0
        # Order: 0 < 1 < 2 < ... < n
        val_i = pm_dict.get(i, 0)
        val_ip1 = pm_dict.get(i + 1, 0)
        
        if val_i == val_ip1:
            typ = 'G'
        elif val_i < val_ip1:
            typ = 'U-'
        else:
            typ = 'U+'
            
        # Companion: (s_i, id) * sigma
        # If (k, j) in pm, then (s_i(k), j) in companion
        companion_list = []
        for k, j in pm:
            if k == i:
                companion_list.append((i + 1, j))
            elif k == i + 1:
                companion_list.append((i, j))
            else:
                companion_list.append((k, j))
        companion = frozenset(companion_list)
        return (typ, [pm, companion])
        
    elif side == 'right':
        # sigma_lower_dagger(i) = mu_inv(i) if i in J else infinity
        # Order: 1 < 2 < ... < m < infinity
        inf = float('inf')
        val_i = inv_pm_dict.get(i, inf)
        val_ip1 = inv_pm_dict.get(i + 1, inf)
        
        if val_i == val_ip1:
            typ = 'G'
        elif val_i < val_ip1:
            typ = 'U-'
        else:
            typ = 'U+'
            
        # Companion: (id, s_i') * sigma
        # If (k, j) in pm, then (k, s_i'(j)) in companion
        companion_list = []
        for k, j in pm:
            if j == i:
                companion_list.append((k, i + 1))
            elif j == i + 1:
                companion_list.append((k, i))
            else:
                companion_list.append((k, j))
        companion = frozenset(companion_list)
        return (typ, [pm, companion])
    
    else:
        raise ValueError("side must be 'left' or 'right'")

def get_all_types_companions(pm, m, n):
    """
    Get all types and companions for all simple reflections s_i and s_i'.
    """
    result = {'left': {}, 'right': {}}
    for i in range(1, m):
        result['left'][i] = get_type_companion(pm, 'left', i)
    for i in range(1, n):
        result['right'][i] = get_type_companion(pm, 'right', i)
    return result


def descent_set_pm(pm,m,n):
    """
    Compute the descent set of a partial matching.
    
    The formula is:
    descent_set(pm) = {i : i in [1,m] and there exists j in [1,n] such that (i,j) in pm and j < i}
    """
    DescentL = [] 
    DescentR = []
    pm_dict = dict(pm)
    inv_pm_dict = {j: k for k, j in pm}
    for i in range(1, m):
        val_i = pm_dict.get(i, 0)
        val_ip1 = pm_dict.get(i + 1, 0)
        if val_i <= val_ip1:
            DescentL.append(i)
    for i in range(1, n):
        val_i = inv_pm_dict.get(i, 0)
        val_ip1 = inv_pm_dict.get(i + 1, 0)
        if val_i <= val_ip1:
            DescentR.append(i)
    return DescentL, DescentR

def ell_pm(pm,m,n):
    """
    Compute the dimension of the orbit of determined by a partial matching.
    the dimension depends on (m,n)

    The formula is: 
    dim(O_pm) = sum_{(i,j)\in pm } (i + n - j) - sum_{(i,j),(i',j')\in pm, i < i', j < j'} 1
    """
    if not pm:
        return 0

    pairs = sorted(pm, key=lambda pair: pair[0])
    dim_sum = sum(i + n - j for i, j in pairs)

    increasing_pairs = 0
    for idx in range(len(pairs)):
        _, j1 = pairs[idx]
        for jdx in range(idx + 1, len(pairs)):
            _, j2 = pairs[jdx]
            if j1 < j2:
                increasing_pairs += 1

    return dim_sum - increasing_pairs

def action_pm_left(pm, w):
    """
    We use the list notation of a permutation w.
    left action of w on pm
    where w in a permutation of {1, ..., m}
    the result is a new partial matching pm'
    """
    result = frozenset((w[i-1], j) for i, j in pm)
    return result

def action_pm_right(pm, w):
    """
    Right action of w on pm
    where w is a permutation of {1, ..., n} (given as a list of length n)
    The result is a new partial matching pm':
    if pm = {(i_k, j_k)}
    the result is {(i_k, w^{-1}(j_k))}

    Args:
        pm: an iterable of pairs (i, j)
        w: a permutation list of length n

    Returns:
        A frozenset representing the right action, i.e., {(i_k, w^{-1}(j_k))}
    """
    n = len(w)
    # Build value_to_index mapping: w[j-1] = value, so value_to_index[value] = j
    value_to_index = {value: idx + 1 for idx, value in enumerate(w)}
    result = frozenset((i, value_to_index[j]) for (i, j) in pm)
    return result

def longest_element(m):
    """
    Return the longest element in the symmetric group S_m.
    """
    return tuple(range(m, 0, -1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enumerate PM(m,n) and print type+companion data."
    )
    parser.add_argument("m", type=int, nargs="?", default=2)
    parser.add_argument("n", type=int, nargs="?", default=2)
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only print |PM(m,n)| from formula and enumeration.",
    )
    parser.add_argument(
        "--special-basic",
        action="store_true",
        help="Print only special basic elements sigma_i (i=0..min(m,n)).",
    )
    args = parser.parse_args()

    m, n = args.m, args.n
    if m < 0 or n < 0:
        raise ValueError(f"Require m,n >= 0, got m={m}, n={n}")

    enum_count = count_pm(m, n)
    formula_count = count_pm_formula(m, n)
    print(f"|PM({m},{n})| by enumeration: {enum_count}")
    print(f"|PM({m},{n})| by formula:     {formula_count}")

    if args.count_only:
        raise SystemExit(0)
    if args.special_basic:
        sigmas = all_special_sigmas(m, n)
        print(f"Special basic elements count: {len(sigmas)}")
        for i, sigma in enumerate(sigmas):
            print(f"sigma_{i} = {sigma}")
        raise SystemExit(0)

    pms = generate_all_pm(m, n)
    for pm in pms:
        print(f"\nPM: {pm}")
        info = get_all_types_companions(pm, m, n)
        for side in ['left', 'right']:
            for i, res in info[side].items():
                typ, companions = res
                print(f"  {side} s_{i}: type {typ}, companions {companions}")
