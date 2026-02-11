import itertools

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
        result['left'][i] = get_type_companion(pm, m, n, 'left', i)
    for i in range(1, n):
        result['right'][i] = get_type_companion(pm, m, n, 'right', i)
    return result

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
        i1, j1 = pairs[idx]
        for jdx in range(idx + 1, len(pairs)):
            i2, j2 = pairs[jdx]
            if j1 < j2:
                increasing_pairs += 1

    return dim_sum - increasing_pairs

if __name__ == "__main__":
    # Example usage and verification
    m, n = 2, 2
    pms = generate_all_pm(m, n)
    print(f"Total partial matchings for m={m}, n={n}: {len(pms)}")
    for pm in pms:
        print(f"\nPM: {pm}")
        info = get_all_types_companions(pm, m, n)
        for side in ['left', 'right']:
            for i, res in info[side].items():
                print(f"  {side} s_{i}: type {res['type']}, companion {res['companion']}")
