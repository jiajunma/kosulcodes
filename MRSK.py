from RB import (
    generate_permutations,
    generate_all_beta,
    sigma_to_beta,
    str_colored_partition,
)


AT = "@" 


def _lt(a, b):
    if a is AT:
        return False
    if b is AT:
        return True
    return a < b


def row_insert(tableau, x):
    """
    Standard row bumping insertion into a tableau.

    Args:
        tableau: list of rows
        x: value to insert (number or AT)

    Returns:
        The row index where a new box was added.
    """
    row_idx = 0
    while True:
        if row_idx == len(tableau):
            tableau.append([x])
            return row_idx
        row = tableau[row_idx]
        for j, y in enumerate(row):
            if _lt(x, y):
                row[j], x = x, y
                break
        else:
            row.append(x)
            return row_idx
        row_idx += 1


def insert_into_infinite_row(row, x):
    """
    Insert x into the infinite row r^@ and return the bumped element.

    The row is modeled by a finite list of numbers in nondecreasing order.
    If x is >= all entries, it appends x and bumps AT.
    """
    for j, y in enumerate(row):
        if _lt(x, y):
            row[j] = x
            return y
    row.append(x)
    return AT


def shape(tableau):
    return [len(row) for row in tableau]


def tableau_to_string(tableau):
    """
    Format a tableau as a labeled Young diagram.
    """
    if not tableau:
        return "(empty)"
    width = max(len(str(x)) for row in tableau for x in row)
    return "\n".join(" ".join(f"{x:>{width}}" for x in row) for row in tableau)


def print_tableau(tableau, label=None):
    if label:
        print(label)
    print(tableau_to_string(tableau))


def strip_at(tableau):
    stripped = []
    for row in tableau:
        new_row = [x for x in row if x is not AT]
        if new_row:
            stripped.append(new_row)
    return stripped


def mirabolic_rsk(w, beta):
    """
    Mirabolic RSK for w~ = (w, beta).

    Args:
        w: permutation (tuple/list of ints)
        beta: subset of positions (1-indexed)

    Returns:
        dict with tableaux and shapes:
            T_at, T1, T2, nu, theta, nu_prime
    """
    beta_set = set(beta)
    T_at = []
    T1 = []
    r_row = []

    for i, val in enumerate(w, start=1):
        if i in beta_set:
            new_row = row_insert(T_at, val)
        else:
            bumped = insert_into_infinite_row(r_row, val)
            new_row = row_insert(T_at, bumped)
        while len(T1) <= new_row:
            T1.append([])
        T1[new_row].append(i)

    for val in r_row:
        row_insert(T_at, val)

    T2 = strip_at(T_at)
    nu = shape(T1)
    nu_prime = shape(T2)
    theta = shape(T_at)[1:]

    return {
        "T_at": T_at,
        "T1": T1,
        "T2": T2,
        "nu": nu,
        "theta": theta,
        "nu'": nu_prime,
    }


def mirabolic_rsk_from_sigma(w, sigma):
    beta = sigma_to_beta(w, sigma)
    return mirabolic_rsk(w, beta)


def generate_all_w_beta_pairs(n):
    for w in generate_permutations(n):
        for beta in generate_all_beta(w):
            yield w, beta


def count_microlocal_packets(n, verbose=1):
    """
    Count microlocal packets by grouping (w, beta) pairs by their (nu, theta, nu') output.
    
    A microlocal packet is the collection of all (w, beta) pairs that map to the same
    (nu, theta, nu') triple under the Mirabolic RSK algorithm.
    
    Args:
        n: size of permutations
        verbose: output level
            0 - only summary statistics
            1 - print each packet with count
            2 - print each packet with all (w, beta) pairs
    
    Returns:
        dict: mapping from (nu, theta, nu') to list of (w, beta) pairs
    """
    packets = {}
    
    for w, beta in generate_all_w_beta_pairs(n):
        result = mirabolic_rsk(w, beta)
        # Create a hashable key from the shapes
        key = (tuple(result["nu"]), tuple(result["theta"]), tuple(result["nu'"]))
        
        if key not in packets:
            packets[key] = []
        packets[key].append((w, beta))
    
    # Print results based on verbosity
    if verbose >= 1:
        print(f"Microlocal packets for n={n}:")
        print(f"Total number of packets: {len(packets)}")
        print(f"Total (w,beta) pairs: {sum(len(v) for v in packets.values())}")
        print("=" * 60)

        if verbose >= 2: 
         for key, pairs in sorted(packets.items(), key=lambda x: len(x[1]), reverse=True):
            nu, theta, nu_prime = key
            print(f"\nPacket: nu={list(nu)}, theta={list(theta)}, nu'={list(nu_prime)}")
            print(f"  Size: {len(pairs)}")
            
            print("  Members:")
            for w, beta in pairs:
                print(f"    {str_colored_partition(w, beta)}")
    
    return packets


def test_all_rb(n, verbose=1):
    count = 0
    for w, beta in generate_all_w_beta_pairs(n):
        count += 1
        result = mirabolic_rsk(w, beta)
        if verbose >= 2:
            print("w,beta =", str_colored_partition(w, beta))
            print("nu =", result["nu"])
            print("theta =", result["theta"])
            print("nu' =", result["nu'"])
            if verbose > 3:
                print_tableau(result["T_at"], label="T_at:")
                print_tableau(result["T1"], label="T1:")
                print_tableau(result["T2"], label="T2:")
            print("-" * 30)
    print(f"Total RB pairs for n={n}: {count}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage:\n"
            "  python MRSK.py <n> [verbose]\n"
            "  python MRSK.py count <n> [verbose]\n"
            "  python MRSK.py wbeta \"(w1,w2,...)\" \"{i,j,...}\"\n"
            "  python MRSK.py wsigma \"(w1,w2,...)\" \"{i,j,...}\""
        )
    if sys.argv[1] == "count":
        if len(sys.argv) < 3:
            raise SystemExit("Usage: python MRSK.py count <n> [verbose]")
        n = int(sys.argv[2])
        verbose = int(sys.argv[3]) if len(sys.argv) > 3 else 1
        count_microlocal_packets(n, verbose)
    elif sys.argv[1] == "wbeta":
        if len(sys.argv) != 4:
            raise SystemExit("Usage: python MRSK.py wbeta \"(w1,w2,...)\" \"{i,j,...}\"")
        w = eval(sys.argv[2], {"__builtins__": {}})
        beta = eval(sys.argv[3], {"__builtins__": {}})
        if not isinstance(w, tuple):
            raise SystemExit("wbeta expects a tuple for w, e.g. \"(3,1,4,2)\"")
        if not isinstance(beta, set):
            raise SystemExit("wbeta expects a set for beta, e.g. \"{1,3}\"")
        result = mirabolic_rsk(w, beta)
        print("w,beta =", str_colored_partition(w, beta))
        print("nu =", result["nu"])
        print("theta =", result["theta"])
        print("nu' =", result["nu'"])
        print_tableau(result["T_at"], label="T_at:")
        print_tableau(result["T1"], label="T1:")
        print_tableau(result["T2"], label="T2:")

    elif sys.argv[1] == "wsigma":
        if len(sys.argv) != 4:
            raise SystemExit("Usage: python MRSK.py wsigma \"(w1,w2,...)\" \"{i,j,...}\"")
        w = eval(sys.argv[2], {"__builtins__": {}})
        sigma = eval(sys.argv[3], {"__builtins__": {}})
        if not isinstance(w, tuple):
            raise SystemExit("wsigma expects a tuple for w, e.g. \"(3,1,4,2)\"")
        if not isinstance(sigma, set):
            raise SystemExit("wsigma expects a set for sigma, e.g. \"{1,3}\"")
        result = mirabolic_rsk_from_sigma(w, sigma)
        print("w,sigma =", str_colored_partition(w, sigma))
        print("nu =", result["nu"])
        print("theta =", result["theta"])
        print("nu' =", result["nu'"])
        print_tableau(result["T_at"], label="T_at:")
        print_tableau(result["T1"], label="T1:")
        print_tableau(result["T2"], label="T2:")
    else:
        n = int(sys.argv[1])
        verbose = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        test_all_rb(n, verbose)
