from RB import generate_permutations, generate_all_beta, str_colored_partition


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
        "nu_prime": nu_prime,
    }


def generate_all_w_beta_pairs(n):
    for w in generate_permutations(n):
        for beta in generate_all_beta(w):
            yield w, beta


def test_all_rb(n, verbose=1):
    count = 0
    for w, beta in generate_all_w_beta_pairs(n):
        count += 1
        result = mirabolic_rsk(w, beta)
        if verbose >= 2:
            print("w,beta =", str_colored_partition(w, beta))
            print("nu =", result["nu"])
            print("theta =", result["theta"])
            print("nu_prime =", result["nu_prime"])
            print("-" * 30)
    print(f"Total RB pairs for n={n}: {count}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python MRSK.py <n> [verbose]")
    n = int(sys.argv[1])
    verbose = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    test_all_rb(n, verbose)
