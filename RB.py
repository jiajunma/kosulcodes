from itertools import permutations, chain, combinations
import csv

from perm import *



def normalize_key(w, sigma):
    return (tuple(w), frozenset(sigma))

def normalize_key_sigma(w, sigma):
    assert is_decreasing_on_subset(w, sigma), f"Sigma={sigma} does not satisfy the decreasing condition on {w}"
    return (tuple(w), frozenset(sigma))


def denormalize_key(key):
    w, beta = key
    return tuple(w), frozenset(beta)

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
    Generate colored permutation string for a permutation w.
    Elements at positions in I are colored red, others are colored blue.
    
    Args:
        w: A permutation (tuple or list)
        I: A set of positions to color red (1-indexed)
    
    Returns:
        A string with ANSI color codes showing the colored permutation
    """
    # ANSI color codes
    BLUE = '\033[94m'
    RED = '\033[91m'
    RESET = '\033[0m'
    
    n = len(w)
    result = []
    
    for i in range(1, n + 1):
        if i in I:
            result.append(f"{RED}{w[i-1]}{RESET}")
        else:
            result.append(f"{BLUE}{w[i-1]}{RESET}")
    
    return f"({', '.join(result)})"


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


def tilde_inverse(w, beta):
    """
    Return (w^{-1}, w(beta)) for w~ = (w, beta).
    """
    w_inv = inverse_permutation(w)
    beta_img = {w[i - 1] for i in beta}
    assert is_beta_on_subset(w_inv, beta_img), f"Beta {beta_img} does not satisfy the beta condition on w_inv={w_inv}"
    return w_inv, beta_img


def tilde_inverse_sigma(w, sigma):
    """
    first convert to beta notation
    Return (w^{-1}, w(beta)) for w~ = (w, beta).
    """
    beta = sigma_to_beta(w, sigma)
    w_inv = inverse_permutation(w)
    beta_img = {w[i - 1] for i in beta}
    assert is_beta_on_subset(w_inv, beta_img), f"Beta {beta_img} does not satisfy the beta condition on w_inv={w_inv}"
    sigma_img = beta_to_sigma(w_inv, beta_img)
    return w_inv, sigma_img


def fourier_transform(w, beta):
    """
    Compute the Fourier transform of (w, beta): 
    F(w~) = (w0 w w0, complement of w0(beta) in {1..n}).
    
    Args:
        w: A permutation (tuple or list) of length n.
        beta: A set of indices (1-based) representing a subset of {1..n}.

    Returns:
        (w0 * w * w0, {1,..,n} \ w0(beta)) as a tuple (permutation, set).
    """
    n = len(w)
    w0 = longest_element(n)
    # Compute w0 * w * w0
    w0ww0 = permutation_prod(w0, permutation_prod(w, w0))
    # Map beta through w0, collecting images
    beta_img = {w0[i - 1] for i in beta}
    beta_comp = set(range(1, n + 1)).difference(beta_img)
    return w0ww0, beta_comp


def right_action_rb(wtilde, x):
    """
    Right action of x in S_n on (w, beta): (w x, x(beta)).

    For simple reflection the action is given by:
    (w, beta) s  :=  (w s, s(beta))

    For general elements the action is given by:
    (w, beta) g  :=  (w g, g^{-1}(beta))

    """
    w, beta = wtilde
    x_inv = inverse_permutation(x)
    return permutation_prod(w, x), {x_inv[i - 1] for i in beta}

def root_type1(w, sigma):
    """
    Determine the root type for each simple reflection s_i in S_1 

    For each i in {1, ..., n-1}, classify (s_i, O_sigma) into U-/U+/T-/T+ and
    compute the corresponding s_i-companion(s).

    Args:
        w: A permutation (tuple or list)
        sigma: A set of positions (1-indexed)

    Returns:
        A dict mapping i -> {"type": str, "companions": list of (w, sigma_set)}
    """
    n = len(w)
    sigma_set = set(sigma)
    w_sigma_image = {w[i - 1] for i in sigma_set}
    w_inv = inverse_permutation(w)
    l_w = length_of_permutation(w)
    result = {}

    for i in range(1, n):
        s_i = transposition(i, i + 1, n)
        s_i_w = permutation_prod(s_i, w)
        l_siw = length_of_permutation(s_i_w)
        inter = w_sigma_image & {i, i + 1}

        if l_siw > l_w:
            if inter == {i} or not inter:
                result[i] = {
                    "type": "U-",
                    "companions": [(s_i_w, set(sigma_set))],
                }
            elif inter == {i + 1}:
                result[i] = {
                    "type": "T-",
                    "companions": [
                        (s_i_w, set(sigma_set)),
                        (s_i_w, set(sigma_set) | {w_inv[i - 1]}),
                    ],
                }
            else:
                raise ValueError(f"Unhandled case for i={i}, inter={inter}, l(s_i w)>l(w), in fact this case should not happen")
        elif l_siw < l_w:
            if inter == {i + 1} or not inter:
                result[i] = {
                    "type": "U+",
                    "companions": [(s_i_w, set(sigma_set))],
                }
            elif inter == {i}:
                result[i] = {
                    "type": "T-",
                    "companions": [
                        (s_i_w, set(sigma_set)),
                        (w, set(sigma_set) | {w_inv[i]}),
                    ],
                }
            elif inter == {i, i + 1}:
                new_sigma = set(sigma_set)
                new_sigma.discard(w_inv[i])
                result[i] = {
                    "type": "T+",
                    "companions": [
                        (w, set(new_sigma)),
                        (s_i_w, set(new_sigma)),
                    ],
                }
            else:
                raise ValueError(f"Unhandled case for i={i}, inter={inter}, l(s_i w)<l(w), in fact this case should not happen")
        else:
            raise ValueError(f"Unexpected length equality for i={i}")

    return result

def root_type2(w, sigma):
    """
    Determine the root type for each simple reflection s'_i in S_2.

    For each i in {1, ..., n-1}, classify (s'_i, O_sigma) into U-/U+/T-/T+
    and compute the corresponding s'_i-companion(s).

    Args:
        w: A permutation (tuple or list)
        sigma: A set of positions (1-indexed)

    Returns:
        A dict mapping i -> {"type": str, "companions": list of (w, sigma_set)}
    """
    n = len(w)
    sigma_set = set(sigma)
    l_w = length_of_permutation(w)
    result = {}

    for i in range(1, n):
        s_i = transposition(i, i + 1, n)
        w_s_i = permutation_prod(w, s_i)
        l_wsi = length_of_permutation(w_s_i)
        inter = sigma_set & {i, i + 1}
        sigma_swapped = {s_i[j - 1] for j in sigma_set}

        if l_wsi > l_w:
            if inter == {i} or not inter:
                result[i] = {
                    "type": "U-",
                    "companions": [(w_s_i, set(sigma_swapped))],
                }
            elif inter == {i + 1}:
                result[i] = {
                    "type": "T-",
                    "companions": [
                        (w_s_i, set(sigma_swapped)),
                        (w_s_i, set(sigma_swapped) | {i + 1}),
                    ],
                }
            else:
                raise ValueError(
                    f"Unhandled case for i={i}, inter={inter}, l(w s_i)>l(w), in fact this case should not happen"
                )
        elif l_wsi < l_w:
            if inter == {i + 1} or not inter:
                result[i] = {
                    "type": "U+",
                    "companions": [(w_s_i, set(sigma_swapped))],
                }
            elif inter == {i}:
                result[i] = {
                    "type": "T-",
                    "companions": [
                        (w_s_i, set(sigma_swapped)),
                        (w, set(sigma_swapped) | {i}),
                    ],
                }
            elif inter == {i, i + 1}:
                sigma_without_ip1 = set(sigma_set)
                sigma_without_ip1.discard(i + 1)
                sigma_without_i = set(sigma_set)
                sigma_without_i.discard(i)
                result[i] = {
                    "type": "T+",
                    "companions": [
                        (w, set(sigma_without_ip1)),
                        (w_s_i, set(sigma_without_i)),
                    ],
                }
            else:
                raise ValueError(
                    f"Unhandled case for i={i}, inter={inter}, l(w s_i)<l(w), in fact this case should not happen"
                )
        else:
            raise ValueError(f"Unexpected length equality for i={i}")

    return result

def root_type_right(w, sigma):
    """
    Determine the root type for each simple reflection s_i for the right action .

    For each i in {1, ..., n-1}, classify (s'_i, O_sigma) into U-/U+/T-/T+
    and compute the corresponding s'_i-companion(s).

    Args:
        w: A permutation (tuple or list)
        sigma: A set of positions (1-indexed)

    Returns:
        a list ->  (length of w)+ (i -> (type, list of companion (w, sigma_set)))
    """
    n = len(w)
    sigma_set = set(sigma)
    beta = sigma_to_beta(w, sigma)
    l_w = length_of_permutation(w)
    result = [None]* (n+1) 
    # Store the length of (w, beta) at the beginning of the list
    result[0] = l_w + len(beta)

    for i in range(1, n):
        s_i = transposition(i, i + 1, n)
        w_s_i = permutation_prod(w, s_i)
        inter = sigma_set & {i, i + 1}
        sigma_swapped = {s_i[j - 1] for j in sigma_set}

        C = [normalize_key_sigma(w,sigma)]
        T = ""
        # If l(w s_i) > l(w) <===> w(i)<w(i+1).  
        if w[i-1] < w[i]:
            if inter == {i} or not inter:
                T = "U-"
                C.append(normalize_key_sigma(w_s_i, sigma_swapped))
            elif inter == {i + 1}:
                T = "T-"
                C.append(normalize_key_sigma(w_s_i, sigma_swapped))
                C.append(normalize_key_sigma(w_s_i, sigma_swapped | {i + 1}))
            else:
                raise ValueError(
                    f"Unhandled case for i={i}, inter={inter}, l(w s_i)>l(w), in fact this case should not happen"
                )
        else:
            if inter == {i + 1} or not inter:
                T = "U+"
                C.append(normalize_key_sigma(w_s_i, sigma_swapped))
            elif inter == {i}:
                T = "T-"
                C.append(normalize_key_sigma(w_s_i, sigma_swapped))
                C.append(normalize_key_sigma(w, sigma_swapped | {i}))
            elif inter == {i, i + 1}:
                sigma_without_ip1 = set(sigma_set)
                sigma_without_ip1.discard(i + 1)
                sigma_without_i = set(sigma_set)
                sigma_without_i.discard(i)
                T = "T+"
                C.append(normalize_key_sigma(w, sigma_without_ip1))
                C.append(normalize_key_sigma(w_s_i, sigma_without_i))
            else:
                raise ValueError(
                    f"Unhandled case for i={i}, inter={inter}, l(w s_i)<l(w), in fact this case should not happen"
                )
        result[i] = (T, C)

    return result

def dim_Omega_w_beta(w, beta):
    """
    Compute the dimension of the orbit of (w, beta).

    Formula: N(N-1)/2 + l(w) + |beta|
    """
    n = len(w)
    return n * (n - 1) // 2 + length_of_permutation(w) + len(beta)

def dim_Omega_w_sigma(w, sigma):
    """
    Compute the dimension of the orbit of (w, sigma) 
    This is the same as the dimension of the orbit of (w, beta), but one computes beta from sigma first.
    Formula: N(N-1)/2 + l(w) + |beta|
    """
    beta = sigma_to_beta(w, sigma)
    return dim_Omega_w_beta(w, beta)

def pretty_print_root_type_results(w, sigma, root_type_func):
    """
    Nicely print the output of root_type1 or root_type2 for a given (w, sigma).

    Args:
        w: A permutation (tuple or list)
        sigma: A set or list of positions (1-indexed)
        root_type_func: Function, either root_type1 or root_type2

    Output:
        Prints a table/report to stdout.
    """
    result = root_type_func(w, sigma)
    n = len(w)
    print(f"(w,σ) = {str_colored_partition(w, sigma)}")
    print(f"Resulting types for each i in 1..{n-1}:")
    print("-" * 70)
    print(f"{'i':^3} | {'type':^8} | {'companions (w,sigma)':^50}")
    print("-" * 70)
    for i in range(1, n):
        typ = result[i]["type"]
        companions = result[i]["companions"]
        companions_colored = "; ".join(str_colored_partition(ww, ss) for (ww, ss) in companions)
        print(f"{i:^3} | {typ:^8} | {companions_colored}")
    print("-" * 70)


def _sigma_sort_key(sigma):
    return (len(sigma), tuple(sorted(sigma)))


def _format_tuple(values):
    return "(" + ",".join(str(x) for x in values) + ")"


def _format_set(values):
    if not values:
        return "{}"
    return "{" + ",".join(str(x) for x in sorted(values)) + "}"


def _format_w_sigma_label(w, sigma):
    return f"w={_format_tuple(w)};sigma={_format_set(sigma)}"

def _format_colored_w_label(w, sigma):
    return str_colored_partition(w, sigma)

def _format_colored_w_html(w, sigma):
    parts = []
    for i in range(1, len(w) + 1):
        val = w[i - 1]
        if i in sigma:
            parts.append(f"<span class=\"red\">{val}</span>")
        else:
            parts.append(f"<span class=\"blue\">{val}</span>")
    return "(" + ", ".join(parts) + ")"


def collect_w_sigma_pairs(n):
    """
    Collect all (w, sigma) pairs in a deterministic order.

    Args:
        n: The size of the permutation.

    Returns:
        List of (w, sigma) pairs.
    """
    pairs = []
    for w in generate_permutations(n):
        sigmas = list(generate_all_sigma(w))
        sigmas.sort(key=_sigma_sort_key)
        for sigma in sigmas:
            pairs.append((w, sigma))
    return pairs


def write_root_type_table(n, root_type_func, output_path):
    """
    Write a CSV table for root types.

    Columns: all (w, sigma) pairs.
    Rows: simple reflections s_i for i in 1..n-1.
    Cells: root type.

    Args:
        n: The size of the permutation.
        root_type_func: Function, either root_type1 or root_type2.
        output_path: Path to write the CSV file.
    """
    pairs = collect_w_sigma_pairs(n)
    headers = ["simple_reflection"] + [_format_colored_w_label(w, sigma) for w, sigma in pairs]
    rows = [[f"s_{i}"] for i in range(1, n)]

    for w, sigma in pairs:
        result = root_type_func(w, sigma)
        for i in range(1, n):
            rows[i - 1].append(result[i]["type"])

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def write_root_type_table_orbit_rows(n, output_path):
    """
    Write a CSV table for both root type 1 and 2.

    Rows: all (w, sigma) pairs (orbits).
    Columns: simple reflections s_i for i in 1..n-1, with type1/type2.
    Cells: root type.

    Args:
        n: The size of the permutation.
        output_path: Path to write the CSV file.
    """
    pairs = collect_w_sigma_pairs(n)
    headers = ["orbit"]
    for i in range(1, n):
        headers.append(f"s_{i}_type1")
        headers.append(f"s_{i}_type2")

    rows = []
    for w, sigma in pairs:
        result1 = root_type1(w, sigma)
        result2 = root_type2(w, sigma)
        row = [_format_colored_w_label(w, sigma)]
        for i in range(1, n):
            row.append(result1[i]["type"])
            row.append(result2[i]["type"])
        rows.append(row)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def write_root_type_table_orbit_rows_html(n, output_path):
    """
    Write an HTML table for both root type 1 and 2.

    Rows: all (w, sigma) pairs (orbits).
    Columns: simple reflections s_i for i in 1..n-1, with type1/type2.
    Cells: root type.

    Args:
        n: The size of the permutation.
        output_path: Path to write the HTML file.
    """
    pairs = collect_w_sigma_pairs(n)

    headers = ["orbit"]
    for i in range(1, n):
        headers.append(f"s_{i}_type1")
        headers.append(f"s_{i}_type2")

    rows = []
    for w, sigma in pairs:
        result1 = root_type1(w, sigma)
        result2 = root_type2(w, sigma)
        row = [_format_colored_w_html(w, sigma)]
        for i in range(1, n):
            row.append(result1[i]["type"])
            row.append(result2[i]["type"])
        rows.append(row)

    with open(output_path, "w") as f:
        f.write("<!doctype html>\n")
        f.write("<html>\n<head>\n<meta charset=\"utf-8\">\n")
        f.write("<style>\n")
        f.write("table{border-collapse:collapse;font-family:Arial, sans-serif;}\n")
        f.write("th,td{border:1px solid #999;padding:4px 8px;text-align:left;}\n")
        f.write(".red{color:#c00;}\n")
        f.write(".blue{color:#06c;}\n")
        f.write("</style>\n</head>\n<body>\n")
        f.write("<table>\n<thead>\n<tr>\n")
        for h in headers:
            f.write(f"<th>{h}</th>\n")
        f.write("</tr>\n</thead>\n<tbody>\n")
        for row in rows:
            f.write("<tr>\n")
            for cell in row:
                f.write(f"<td>{cell}</td>\n")
            f.write("</tr>\n")
        f.write("</tbody>\n</table>\n</body>\n</html>\n")



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
        if verbose >= 3:
            pretty_print_root_type_results(w, sigma, root_type1)
            pretty_print_root_type_results(w, sigma, root_type2)
    
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
    
    # Table mode: python3 RB.py <n> 4
    # Only mode=4 writes the combined table (orbit rows, simple reflections columns).
    if len(sys.argv) > 2:
        n = int(sys.argv[1])
        mode = int(sys.argv[2])
        if mode == 4:
            output_path = f"rb_table_root_types_n{n}.csv"
            write_root_type_table_orbit_rows(n, output_path)
            print(f"Wrote table: {output_path}")
            sys.exit(0)
        if mode == 5:
            output_path = f"rb_table_root_types_n{n}.html"
            write_root_type_table_orbit_rows_html(n, output_path)
            print(f"Wrote table: {output_path}")
            sys.exit(0)

    # Default behavior
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 4

    if len(sys.argv) > 2:
        verbose = int(sys.argv[2])
    else:
        verbose = 1

    verify_sigma_beta_equality(n, verbose)
    print_all_w_sigma_pairs(n, verbose)
