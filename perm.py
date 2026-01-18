import itertools

def generate_permutations(n):
    """
    Generate all permutations of 1..n as tuples.

    Args:
        n: integer, the size of permutation

    Yields:
        tuple of int, each permutation of 1..n

    Examples:
        >>> list(generate_permutations(2))
        [(1, 2), (2, 1)]
    """
    yield from itertools.permutations(range(1, n+1))


def length_of_permutation(w):
    """
    Compute the length (inversion number) of a permutation w.

    Args:
        w: A permutation tuple representing the permutation (1-based values)

    Returns:
        The number of inversions in w.

    Examples:
        >>> length_of_permutation((3, 2, 1))
        3
        >>> length_of_permutation((1, 2, 3))
        0
    """
    n = len(w)
    inversions = 0
    for i in range(n):
        for j in range(i+1, n):
            if w[i] > w[j]:
                inversions += 1
    return inversions


def inversions(w):
    """
    Backward-compatible alias for length_of_permutation.

    Examples:
        >>> inversions((2, 1, 3))
        1
    """
    return length_of_permutation(w)

def permutation_prod(w1, w2):
    """
    Return the composition (product) of two permutations w1 and w2.

    Both are tuples of length n, representing permutations of 1..n.
    Computes w1∘w2, i.e., result[i] = w1[w2[i]-1]

    Args:
        w1: The first permutation (tuple of length n, 1-based values).
        w2: The second permutation (tuple of length n, 1-based values).

    Returns:
        A tuple of length n representing the composition permutation.

    Examples:
        >>> permutation_prod((2, 3, 1), (3, 1, 2))
        (1, 2, 3)
    """
    assert len(w1) == len(w2), "Permutations must have the same length"
    n = len(w1)
    return tuple(w1[w2[i] - 1] for i in range(n))
    

def transposition(i, j, n):
    """
    Return the transposition of i and j in the permutation of 1..n.

    Args:
        i: The first position (1-based)
        j: The second position (1-based)
        n: The size of the permutation

    Returns:
        A permutation tuple of length n representing the transposition.

    Examples:
        >>> transposition(2, 4, 4)
        (1, 4, 3, 2)
    """
    return tuple(i if x == j else j if x == i else x for x in range(1, n+1))

def simple_reflection(i, n):
    """
    Return the simple reflection of i in the permutation of 1..n.

    Returns:
        A permutation tuple of length n representing the reflection.

    Examples:
        >>> simple_reflection(2, 5)
        (1, 3, 2, 4, 5)
    """
    return transposition(i, i+1, n)

def inverse_permutation(w):
    """
    Return the inverse of a permutation w.

    Args:
        w: A permutation tuple of length n, 1-based values.

    Returns:
        A permutation tuple of length n representing w^{-1}, so that w[w^{-1}[i]-1] = i+1.

    Examples:
        >>> inverse_permutation((3, 1, 2))
        (2, 3, 1)
        >>> w = (4, 2, 5, 1, 3)
        >>> inv = inverse_permutation(w)
        >>> permutation_prod(w, inv)
        (1, 2, 3, 4, 5)
    """
    n = len(w)
    inv = [0] * n
    for i, val in enumerate(w):
        inv[val - 1] = i + 1
    return tuple(inv)


def is_in_right_descent_set(w, i):
    """
    Test if i is in the right descent set of permutation w (1-based index).

    Args:
        w: Tuple/list representing the permutation of 1..n
        i: Position (1-based) to test for right descent

    Returns:
        True if i is a right descent (i.e., w[i-1] > w[i]), False otherwise.
        Raises ValueError if i is out of range.

    Examples:
        >>> is_in_right_descent_set((3, 1, 2), 1)
        True
        >>> is_in_right_descent_set((3, 1, 2), 2)
        False
        >>> is_in_right_descent_set((3, 1, 2), 3)
        Traceback (most recent call last):
        ...
        ValueError: index i=3 is out of bounds for permutation of length n=3
    """
    n = len(w)
    if i < 1 or i >= n:
        raise ValueError(f"index i={i} is out of bounds for permutation of length n={n}")
    return w[i-1] > w[i]

def is_in_left_descent_set(w, i):
    """
    Test if i is in the left descent set of permutation w (1-based index).
    The left descent set is the right descent set of the inverse permutation w^{-1}.

    Examples:
        >>> is_in_left_descent_set((3, 1, 2), 1)
        False
    """
    return is_in_right_descent_set(inverse_permutation(w), i)

def right_descent_set(w):
    """
    Compute the right descent set of permutation w (1-based index).
    The right descent set is { i in 1..n-1 | w[i-1] > w[i] }.

    Here i corresponds to the simple reflection (i, i+1).  

    Examples:
        >>> right_descent_set((3, 1, 2))
        {1}
    """
    return {i for i in range(1, len(w)) if is_in_right_descent_set(w, i)}

def left_descent_set(w):
    """
    The left descent set is the right descent set of the inverse permutation w^{-1}.

    Examples:
        >>> left_descent_set((3, 1, 2))
        {2}
    """
    return right_descent_set(inverse_permutation(w))

def right_descending_element(w):
    """
    Return an element of the right descending set of w.

    Args:
        w: A permutation tuple or list.

    Returns:
        An integer in the right descending set of w, or None if w is the identity permutation.

    Examples:
        >>> right_descending_element((3, 1, 2))
        1
        >>> right_descending_element((1, 3, 2))
        2
        >>> right_descending_element((1, 2, 3))
    """
    n = len(w)
    for j in range(n - 1):
        if w[j] > w[j + 1]:
            return j+1
    return None

def reduced_word(w):
    """
    Return a reduced word (as simple reflections) for permutation w.

    Examples:
        >>> w = (3, 1, 4, 2)
        >>> word = reduced_word(w)
        >>> perm = (1, 2, 3, 4)
        >>> for s in word:
        ...     perm = permutation_prod(s, perm)
        >>> perm == w
        True
    """
    w = list(w)
    n = len(w)
    word = []
    # Bubble-sort to identity, recording adjacent swaps.
    for _ in range(n):
        swapped = False
        for j in range(n - 1):
            if w[j] > w[j + 1]:
                w[j], w[j + 1] = w[j + 1], w[j]
                word.append(simple_reflection(j + 1, n))
                swapped = True
        if not swapped:
            break
    return word


def is_bruhat_leq(u, v):
    """
    Decide whether permutation u is <= v in the (strong) Bruhat order.

    This uses the rank-matrix criterion:
    For all i,j in [n], define x[i,j] = |{a <= i : x(a) >= j}|.
    Then u <= v iff u[i,j] <= v[i,j] for all i,j.

    Args:
        u: Permutation as a tuple or list, length n.
        v: Permutation as a tuple or list, length n.

    Returns:
        True if u <= v in the (strong) Bruhat order, else False.

    Examples:
        >>> is_bruhat_leq((1,2,3), (3,2,1))
        True
        >>> is_bruhat_leq((2,1,3), (3,1,2))
        True
        >>> is_bruhat_leq((3,1,2), (2,3,1))
        False
    """
    key = (tuple(u), tuple(v))
    cache = getattr(is_bruhat_leq, "_cache", None)
    if cache is None:
        cache = {}
        setattr(is_bruhat_leq, "_cache", cache)
    if key in cache:
        return cache[key]
    n = len(u)
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            u_count = sum(1 for x in u[:i] if x >= j)
            v_count = sum(1 for x in v[:i] if x >= j)
            if u_count > v_count:
                cache[key] = False
                return False
    cache[key] = True
    return True


if __name__ == "__main__":
    # Quick test of reduced_word using the functions above
    w = (3, 1, 4, 2)
    n = len(w)
    word = reduced_word(w)
    print("Permutation w:", w)
    print("Reduced word:")
    for s in word:
        print(s)
    # Test that applying word to identity gives w
    perm = tuple(range(1, n+1))
    for s in word:
        perm = permutation_prod(s, perm)
    print("Product of reduced word (should be w):", perm)
    print("Test passed:", perm == w)




