from perm import length_of_permutation


def row_insert(tableau, x):
    """
    Insert x into tableau using row insertion.

    Args:
        tableau: list of rows (each row is a list of ints, increasing).
        x: integer to insert.

    Returns:
        The row index where the new box was appended.
    """
    for row_idx, row in enumerate(tableau):
        for col_idx, y in enumerate(row):
            if x < y:
                row[col_idx], x = x, y
                break
        else:
            row.append(x)
            return row_idx
    tableau.append([x])
    return len(tableau) - 1


def rsk(w):
    """
    Compute the RSK insertion and recording tableaux for a permutation/word w.

    Args:
        w: iterable of integers (typically a permutation of 1..n)

    Returns:
        (P, Q) where P and Q are tableaux represented as list of rows.

    Examples:
        >>> rsk([3, 1, 4, 2])
        ([[1, 2], [3, 4]], [[1, 3], [2, 4]])
        >>> rsk([2, 1, 3])
        ([[1, 3], [2]], [[1, 3], [2]])
    """
    P = []
    Q = []
    for idx, x in enumerate(w, start=1):
        row_idx = row_insert(P, x)
        while len(Q) <= row_idx:
            Q.append([])
        Q[row_idx].append(idx)
    return P, Q


def partition_from_tableau(tableau):
    """
    Return the partition (row lengths) from a tableau.
    """
    return [len(row) for row in tableau]


def tableau_to_string(tableau):
    """
    Format a tableau as aligned rows for printing.
    """
    if not tableau:
        return "(empty)"
    width = max(len(str(x)) for row in tableau for x in row)
    return "\n".join(" ".join(f"{x:>{width}}" for x in row) for row in tableau)


def print_tableau(tableau, label=None):
    """
    Print a tableau with an optional label.
    """
    if label:
        print(label)
    print(tableau_to_string(tableau))


def column_lengths_from_partition(partition):
    """
    Compute column lengths from a partition given as row lengths.
    """
    if not partition:
        return []
    max_len = max(partition)
    return [sum(1 for row_len in partition if row_len >= j) for j in range(1, max_len + 1)]


def sum_of_column_squares(partition):
    """
    Sum of squares of column lengths of a partition.
    """
    cols = column_lengths_from_partition(partition)
    return sum(c * c for c in cols)


def rsk_partition(w):
    """
    Compute the RSK partition (shape) for w.
    """
    P, _ = rsk(w)
    return partition_from_tableau(P)


def inversion_and_column_square_check(w):
    """
    Compute inversion number of w and sum of column squares of the RSK partition.

    Returns:
        (inv_count, col_sq_sum, equal_flag)
    """
    inv = length_of_permutation(tuple(w))
    partition = rsk_partition(w)
    col_sq = sum_of_column_squares(partition)
    return inv, col_sq, inv == col_sq


if __name__ == "__main__":
    import doctest

    doctest.testmod()