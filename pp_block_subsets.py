import argparse
import itertools

from PP_to_PM import normalize_subset, parse_csv_ints
from plot_block_configuration import column_caps_from_mu


def allowed_cells_for_mu(m, n, I, J, rule="ge"):
    """
    Allowed box cells for block configuration of mu=(I,J):
      rule='le': A(mu) = {(lambda, j) | j <= cap(lambda)}
      rule='ge': A(mu) = {(lambda, j) | j >= cap(lambda)}
    where cap(lambda) comes from mu.
    """
    I = normalize_subset(I, m)
    J = normalize_subset(J, n)
    caps = column_caps_from_mu(m, n, I, J)
    cells = []
    for lam in range(1, m + 1):
        cap = caps[lam - 1]
        if rule == "le":
            row_iter = range(1, cap + 1)
        elif rule == "ge":
            # use the opposite side of staircase (user's convention)
            start = max(1, cap)
            row_iter = range(start, n + 1)
        else:
            raise ValueError(f"Unknown rule={rule}, expected 'le' or 'ge'")
        for j in row_iter:
            cells.append((lam, j))
    return cells


def to_row_col_cells(cells_lambda_j):
    """
    Convert (lambda, j) to paper-style (j, i) = (row, col).
    """
    return [(j, lam) for (lam, j) in cells_lambda_j]


def iter_all_subsets(items):
    """
    Yield all subsets of items as tuples, in size-then-lex order.
    """
    for r in range(0, len(items) + 1):
        for comb in itertools.combinations(items, r):
            yield comb


def main():
    parser = argparse.ArgumentParser(
        description="Given mu in PP(m,n), list allowed box cells and subsets inside the block configuration."
    )
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("--I", help='CSV subset, e.g. "2,4"')
    parser.add_argument("--J", help='CSV subset, e.g. "1,3"')
    parser.add_argument(
        "--i",
        type=int,
        default=None,
        help="Use special mu_i with I={m-i+1,...,m}, J={1,...,i}.",
    )
    parser.add_argument(
        "--enumerate",
        action="store_true",
        help="Enumerate all subsets of allowed cells (can be huge).",
    )
    parser.add_argument(
        "--max-cells-for-enum",
        type=int,
        default=20,
        help="Safety threshold for --enumerate. Override with --force.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force enumeration even if too many allowed cells.",
    )
    parser.add_argument(
        "--max-print",
        type=int,
        default=200,
        help="Max subsets to print (when --enumerate).",
    )
    parser.add_argument(
        "--rule",
        choices=["ge", "le"],
        default="le",
        help="Block side convention: ge means j>=cap(lambda), le means j<=cap(lambda).",
    )
    args = parser.parse_args()

    m, n = args.m, args.n
    if m <= 0 or n <= 0:
        raise ValueError(f"Require m,n >= 1, got m={m}, n={n}")

    if args.i is not None:
        i = args.i
        if i < 0 or i > min(m, n):
            raise ValueError(f"Require 0 <= i <= min(m,n), got i={i}, m={m}, n={n}")
        I = list(range(m - i + 1, m + 1))
        J = list(range(1, i + 1))
    else:
        if args.I is None or args.J is None:
            raise ValueError("Provide either --i, or both --I and --J.")
        I = parse_csv_ints(args.I)
        J = parse_csv_ints(args.J)
    I = normalize_subset(I, m)
    J = normalize_subset(J, n)
    if len(I) != len(J):
        raise ValueError(f"Require |I|=|J|, got |I|={len(I)}, |J|={len(J)}")

    cells_lambda_j = allowed_cells_for_mu(m, n, I, J, rule=args.rule)
    cells = to_row_col_cells(cells_lambda_j)
    cell_count = len(cells)
    total_subsets = 1 << cell_count

    print(f"mu = (I={tuple(I)}, J={tuple(J)}) in PP({m},{n})")
    print(f"rule = {args.rule}")
    print(f"Allowed cells A(mu) size = {cell_count}")
    print(f"A(mu) in paper coordinates (j,i) = {cells}")
    print(f"Number of subsets in block space = 2^{cell_count} = {total_subsets}")

    if not args.enumerate:
        return

    if cell_count > args.max_cells_for_enum and not args.force:
        print(
            f"\nSkip enumeration: |A(mu)|={cell_count} > {args.max_cells_for_enum}. "
            f"Use --force to override."
        )
        return

    print("\nEnumerating subsets:")
    printed = 0
    for idx, subset in enumerate(iter_all_subsets(cells), start=1):
        print(f"{idx}. {subset}")
        printed += 1
        if printed >= args.max_print:
            print(f"... stopped at {printed} lines (set --max-print for more)")
            break


if __name__ == "__main__":
    main()
