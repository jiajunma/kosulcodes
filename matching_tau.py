from RB import (
    generate_permutations,
    generate_all_beta,
    beta_to_sigma,
    str_colored_partition,
    tilde_inverse_sigma,
    normalize_key,
)
from HeckeRB import HeckeRB
from PM import descent_set_pm, action_pm_left, action_pm_right
from perm import longest_element
from Maze import (
    RB_to_maze,
    maze_to_rook_board,
    str_R,
    str_maze,
    count_R,
    generate_R,
    format_R,
    RB_to_R,
)


def count_RB_to_rook_board_bijection(n, verbose=False, max_print=100):
    """
    Count sizes to show RB -> maze -> rook_board is bijective for size n.
    Print the map RB --> maze --> rook_board.
    Returns True if counts match and the map hits all rook boards.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    rook_from_rb = set()
    rb_count = 0
    for w in generate_permutations(n):
        for beta in generate_all_beta(w):
            rb_count += 1
            maze = RB_to_maze(w, beta, n, n)
            rb = frozenset(maze_to_rook_board(maze))
            rook_from_rb.add(rb)
            sigma = beta_to_sigma(w, beta)
            print(
                f"(w,beta)={str_colored_partition(w, beta)} --> "
                f"(w,sigma)={str_colored_partition(w, sigma)} --> rook_board\n"
                f"{str_R(frozenset(maze_to_rook_board(maze)))}"
            )
            print(str_maze(maze))
             
    rook_count = count_R(n, n)
    image_count = len(rook_from_rb)

    ok = (rb_count == rook_count == image_count)

    if verbose or not ok:
        print(f"RB elements count: {rb_count}")
        print(f"Rook boards count: {rook_count}")
        print(f"Image size (RB -> maze -> R): {image_count}")
        print("Count check:", "PASS" if ok else "FAIL")
        if not ok and max_print > 0:
            missing = [rb for rb in generate_R(n, n) if frozenset(rb) not in rook_from_rb]
            if missing:
                print(f"Missing rook boards (showing up to {max_print}):")
                for rb in missing[:max_print]:
                    print(f"  {format_R(rb)}")

    return ok


def _descents_wbeta(hrb, w, beta):
    sigma = beta_to_sigma(w, beta)
    key = normalize_key(w, sigma)
    d_right = {i for i in range(1, hrb.n) if hrb.is_in_Phi_i(key, i)}
    w_inv, sigma_inv = tilde_inverse_sigma(w, sigma)
    key_tau = normalize_key(w_inv, sigma_inv)
    d_left = {i for i in range(1, hrb.n) if hrb.is_in_Phi_i(key_tau, i)}
    return d_left, d_right


def _complement(s, n):
    return set(range(1, n)).difference(s)


def check_descent_complements(n, verbose=False, max_print=5):
    """
    Compare descent sets of (w,beta) with complements of descent sets of rb,
    rb*w0, and w0*rb. Returns True if all checks pass.
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")

    hrb = HeckeRB(n)
    w0 = longest_element(n)

    failures = []
    total = 0
    counts_left = {"rb": 0, "w0*rb": 0, "rb*w0": 0}
    counts_right = {"rb": 0, "w0*rb": 0, "rb*w0": 0}
    for w in generate_permutations(n):
        for beta in generate_all_beta(w):
            total += 1
            dL_wb, dR_wb = _descents_wbeta(hrb, w, beta)

            rb = RB_to_R(w, beta)
            dL_rb, dR_rb = descent_set_pm(rb, n, n)
            dL_rb = set(dL_rb)
            dR_rb = set(dR_rb)

            rb_left = action_pm_left(rb, w0)
            dL_rb_left, dR_rb_left = descent_set_pm(rb_left, n, n)
            dL_rb_left = set(dL_rb_left)
            dR_rb_left = set(dR_rb_left)

            rb_right = action_pm_right(rb, w0)
            dL_rb_right, dR_rb_right = descent_set_pm(rb_right, n, n)
            dL_rb_right = set(dL_rb_right)
            dR_rb_right = set(dR_rb_right)

            print(f"(w,beta)={str_colored_partition(w, beta)}")
            print(f"  rb: {format_R(rb)}")
            print(f"  dL(w,beta)={sorted(dL_wb)} dR(w,beta)={sorted(dR_wb)}")
            print(f"  dL(rb)={sorted(dL_rb)} dR(rb)={sorted(dR_rb)}")
            print(f"  dL(rb*w0)={sorted(dL_rb_right)} dR(rb*w0)={sorted(dR_rb_right)}")
            print(f"  dL(w0*rb)={sorted(dL_rb_left)} dR(w0*rb)={sorted(dR_rb_left)}")

            if dL_wb == _complement(dL_rb, n) :
                counts_left["rb"] += 1
            if dR_wb == _complement(dR_rb, n) :
                counts_right["rb"] += 1
            if dL_wb == _complement(dL_rb_left, n) :
                counts_left["w0*rb"] += 1
            if dR_wb == _complement(dR_rb_left, n) :
                counts_right["w0*rb"] += 1
            if dL_wb == _complement(dL_rb_right, n) :
                counts_left["w0*rb"] += 1
            if dL_wb == _complement(dL_rb_right, n) and dR_wb == _complement(dR_rb_right, n) :
                counts_left["w0*rb"] += 1
        
    print(f"rb: {counts_left['rb']}/{total} {counts_right['rb']}/{total}")
    print(f"w0*rb: {counts_left['w0*rb']}/{total} {counts_right['w0*rb']}/{total}")
    print(f"rb*w0: {counts_left['rb*w0']}/{total} {counts_right['rb*w0']}/{total}")
    return len(failures) == 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Utilities for RB/maze/rook-board checks."
    )
    parser.add_argument("-n", type=int, required=True, help="Size parameter n")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--max-print",
        type=int,
        default=5,
        help="Max missing rook boards to print on failure",
    )
    parser.add_argument(
        "--check-descents",
        action="store_true",
        help="Check descent complement relations for (w,beta) vs rook boards.",
    )
    args = parser.parse_args()

    if args.check_descents:
        check_descent_complements(args.n, verbose=args.verbose, max_print=args.max_print)
    else:
        count_RB_to_rook_board_bijection(
            args.n, verbose=args.verbose, max_print=args.max_print
        )
