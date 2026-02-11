from RB import generate_permutations, generate_all_beta, beta_to_sigma, str_colored_partition
from Maze import (
    RB_to_maze,
    maze_to_rook_board,
    str_R,
    str_maze,
    count_R,
    generate_R,
    format_R,
)


def count_RB_to_rook_board_bijection(n, verbose=False, max_print=5):
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
            rook_from_rb.add(frozenset(maze_to_rook_board(maze)))
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Check RB -> maze -> rook_board bijection for size n."
    )
    parser.add_argument("-n", type=int, required=True, help="Size parameter n")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--max-print",
        type=int,
        default=5,
        help="Max missing rook boards to print on failure",
    )
    args = parser.parse_args()

    count_RB_to_rook_board_bijection(
        args.n, verbose=args.verbose, max_print=args.max_print
    )
