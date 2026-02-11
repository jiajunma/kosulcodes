import argparse
import itertools


def normalize_subset(values, upper):
    vals = sorted(set(values))
    if any(v < 1 or v > upper for v in vals):
        raise ValueError(f"subset value out of range [1,{upper}]: {vals}")
    return vals


def pp_to_pm(m, n, I, J):
    """
    Lemma D.9 (stack algorithm):
      input  mu=(I,J), I={i1<...<ik} subset [m], J={j1<...<jk} subset [n]
      output sigma in PM(m,n)
    """
    I = normalize_subset(I, m)
    J = normalize_subset(J, n)
    if len(I) != len(J):
        raise ValueError(f"Require |I|=|J|, got |I|={len(I)}, |J|={len(J)}")

    k = len(I)
    if k == 0:
        return frozenset()

    sigma = set()
    stack = []
    j_prev = 0

    for p in range(k):
        ip = I[p]
        jp = J[p]
        i_next = I[p + 1] if p + 1 < k else (m + 1)

        # push j_{p-1}+1, ..., j_p into stack
        for y in range(j_prev + 1, jp + 1):
            stack.append(y)

        # for x in i_p, ..., i_{p+1}-1, pop and pair (x,y)
        for x in range(ip, i_next):
            if not stack:
                break
            y = stack.pop()
            sigma.add((x, y))

        j_prev = jp

    return frozenset(sigma)


def generate_all_pp(m, n):
    """
    Generate PP(m,n) = {(I,J) | I subset [m], J subset [n], |I|=|J|}.
    """
    for k in range(0, min(m, n) + 1):
        for I in itertools.combinations(range(1, m + 1), k):
            for J in itertools.combinations(range(1, n + 1), k):
                yield (tuple(I), tuple(J))


def parse_csv_ints(s):
    if s is None or s.strip() == "":
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Map PP(m,n) to PM(m,n) by the stack algorithm (Lemma D.9)."
    )
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("--I", type=str, default=None, help='CSV subset, e.g. "1,3,5"')
    parser.add_argument("--J", type=str, default=None, help='CSV subset, e.g. "2,4,7"')
    parser.add_argument("--all", action="store_true", help="Enumerate all PP(m,n) and map them.")
    args = parser.parse_args()

    m, n = args.m, args.n
    if m < 0 or n < 0:
        raise ValueError(f"Require m,n >= 0, got m={m}, n={n}")

    if args.all:
        items = list(generate_all_pp(m, n))
        print(f"PP({m},{n}) size = {len(items)}")
        for idx, (I, J) in enumerate(items, start=1):
            sigma = pp_to_pm(m, n, I, J)
            print(f"{idx}. (I={I}, J={J}) -> {sigma}")
        return

    if args.I is None or args.J is None:
        raise ValueError("Provide both --I and --J, or use --all")

    I = parse_csv_ints(args.I)
    J = parse_csv_ints(args.J)
    sigma = pp_to_pm(m, n, I, J)
    print(f"mu=(I={tuple(sorted(set(I)))}, J={tuple(sorted(set(J)))})")
    print(f"sigma={sigma}")


if __name__ == "__main__":
    main()
