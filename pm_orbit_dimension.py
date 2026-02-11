import argparse
from collections import deque

from PM import generate_all_pm, get_type_companion, all_special_sigmas, special_sigma_i
from pp_block_subsets import allowed_cells_for_mu


def mu_i_for_special_sigma(m, n, i):
    if i < 0 or i > min(m, n):
        raise ValueError(f"Require 0 <= i <= min(m,n), got i={i}, m={m}, n={n}")
    I = tuple(range(m - i + 1, m + 1))
    J = tuple(range(1, i + 1))
    return (I, J)


def basic_seed_lengths(m, n):
    """
    For special sigma_i, length = |A(mu_i)| with A from block configuration.
    """
    seeds = {}
    for i in range(0, min(m, n) + 1):
        sigma_i = special_sigma_i(m, n, i)
        I, J = mu_i_for_special_sigma(m, n, i)
        a_mu = allowed_cells_for_mu(m, n, I, J, rule="le")
        seeds[sigma_i] = len(a_mu)
    return seeds


def build_type_constraints(pms, m, n):
    """
    Build directed constraints:
      geometric convention:
      if type is U+,  ell(comp)=ell(pm)-1
      if type is U-,  ell(comp)=ell(pm)+1
      if type is G,   ell(comp)=ell(pm)
    """
    constraints = []
    for pm in pms:
        for side in ("left", "right"):
            upper = m if side == "left" else n
            for i in range(1, upper):
                typ, companions = get_type_companion(pm, side, i)
                comp = companions[1]
                if typ == "U+":
                    delta = -1
                elif typ == "U-":
                    delta = 1
                elif typ == "G":
                    delta = 0
                else:
                    continue
                constraints.append((pm, comp, delta, side, i, typ))
    return constraints


def propagate_lengths(pms, seeds, constraints):
    """
    Propagate lengths through weighted relations ell(v)=ell(u)+delta.
    """
    adj = {}
    for pm in pms:
        adj[pm] = []
    for u, v, d, side, i, typ in constraints:
        adj[u].append((v, d, side, i, typ))
        adj[v].append((u, -d, side, i, typ))

    ell = dict(seeds)
    q = deque(seeds.keys())
    conflicts = []

    while q:
        u = q.popleft()
        for v, d, side, i, typ in adj[u]:
            cand = ell[u] + d
            if v not in ell:
                ell[v] = cand
                q.append(v)
            elif ell[v] != cand:
                conflicts.append(
                    {
                        "u": u,
                        "v": v,
                        "known_v": ell[v],
                        "candidate_v": cand,
                        "delta": d,
                        "side": side,
                        "i": i,
                        "type": typ,
                    }
                )

    unresolved = [pm for pm in pms if pm not in ell]
    return ell, unresolved, conflicts


def fmt_pm(pm):
    pairs = sorted(pm)
    return "{" + ", ".join(f"({a},{b})" for a, b in pairs) + "}"


def main():
    parser = argparse.ArgumentParser(
        description="Compute PM orbit dimensions (lengths) from basic seeds via type propagation."
    )
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("--max-print", type=int, default=80)
    parser.add_argument("--show-seeds", action="store_true")
    parser.add_argument(
        "--shift-min-to-zero",
        action="store_true",
        help="Shift all computed lengths by a constant so the global minimum becomes 0.",
    )
    args = parser.parse_args()

    m, n = args.m, args.n
    if m < 0 or n < 0:
        raise ValueError(f"Require m,n >= 0, got m={m}, n={n}")

    pms = generate_all_pm(m, n)
    seeds = basic_seed_lengths(m, n)
    constraints = build_type_constraints(pms, m, n)
    ell, unresolved, conflicts = propagate_lengths(pms, seeds, constraints)
    shift = 0
    if ell and args.shift_min_to_zero:
        shift = -min(ell.values())
        ell = {k: v + shift for k, v in ell.items()}

    print(f"PM({m},{n}) size = {len(pms)}")
    print(f"Basic seed count = {len(seeds)}")
    if args.show_seeds:
        for i in range(0, min(m, n) + 1):
            s = special_sigma_i(m, n, i)
            print(f"  sigma_{i}: ell={seeds[s]}, sigma={fmt_pm(s)}")

    print(f"Assigned lengths = {len(ell)}")
    print(f"Unresolved = {len(unresolved)}")
    print(f"Conflicts = {len(conflicts)}")
    if args.shift_min_to_zero:
        print(f"Applied shift = {shift}")

    if conflicts:
        print("\nSample conflicts:")
        for c in conflicts[: min(args.max_print, len(conflicts))]:
            print(
                f"  {fmt_pm(c['u'])} --({c['side']} s_{c['i']}, {c['type']}, d={c['delta']})--> "
                f"{fmt_pm(c['v'])}: known={c['known_v']}, candidate={c['candidate_v']}"
            )

    print("\nLengths:")
    sorted_pms = sorted(pms, key=lambda x: (len(x), sorted(x)))
    shown = 0
    for idx, pm in enumerate(sorted_pms, start=1):
        if pm in ell:
            print(f"{idx}. ell={ell[pm]:>2}  {fmt_pm(pm)}")
        else:
            print(f"{idx}. ell=??  {fmt_pm(pm)}")
        shown += 1
        if shown >= args.max_print and len(sorted_pms) > args.max_print:
            print(f"... ({len(sorted_pms) - args.max_print} more)")
            break

    raise SystemExit(0 if (not conflicts and not unresolved) else 1)


if __name__ == "__main__":
    main()
