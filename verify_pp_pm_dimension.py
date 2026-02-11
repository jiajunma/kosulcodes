import argparse

from PP_to_PM import generate_all_pp, pp_to_pm
from pp_block_subsets import allowed_cells_for_mu
from PM import generate_all_pm
from pm_orbit_dimension import basic_seed_lengths, build_type_constraints, propagate_lengths


def verify_pp_pm_dimension(m, n, max_print=20):
    # Compute PM dimensions by propagation.
    pms = generate_all_pm(m, n)
    seeds = basic_seed_lengths(m, n)
    constraints = build_type_constraints(pms, m, n)
    ell, unresolved, conflicts = propagate_lengths(pms, seeds, constraints)

    if unresolved:
        raise RuntimeError(f"PM propagation unresolved elements: {len(unresolved)}")
    if conflicts:
        raise RuntimeError(f"PM propagation has conflicts: {len(conflicts)}")

    # Compare with direct |A(mu)| for each mu in PP(m,n).
    bad = []
    checked = 0
    for mu in generate_all_pp(m, n):
        I, J = mu
        sigma = pp_to_pm(m, n, I, J)
        direct = len(allowed_cells_for_mu(m, n, I, J, rule="le"))
        via_pm = ell[sigma]
        checked += 1
        if direct != via_pm:
            bad.append((mu, sigma, direct, via_pm))

    print(f"PP({m},{n}) size = {checked}")
    print(f"PM({m},{n}) size = {len(pms)}")
    print(f"Mismatches = {len(bad)}")
    if bad:
        print("\nSample mismatches:")
        for mu, sigma, direct, via_pm in bad[:max_print]:
            print(f"  mu={mu} -> sigma={sigma}, |A(mu)|={direct}, ell_PM={via_pm}")
    return len(bad) == 0


def main():
    parser = argparse.ArgumentParser(
        description="Verify |A(mu)| equals PM propagated dimension after mu->sigma mapping."
    )
    parser.add_argument("m", type=int)
    parser.add_argument("n", type=int)
    parser.add_argument("--max-print", type=int, default=20)
    args = parser.parse_args()

    ok = verify_pp_pm_dimension(args.m, args.n, max_print=args.max_print)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
