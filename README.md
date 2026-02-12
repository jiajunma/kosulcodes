# Theta/Bessel Package (for collaborator)

This folder collects the scripts we discussed, focused on:
- Theta-side Hecke/KL/W-graph code
- Updated PM Fourier convention (our convention)
- RB -> PM / RB -> Theta maps, including Bessel-restricted map

## 1) Theta W-graph
- `WGraphTheta2.py` : canonical-basis/action-based W-graph
- `WGraphTheta3.py` : mu + descent-difference W-graph
- `WGraphThetaUtils.py` : shared utilities

Run:
```bash
python3 WGraphTheta2.py 2 3 WGraphTheta2.svg
python3 WGraphTheta3.py 2 3 WGraphTheta3.svg
```

## 2) Theta Hecke/KL/bar
- `Hecketheta.py`
- `verify_bar_theta.py`
- `verify_canonical_theta.py`
- `PM.py`, `PM_Bruhat.py`
- `PP_to_PM.py`, `PP_downset.py`, `pp_block_subsets.py`
- `pm_orbit_dimension.py`, `verify_pp_pm_dimension.py`

## 3) Fourier convention (updated)
- `Maze.py`

Current `Fourier(...)` in `Maze.py` is **our convention**:
\[
\mathrm{Fourier} = r \circ F_{\text{base}} \circ r, \quad r(i,j)=(i,n+1-j).
\]

## 4) RB -> PM / RB -> Theta maps
- `rb_to_pm_table.py`
  - current RB -> PM table (our convention)
- `rb_to_pm_fourier_table.py`
  - current RB -> PM, then PM Fourier
- `rb_nminus1_n_to_pm_table.py`
  - map from `RB_{n-1,n}` to `PM(n-1,n)` via Fourier and row-shift `i -> i-1`
- `rb_n_np1_to_pm_table.py`
  - helper script for `RB_{n,n+1}` experiments

Useful runs:
```bash
python3 rb_to_pm_table.py 3
python3 rb_to_pm_fourier_table.py 3
python3 rb_nminus1_n_to_pm_table.py 5
```

## 5) Bessel-side file used in comparison
- `WGraphBessel2.py`
- `HeckeBessel.py`

Run:
```bash
python3 WGraphBessel2.py -n 3 -o WGraphBessel2.svg
```

## Notes
- All scripts here are kept in the current "our convention" branch state.
- If you want a zipped copy of this folder, run:
```bash
zip -r share_theta_bessel_collab.zip share_theta_bessel_collab
```
