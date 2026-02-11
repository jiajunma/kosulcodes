# PM(m,n) Dimension Formula (Induction)

## Files
- `pm_orbit_dimension.py` : main induction propagation for PM orbit dimensions (lengths)
- `verify_pp_pm_dimension.py` : cross-check with PP->PM and block count `|A(mu)|`
- `PM.py` : PM generation + type/companion rules
- `PP_to_PM.py` : PP(m,n) to PM(m,n) map
- `pp_block_subsets.py` : compute allowed block cells `A(mu)`
- `plot_block_configuration.py` : helper used by `pp_block_subsets.py`

## Run
```bash
python3 pm_orbit_dimension.py 3 4
python3 verify_pp_pm_dimension.py 3 4
```

Replace `3 4` by your `(m,n)`.
