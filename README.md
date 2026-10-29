# how-measures-lie

Code and figures for [How image measures lie quietly](https://www.parallaxmetrology.com/how_measures_lie.html),
an essay by Parallax Metrology: nineteen ways a hand-built image measure returns a plausible,
wrong number without raising an error, and the check that catches each one.

Thirteen of the nineteen failures are reproduced here on synthetic images, with deliberately
ordinary code (percentile stretch, percentile threshold, weighted centroid, capped peak finding,
phase-scrambled nulls), so that each figure shows the pattern rather than one implementation.

| # | Failure | Figure |
|---|---|---|
| 1 | The dead field | `figures/dead_field.png` |
| 2 | Stretched dust | `figures/stretched_dust.png` |
| 3 | The starved mask | `figures/starved_mask.png` |
| 4 | Ceilings that fake agreement | `figures/ceiling.png` |
| 5 | Ratios with tiny denominators | `figures/ratio.png` |
| 6 | Resolution is part of the measurement | `figures/resolution.png` |
| 7 | Minimum mark size | `figures/mark_size.png` |
| 10 | A control has free parameters (ablation fill) | `figures/ablation_fill.png` |
| 12 | Single random draws mislead | `figures/single_draw.png` |
| 13 | A hand-picked comparison makes a false extreme | `figures/foils.png` |
| 14 | Unseeded randomness | `figures/unseeded.png` |
| 15 | The default lens is blind to the construction | `figures/default_lens.png` |
| 16 | The measure reads the substrate | `figures/substrate.png` |

The other six (cleaning, records, control direction, situation not object, same name two
things, worked not contrasting) are inference failures that need a real painting to show; the
essay gives their cases from the Image Structure Reader's lab log.

## Rebuild

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy scikit-image Pillow matplotlib
.venv/bin/python demos/make_figures.py     # figures/*.png and figures/numbers.json
.venv/bin/python build_page.py             # how_measures_lie.html, numbers filled from the JSON
```

Entries 2, 3, 15 and 16 use the Notan mass from
[notan](https://github.com/parallaxmetrology/notan); the scripts expect it checked out
beside this folder (`../notan`).

Entry 14 is unseeded on purpose, so its numbers change on every run.

## Licence

Code: MIT (see [LICENSE](LICENSE)). The essay, README and figures: CC BY 4.0 (see
[LICENSE-CONTENT](LICENSE-CONTENT)).
