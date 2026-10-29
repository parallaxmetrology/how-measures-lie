# Contributing

- **A failure we missed.** A way an image measure returned a plausible, wrong number in your own work, with the check that caught it. The bar is the same as for the existing entries: a real case, and an input whose right answer is not in doubt.
- **A better demonstration.** A smaller or clearer synthetic case for any entry, especially the six without a figure.
- **Corrections.** If a check is wrong, or too strict, open an issue with the counterexample.

## Rebuild

```bash
git clone https://github.com/parallaxmetrology/notan
git clone https://github.com/parallaxmetrology/how-measures-lie
cd how-measures-lie
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy scikit-image Pillow matplotlib
python demos/make_figures.py && python build_page.py
```

The demos expect `notan` checked out beside this folder (entries 2, 3, 15 and 16 use its Notan mass).

Be kind and specific. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
