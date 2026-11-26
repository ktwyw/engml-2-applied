# Contributing

Corrections and suggestions are welcome through the
[issue tracker](https://github.com/ktwyw/engml-2-applied/issues).

1. **Reporting an error.** Give the lecture and section, what is stated, and what you believe is
   correct, with a short calculation if possible.
2. **The package.** Code in `engml_2_applied/` uses NumPy alone (Matplotlib only in `engml_2_applied/plotting.py`).
   A new function needs a docstring that states what it computes, a test in `tests/`, and, where an
   exact result exists, a check in `scripts/validate.py`.
3. **The notebooks** are built from plain-text sources and are not edited directly. The text and
   code of the lectures are in `scripts/lectures_00_05.py`, `lectures_06_11.py` and
   `lectures_12_17.py`; the figure cells are in `scripts/lecture_figures.py`. After a change, run

   ```bash
   python scripts/build_notebooks.py          # all 36 notebooks, each in a fresh kernel
   python scripts/build_notebooks.py 05 14    # or only some lectures
   ```

4. **Numbers in the text.** A number quoted in a Markdown cell must appear in the output of the
   code above it. If a change to the code changes an output, change the text that describes it.
5. **Before a pull request:**

   ```bash
   pip install -e ".[test,book]"
   pytest -q
   python scripts/validate.py                 # rewrites docs/validation.md
   python scripts/make_readme_gallery.py      # if a figure has changed
   ```
