# Licence for text, figures and data

Copyright (c) 2026 Yanwei Wang

The following material in this repository is licensed under the
[Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/):

- the explanatory text (Markdown cells) of the notebooks in `notebooks/` and `solutions/`;
- the figures and other outputs stored in those notebooks;
- `README.md` and the pages in `docs/`.

You are free to share and adapt this material for any purpose, provided that
you give appropriate credit, provide a link to the licence, and indicate if
changes were made. A suggested form of credit is given in `CITATION.cff`.

The source code (`engml_2_applied/`, `tests/`, `scripts/`, and the code cells of the
notebooks) is licensed separately under the MIT licence; see `LICENSE`.

All numerical results in the notebooks are computed within the notebooks
themselves, and all data sets used in the lectures of this volume are synthetic:
the analyses and yields of a methanol reactor, the flows of a reactor, separator
and recycle, the isotherms and adsorption curves, the calibration data with
spikes, the near-infrared spectra, the readings of correlated sensors, the points
that stand for phases, the conversion and rate-constant curves, the hardness
curve of a binary alloy and the data of the capstone are generated in the
notebooks from stated formulas and random-number seeds, with assumed
parameters, and are labelled as such. They are not measurements.

The only tabulated data are in `engml_2_applied/constants.py`: the gas constant, the
Boltzmann constant and the Avogadro constant (CODATA 2018 values), and a table of
16 compounds with their molecular weights, normal boiling points, critical
temperatures, critical pressures and acentric factors, which belongs to Volume I
and is used by the tests of the package. These are rounded values of the kind found
in standard compilations of the properties of pure compounds; they are facts and
are not subject to copyright, but they are given for teaching and should be checked
against a primary source before they are used in design. No third-party data sets
are included in this release.
