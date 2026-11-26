# A 14-week syllabus

The 18 lectures fit a semester of 14 weeks with one lecture and one computing session a week. In
the weeks with two notebooks, the first is the lecture and the second is worked in the computing
session.

**For whom.** Students in the third or fourth year of a chemical, materials or process
engineering degree, first-year graduate students, and practising engineers who work with process
or laboratory data.

**Prerequisites.** Volume I of this series, or: linear algebra up to the singular value
decomposition, least squares, Python with NumPy, and material balances with some thermodynamics.

| Week | Subject | Lectures |
|:-:|---|---|
| 1 | Atom, mass and charge balances as a linear system | 0 |
| 2 | Least squares with exact constraints | 1 |
| 3 | Data reconciliation and gross errors | 2 |
| 4 | Monotone fits; the pool-adjacent-violators algorithm | 3 |
| 5 | Smooth monotone fits; robust and quantile regression | 4, 5 |
| 6 | Spectra and their preprocessing; principal components | 6, 7 |
| 7 | Partial least squares | 8 |
| 8 | Process monitoring with $T^2$ and $Q$ | 9 |
| 9 | Clustering with K-means | 10 |
| 10 | Spectral clustering; Gaussian mixtures | 11, 12 |
| 11 | Regression trees | 13 |
| 12 | Random forests; gradient boosting | 14, 15 |
| 13 | Conformal intervals and out-of-distribution checks | 16 |
| 14 | Capstone project | 17 |

## A suggested assessment

- **Weekly notebooks (35 %).** The three exercises of each lecture, handed in as an executed
  notebook. The exercises end in `assert` statements, so a submission can be checked by running it
  in a fresh kernel; the written interpretation is marked by hand.
- **Practical examination in week 7 (25 %).** On data not seen before: reconcile a set of measured
  flows and test it for a gross error, fit a monotone curve, and calibrate a PLS model with a
  justified number of latent variables.
- **Capstone project with an oral examination (40 %).** A soft sensor built on the pattern of
  Lecture 17, with a constraint that its predictions satisfy, an interval, and a check that
  withholds a prediction; and a 15-minute explanation at the board of one method used in it.

## On the use of AI assistants

The exercises of this course can be completed by an AI assistant. The assessment above is designed
with that in mind: the weight is on the verification of results, on the interpretation of an
output, and on explaining a method in person. A useful rule for students is the one this
book applies to itself: a number in a report must be traceable to an output, and a claim must be
checked against something independent of the code that produced it.
