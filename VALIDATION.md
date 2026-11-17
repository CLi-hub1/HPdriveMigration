# Package validation

The checks below were performed on the packaged inputs using Python 3.12.9, NumPy 2.1.3, pandas 2.2.3, SciPy 1.15.3, statsmodels 0.14.4, linearmodels 6.1 and Matplotlib 3.10.5. These checks do not establish that the observational associations are causal.

- The aggregate and 16-term baseline models were re-estimated for 16,709 observations; maximum absolute coefficient differences from the saved references were below 1.1 × 10⁻¹⁶.
- For type sensitivity, 128 saved coefficients, standard errors and P values matched re-estimation within 1 × 10⁻⁹.
- The ten-moderator analysis generated 480 coefficient rows. Coefficients matched the saved results; the largest absolute differences were approximately 1.51 × 10⁻⁸ for standard errors and 9.86 × 10⁻¹⁰ for P values.
- All 640 rows in the full 16-policy income and flow model output matched the saved coefficients, standard errors and P values. Rebuilt IRS linkage keys and the checked `n1`, `n2`, AGI and transformed income fields matched the supplied aligned panel.
- The trained classifier weights are not distributed; the historical checksum and model configuration remain in `inputs/classifier/model_manifest.json`. Aggregating the archived action predictions reproduced all six supplied main and sensitivity county-year policy tables. The pooled out-of-fold micro F1 computed from the saved validation predictions matched the recorded value. A separate annotation audit reproduced all 964 training rows, connected-group assignments and five-fold assignments.
- Plotting commands for Figures 1–5, Supplementary Figures S1–S11, and Extended Data Figures 1–2 completed using the packaged inputs. The geographic command was tested separately with Python 3.12.9, pyshp 2.3.1, pyproj 3.7.1, shapely 2.1.0, NumPy 2.2.4 and Matplotlib 3.10.1. All 112 data files listed in `Reproduction/data_manifest.csv` matched their recorded hashes. The supplementary table export was checked against a separately supplied SI Word file.

Classifier retraining and new GPU inference were not run in the tested Python environments. The annotations, validation records and scripts are supplied, but trained weights are omitted. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for this limitation and the redaction and raw-source constraints.

`python code4manuscript/verify_package.py` verifies the file hashes after download. `python code4manuscript/verify_core_models.py` repeats selected statistical checks.
