# Figure and table source data

The figure and table numbers follow the final `manuscript_finalversion/Manuscript.docx` and `manuscript_finalversion/Supplymentary information.docx`.

- Manuscript SHA-256: `493046dd534c8193cfe2464b0d613b73c0b9c9a341a0ff99025347315d1ca65d`
- Supplementary information SHA-256: `aeaa5d4bc09c8399cbf713b166faedd19d4ce1b86be60b3ab144adeee884753f`

## Folders

- `00_Shared_plot_data/`: simplified 2020 county and state boundaries for the maps.
- `01_Main_and_extended_figures/`: Figures 1–5 and Extended Data Figures 1–2.
- `02_Supplementary_figures/`: Figures S1–S11.
- `03_Supplementary_tables/`: one folder for each Table S1–S30.
- `data_manifest.csv`: description, original source path, file size and SHA-256 for every data file.

## Using the data

Figure folders contain plot-ready observations, coefficients, intervals, fitted curves or map coordinates. The final Fig. 2 uses the source file formerly named `FigS6_Aggregate_pair.csv`; the final Fig. S6 uses the source file formerly named `Fig2_TypeEffects_pair.csv`.

Each supplementary table folder includes `Table_SXX_Word_display_values.csv` with the displayed text extracted from the final SI, including rounding. Other CSV files contain the corresponding statistical results at the precision of the original output. Merged cells are followed by empty fields, and line breaks within cells are retained. Consult the Word file for footnotes and formatting. Table S29 is a descriptive variable dictionary, so the displayed table is its direct source.

CSV files use UTF-8 or UTF-8 with BOM. Import county FIPS identifiers as text and pad to five digits where needed; Excel may otherwise remove leading zeros.

The fitted lines and confidence bands for Fig. 1c–f and the Fig. 1b boxplot statistics were recomputed from the saved observations using the original plotting code; all four slopes were checked against the archived statistics. Figs. S4–S5 and Figs. 4/S7/S8 use the existing saved fitted or marginal curves. Map backgrounds use `00_Shared_plot_data/us_2020_epsg4326_plot.json`. Plot layout, fonts and colors require plotting code and are not contained in the CSV files.

The original Fig. 1 code labels Pearson correlation as r, Spearman correlation as β and the OLS slope as ρ. The caption follows those plotted labels; do not swap β and ρ solely on the basis of conventional notation.

Some files retain historical source names and may contain rows for more than one figure; use `data_manifest.csv` to identify the relevant subset. Shared results may be copied into multiple folders. This package contains direct figure and table data, not the original CDP, IRS or ACS downloads.

## Coverage and checks

The package covers 7 main/extended-data figures, 11 supplementary figures and all 30 supplementary tables. All 112 data-file hashes were verified. For Tables S4/S5 and S21–S28, 2,080 coefficients, standard errors and P values matched the displayed Word values within rounding precision. For S6–S15, 480 coefficient/standard-error pairs were found in the corresponding model outputs; these checks did not independently validate every row label. All 40 quintile-range rows in S19/S20 matched, and 118 values were checked in S2, S3 and S30. Not every cell in every table was independently checked. The regressions were not refitted, and the figures were not all redrawn or compared pixel by pixel.

Two captions in the final SI may need review. Figure S3 is titled and plotted for destination counties, but its caption says “origin county-years”. Figure S5 is titled and plotted for destination partners, but its caption says “origin policy-type label counts”. The original Word files and figures were not modified.

## Supplementary table titles

- Table S1 County coverage by migration endpoint
- Table S2 Annual action groups and predicted labels
- Table S3 Aggregate four-term baseline
- Table S4 Coefficients across baseline and robustness specifications
- Table S5 Raw P, BH q and Holm P across baseline and robustness specifications
- Table S6 Log median household income — all model coefficients
- Table S7 Age 65+ share — all model coefficients
- Table S8 Racial diversity — all model coefficients
- Table S9 Tree-canopy share — all model coefficients
- Table S10 Bachelor-or-higher share — all model coefficients
- Table S11 Log FHFA house-price index — all model coefficients
- Table S12 Rent-burden share — all model coefficients
- Table S13 Log median gross rent — all model coefficients
- Table S14 Mapped indoor-facility footprint share — all model coefficients
- Table S15 Renewable-generation share — all model coefficients
- Table S16 Interaction-test adjustments across contextual moderators
- Table S17 Joint interaction tests for all ten contextual indicators
- Table S18 Moderator support and model sample sizes
- Table S19 Full sample — economic and flow quintile boundaries
- Table S20 Excluding the 2022–2023 IRS cohort — economic and flow quintile boundaries
- Table S21 Asinh(mean AGI per return / $10,000): full-sample and cohort-exclusion estimates
- Table S22 P values for asinh(mean AGI per return / $10,000) with 40- and 160-test adjustments
- Table S23 Asinh(AGI per 1,000 origin residents / $10,000): full-sample and cohort-exclusion estimates
- Table S24 P values for asinh(AGI per 1,000 origin residents / $10,000) with 40- and 160-test adjustments
- Table S25 Individuals per origin population: full-sample and cohort-exclusion estimates
- Table S26 P values for individuals per origin population: within- and across-outcome adjustment families
- Table S27 Returns per origin population: full-sample and cohort-exclusion estimates
- Table S28 P values for returns per origin population: within- and across-outcome adjustment families
- Table S29 Data variables and sources
- Table S30 Nominal and adjusted result counts in both samples
