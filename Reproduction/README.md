# Figure and table source data

This folder contains the source data for manuscript Figures 1–5, Extended Data Figures 1–2, Supplementary Figures S1–S11 and Supplementary Tables S1–S30. Folders follow the final figure and table numbers. [data_manifest.csv](data_manifest.csv) describes the files and records their SHA-256 checksums.

- `00_Shared_plot_data/`: simplified county and state boundaries.
- `01_Main_and_extended_figures/`: observations and model results for main and extended figures.
- `02_Supplementary_figures/`: observations, estimates and curves for supplementary figures.
- `03_Supplementary_tables/`: displayed Word values and, where available, full-precision results.

For historical reasons, the final Figure 2 uses a source named `FigS6_Aggregate_pair.csv`, while the final Supplementary Figure S6 uses `Fig2_TypeEffects_pair.csv`. Historical source paths in the manifest indicate provenance; the plotting commands and available inputs are described in the [reproducibility guide](../REPRODUCIBILITY.md).

The `Word_display_values.csv` files retain the values and rounding displayed in the Supplementary Information. Empty cells can represent merged cells; CSV does not reproduce Word formatting or footnotes. Read county FIPS codes as text and pad to five digits when needed.

Figure S3 maps destination counties, not origin counties. Figure S5 relates destination-partner policy labels, not origin policy labels, to origin county-year outflow rates.
