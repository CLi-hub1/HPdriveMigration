# Code for the manuscript

This folder contains the Python code for **“Heat-related policies are associated with contrasting migration flows across US counties.”** The analysis links reported local heat adaptation and emissions-reduction actions to county-to-county migration in the United States.

The code covers policy classification, county-pair migration models, sensitivity analyses, contextual moderation, migration-flow income analyses, and figure and table generation. The models estimate associations between reported policy portfolios and migration flows.

## Contents

- [Inputs and directory layout](#inputs-and-directory-layout)
- [Policy preparation and classification](#policy-preparation-and-classification)
- [Migration analyses](#migration-analyses)
- [Figures and tables](#figures-and-tables)
- [Output location](#output-location)
- [Data sources](#data-sources)

## Inputs and directory layout

The scripts use the parent of `code4manuscript` as the project root. Run the commands below from that directory. The code requires the relevant input data to be placed alongside the code in the following layout:

```text
project-root/
├── code4manuscript/            # Python scripts and requirements.txt
├── data4results/               # Plot-ready data, when available
├── results1/                   # Descriptive inputs and geographic data
├── results2/                   # Baseline and lagged model inputs
├── results3/                   # Sensitivity-model inputs and results
├── results4/                   # Contextual-moderator inputs and results
├── results5/                   # IRS income-flow inputs and results
└── manuscript_finalversion/    # Supplementary Information, for table export
```

The code files alone do not include these data directories, the manually annotated policy texts, or the trained classifier weights. Provide the inputs needed for the analyses you wish to run. The required filenames are specified in each script; the table below lists the principal model inputs.

| Analysis | Principal input files, relative to the project root |
| --- | --- |
| Aggregate and policy-type models | `results2/data/frozen_inputs/panel_lag0.csv`, `panel_lag1.csv` and `panel_lag2.csv` |
| Sensitivity models | `results3/data/panel_lag0.csv`, `panel_lag1.csv`, `panel_lag2.csv` and `all_aggregate_and_type_models.csv` |
| Contextual moderation | `results4/data/panel_lag0.csv`, `results5_mediator_dataset_2016_2023.csv` and `results3_selection_pair8.csv` |
| Income-flow models | `results5/data/results5_economiccomposition_aligned_panel.csv` |

The Python dependencies are listed in `requirements.txt`. The statistical scripts were checked with Python 3.12; classifier training and the archived FP32 inference workflow require a compatible CUDA-enabled PyTorch environment.

## Policy preparation and classification

The policy workflow uses reported action text to assign four potentially overlapping types: technological/infrastructural, institutional, behavioral/cultural and nature-based. Adaptation actions are screened for heat-related hazards; the mitigation group includes eligible emissions-reduction actions.

| Script | Purpose |
| --- | --- |
| `prepare_policy_reports.py` | Harmonize and group staged policy reports. |
| `prepare_geocoded_panel.py` | Prepare geocoded county policy and migration inputs. |
| `rebuild_classifier.py` | Audit annotations, train the DistilBERT classifier and classify actions. |
| `compare_classifier_search.py` | Compare grid, random and Bayesian hyperparameter searches. |
| `predict_policy_labels_fp32.py` | Apply the frozen classifier to shared action texts. |
| `aggregate_policy_labels.py` | Aggregate predicted labels by county and reporting year. |

For the annotation audit or classifier training, place `human_labels.csv` in `manuscript_code_outputs/classifier/`, then run:

```bash
python code4manuscript/rebuild_classifier.py audit
python code4manuscript/rebuild_classifier.py train
```

Training also requires the pretrained DistilBERT model. To classify a staged action file after the final model and its manifest are available in the classifier working directory, run:

```bash
python code4manuscript/rebuild_classifier.py infer --input path/to/policy_actions.csv
```

The input file must contain `Action` and `Description` columns; `Title` and `selected` are used when present. Policy preparation, shared FP32 inference and aggregation require the staged files and support records named in their respective scripts.

## Migration analyses

The model inputs are directed county-pair–year panels. The principal outcome is the natural logarithm of the number of moving individuals divided by origin-county population. The fixed-effects models include county-pair and year effects, with standard errors clustered by directed county pair.

Run the analyses after placing the inputs in the paths above:

```bash
# Descriptive statistics
python code4manuscript/build_descriptive_tables.py

# Aggregate and policy-type baseline and lagged models
python code4manuscript/fit_aggregate_and_policy_types.py

# Aggregate and policy-type sensitivity analyses
python code4manuscript/fit_aggregate_sensitivity.py
python code4manuscript/fit_type_sensitivity.py

# Contextual moderation
python code4manuscript/fit_contextual_moderation.py

# IRS income-flow linkage and outcome-defined strata
python code4manuscript/link_irs_income_flows.py
python code4manuscript/fit_income_flow_strata.py
```

The linkage script writes a new aligned panel under `manuscript_code_outputs/results5_linkage/data/`. The income-flow model reads the prepared panel at `results5/data/results5_economiccomposition_aligned_panel.csv`; place the intended panel there before running that model. The `verify_core_models.py` script compares selected re-estimated coefficients, standard errors and P values with saved reference results, provided the relevant model inputs and reference tables are available.

The economic analysis divides county-pair–year observations into five groups separately for each outcome and estimates associations within each group. These groups describe flow records and are not individual income groups.

## Figures and tables

| Manuscript figure | Script |
| --- | --- |
| Fig. 1; Figs. S2–S3; Extended Data Fig. 1 | `plot_geographic_and_network.py` |
| Fig. 2 and Fig. S6 | `plot_aggregate_and_policy_types.py` |
| Fig. 3 | `plot_type_sensitivity.py` |
| Fig. 4; Figs. S7–S8 | `plot_moderation_and_income.py` |
| Fig. 5; Figs. S9–S11 | `plot_moderation_and_income.py` |
| Figs. S1, S4 and S5 | `plot_disclosure_and_correlations.py` |
| Extended Data Fig. 2 | `plot_classifier_search.py` |

For example, with the corresponding `data4results` files in place:

```bash
python code4manuscript/plot_moderation_and_income.py 4
python code4manuscript/plot_moderation_and_income.py S10
python code4manuscript/plot_disclosure_and_correlations.py S4
```

`plot_moderation_and_income.py` also accepts `5`, `S7`, `S8`, `S9` and `S11`; `plot_disclosure_and_correlations.py` accepts `S1`, `S4` and `S5`. The other figure scripts use the corresponding saved data under `results1`–`results3`. Some saved source filenames use earlier figure numbers: the final Fig. 2 uses the aggregate source formerly named `FigS6`, and the final Fig. S6 uses the type-specific source formerly named `Fig2`.

The scripts for Figs. S1, S4 and S5 reconstruct the figures from saved observations and curves using a new layout. The numerical source data are retained, but the visual layout may differ from the manuscript. Inspect generated figures before reuse.

To extract the displayed values of Supplementary Tables S1–S30 from the Word file, run:

```bash
python code4manuscript/export_si_tables.py --input path/to/supplementary_information.docx
```

The exported CSV files preserve the values displayed in Word, including rounding; they do not reproduce the Word formatting or footnotes. Where available, full-precision results are stored separately in the model output tables or the plot-data package.

## Output location

Generated files are written by default under `manuscript_code_outputs/` in the project root. To use a different project or output directory, set `MANUSCRIPT_PROJECT_ROOT` or `MANUSCRIPT_OUTPUT_ROOT` before running a script. Some commands require an empty output directory; use a new location when retaining an earlier run.

Fixed-effects baseline and type-sensitivity results can be checked with `verify_core_models.py`. The aggregate random-effects comparator can differ slightly from the saved figure source on re-estimation; in the checked environment the largest absolute coefficient difference was approximately 1.17 × 10⁻⁷ and the largest P-value difference was approximately 0.00028. The cause of this difference has not been established. The figure script reads the saved plot data.

## Data sources

- [CDP Open Data Portal](https://data.cdp.net/) — reported adaptation and mitigation actions.
- [Internal Revenue Service](https://www.irs.gov/statistics/soi-tax-stats-migration-data) — county-to-county migration records.
- [US Census Bureau](https://data.census.gov/) — population and socioeconomic estimates; [TIGER/Line](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) — county boundaries.
- [US Forest Service](https://data.fs.usda.gov/geodata/rastergateway/treecanopycover/) — tree-canopy cover.
- [Federal Housing Finance Agency](https://www.fhfa.gov/data/hpi/datasets) — house-price indices.
- [US Energy Information Administration](https://www.eia.gov/electricity/data/eia923/) and [EIA-860](https://www.eia.gov/electricity/data/eia860/) — electricity generation and plant locations.
- [OpenStreetMap / ohsome](https://heigit.org/technology/) — mapped indoor facilities.

## Contact

For questions about the study or code, contact **Xing Su** (xsu@zju.edu.cn) or **Chao Fan** (cfan@g.clemson.edu).
