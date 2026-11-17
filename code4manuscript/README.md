# Manuscript analysis and figure code

This directory contains English-language Python code for the methods, statistical results, figures and supplementary tables in the final manuscript. It uses relative project paths or environment variables; no researcher-specific absolute paths or credentials are embedded in the scripts. The manuscript and its underlying data are not modified.

## Inputs and outputs

The default project root is the parent of this directory. The scripts read the existing `results1`–`results5`, `data4results`, and manuscript directories where applicable. To use another copy of the data, set `MANUSCRIPT_PROJECT_ROOT` to that directory. Generated outputs go to `manuscript_code_outputs` under the project root; set `MANUSCRIPT_OUTPUT_ROOT` to choose another location. Dataset staging and model outputs require writable directories.

`data4results` provides plot-ready inputs and the displayed supplementary tables. It does not contain all observation-level model inputs, original disclosure records, county-level resources, manually labelled texts, trained model weights or official IRS files. Those inputs must be made available separately with appropriate access and licensing. Full re-estimation is therefore not possible from the plot-data package alone.

The statistical scripts were checked in Python 3.12 with NumPy 2.1.3, pandas 2.2.3, SciPy 1.15.3, statsmodels 0.14.4, linearmodels 6.1 and Matplotlib 3.10.5. `requirements.txt` also lists packages needed by the classifier, geographic processing and downloads. Model training and FP32 inference require suitable PyTorch/Transformers installations, model files and compatible hardware; the archived inference uses CUDA. Some optional packages were unavailable in the test environment, so those stages were not rerun.

## Code by method

| Script | Function |
| --- | --- |
| `prepare_policy_reports.py` | Harmonize annual policy reports, screen adaptation hazards, recover the 2018 mitigation responses from staged questionnaire records, and group action text. |
| `prepare_geocoded_panel.py` | Prepare historical geocoded county policy and migration inputs from staged sources. |
| `rebuild_classifier.py` | Audit annotations, create grouped folds, train the multilabel DistilBERT classifier and run inference. |
| `compare_classifier_search.py` | Compare grid, random and Bayesian hyperparameter search with grouped folds. |
| `predict_policy_labels_fp32.py` | Apply the frozen model to shared action texts using deterministic FP32 inference. |
| `aggregate_policy_labels.py` | Aggregate predicted policy labels and eligible reporting support by county and year. |
| `download_acs_demographics.py` | Retrieve and process the later-year ACS demographic tables used in the archived preparation workflow. |
| `download_acs_legacy.py`, `download_acs_housing.py`, `download_utils.py` | Retrieve the staged ACS housing tables and preserve download metadata. |
| `prepare_acs_housing.py` | Construct county-year gross rent, housing value and rent-burden measures from the staged ACS tables. |
| `prepare_fhfa_hpi.py` | Extract annual FHFA county HPI for the observed county support. |
| `prepare_environmental_resources.py` | Obtain and construct the archived resource panel from staged support and official geographic, canopy, OSM and energy sources. |
| `build_descriptive_tables.py` | Recreate descriptive county-year and annual tables from frozen inputs. |
| `fit_aggregate_and_policy_types.py`, `model_helpers.py` | Estimate aggregate and 16-term baseline/lag models, and select the fixed follow-up set. |
| `fit_aggregate_sensitivity.py` | Estimate the four aggregate terms across the eight Figure 2 specifications. |
| `fit_type_sensitivity.py` | Estimate all 16 policy types across the sensitivity specifications and retain the eight focal terms. |
| `fit_contextual_moderation.py` | Estimate separate linear and quadratic interactions for ten moderators and compute conditional curves. |
| `link_irs_income_flows.py` | Link IRS returns and AGI to the frozen directed migration panel. |
| `fit_income_flow_strata.py` | Estimate the outcome-defined quintile models, including the 2022–2023 exclusion and multiple-testing adjustments. |
| `verify_core_models.py` | Re-estimate and check the main fixed-effects models, all type-sensitivity terms and one economic-flow model against frozen outputs. |

The policy preparation and resource scripts expect staged raw files in the relative directory layout named in their code. The classifier expects annotation data and weights under the configured classifier working directory. Exact re-creation of the upstream data requires the appropriate historical source files and model assets, not simply the `.py` files.

## Figure and table mapping

| Figure or table | Script |
| --- | --- |
| Fig. 1; Figs. S2–S3; Extended Data Fig. 1 | `plot_geographic_and_network.py` |
| Fig. 2 and Fig. S6 | `plot_aggregate_and_policy_types.py` |
| Fig. 3 | `plot_type_sensitivity.py` |
| Fig. 4; Figs. S7–S8 | `plot_moderation_and_income.py` |
| Fig. 5; Figs. S9–S11 | `plot_moderation_and_income.py` |
| Figs. S1, S4 and S5 | `plot_disclosure_and_correlations.py` |
| Extended Data Fig. 2 | `plot_classifier_search.py` |
| Tables S1–S30, displayed values | `export_si_tables.py` |

The original plotting modules for Figs. S1, S4 and S5 were not available in the source package. Their replacement uses the retained observations and fitted curves but does not claim an exact visual reproduction. The available geographic and statistical plots retain their original plotting logic, but the separate historical panel-alignment checking tool was unavailable; inspect the rendered figures before use. Some output filenames retain historical figure identifiers; the mapping above follows the final manuscript. In particular, the aggregate figure was formerly named S6 and the 16-term type figure was formerly named Fig. 2.

The table export reads the final SI and writes the displayed values, which include rounding. Full-precision coefficient and test data are in the corresponding `data4results/03_Supplementary_tables` folders or can be regenerated with the model scripts. This export does not reconstruct Word formatting or footnotes.

## Example commands

Run commands from the project root:

```text
python code4manuscript/build_descriptive_tables.py
python code4manuscript/fit_aggregate_and_policy_types.py
python code4manuscript/fit_aggregate_sensitivity.py
python code4manuscript/fit_type_sensitivity.py
python code4manuscript/fit_contextual_moderation.py
python code4manuscript/link_irs_income_flows.py
python code4manuscript/fit_income_flow_strata.py
python code4manuscript/plot_aggregate_and_policy_types.py
python code4manuscript/plot_type_sensitivity.py
python code4manuscript/plot_moderation_and_income.py 4
python code4manuscript/plot_moderation_and_income.py S10
python code4manuscript/plot_disclosure_and_correlations.py S4
python code4manuscript/export_si_tables.py
python code4manuscript/verify_core_models.py
```

The classifier audit expects `human_labels.csv` in the configured `classifier` working directory. Training also needs access to the pretrained model. Search additionally expects a staged local base-model snapshot; set `MANUSCRIPT_BASE_MODEL` to its location. Shared FP32 inference expects the staged action files and final model specified in the scripts. The optional BF16 comparison is run only when prior comparison files are present. Public-data download scripts make network requests when run.

## Verification performed

- All Python files were parsed and scanned for Chinese characters and hard-coded personal paths.
- The aggregate and 16-term baseline fixed-effects coefficients reproduced the saved estimates to approximately 1.1 × 10⁻¹⁶. All 128 type-sensitivity coefficients and standard errors matched the frozen output in the test environment.
- The quadratic income moderation model and the highest-AGI-intensity-stratum model reproduced the saved coefficients to approximately 4.5 × 10⁻¹⁶ and 9.1 × 10⁻¹⁷, respectively.
- The IRS linkage produced 16,709 aligned records with no n2 mismatches. The ACS housing reconstruction matched all 864 saved county-year records for gross rent, rent burden and home value. All 30 exported Word tables matched the prepared display-value CSV files.
- The aggregate random-effects comparator differed slightly from the saved figure source: the maximum absolute coefficient difference was approximately 1.17 × 10⁻⁷ and the largest P-value difference approximately 0.00028. Its exact source of numerical variation has not been established. The plot script reads the frozen figure data, so it does not replace those values with a rerun.
- Descriptive, selected moderation and income plots, the classifier-search figure and table extraction were exercised. Geographic rendering, original-source retrieval, classifier training and full end-to-end rebuilding were not executed in the available environment.

The code preserves the selected model specifications, adjustments, exclusions and uncertainty calculations. It does not by itself establish causal identification or independently validate policy implementation dates.
