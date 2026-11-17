# Reproducibility guide

## Analysis units and inputs

The main panel has 16,709 directed county-pair–year observations and 5,065 directed county pairs. `Year` is the policy-report year; the contemporaneous model links reports to migration beginning in that year. Lagged models use exact calendar lags. Origin and destination FIPS codes are `GEOID` and `GEOID_destination`. Read FIPS codes as strings and pad to five digits when presenting identifiers.

The 16 policy regressors comprise four content types for each of adaptation (AP) and mitigation (MP), at origins and destinations. A `_dest` suffix identifies destination terms. The totals sum potentially overlapping type labels, so a total is not necessarily the number of unique action groups. The main outcome is `ln_outflow_rate`, the log of moving individuals divided by origin population. The main models have directed pair and year fixed effects and standard errors clustered by directed pair.

| Analysis | Input | Main script | Output under `manuscript_code_outputs/` |
| --- | --- | --- | --- |
| Descriptive statistics | `results1/data/frozen_inputs/` | `build_descriptive_tables.py` | `descriptive_tables/` |
| Aggregate and 16-term baseline and lag models | `results2/data/frozen_inputs/panel_lag*.csv` | `fit_aggregate_and_policy_types.py` | `results2/` |
| Aggregate sensitivity | `results3/data/panel_lag*.csv` | `fit_aggregate_sensitivity.py` | `aggregate_sensitivity/` |
| Type sensitivity | `results3/data/` | `fit_type_sensitivity.py` | `results3/tables/` |
| Contextual moderation | `results4/data/` | `fit_contextual_moderation.py` | `results4/` |
| IRS outcome linkage | `results5/data/raw_irs/`, `panel_lag0.csv` | `link_irs_income_flows.py` | `results5_linkage/` |
| Income and flow strata | `results5/data/results5_economiccomposition_aligned_panel.csv` | `fit_income_flow_strata.py` | `results5/` |

Scripts are in `code4manuscript/`. Run them from the repository root. Some require an empty output directory; remove or move a previous generated output, or set `MANUSCRIPT_OUTPUT_ROOT` to another directory before rerunning. The IRS linkage creates a new aligned panel under `results5_linkage/`; the model script reads the packaged reference panel, so comparing a rebuilt panel with the reference is a separate step.

```bash
python code4manuscript/link_irs_income_flows.py
python code4manuscript/verify_core_models.py
```

The eight focal policy terms were selected once from the contemporaneous 16-term baseline using nominal two-sided P < 0.10. Sensitivity and income-flow models retain all 16 terms. Contextual models include the eight focal terms and their linear and squared interactions. The income-flow analysis defines five groups separately by each outcome; these are groups of flow observations, not individual income categories. See the manuscript and supplementary tables for definitions, inferential families and interpretation.

The full 16-term coefficient tables are `results3/tables/results3_sensitivity_full16.csv` and `results5/tables/results5_quintile_all_models.csv`. Saved values are full precision. The aggregate random-effects comparator has shown a small numerical difference on re-estimation in the checked environment (maximum absolute coefficient difference approximately 1.17 × 10⁻⁷; maximum P-value difference approximately 0.00028); its cause has not been established. The figure uses the saved source values.

## Figure commands

| Figure | Command |
| --- | --- |
| Figure 1, Supplementary Figures S2–S3, Extended Data Figure 1 | `python code4manuscript/plot_geographic_and_network.py` |
| Figure 2 and Supplementary Figure S6 | `python code4manuscript/plot_aggregate_and_policy_types.py` |
| Figure 3 | `python code4manuscript/plot_type_sensitivity.py` |
| Figure 4 and Supplementary Figures S7–S8 | `python code4manuscript/plot_moderation_and_income.py 4` (or `S7`, `S8`) |
| Figure 5 and Supplementary Figures S9–S11 | `python code4manuscript/plot_moderation_and_income.py 5` (or `S9`, `S10`, `S11`) |
| Supplementary Figures S1, S4–S5 | `python code4manuscript/plot_disclosure_and_correlations.py S1` (or `S4`, `S5`) |
| Extended Data Figure 2 | `python code4manuscript/plot_classifier_search.py` |

Figure 2 uses the aggregate source with the historical filename `FigS6_Aggregate_pair.csv`; Supplementary Figure S6 uses the type-specific source formerly named `Fig2_TypeEffects_pair.csv`. Figures S1, S4 and S5 are reconstructed from saved observations and curves, so their layout can differ from the manuscript. Fonts, graphics libraries and software versions can also affect visual appearance.

Supplementary table folders contain CSVs of the displayed Word values. `export_si_tables.py` can also extract table values from an independently supplied Supplementary Information Word document using its `--input` argument; the Word document is not required to inspect the packaged table CSVs.

## Policy classification and aggregation

`inputs/classifier/human_labels.csv` contains the annotated source rows; `training_rows.csv` records eligible texts and grouped folds. The classifier uses four overlapping labels. Five-fold validation groups related reporting accounts and matching text. The final classifier was trained on 964 eligible unique texts; its parameters, threshold and historical model checksum are recorded in `model_manifest.json`. The trained weights are not included. Fold predictions and validation metrics are supplied. These are internal grouped validation results; the hyperparameter-search comparison is separate from the fixed classifier used for the policy measures.

Stage the annotations and grouped policy inputs (without model weights):

```bash
python code4manuscript/stage_classifier_inputs.py --policy
```

Large grouped action CSVs are stored as gzip files and decompressed while staging. Staged inputs are placed under `manuscript_code_outputs/classifier/` and `manuscript_code_outputs/policy_work/policy_inputs/data/`. Use a new output root if files with different contents already exist. The archived action-level predictions are included, so the policy-count aggregation and downstream analysis remain reproducible from the supplied files.

A local file path embedded in one publicly reported action description has been redacted in the released grouped texts and prediction cache. Archived predictions and policy counts are unchanged and refer to the original text. The inference metadata retain both original and released-file hashes; text-level hashes in the action records refer to the original text. Re-inference on the redacted text cannot exactly reproduce the original input for that action.

```bash
# Rebuild the annotation audit and grouped folds.
python code4manuscript/rebuild_classifier.py audit

# Aggregate the staged archived action-level predictions.
python code4manuscript/aggregate_policy_labels.py
```

The classifier training and inference scripts are retained. `predict_policy_labels_fp32.py` loads a model from `manuscript_code_outputs/classifier/final_model/`; the historical weight file is not distributed. To train a new model, run `python code4manuscript/rebuild_classifier.py train` after staging the annotations and providing access to the public base model/tokenizer. Training and inference require a compatible CUDA GPU. A newly trained model will not reproduce the historical model or its predictions exactly; exact historical re-inference requires access to the omitted weights. GPU, precision and package versions can also affect new runs.

For the 2019-questionnaire and title-only sensitivity variants, use the corresponding `--input` and `--tag` arguments to `aggregate_policy_labels.py`. `compare_classifier_search.py` additionally requires a local copy of the public base model; set `MANUSCRIPT_BASE_MODEL` to that directory. The three-strategy search consists of 180 candidate–fold fits and can be computationally demanding.

## Scope and source reconstruction

This package includes the prepared model panels, analysis scripts, classifier annotations and validation records, grouped policy records, archived predictions, IRS county outflow downloads and plot/table data. The trained classifier weights are omitted; therefore, the archived classifications can be aggregated and analyzed, but exact re-inference with the historical classifier is not possible from this package alone. The package also does not include every historical CDP questionnaire export, all original geographic and resource downloads, or the complete staged workspace for the earliest raw-source preparation scripts. Those scripts document transformations but require the original downloads and staged inputs named within them. See [DATA_SOURCES.md](DATA_SOURCES.md). Thus the principal model and figure workflows start from the supplied analysis-ready inputs; a complete raw-download-to-panel rebuild requires those additional source files.
