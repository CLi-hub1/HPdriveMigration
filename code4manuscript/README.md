# Analysis code

Run scripts from the repository root. The root [README](../README.md) introduces the project and the [reproducibility guide](../REPRODUCIBILITY.md) maps inputs, scripts and outputs, including all figure commands.

The scripts use the project root by default. Set `MANUSCRIPT_PROJECT_ROOT` to use another input root or `MANUSCRIPT_OUTPUT_ROOT` to use another output directory. The default output directory is `manuscript_code_outputs/`. The supplied statistical environment is listed in [requirements.txt](requirements.txt). Classifier training and inference also require CUDA-enabled PyTorch.

## Main scripts

| Step | Scripts |
| --- | --- |
| Prepare source data | `prepare_policy_reports.py`, `prepare_geocoded_panel.py`, `download_acs_demographics.py`, `download_acs_housing.py`, `download_acs_legacy.py`, `prepare_acs_housing.py`, `prepare_fhfa_hpi.py`, `prepare_environmental_resources.py` |
| Classify and aggregate policies | `stage_classifier_inputs.py`, `rebuild_classifier.py`, `compare_classifier_search.py`, `predict_policy_labels_fp32.py`, `aggregate_policy_labels.py` |
| Summarize data | `build_descriptive_tables.py` |
| Fit policy–migration models | `fit_aggregate_and_policy_types.py`, `fit_aggregate_sensitivity.py`, `fit_type_sensitivity.py` |
| Contextual moderation | `fit_contextual_moderation.py` |
| Income and flow outcomes | `link_irs_income_flows.py`, `fit_income_flow_strata.py` |
| Plot figures | `plot_geographic_and_network.py`, `plot_aggregate_and_policy_types.py`, `plot_type_sensitivity.py`, `plot_moderation_and_income.py`, `plot_disclosure_and_correlations.py`, `plot_classifier_search.py` |
| Verify or export | `verify_core_models.py`, `verify_package.py`, `export_si_tables.py` |

`config.py`, `model_helpers.py` and `download_utils.py` provide shared utilities. Preparation from original downloads requires the staged raw sources described in the guide; the included prepared panels support model re-estimation without rebuilding every upstream data source.
