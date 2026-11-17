# Heat-related policies are associated with contrasting migration flows across US counties

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains code, analysis inputs, classifier records and figure and table data for the study **“Heat-related policies are associated with contrasting migration flows across US counties.”** It links reported local climate actions with directed county-to-county migration flows in the United States. Estimated policy–migration relationships are observational associations.

## Contents

- [Overview](#overview)
- [Repository contents](#repository-contents)
- [Reproduce the statistical analyses](#reproduce-the-statistical-analyses)
- [Reproduce figures and examine tables](#reproduce-figures-and-examine-tables)
- [Policy classification](#policy-classification)
- [Data sources and terms](#data-sources-and-terms)
- [Contact](#contact)

## Overview

The workflow classifies heat-related adaptation actions and emissions-reduction actions into four overlapping content types, builds county-level policy portfolios, and links them to IRS migration flows. The analysis estimates aggregate and type-specific associations at migration origins and destinations, tests sensitivity to model and sample specifications, examines ten contextual moderators, and explores income and migration-flow outcomes.

## Repository contents

| Location | Contents |
| --- | --- |
| [code4manuscript](code4manuscript/) | Analysis, classification, preparation and plotting scripts; [requirements](code4manuscript/requirements.txt). |
| [Reproduction](Reproduction/) | Source data for Figures 1–5, Extended Data Figures 1–2, Supplementary Figures S1–S11 and Supplementary Tables S1–S30. |
| [inputs/classifier](inputs/classifier/) | Human annotations, validation records, classifier configuration and tokenizer files. Trained classifier weights are not included. |
| [inputs/policy](inputs/policy/) | Grouped action texts, archived label predictions and county-year policy counts. |
| `results1`–`results5` | Prepared panels, reference model results and additional inputs required by the scripts. |
| [DATA_SOURCES.md](DATA_SOURCES.md) | Source descriptions, reuse terms, variable definitions and provenance. |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Input–script–output map, commands and scope of the reproduction. |
| [VALIDATION.md](VALIDATION.md) | Checks performed on the packaged files and statistical results. |

Run the commands below from the repository root. The scripts write new results to `manuscript_code_outputs/`; they do not overwrite the packaged inputs. Python 3.12 was used for the statistical verification. The required Python packages are listed in [requirements.txt](code4manuscript/requirements.txt); classifier training and inference additionally require a compatible CUDA-enabled PyTorch installation.

## Reproduce the statistical analyses

The prepared panels support re-estimation of the main models and their sensitivity analyses:

```bash
python code4manuscript/build_descriptive_tables.py
python code4manuscript/fit_aggregate_and_policy_types.py
python code4manuscript/fit_aggregate_sensitivity.py
python code4manuscript/fit_type_sensitivity.py
python code4manuscript/fit_contextual_moderation.py
python code4manuscript/fit_income_flow_strata.py
```

To independently check the aggregate and 16-term baseline models, the 16-term sensitivity results and the highest AGI-intensity stratum against archived coefficients, standard errors and P values:

```bash
python code4manuscript/verify_core_models.py
```

To check that the downloaded files match the packaged checksums, run `python code4manuscript/verify_package.py`.

The full 16-policy coefficient results for the income and migration-flow models are in [`results5/tables/results5_quintile_all_models.csv`](results5/tables/results5_quintile_all_models.csv). See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for specifications, output locations and additional commands.

## Reproduce figures and examine tables

[Reproduction](Reproduction/) is organized by final manuscript figure and table number. Its [manifest](Reproduction/data_manifest.csv) describes the data files, sources and SHA-256 checksums. Table folders include the values displayed in the Supplementary Information and, where available, full-precision model results.

Examples using the packaged source data:

```bash
python code4manuscript/plot_disclosure_and_correlations.py S4
python code4manuscript/plot_moderation_and_income.py 4
python code4manuscript/plot_moderation_and_income.py 5
```

Commands for all figures and a description of historical source filenames are provided in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). Reconstructed layouts may differ from the manuscript; the underlying observations and saved estimates are supplied.

## Policy classification

The fixed DistilBERT model used for the policy measures has a learning rate of 1 × 10⁻⁵, five epochs and a batch size of 8. Hyperparameter-search comparisons are provided separately and did not replace this classifier. The selected grid-search configuration used 5 × 10⁻⁵, five epochs and a batch size of 16.

The historical classifier weights are not included in this repository. The classifier code, annotations, validation predictions and archived action-level predictions remain available. To stage the annotations and grouped action records, run:

```bash
python code4manuscript/stage_classifier_inputs.py --policy
```

The archived action predictions support reproduction of the published policy counts without model weights. Training a new classifier requires the public base model and a compatible CUDA-enabled PyTorch installation; it will not reproduce the historical classifier exactly. For training, inference and model limitations, see [REPRODUCIBILITY.md](REPRODUCIBILITY.md#policy-classification-and-aggregation).

## Data sources and terms

Principal sources include [CDP](https://data.cdp.net/), [IRS migration data](https://www.irs.gov/statistics/soi-tax-stats-migration-data), the [American Community Survey](https://data.census.gov/), and federal and open geospatial resource datasets described in [DATA_SOURCES.md](DATA_SOURCES.md).

**This data was collected in partnership by [CDP and ICLEI - Local Governments for Sustainability](https://data.cdp.net/).** CDP-derived material is subject to the applicable CDP terms, including non-commercial-use and attribution conditions; the archived April 2024 terms are included in [DATA_LICENSES](DATA_LICENSES/).

The [MIT License](LICENSE) applies to the code. Third-party data and pretrained model components remain subject to their own terms; see [DATA_SOURCES.md](DATA_SOURCES.md).

## Contact

**Xing Su** — [20250612@hhu.edu.cn](mailto:20250612@hhu.edu.cn)  
