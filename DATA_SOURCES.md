# Data sources and reuse

## Data included in this repository

| Data | Main files | Source and interpretation |
| --- | --- | --- |
| Policy actions and classifications | `inputs/policy/`; `inputs/classifier/` | Grouped CDP action reports, manual labels, classification predictions and county-year policy counts. Adaptation actions were screened using heat, hot and drought hazard text; mitigation includes eligible substantive emissions-reduction actions. The grouped files retain source text and classification audit fields. |
| County-pair migration | `results1/data/frozen_inputs/`; `results2/data/frozen_inputs/`; `results3/data/`; `results5/data/raw_irs/` | [IRS Statistics of Income migration data](https://www.irs.gov/statistics/soi-tax-stats-migration-data). IRS `n1` is moving returns, `n2` is moving individuals, and `agi` is adjusted gross income in thousands of US dollars. County-pair values reflect IRS reporting and suppression rules. |
| Population and socioeconomic characteristics | County and county-pair panel files | [American Community Survey](https://data.census.gov/) five-year estimates: population, median household income, age 65+ share, racial diversity, educational attainment, rent burden and median gross rent. Racial diversity is one minus the sum of squared race-group shares. |
| Housing prices | `results4/data/results5_mediator_dataset_2016_2023.csv` | [Federal Housing Finance Agency house-price index](https://www.fhfa.gov/data/hpi/datasets). |
| Tree canopy | Contextual panel above | [US Forest Service tree-canopy data](https://data.fs.usda.gov/geodata/rastergateway/treecanopycover/). |
| Mapped indoor facilities | Contextual panel above | [OpenStreetMap and ohsome](https://heigit.org/technology/) mapped building footprints. Area share is a mapped resource proxy, not a measure of actual access or use. |
| Renewable generation | Contextual panel above | [US Energy Information Administration EIA-923](https://www.eia.gov/electricity/data/eia923/) generation and [EIA-860](https://www.eia.gov/electricity/data/eia860/) plant information. |
| County boundaries | `results1/data/boundaries/`; `Reproduction/00_Shared_plot_data/` | Simplified county and state geometry from [US Census TIGER/Line files](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html). |
| Plot and table results | `Reproduction/`; selected reference tables under `results2`–`results5` | Data and saved estimates used in manuscript figures and SI tables. [`Reproduction/data_manifest.csv`](Reproduction/data_manifest.csv) lists file descriptions and SHA-256 checksums. |

Detailed study variable definitions and units are provided in [Supplementary Table S29](Reproduction/03_Supplementary_tables/). The main migration and income transformations, sample construction, and model details are described in the manuscript and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## CDP attribution and terms

**This data was collected in partnership by [CDP and ICLEI - Local Governments for Sustainability](https://data.cdp.net/).**

The archived [CDP Open Data Portal Terms of Use and License, April 2024](DATA_LICENSES/CDP_Open_Data_Portal_Terms_April2024.pdf), sets conditions on the use and redistribution of CDP material and derivative databases, including non-commercial use and attribution. CDP-derived action texts, annotations and policy measures in this repository are not relicensed under MIT. Users should consult CDP for current terms and retain source attribution and applicable notices.

## Other reuse terms

The repository's [MIT License](LICENSE) applies to the authors' code; it does not grant rights to third-party datasets or pretrained model components. Users should review the terms attached to each original source and the [DistilBERT base model](https://huggingface.co/distilbert/distilbert-base-uncased). The tokenizer/configuration files and historical model checksum are retained, but the trained classifier weights are not distributed.

The package includes prepared analysis inputs and grouped action records; it does not contain every original third-party download. Historical source paths in manifests identify provenance and are not promises that all historical working directories are included.

One local file path embedded in a public action description has been redacted from the distributed texts. The archived classifications and downstream policy counts are unchanged; inference metadata retain original and released-file hashes. See the [reproducibility guide](REPRODUCIBILITY.md#policy-classification-and-aggregation).
