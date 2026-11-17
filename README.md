# Heat-related policies are associated with contrasting migration flows across US counties [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
This repository contains the code for the manuscript **“Heat-related policies are associated with contrasting migration flows across US counties.”**

- [Overview](#overview)
- [Code description](#code-description)
- [Data sources](#data-sources)
- [Instructions for use](#instructions-for-use)
- [Contact](#contact)

## Overview

Climate change influences human migration, while local governments are increasingly adopting actions to adapt to climate risks and reduce greenhouse gas emissions. However, how these reported actions relate to observed migration remains unclear.

In this study, we link reported local climate actions to county-to-county migration flows in the United States. We examine how policy–migration associations vary across **policy types**, **migration origins and destinations**, and **local socioeconomic and environmental conditions**. We also explore the income recorded with migration flows.

## Code description

The `code4manuscript` folder contains the Python scripts used for the following analyses:

- **Policy data preparation and classification:** Process reported actions and use a supervised DistilBERT classifier to assign actions to four potentially overlapping types: technological/infrastructural, institutional, behavioral/cultural and nature-based.

- **County-pair migration models:** Estimate associations between policy labels and bilateral migration flows using county-pair and year fixed effects.

- **Sensitivity analyses:** Examine alternative samples, outcome-tail treatments, a random-effects comparator and lagged policy specifications.

- **Contextual moderation:** Assess how policy–migration associations vary across socioeconomic and environmental conditions.

- **Economic characteristics of migration flows:** Explore adjusted gross income and migration-flow outcomes across outcome-defined groups.

- **Figures and tables:** Generate figures and extract supplementary table values.

A complete list of scripts and their corresponding analyses is provided in [`code4manuscript/README.md`](code4manuscript/README.md).

## Data sources

The study draws on the following data sources:

| Data | Source |
| --- | --- |
| Reported adaptation and mitigation actions | [CDP Open Data Portal](https://data.cdp.net/) |
| County-to-county migration flows | [Internal Revenue Service](https://www.irs.gov/statistics/soi-tax-stats-migration-data) |
| Population and socioeconomic characteristics | [US Census Bureau: American Community Survey](https://data.census.gov/) |
| County boundaries | [US Census Bureau: TIGER/Line Shapefiles](https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html) |
| Tree-canopy cover | [US Forest Service](https://data.fs.usda.gov/geodata/rastergateway/treecanopycover/) |
| House-price indices | [Federal Housing Finance Agency](https://www.fhfa.gov/data/hpi/datasets) |
| Electricity generation and plant locations | [EIA-923](https://www.eia.gov/electricity/data/eia923/) and [EIA-860](https://www.eia.gov/electricity/data/eia860/) |
| Mapped indoor facilities | [OpenStreetMap data accessed through the ohsome API](https://heigit.org/technology/) |

## Instructions for use

Download or clone this repository using the **Code** button on GitHub.

The scripts require the relevant prepared input data. Classifier training and inference also require annotated examples and model files. Figure scripts require the corresponding plot-ready data.

For script descriptions and input requirements, see [`code4manuscript/README.md`](code4manuscript/README.md).

## Contact

For questions about the study or code, please contact:

- **Xing Su:** xsu@zju.edu.cn
