# Heat-related policies drive population out-migration in the U.S. [![PyPI Version](https://img.shields.io/pypi/v/projectname.svg)](https://pypi.org/projectprojectname/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/license/MIT)
This repository includes the code and data necessary to reproduce the results and figures for the article "Heat-related policies drive population out-migration in the U.S." 

- [Overview](#Overview)
- [System Requirements](#SystemRequirements)
- [Installation guide](#Installationguide)
- [Demo](#Demo)
- [Instructions for use](#Instructionsforuse)
- [Contact](#Contact)

## Overview <a id="Overview"></a>
### **Background** 
Migration is becoming a major force in the dynamics of population allocation, with climate-induced migration receiving growing attention. However, the role played by human climate endeavors is continually overlooked in existing climate-induced migration studies, including empirical studies, meta-analyses, and future projections. To address this gap, we combine machine learning with attribution mapping to provide robust evidence on the impacts of heat-related policies (HPs) on migration patterns across U.S. counties. 
### **Discription** 
This repository consists of four main parts of code. The first one is to fine-tune the pre-trained DistilBERT model using our annotated dataset, which in turn predicts the classification to which the heat-related policies(HPs) belongs; the second one is to investigate the causal effect of HPs on migration, which mainly consists of our proposed G2SLS method and the traditional TWFE, OLS method; the third is to estimate the heterogeneity of the effect across geographical regions and policy types; and the fourth is to calculate the mediating impact of climate-related public opinion on migration due to HPs. 

## System Requirements <a id="SystemRequirements"></a>
The code to categorize heat-related policies ran on a 1x NVIDIA GeForce RTX 4090.  
Other codes were employed on a system with the following specifications:
- operating system: Windows 10
- CPU: AMD Ryzen 7 5800H
- memory (RAM): 16GB
- disk storage: 500GB
- GPU: 1x NVIDIA GeForce GTX 1650

The main software requirements are Python 3.9 with transformers 4.31 and torch 2.0. Other libraries used include pandas, numpy, matplotlib, sklearn, scipy, statsmodels, shapely, and geopandas.

## Installation guide <a id="Installationguide"></a>
The implementation is distributed as source code rather than a packaged module. Researchers may either:

1. Download the compressed archive via the GitHub interface

2. Clone the repository using standard Git workflows

## Demo <a id="Demo"></a>
The code provided in this repository can be run directly under the appropriate compilation environment.

Expected outputs vary depending on the code and generally include categorized and attributed outputs.

## Instructions for use <a id="Instructionsforuse"></a>
The source of the raw data has been provided as the link in the paper, and the authors did not use software to collect the raw data but downloaded it directly through the link. 

For the data preprocessing section, the relevant code has been provided in the Data preparation folder; for the modeling section, the relevant code has been provided in the Model folder. This folder includes a machine learning-based classification model for HPs, attribution models based on OLS, TWFE, and G2SLS, and multiple validations for the robustness of the attribution results; for the reproduction section, the Reproduction folder includes the data that produces the plots in the manuscript.

## Contact <a id="Contact"></a>
For questions and support:
📧 Email: xsu@zju.edu.cn; cfan@g.clemson.edu
