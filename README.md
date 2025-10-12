# FedEx Supply Chain Project  

## Table of Contents  
- [Project Overview](#project-overview)  
- [Directory Structure](#directory-structure)  
- [Data](#data)  
- [Scripts & Source Code](#scripts--source-code)  
- [Analyses & Notebooks](#analyses--notebooks)  
- [Reports & Figures](#reports--figures)  
- [Configuration & Dependencies](#configuration--dependencies)  
- [How to Run / Reproduce](#how-to-run--reproduce)  
- [Key Findings & Deliverables](#key-findings--deliverables)  
- [Future Work & To‑dos](#future-work--to‑dos)  
- [Contributors](#contributors)  
- [License & Acknowledgments](#license--acknowledgments)  

---

## Project Overview

This repository contains the end-to-end work for a supply chain / logistics analytics project, likely in collaboration with FedEx (or inspired use case). The main goal is to analyze, model, simulate, and derive insights from a supply chain dataset (DataCo Supply Chain), covering aspects such as delivery risk, forecasting, profit optimization, customer segmentation, and operations.

So far, the work includes:

- Data exploration, cleaning, and preprocessing  
- Feature engineering  
- Modeling / simulation (Monte Carlo, MDP / reinforcement‑learning style)  
- Sensitivity analysis  
- Reports, figures, and final deliverables  
- Documentation of methods  

---

## Directory Structure  

```
├── configs  
├── csv_results  
├── figures  
├── monthly_reports  
├── project_docs  
├── scripts  
├── src  
├── tests  
├── DataCoSupplyChainDataset.csv  
├── DescriptionDataCoSupplyChain.csv  
├── data_dictionary.xlsx  
├── data_quality_issues.ipynb  
├── empirical_data.csv  
├── exploratory_statistics.ipynb  
├── feature_eng.md  
├── mc_simulation_v1.ipynb  
├── mc_simulation_v2.ipynb  
├── params.yaml  
├── README.md  
├── rl_mdpspec.md  
├── scri_draft.md  
├── scri_final.ipynb  
├── scri_method.md  
├── scri_results.csv  
├── scri_results_labeled.csv  
├── sensitivity_analysis.ipynb  
├── requirements.txt  
└── … (other Jupyter notebooks as above)
```

---

## Data

- **DataCoSupplyChainDataset.csv** — main dataset (180,519 records, ~53 features)  
- **DescriptionDataCoSupplyChain.csv** — metadata / descriptions of features  
- **data_dictionary.xlsx** — additional documentation of data fields  
- **empirical_data.csv** — possibly a derived or processed subset used in modeling/analysis  

### Data Quality & Preprocessing Notes

- Missing values: ~336,209 missing entries across various columns  
- Duplicates: no duplicate full rows detected  
- Data types: mixture of categorical / object, numeric (float64, int64)  
- Approach to missingness: imputation (mode, “Unknown”, or domain‑appropriate)  
- Outliers: flagged and handled via winsorization or clipping (1st–99th percentile)  
- Some overlapping / redundant columns were flagged during exploratory work  
- Additional data cleaning steps are recorded in notebooks like `data_quality_issues.ipynb`

---

## Scripts & Source Code

- `src/` — modular code for data preprocessing, feature engineering, modeling, simulation  
- `scripts/` — standalone Python scripts for batch runs or experiments  
- `tests/` — validation / testing scripts  
- `configs/` — configuration YAML/JSON files for parameterization  
- `params.yaml` — stores model parameters / experiment settings  

---

## Analyses & Notebooks

- `exploratory_statistics.ipynb` — data overview and visualization  
- `data_quality_issues.ipynb` — missingness and data issues  
- `mc_simulation_v1.ipynb`, `mc_simulation_v2.ipynb` — Monte Carlo simulations  
- `sensitivity_analysis.ipynb` — tests robustness of parameters  
- `scri_final.ipynb` — integrated report combining all results  
- `.md` docs like `feature_eng.md`, `rl_mdpspec.md`, etc. describe methodology  

---

## Reports & Figures

- Visual outputs (charts/plots) in `figures/`  
- Monthly summaries in `monthly_reports/`  
- Comprehensive narrative + visuals in `scri_final.ipynb`  

---

## Configuration & Dependencies

- Install dependencies: `pip install -r requirements.txt`  
- Edit configurations in `params.yaml` / `configs/` before running experiments  

---

## How to Run / Reproduce

1. **Setup environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure parameters**
   - Edit `params.yaml` or `configs/`

3. **Run analysis**
   - Execute relevant notebooks or scripts (`src/` / `scripts/`)

4. **Generate reports**
   - Use `scri_final.ipynb` to consolidate visuals & results

---

## Key Findings & Deliverables

- Cleaned dataset ready for modeling  
- Monte Carlo & sensitivity simulations  
- Final labeled results (`scri_results.csv`)  
- Documentation of MDP & RL‑based modeling approaches  
- Visuals and analytical summaries  

---

## Contributors

- **Puneet (brpuneet898)** — Project Manager  
- **Devanshu Bhatnagar** — Intern
