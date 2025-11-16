## Activity Cliff-Aware Reverse QSAR Modeling 

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2023.09.5-forestgreen.svg)](https://www.rdkit.org/)
[![Cliff_Aware](https://img.shields.io/badge/Cliff__Aware-CAFE%20%2B%20CAFE%20LATE-orange.svg)](#cafe--cafe-late-at-a-glance)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](#license)

A compact workflow that turns activity cliffs into a design asset: CAFE mines cliff-enriched fragments from QSAR data, while CAFE LATE uses them to steer fragment-based molecular generation.

### CAFE / CAFE LATE at a glance

| 🧬 **CAFE** – Cliff-Aware Fragment Extraction | 🧪 **CAFE LATE** – Late Activity Tuning | 🔍 **Reverse QSAR** – Fragment Mining | 🧱 **AC-Aware Generation** |
| :------------------------------------------- | :-------------------------------------- | :------------------------------------ | :-------------------------- |
| Mines activity-cliff-enriched fragments from QSAR data using BRICS, classical ML models and SHAP-based importance. | Adjusts QSAR probabilities for new molecules that contain AC-enriched fragments, sharpening predictions near cliffs. | Learns which fragments drive activity jumps and ranks models by AUPRC₍active₎ with bootstrap CIs. | Assembles new molecules around core scaffolds, scores them with QSAR + SA + QED + CAFE LATE and returns a diverse, AC-aware hit list. |

## Key Features

- **CAFE / CAFE LATE built in**: Two-stage use of activity cliffs – first for fragment mining, then for late-stage reweighting of QSAR scores.
- **Cliff-aware QSAR**: Models evaluated with Cliff_RMSE, AUPRC₍active₎ and grouped splits (cliff_group / Butina) to avoid leakage.
- **Rich molecular representations**: ECFP (1024/2048 bits), MACCS keys, RDKit and Mordred descriptors with Boruta feature selection.
- **Reverse QSAR engine**: BRICS fragmentation plus up to 9 classical ML models and SHAP-driven fragment ranking.
- **Fragment-based molecular generation**: Island algorithm over user-defined cores with multi-objective scoring (activity, synthetic accessibility, drug-likeness).


## Table of Contents

 - [Installation](#installation)
 - [Quick Start](#quick-start)
 - [Pipeline Overview](#pipeline-overview)
 - [Configuration](#configuration)
 - [Usage](#usage)
 - [Adaptability](#adaptability)
 - [Citation](#citation)
 - [License](#license)

## Installation
### Setup

1. **Clone the repository**:
```bash
  git clone https://github.com/yourusername/AC-aware-modeling.git
  cd AC-aware-modeling
```

2. **Create and activate conda environment**:
```bash
  conda env create -f environment.yaml
  conda activate ac-aware-modeling
```

## Quick Start

### Complete Pipeline

Run all stages in sequence:
```bash
  python main.py
```

This executes:
1. Dataset preparation (fetches from ChEMBL or uses custom data)
2. Activity cliffs analysis (optional, configurable)
3. Reverse QSAR fragment mining (CAFE)
4. QSAR model evaluation (cliff-aware metrics)
5. Molecular generation (CAFE LATE)

### Step-by-Step Execution

**1. Prepare dataset only:**
```bash
  # Set REBUILD_DATASET: true in config.yml, then:
  python main.py
  # Pipeline stops after dataset creation
```

**2. Run remaining stages:**
```bash
  # Set REBUILD_DATASET: false, enable desired stages in config.yml:
  python main.py
```

The pipeline automatically skips completed stages (resume functionality).

## Pipeline Overview

The pipeline consists of five main stages:

### 1. Dataset preparation (`data_preparation.py`)

- **Input**: ChEMBL target ID (`TARGET_ID` in config) or custom CSV with columns: `ChEMBL_ID`, `canonical_smiles`, `pIC50` (or `IC50_nM`)
- **Process**:
  - Fetches activity data from ChEMBL API (if `TARGET_ID` provided)
  - Merges with chemical space metadata (optional)
  - Standardizes SMILES (RDKit canonicalization, tautomer removal)
  - Filters: MW < 900 Da, LogP < 8
  - Converts activities to pIC50 and binary labels (threshold: `THRESHOLD_NM` in config)
- **Output**: `data/processed/final_dataset.csv` (required for all downstream stages)
- **Standalone**: Set `REBUILD_DATASET: true` in config, run `python main.py` (stops after dataset creation)

### 2. Activity cliffs analysis (`AC_analysis/ac_analysis.py`)

- **Purpose**: Identify structurally similar molecule pairs with large activity differences
- **Definition**: Activity cliff = Tanimoto similarity ≥ 0.8 (ECFP4) AND |ΔpIC50| ≥ 1.0 log unit
- **Metrics**: SALI (Structure-Activity Landscape Index) quantifies cliff severity
- **Output**: `results/AC_analysis/activity_cliffs.csv`, SALI scatterplot, PCA/t-SNE visualizations
- **Usage**: Optional (enable via `AC_Analysis.enable: true` in config)

### 3. Reverse QSAR – CAFE (`reverse_qsar/defragmentation.py`)

- **Purpose**: Extract fragments enriched in activity cliffs (CAFE: Cliff-Aware Fragment Extraction)
- **Methodology**:
  - BRICS fragmentation → binary fragment-molecule matrix
  - Train 9 ML models (LogReg, KNN, SVC, RF, BRF, ExtraTrees, GB, XGB, CatBoost) on fragment presence vs. activity
  - Rank models by AUPRC₍active₎ with bootstrap CIs
  - SHAP importance → AC enrichment: fragments appearing only in active cliff members get up-weighted
- **Output**: `results/reverse_QSAR/*/selected_fragments_with_ACflag.csv` (fragments + AC enrichment flags)
- **Key Feature**: Fragments driving activity jumps are prioritized for generation

### 4. Predictive modeling (`predictive_model/cliffaware_qsar.py`)

- **Purpose**: Train QSAR models with explicit activity cliff awareness
- **Features**:
  - Multiple backbones: ECFP1024/2048, MACCS keys, RDKit/Mordred descriptors
  - Group-based splitting: `cliff_group` (keeps cliff-connected molecules together) or `Butina` clustering
  - Primary metric: **Cliff_RMSE** (RMSE computed only on molecules involved in activity cliffs)
  - Validation: Bootstrap CIs, y-scrambling, repeated CV
  - Feature selection: Boruta (optimized for small datasets ~350 compounds)
- **Output**: Best model per backbone (`best_model.joblib`), metrics, SHAP plots, permutation importance
- **Model Selection**: Ranked by Cliff_RMSE (minimize) → best overall model saved as `best_overall_model.joblib`

### 5. Molecular generation – CAFE LATE (`generator/`)

- **Purpose**: Generate novel molecules using AC-enriched fragments with late-stage QSAR correction
- **Methodology**:
  - **Island Algorithm**: Systematic exploration of user-defined core scaffolds with balanced fragment coverage
  - **Multi-objective scoring**: QSAR (activity prediction) + SA (synthetic accessibility, lower is better) + QED (drug-likeness)
  - **CAFE LATE**: Molecules containing AC-enriched fragments get QSAR probability boost (configurable weight)
  - **Selection**: Pareto front optimization → aggregate score ranking → diversity filtering (Butina clustering or maxmin greedy)
- **Output**: `results/generation/hits.csv` (diverse candidate molecules with scores and AC fragment flags)

## Configuration

The pipeline is fully configurable via `config.yml`. Key sections:

### Dataset Configuration
```yaml
TARGET_ID: "CHEMBL392"        # ChEMBL target ID for data fetching
THRESHOLD_NM: 10000            # Activity threshold (nM) for binary classification
```

### QSAR Evaluation
```yaml
QSAR_Eval:
  enable: true
  backbones: ["ecfp1024", "ecfp2048", "maccs", "descriptors"]
  engines: ["LogReg", "KNN", "SVC", "RF", "ExtraTrees", "GB", "XGB", "CatBoost", "BRF"]
  primary_metric: "Cliff_RMSE"  # Primary metric for model ranking
  split:
    mode: cliff_group           # Prevents data leakage
    test_size: 0.20
```

### Reverse QSAR
```yaml
ReverseQSAR:
  enable: true
  primary_metric: "AUPRC_active"  # Best for imbalanced data
  selection:
    cumsum_active_threshold: 0.9  # Cumulative importance threshold
```

### Molecular Generation
```yaml
Generator:
  enable: true
  generation:
    n_samples: 5000              # Number of molecules to generate
  scoring:
    qsar_weight: 0.4            # Weight for activity prediction
    sa_weight: 0.4              # Weight for synthetic accessibility
    qed_weight: 0.2             # Weight for drug-likeness
```

## Usage

### Complete Pipeline

Run all enabled stages:
```bash
  python main.py
```

### Step-by-Step Execution

**1. Prepare dataset first:**
```bash
  # In config.yml:
  REBUILD_DATASET: true
  AC_Analysis.enable: false
  ReverseQSAR.enable: false
  QSAR_Eval.enable: false
  Generator.enable: false

python main.py  # Stops after dataset creation
```

**2. Run remaining stages:**
```bash
  # In config.yml:
  REBUILD_DATASET: false
  AC_Analysis.enable: true
  ReverseQSAR.enable: true
  QSAR_Eval.enable: true
  Generator.enable: true

  
 python main.py  # Skips dataset, runs enabled stages
```

### Resume Functionality

Pipeline automatically skips completed stages. To force recomputation:
```yaml
QSAR_Eval:
  resume:
    enable: false     # Force recomputation
    overwrite: true    # Overwrite existing models
```

### Using Your Own Dataset

1. **Prepare CSV** with columns:
   - `ChEMBL_ID` (or `ID`)
   - `canonical_smiles`
   - `pIC50` (or `IC50_nM` for automatic conversion)
   - `activity_flag` (optional, computed from `THRESHOLD_NM`)

2. **Update `config.yml`**:
   ```yaml
   Paths:
     dataset: "path/to/your/dataset.csv"
   REBUILD_DATASET: false  # Use existing dataset
   ```



## Adaptability

The pipeline can be adapted for different targets and datasets:

### Using Different Targets

1. Change `TARGET_ID` in `config.yml` to your ChEMBL target
2. Or provide custom dataset (see "Using with Your Own Data")

### Adjusting for Different Dataset Sizes

- **Small datasets** (<200 compounds): Adjust `Cheminformatics.default_top_k` lower
- **Large datasets** (>1000 compounds): Can increase `default_top_k` and enable more features

### Custom Metrics

Primary metric can be changed in `config.yml`:
- QSAR: `primary_metric: "Cliff_RMSE"` (default)
- Reverse QSAR: `primary_metric: "AUPRC_active"` (default)

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{ac_aware_modeling,
  title = {Activity Cliff-Aware QSAR Modeling Pipeline},
  author = {Szostek, Tomasz},
  year = {2024},
  url = {https://github.com/yourusername/AC-aware-modeling},
  version = {3.0}
}
```


## License

MIT License - see LICENSE file for details.



