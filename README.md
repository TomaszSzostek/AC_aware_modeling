## Activity Cliff-Aware QSAR Modeling Pipeline

A Python framework for activity cliff-aware quantitative structure–activity relationship (QSAR) modeling and fragment-based molecular generation. The pipeline integrates cheminformatics and machine learning components to build predictive models that explicitly account for activity cliffs—structurally similar molecules with markedly different potencies.

## Key Features

- **Activity cliff-aware QSAR modeling**: Models evaluated with dedicated cliff-aware metrics (Cliff_RMSE).
- **Multiple molecular representations**: ECFP (1024/2048 bits), MACCS keys, RDKit and Mordred descriptors.
- **Feature selection**: Boruta-based feature selection tailored to small datasets (~350 compounds).
- **Group-based data splitting**: Splits that keep activity cliff groups together (cliff_group or Butina clustering) to limit data leakage.
- **Reverse QSAR**: Fragment extraction using BRICS, multi-model evaluation, SHAP importance and AC enrichment.
- **AC-aware molecular generation**: Fragment-based generator using AC-enriched fragments with multi-objective scoring (QSAR, SA, QED).
- **Reproducible evaluation**: Bootstrap confidence intervals, y-scrambling and repeated cross-validation.

## Table of Contents

 - [Installation](#installation)
 - [Quick Start](#quick-start)
 - [Pipeline Overview](#pipeline-overview)
 - [Configuration](#configuration)
 - [Usage](#usage)
 - [Methodology](#methodology)
 - [Output](#output)
 - [Reproducibility](#reproducibility)
 - [Citation](#citation)
 - [License](#license)

## Installation

### Prerequisites

- Python 3.10
- Conda (Miniconda or Anaconda)

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

3. **Verify installation**:
```bash
python -c "import rdkit; import sklearn; import shap; print('All dependencies imported successfully')"
```

### Dependencies

The pipeline requires the following key packages:
- **RDKit** (2023.09.5): Cheminformatics toolkit
- **scikit-learn** (≥1.4): Machine learning algorithms
- **SHAP** (≥0.45): Model interpretability
- **pandas**, **numpy**: Data manipulation
- **matplotlib**, **seaborn**: Visualization
- **networkx**: Graph algorithms for activity cliff grouping
- **Mordred**: Molecular descriptor calculation
- **Boruta_py**: Feature selection

Optional dependencies (for specific models):
- **xgboost**: XGBoost gradient boosting
- **catboost**: CatBoost gradient boosting
- **imbalanced-learn**: Balanced Random Forest

## Quick Start

1. **Prepare your dataset** (or use the provided example):
```bash
# Edit config.yml to set your ChEMBL target ID or provide custom data
# Then run:
python main.py
```

2. **Run the complete pipeline**:
```bash
python main.py
```

The pipeline will:
1. Fetch and prepare the dataset (if needed)
2. Detect activity cliffs (optional)
3. Extract AC-enriched fragments via Reverse QSAR
4. Evaluate QSAR models with activity cliff awareness
5. Generate novel molecules using AC-enriched fragments

## Pipeline Overview

The pipeline consists of five main stages:

### 1. Dataset preparation (`data_preparation.py`)

- **Input**: ChEMBL target ID or custom CSV files
- **Process**:
  - Fetches data from ChEMBL API
  - Merges with chemical space metadata
  - Cleans and standardizes compounds (MW < 900, LogP < 8)
  - Normalizes activity values to nM
  - Removes tautomer duplicates
- **Output**: `data/processed/final_dataset.csv`

### 2. Activity cliffs analysis (`AC_analysis/ac_analysis.py`)

- **Purpose**: Detect and visualize activity cliffs
- **Definition**: Pairs of molecules with:
  - Tanimoto similarity ≥ threshold (typically 0.8)
  - Potency difference ≥ threshold (typically 1.0 log units, i.e., 10-fold)
- **Output**: Activity cliff pairs, SALI scatterplot, t-SNE visualization
- **Usage**: Optional (can be skipped if cliffs are pre-computed)

### 3. Reverse QSAR (`reverse_qsar/defragmentation.py`)

- **Purpose**: Extract activity cliff-enriched molecular fragments
- **Methodology**:
  - BRICS fragmentation of molecules
  - Multi-model evaluation (9 ML algorithms)
  - SHAP-based feature importance
  - AC enrichment analysis
- **Output**: Selected fragments with AC flags (`results/reverse_QSAR/`)
- **Key Feature**: Fragments selected for their contribution to activity cliffs

### 4. Predictive modeling (`predictive_model/cliffaware_qsar.py`)

- **Purpose**: Build and evaluate QSAR models with activity cliff awareness
- **Features**:
  - Multiple molecular representations (ECFP, MACCS, descriptors)
  - Group-based data splitting (prevents leakage)
  - Activity cliff-aware metrics (Cliff_RMSE as primary metric)
  - Comprehensive validation (bootstrap, y-scrambling, CV)
- **Output**: Trained models, metrics, visualizations (`results/predictive_model/`)
- **Model Selection**: Models ranked by Cliff_RMSE (lower is better)

### 5. Molecular generation (`generator/`)

- **Purpose**: Generate novel molecules using AC-enriched fragments
- **Methodology**:
  - Island Algorithm: Systematic exploration of core scaffolds
  - Multi-objective scoring: QSAR (activity) + SA (synthetic accessibility) + QED (drug-likeness)
  - Pareto front optimization
  - Diversity selection (Butina clustering)
- **Output**: Candidate molecules (`results/generation/hits.csv`)

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

### File Paths
```yaml
Paths:
  dataset: "data/processed/final_dataset.csv"
  results_root: "results/predictive_model"
  fragments: "results/reverse_QSAR"
  ac_analysis: "results/AC_analysis"
```

**See `config.yml` for complete configuration options and detailed descriptions.**

## Usage

### Basic Usage

Run the complete pipeline:
```bash
python main.py
```

### Step-by-Step Execution

Each stage can be enabled/disabled via `config.yml`:

1. **Dataset Preparation**: Set `REBUILD_DATASET: true` to force rebuild
2. **Activity Cliffs Analysis**: Set `AC_Analysis.enable: true`
3. **Reverse QSAR**: Set `ReverseQSAR.enable: true`
4. **Predictive Modeling**: Set `QSAR_Eval.enable: true`
5. **Molecular Generation**: Set `Generator.enable: true`

### Resume Functionality

The pipeline supports resume functionality to avoid recomputing existing results:

```yaml
QSAR_Eval:
  resume:
    enable: true      # Skip existing artifacts, only recompute missing ones
    overwrite: false  # Set to true to force recomputation
```

### Using with Your Own Data

1. **Prepare your dataset CSV** with columns:
   - `ChEMBL_ID` (or `ID`)
   - `canonical_smiles`
   - `pIC50` (or provide `IC50_nM` for conversion)
   - `activity_flag` (optional, will be computed from threshold)

2. **Update `config.yml`**:
   - Set `Paths.dataset` to your dataset path
   - Disable ChEMBL fetching: Set `REBUILD_DATASET: false`
   - Adjust thresholds as needed

3. **Run the pipeline**:
```bash
python main.py
```

## Methodology

### Activity Cliff Definition

Activity cliffs are defined as pairs of structurally similar molecules (high Tanimoto similarity) that exhibit significantly different potency:

- **Similarity threshold**: Tanimoto similarity ≥ 0.8 (ECFP4 fingerprints)
- **Potency difference**: |ΔpIC50| ≥ 1.0 log unit (10-fold difference)

### Data Splitting Strategy

The pipeline uses **group-based splitting** to prevent data leakage:

- **Cliff-group mode**: Molecules connected by activity cliff edges stay in the same split
- **Butina mode**: Molecules clustered by structural similarity (scaffold fingerprints)

This ensures that structurally similar compounds (including activity cliffs) remain together, preventing information leakage between train and test sets.

### Model Evaluation

Models are evaluated using activity cliff-aware metrics:

- **Cliff_RMSE** (primary metric): RMSE computed only on molecules involved in activity cliffs
  - Lower Cliff_RMSE indicates better prediction of activity cliffs
  - Models are ranked by Cliff_RMSE (minimize)
  
- **Standard metrics**: AUROC, AUPRC_active, F1, MCC, Accuracy, Brier score, ECE

- **Validation**: Bootstrap confidence intervals, y-scrambling, repeated cross-validation

### Feature Engineering

**Fingerprints**:
- ECFP1024/2048: Extended Connectivity Fingerprints (radius=2, ECFP4)
- MACCS: 166 structural keys

**Descriptors**:
- RDKit 2D descriptors (~200 descriptors)
- Mordred 2D descriptors (if available)
- Boruta feature selection (optimized for small datasets)

### Reverse QSAR

1. **BRICS Fragmentation**: Breaks molecules into chemically meaningful fragments
2. **Multi-Model Training**: 9 ML algorithms evaluated
3. **SHAP Analysis**: Computes feature importance for fragments
4. **AC Enrichment**: Identifies fragments enriched in activity cliffs
5. **Fragment Selection**: Selects fragments until cumulative importance reaches threshold

### Molecular Generation

1. **Island Algorithm**: Systematic exploration of core scaffolds
   - Ensures balanced coverage of all fragments and cores
   - Adaptive (bandit) sampling for exploration-exploitation balance

2. **Multi-Objective Scoring**:
   - **QSAR**: Activity prediction from best model
   - **SA**: Synthetic accessibility (lower is better)
   - **QED**: Drug-likeness score

3. **Selection Strategy**:
   - Pareto front optimization
   - Ranking by aggregate score
   - Diversity selection (Butina clustering or maxmin greedy)

## Output

### Predictive Modeling Outputs

Located in `results/predictive_model/`:

- **Per-backbone subdirectories** (`ecfp1024/`, `ecfp2048/`, `maccs/`, `descriptors/`):
  - Trained models (`.joblib`)
  - Metrics (`.json`, `.csv`)
  - Visualizations (ROC, PR, confusion matrix, probability histograms)
  - SHAP plots (bar, beeswarm)
  - Permutation importance results
  
- **Global outputs**:
  - `best_overall_model.joblib`: Best model across all backbones
  - `all_backbones_metrics.csv`: Comparison across all backbones
  - `backbone_comparison_RMSEcliff.png`: Visualization of backbone performance

### Reverse QSAR Outputs

Located in `results/reverse_QSAR/`:

- `model_comparison.png`: Model performance comparison
- `model_metrics.csv`: Detailed metrics per model
- `reinvent_fragments_all.csv`: All AC-enriched fragments
- Per-model subdirectories:
  - `selected_fragments_with_ACflag.csv`: Selected fragments with AC enrichment flags
  - `ac_enrichment.csv`: AC enrichment analysis
  - `plots/`: SHAP visualizations

### Molecular Generation Outputs

Located in `results/generation/`:

- `hits.csv`: Final diverse set of candidate molecules
  - Columns: SMILES, core, fragments (AC-flagged), qsar, qed, sa, aggregate_score
- `generation_summary.json`: Generation statistics
- `post_score.csv`: All scored molecules (optional)

### Activity Cliffs Analysis Outputs

Located in `results/AC_analysis/`:

- `activity_cliffs.csv`: Detected activity cliff pairs
- `sali_scatter.png`: SALI scatterplot visualization
- `tsne_cliffs.png`: t-SNE embedding with cliff highlighting

## Reproducibility

The pipeline is designed for full reproducibility:

1. **Random seeds**: All random operations use configurable seeds (default: 42)
2. **Deterministic algorithms**: Deterministic implementations where possible
3. **Configuration-driven**: All parameters controlled via `config.yml`
4. **Version tracking**: Environment locked via `environment.yaml`

To reproduce results:
```bash
# Use exact same environment
conda env create -f environment.yaml
conda activate ac-aware-modeling

# Use same config.yml (ensure all seeds match)
python main.py
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

### Key References

- **Activity Cliffs**: Stumpfe, D., & Bajorath, J. (2011). Exploring activity cliffs in medicinal chemistry. *Journal of Medicinal Chemistry*, 54(1), 26-47.
- **Boruta Feature Selection**: Kursa, M. B., & Rudnicki, W. R. (2010). Feature selection with the Boruta package. *Journal of Statistical Software*, 36(11), 1-13.
- **Group-Based Splitting**: Sheridan, R. P. (2013). Time-split based validation as a practical approach to QSAR model validation. *Journal of Chemical Information and Modeling*, 53(4), 783-790.
- **SHAP Explanations**: Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NIPS*, 30, 4765-4774.

## License

MIT License - see LICENSE file for details.

## Authors

- **Tomasz Szostek** - Medical University of Warsaw
- AC-Aware Modeling Team

## Acknowledgments

- RDKit community for cheminformatics tools
- ChEMBL database for compound data
- Contributors to scikit-learn, SHAP, and other open-source libraries

## Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**Version**: 3.0  
**Last Updated**: 2024  
**Status**: Production-ready for publication

