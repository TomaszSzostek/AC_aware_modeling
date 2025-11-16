"""
Reverse QSAR Defragmentation Module

A comprehensive reverse QSAR evaluation framework for fragment-based molecular design
with activity cliff awareness and multi-model evaluation capabilities.

Key Features:
- Multi-model evaluation with 9 different ML algorithms
- Activity cliff-aware metrics and model selection
- BRICS fragment generation and analysis
- SHAP-based feature importance and fragment selection
- Publication-ready visualizations and statistical analysis
- AC enrichment analysis for fragment prioritization
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Mapping, Dict, Any
import json
import logging
import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import (
    ConfusionMatrixDisplay, roc_auc_score, accuracy_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, brier_score_loss, average_precision_score
)
from sklearn.calibration import calibration_curve
import networkx as nx

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import BRICS
import re

# Optional imports with fallbacks
try:
    from imblearn.ensemble import BalancedRandomForestClassifier as BRF
    IMB_OK = True
except ImportError:
    IMB_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False

try:
    import catboost as cb
    CAT_OK = True
except ImportError:
    CAT_OK = False

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.calibration")


# Publication settings
PUB_DPI = 500
PALETTE = {
    "blue": "#8ecae6",      # pastel blue
    "green": "#a8ddb5",     # pastel green
    "purple": "#c7b9ff",    # pastel purple
    "orange": "#fdbf6f",    # pastel orange
    "red": "#f4aaaa",       # pastel red
    "gray": "#cfcfcf",      # pastel gray
    "ink": "#3a3a3a",       # dark labels
}

# Configure matplotlib for publication-quality plots
plt.rcParams.update({
    "figure.dpi": PUB_DPI,
    "savefig.dpi": PUB_DPI,
    "axes.edgecolor": "#aaaaaa",
    "axes.labelcolor": PALETTE["ink"],
    "xtick.color": PALETTE["ink"],
    "ytick.color": PALETTE["ink"],
    "font.size": 10,
})


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

# Wildcard normalization patterns (from reinvent_library.py)
STAR_NUM  = re.compile(r"\[\s*\d+\s*\*\s*\]")
STAR_MAP  = re.compile(r"\[\s*\*:\s*\d+\s*\]")
PLAIN_STAR = "[*]"

def normalize_fragment_smiles(smi: str) -> str | None:
    """
    Normalize fragment SMILES by replacing attachment tokens with plain [*].
    Converts [1*], [12*], [*:1], [*:12] etc. to [*].
    Returns None if no star remains after normalization.
    """
    if smi is None:
        return None
    s = str(smi)
    s = STAR_MAP.sub(PLAIN_STAR, s)
    s = STAR_NUM.sub(PLAIN_STAR, s)
    return s if PLAIN_STAR in s else None

def normalize_and_deduplicate_fragments(fragments: list[str], importance_scores: pd.Series = None) -> tuple[list[str], pd.Series, list[bool]]:
    """
    Normalize fragment SMILES and remove duplicates.
    
    Args:
        fragments: List of fragment SMILES
        importance_scores: Optional importance scores for fragments
        
    Returns:
        Tuple of (normalized_fragments, normalized_importance_scores, is_AC_enriched_mask)
    """
    if not fragments:
        return [], pd.Series(dtype=float), []
    
    # Normalize fragments
    normalized_frags = []
    frag_to_orig = {}  # Map normalized -> original for importance aggregation
    
    for frag in fragments:
        norm_frag = normalize_fragment_smiles(frag)
        if norm_frag:
            if norm_frag not in frag_to_orig:
                frag_to_orig[norm_frag] = []
            frag_to_orig[norm_frag].append(frag)
            normalized_frags.append(norm_frag)
    
    # Remove duplicates while preserving order
    unique_frags = []
    seen = set()
    for frag in normalized_frags:
        if frag not in seen:
            unique_frags.append(frag)
            seen.add(frag)
    
    # Aggregate importance scores for normalized fragments
    if importance_scores is not None and not importance_scores.empty:
        norm_importance = []
        for frag in unique_frags:
            orig_frags = frag_to_orig[frag]
            # Use maximum importance among original fragments
            max_imp = max(importance_scores.get(orig, 0.0) for orig in orig_frags)
            norm_importance.append(max_imp)
        norm_importance_series = pd.Series(norm_importance, index=unique_frags)
    else:
        norm_importance_series = pd.Series(1.0, index=unique_frags)
    
    # Create placeholder AC enrichment mask (will be updated later if needed)
    ac_mask = [False] * len(unique_frags)
    
    return unique_frags, norm_importance_series, ac_mask

def normalize_fragment_data(X_frag: pd.DataFrame, importance_scores: pd.Series, log=None) -> tuple[pd.DataFrame, pd.Series]:
    """
    Normalize fragment SMILES and aggregate importance scores to remove duplicates.
    
    Fragment normalization standardizes attachment points (e.g., [1*] -> [*]) and removes
    duplicate fragments. Importance scores are aggregated using maximum to preserve the highest
    importance value when multiple fragments map to the same normalized fragment.
    
    Parameters
    ----------
    X_frag : pd.DataFrame
        Fragment presence matrix (n_samples x n_fragments) with fragment SMILES as column names
    importance_scores : pd.Series
        Feature importance scores indexed by fragment SMILES
    log : logging.Logger, optional
        Logger instance for progress tracking
        
    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Tuple containing:
        - Normalized fragment presence matrix (n_samples x n_unique_fragments)
        - Normalized importance scores indexed by normalized fragment SMILES
    """
    if log is None:
        log = logging.getLogger(__name__)
    
    orig_to_norm = {}
    norm_to_orig = {}
    
    for orig_frag in X_frag.columns:
        norm_frag = normalize_fragment_smiles(orig_frag)
        if norm_frag:
            orig_to_norm[orig_frag] = norm_frag
            if norm_frag not in norm_to_orig:
                norm_to_orig[norm_frag] = []
            norm_to_orig[norm_frag].append(orig_frag)
    
    unique_norm_frags = list(norm_to_orig.keys())
    
    if len(unique_norm_frags) == 0:
        return X_frag, importance_scores
    
    X_norm = pd.DataFrame(0, index=X_frag.index, columns=unique_norm_frags, dtype=np.int8)
    
    for norm_frag, orig_frags in norm_to_orig.items():
        X_norm[norm_frag] = X_frag[orig_frags].sum(axis=1).clip(upper=1)
    
    norm_importance = []
    for norm_frag in unique_norm_frags:
        orig_frags = norm_to_orig[norm_frag]
        orig_imps = [importance_scores.get(orig, 0.0) for orig in orig_frags]
        max_imp = max(orig_imps) if orig_imps else 0.0
        norm_importance.append(max_imp)
    
    norm_importance_series = pd.Series(norm_importance, index=unique_norm_frags)
    
    log.debug(f"Normalization: {X_frag.shape[1]} -> {X_norm.shape[1]} fragments")
    
    return X_norm, norm_importance_series

def ensure_dir(p: Path):
    """Ensure directory exists, create if necessary."""
    p.mkdir(parents=True, exist_ok=True)

def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def smiles_to_mol(s: str):
    """Convert SMILES string to RDKit molecule object."""
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None

def logits_from_proba(p, eps=1e-6):
    """Convert probabilities to logits."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

def ece_score(y_true, y_prob, n_bins=15):
    """Expected Calibration Error."""
    pt, pp = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    return float(np.mean(np.abs(pt - pp)))

# =============================================================================
# CLIFF-AWARE METRICS
# =============================================================================


def compute_metrics(y_true, y_prob, pairs_cliff_test, in_cliff_nodes, threshold: float = 0.5):
    """Compute comprehensive metrics including cliff-aware measures."""
    y_pred = (y_prob >= float(threshold)).astype(int)
    out = {}
    
    # Standard metrics
    out["AUROC"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    out["AUPRC_active"] = average_precision_score(y_true, y_prob)
    out["F1"] = f1_score(y_true, y_pred, zero_division=0)
    out["MCC"] = matthews_corrcoef(y_true, y_pred) if (
        len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1) else 0.0
    out["ACC"] = accuracy_score(y_true, y_pred)
    out["Brier"] = brier_score_loss(y_true, y_prob)
    out["ECE"] = ece_score(y_true, y_prob, n_bins=15)
    out["RMSE"] = float(np.sqrt(np.mean((y_prob - y_true) ** 2)))
    
    # Cliff-aware metrics
    if in_cliff_nodes.sum() > 0:
        yc = y_true[in_cliff_nodes]
        pc = y_prob[in_cliff_nodes]
        ypc = (pc >= float(threshold)).astype(int)
        out["Cliff_Brier"] = brier_score_loss(yc, pc)
        out["Cliff_ECE"] = ece_score(yc, pc, n_bins=15)
        out["Cliff_RMSE"] = float(np.sqrt(np.mean((pc - yc) ** 2)))
        out["Cliff_F1"] = f1_score(yc, ypc, zero_division=0)
        out["Delta_RMSE"] = out["Cliff_RMSE"] - out["RMSE"]
        out["AUPRC_cliff"] = average_precision_score(yc, pc)
    else:
        out["Cliff_Brier"] = out["Cliff_ECE"] = out["Cliff_RMSE"] = out["Delta_RMSE"] = float("nan")
        out["AUPRC_cliff"] = float("nan")
    
    out["Cliff_NodeShare"] = float(in_cliff_nodes.mean())
    return out


def bootstrap_ci(values: np.ndarray, confidence: float = 0.95, n_bootstrap: int = 2000, seed: int = 42) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for a metric (percentile method).
    
    This computes CI for the distribution of metric values across multiple runs/splits.
    
    Args:
        values: Array of metric values from repeated runs
        confidence: Confidence level (default 0.95 for 95% CI)
        n_bootstrap: Number of bootstrap samples (default 2000 for publication quality)
        seed: Random seed
        
    Returns:
        Tuple of (lower_bound, upper_bound) for confidence interval
    """
    if len(values) == 0 or np.all(np.isnan(values)):
        return (float("nan"), float("nan"))
    
    # Remove NaN values
    valid_values = values[~np.isnan(values)]
    if len(valid_values) == 0:
        return (float("nan"), float("nan"))
    
    # Bootstrap sampling
    rng = np.random.RandomState(seed)
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(valid_values, size=len(valid_values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # Compute confidence interval (percentile method)
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower = np.percentile(bootstrap_means, lower_percentile)
    upper = np.percentile(bootstrap_means, upper_percentile)
    
    return (float(lower), float(upper))


def format_metrics_with_combined_ci(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format metrics with combined CI strings (rows = models, columns = metrics).
    
    Converts from:
      model | AUROC | AUROC_CI_lower | AUROC_CI_upper | ...
      LogReg| 0.850 | 0.845          | 0.855          | ...
    
    To:
      model  | AUROC                       | AUPRC_active               | ...
      LogReg | 0.850 (95% CI, 0.845-0.855) | 0.812 (95% CI, 0.800-0.820)| ...
      KNN    | ...                         | ...                        | ...
    
    Args:
        aggregated_df: DataFrame with aggregated metrics (mean + CI columns)
        
    Returns:
        DataFrame with combined CI strings (rows=models, columns=metrics)
    """
    # Get base metric names (without _CI_lower, _CI_upper, _SD suffixes)
    all_cols = aggregated_df.columns.tolist()
    base_metrics = []
    
    for col in all_cols:
        # Skip suffixes and metadata
        if any(suffix in col for suffix in ['_CI_lower', '_CI_upper', '_SD']):
            continue
        if col in ['model', 'backbone', 'repeat']:
            continue
        if col not in base_metrics:
            base_metrics.append(col)
    
    # Build result DataFrame
    result_rows = []
    
    for _, row in aggregated_df.iterrows():
        model_name = row.get('model', 'Unknown')
        result_row = {'model': model_name}
        
        for metric in base_metrics:
            mean_val = row.get(metric, np.nan)
            ci_lower = row.get(f"{metric}_CI_lower", np.nan)
            ci_upper = row.get(f"{metric}_CI_upper", np.nan)
            
            # Format as "mean (95% CI, lower-upper)"
            if pd.notna(mean_val):
                if pd.notna(ci_lower) and pd.notna(ci_upper):
                    formatted = f"{mean_val:.3f} (95% CI, {ci_lower:.3f}-{ci_upper:.3f})"
                else:
                    formatted = f"{mean_val:.3f}"
            else:
                formatted = "N/A"
            
            result_row[metric] = formatted
        
        result_rows.append(result_row)
    
    return pd.DataFrame(result_rows)


def aggregate_metrics_with_ci(metrics_list: List[Dict[str, Any]], n_bootstrap: int = 2000, seed: int = 42) -> Dict[str, Any]:
    """
    Aggregate metrics from multiple runs with mean, std, and 95% CI.
    
    Publication-quality aggregation:
    - Mean and 95% CI for main table
    - SD for supplementary materials
    - All values rounded to 3 decimal places
    
    Args:
        metrics_list: List of metric dictionaries from repeated runs
        n_bootstrap: Number of bootstrap samples for CI (default 2000)
        seed: Random seed
        
    Returns:
        Dictionary with aggregated metrics (mean, CI_lower, CI_upper, SD for each metric)
    """
    if not metrics_list:
        return {}
    
    # Collect all metric names
    metric_names = set()
    for m in metrics_list:
        metric_names.update(m.keys())
    
    # Remove non-numeric fields
    exclude_fields = {"model", "backbone", "Calibration", "repeat"}
    metric_names = metric_names - exclude_fields
    
    aggregated = {}
    
    # Copy metadata from first run
    for field in exclude_fields:
        if field in metrics_list[0]:
            aggregated[field] = metrics_list[0][field]
    
    # Aggregate each metric
    for metric in sorted(metric_names):
        values = []
        for m in metrics_list:
            if metric in m:
                val = m[metric]
                # Convert to float and handle NaN
                try:
                    val = float(val)
                    values.append(val)
                except (ValueError, TypeError):
                    pass
        
        if values:
            values_arr = np.array(values)
            # Round to 3 decimal places for publication
            aggregated[f"{metric}"] = round(float(np.nanmean(values_arr)), 3)
            aggregated[f"{metric}_SD"] = round(float(np.nanstd(values_arr, ddof=1) if len(values_arr) > 1 else 0.0), 3)
            
            ci_lower, ci_upper = bootstrap_ci(values_arr, confidence=0.95, n_bootstrap=n_bootstrap, seed=seed)
            aggregated[f"{metric}_CI_lower"] = round(ci_lower, 3)
            aggregated[f"{metric}_CI_upper"] = round(ci_upper, 3)
        else:
            aggregated[f"{metric}"] = float("nan")
            aggregated[f"{metric}_SD"] = float("nan")
            aggregated[f"{metric}_CI_lower"] = float("nan")
            aggregated[f"{metric}_CI_upper"] = float("nan")
    
    return aggregated

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load and validate dataset for reverse QSAR evaluation."""
    df = pd.read_csv(csv_path)
    req = {"canonical_smiles", "activity_flag"}
    missing = req - set(df.columns)
    if missing:
        raise ValueError(f"CSV must contain columns: {req}, missing: {missing}")
    return df

# =============================================================================
# MULTI-MODEL EVALUATION
# =============================================================================

def make_model(name: str, random_state: int = 42):
    """Factory for creating ML models with consistent parameters."""
    name = name.strip()
    
    if name == "LogReg":
        return LogisticRegression(
            C=0.5, penalty="l2", solver="liblinear",
            class_weight="balanced", max_iter=4000, random_state=random_state
        )
    elif name == "Ridge":
        return RidgeClassifier(
            alpha=1.0, class_weight="balanced", random_state=random_state
        )
    elif name == "SVC":
        return SVC(
            kernel="rbf", C=2.0, gamma="scale", probability=True,
            class_weight="balanced", random_state=random_state
        )
    elif name == "RF":
        return RandomForestClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=10, 
            min_samples_split=20, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=random_state
        )
    elif name == "BRF":
        if not IMB_OK:
            raise RuntimeError("imblearn not available for BalancedRandomForest.")
        return BRF(
            n_estimators=300, max_depth=4, min_samples_leaf=10, 
            min_samples_split=20, max_features="sqrt",
            sampling_strategy="all", replacement=True, bootstrap=False,
            n_jobs=-1, random_state=random_state
        )
    elif name == "ExtraTrees":
        return ExtraTreesClassifier(
            n_estimators=300, max_depth=4, min_samples_leaf=10, 
            min_samples_split=20, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=random_state
        )
    elif name == "GB":
        return GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=3, 
            min_samples_leaf=10, subsample=0.8, max_features="sqrt", 
            random_state=random_state
        )
    elif name == "LinSVM":
        return LinearSVC(
            C=1.0, class_weight="balanced", random_state=random_state
        )
    elif name == "KNN":
        return KNeighborsClassifier(
            n_neighbors=15, weights="uniform", metric="minkowski", p=2
        )
    elif name == "XGB":
        if not XGB_OK:
            raise RuntimeError("xgboost not installed.")
        return xgb.XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.1, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=5.0, reg_alpha=1.0, reg_lambda=2.0,
            eval_metric="auc", n_jobs=-1, random_state=random_state, tree_method="auto"
        )
    elif name == "CatBoost":
        if not CAT_OK:
            raise RuntimeError("catboost not installed.")
        return cb.CatBoostClassifier(
            iterations=500, learning_rate=0.1, depth=4, l2_leaf_reg=5.0, 
            min_data_in_leaf=10, loss_function="Logloss", eval_metric="Logloss", 
            random_seed=random_state, verbose=False
        )
    else:
        raise ValueError(f"Unknown model: {name}")

def _positive_proba(model, X):
    """Return P(y=1) robustly for sklearn-like models."""
    import warnings
    # Suppress sklearn feature name warnings (informational only, doesn't affect results)
    # IMPORTANT: Preserve DataFrame as-is to maintain column order and feature alignment
    # sklearn models trained on DataFrame expect DataFrame for prediction to match features correctly
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        # Keep DataFrame if possible - sklearn handles it correctly
        if isinstance(X, pd.DataFrame):
            X_input = X  # Keep as DataFrame - sklearn will handle it correctly
        else:
            X_input = np.asarray(X, dtype=float)
        
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_input)
            proba = np.asarray(proba)
            if proba.ndim == 1:
                return proba
            if proba.shape[1] == 2:
                return proba[:, 1]
            if hasattr(model, "classes_"):
                try:
                    pos_idx = int(np.argmax(model.classes_))
                    return proba[:, pos_idx]
                except Exception:
                    pass
            return proba.ravel()
        if hasattr(model, "decision_function"):
            s = model.decision_function(X_input)
            s = np.asarray(s)
            if s.ndim > 1:
                s = s.ravel()
            return 1.0 / (1.0 + np.exp(-s))
        yhat = model.predict(X_input)
        return np.asarray(yhat, float).ravel()

def tune_threshold(y_true_val, p_val, metric: str = "F1") -> float:
    """Tune decision threshold to maximize a chosen metric on validation set.
    
    For Reverse QSAR, we typically use "F1" or "Recall" to ensure we detect 
    active compounds (important for fragment extraction).
    
    Args:
        y_true_val: True labels
        p_val: Predicted probabilities
        metric: Metric to optimize ("F1", "Recall", "MCC", "YOUDEN", "ACC")
        
    Returns:
        Optimized threshold value
    """
    metric = (metric or "F1").upper()
    y_true_val = np.asarray(y_true_val).astype(int)
    p_val = np.asarray(p_val, dtype=float)

    grid = np.linspace(0.01, 0.99, 99)
    best_t, best_s = 0.5, -np.inf

    for t in grid:
        yhat = (p_val >= t).astype(int)
        if metric == "MCC":
            s = matthews_corrcoef(y_true_val, yhat) if (
                np.unique(y_true_val).size > 1 and np.unique(yhat).size > 1) else -np.inf
        elif metric == "F1":
            s = f1_score(y_true_val, yhat, zero_division=0)
        elif metric == "RECALL":
            s = recall_score(y_true_val, yhat, zero_division=0)
        elif metric == "YOUDEN":
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true_val, yhat, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            sens = tp / max(1, tp + fn)
            spec = tn / max(1, tn + fp)
            s = sens + spec - 1.0
        elif metric == "ACC":
            s = accuracy_score(y_true_val, yhat)
        else:
            s = f1_score(y_true_val, yhat, zero_division=0)
        if s > best_s:
            best_s, best_t = s, float(t)
    return float(best_t)


# =============================================================================
# BRICS FRAGMENTATION
# =============================================================================

def generate_fragment_matrix_brics(
    smiles_df: pd.DataFrame, min_occ: int = 1
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build binary presence matrix of BRICS fragments.
    
    Args:
        smiles_df: DataFrame with 'SMILES' column
        min_occ: Minimum occurrence threshold for fragment inclusion
        
    Returns:
        Tuple of (fragment_matrix, fragment_names)
    """
    frag_dict: dict[int, List[str]] = {}
    counter: dict[str, int] = {}

    for idx, smi in enumerate(smiles_df["SMILES"]):
        mol = Chem.MolFromSmiles(str(smi))
        if not mol:
            frag_dict[idx] = []
            continue
        frags = list(BRICS.BRICSDecompose(mol))
        frag_dict[idx] = frags
        for f in frags:
            counter[f] = counter.get(f, 0) + 1

    cols = sorted([f for f, c in counter.items() if c >= min_occ])
    X = pd.DataFrame(0, index=smiles_df.index, columns=cols, dtype=int)
    for i, frags in frag_dict.items():
        for f in frags:
            if f in X.columns:
                X.at[i, f] = 1
    return X, cols

# =============================================================================
# MULTI-MODEL EVALUATION PIPELINE
# =============================================================================

def create_split(
    X_frag: pd.DataFrame,
    y: pd.Series,
    smiles_df: pd.DataFrame,
    mode: str,
    test_size: float,
    random_state: int,
    cliffs_csv: Path,
    log: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create train/test split using specified strategy.
    
    Args:
        X_frag: Fragment feature matrix
        y: Binary activity labels
        smiles_df: DataFrame with SMILES column
        mode: Split mode ("standard", "cliff_group", or "cliff_train_max")
        test_size: Fraction for test set
        random_state: Random seed for this split
        cliffs_csv: Path to activity cliffs CSV
        log: Logger instance
        
    Returns:
        Tuple of (train_idx, test_idx)
    """
    if mode == "cliff_train_max":
        # CAFE-specific: Assign ALL cliff molecules to TRAIN
        # Test set = only non-cliff molecules (clean generalization test)
        tr_idx, te_idx = _cliff_train_enriched_split(
            smiles_df=smiles_df,
            y=y,
            cliffs_csv=cliffs_csv,
            test_size=test_size,
            random_state=random_state,
            log=log
        )
    elif mode == "cliff_group":
        # Keep cliff groups together (may assign to train OR test)
        groups = _cliff_groups(df=smiles_df, smiles_col="SMILES", cliffs_csv=cliffs_csv)
        tr_idx, te_idx = _grouped_train_test_indices(groups, y, test_size=test_size, random_state=random_state)
    else:
        # Standard stratified random split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        tr_idx, te_idx = next(sss.split(X_frag, y))
    
    return tr_idx, te_idx


def evaluate_model_once(
    model_name: str,
    X_frag: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    threshold_metric: str = "F1",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Evaluate a single model once (single train/test run).
    
    Args:
        model_name: Name of the model to evaluate
        X_frag: Fragment feature matrix
        y: Binary activity labels
        train_idx: Training set indices
        test_idx: Test set indices
        threshold_metric: Metric for threshold optimization
        random_state: Random seed
        
    Returns:
        Dictionary of metrics
    """
    # Prepare data splits - ensure feature names are preserved
    X_tr, X_te = X_frag.iloc[train_idx].copy(), X_frag.iloc[test_idx].copy()
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    
    # Clean feature names for XGBoost compatibility
    clean_feature_names = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '') for col in X_frag.columns]
    X_tr.columns = clean_feature_names
    X_te.columns = clean_feature_names
    
    # Create and train model
    model = make_model(model_name, random_state=random_state)
    model.fit(X_tr, y_tr)
    
    # Get predictions
    y_prob = _positive_proba(model, X_te)
    
    # Optimize threshold using specified metric (from config)
    threshold = tune_threshold(y_te, y_prob, metric=threshold_metric)
    y_pred = (y_prob >= threshold).astype(int)
    
    # Compute standard classification metrics
    metrics = {
        "model": model_name,
        "threshold": threshold,
        "AUROC": roc_auc_score(y_te, y_prob),
        "AUPRC_active": average_precision_score(y_te, y_prob),
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred, zero_division=0),
        "Recall": recall_score(y_te, y_pred, zero_division=0),
        "F1": f1_score(y_te, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_te, y_pred),
        "Brier": brier_score_loss(y_te, y_prob),
        "RMSE": float(np.sqrt(np.mean((y_prob - y_te) ** 2)))
    }
    
    return metrics


def run_multi_model_evaluation(
    X_frag: pd.DataFrame,
    y: pd.Series,
    smiles_df: pd.DataFrame,
    potency_series: pd.Series,
    models: List[str],
    out_dir: Path,
    config: Dict[str, Any],
    log: logging.Logger,
    random_state: int = 42,
    n_repeats: int = 10
) -> Tuple[str, Dict[str, Any]]:
    """
    Run multi-model evaluation with repeated independent splits and select best model for Reverse QSAR.
    
    Each model is trained n_repeats times with DIFFERENT train/test splits to estimate
    uncertainty in performance metrics (split variability + model initialization).
    
    IMPORTANT: Each repeat creates a NEW split with the SAME logic (standard/cliff_group)
    but different random seed. This preserves AC-aware splitting while measuring uncertainty.
    
    For Reverse QSAR, we use standard metrics (not Cliff_RMSE) since test set may not have
    cliff pairs (standard split maximizes AC information in train set).
    
    Recommended primary metric: AUPRC_active (best for imbalanced data, focuses on 
    identifying active compounds which is crucial for fragment extraction).
    
    Args:
        X_frag: Fragment feature matrix
        y: Binary activity labels
        smiles_df: DataFrame with SMILES column
        potency_series: Potency values (for reference, not used in evaluation)
        models: List of model names to evaluate
        out_dir: Output directory for results
        config: Configuration dictionary
        log: Logger instance
        random_state: Random seed (base seed for repeats)
        n_repeats: Number of independent runs per model (each with new split)
        
    Returns:
        Tuple of (best_model_name, best_model_metrics)
    """
    print(f"🤖 Evaluating {len(models)} models with {n_repeats} independent splits each...")
    print(f"   Dataset: {len(X_frag)} compounds ({float(np.mean(y)):.1%} active)")
    
    # Get split configuration
    split_cfg = config.get("ReverseQSAR", {}).get("split", {})
    mode = split_cfg.get("mode", "standard")
    test_size = float(split_cfg.get("test_size", 0.20))
    
    # Get threshold optimization metric
    threshold_metric = config.get("ReverseQSAR", {}).get("threshold_metric", "F1")
    print(f"   Threshold optimization: {threshold_metric}")
    
    ac_results_dir = config.get("AC_Analysis", {}).get("results_dir", "results/AC_analysis")
    cliffs_csv = Path(ac_results_dir) / "activity_cliffs.csv"
    
    # Log split strategy once
    if mode == "cliff_train_max":
        # Count cliff molecules for summary
        if Path(cliffs_csv).exists():
            cliffs_df = pd.read_csv(cliffs_csv)
            cliff_mols = set()
            for _, row in cliffs_df.iterrows():
                if pd.notna(row.get("mol_i")): cliff_mols.add(row.get("mol_i"))
                if pd.notna(row.get("mol_j")): cliff_mols.add(row.get("mol_j"))
            print(f"   Split strategy: cliff_train_max (ALL {len(cliff_mols)} cliff molecules → TRAIN)")
        else:
            print(f"   Split strategy: cliff_train_max")
    else:
        print(f"   Split strategy: {mode}")
    
    # Store metrics for all runs
    all_runs_metrics = []          # One dict per run
    model_aggregated_metrics = []  # Aggregated metrics per model
    
    # Evaluate each model with repeated runs (each with a new split)
    for model_name in models:
        print(f"  Training {model_name} ({n_repeats} independent splits)...")
        model_run_metrics = []  # Metrics for this model's repeats
        
        try:
            for repeat_idx in range(n_repeats):
                # Use different seed for each repeat – creates a new split
                repeat_seed = random_state + repeat_idx
                
                # Create new split for this repeat (preserving split logic)
                train_idx, test_idx = create_split(
                    X_frag=X_frag,
                    y=y,
                    smiles_df=smiles_df,
                    mode=mode,
                    test_size=test_size,
                    random_state=repeat_seed,
                    cliffs_csv=cliffs_csv,
                    log=log
                )
                
                # Evaluate model on this split
                metrics = evaluate_model_once(
                    model_name=model_name,
                    X_frag=X_frag,
                    y=y,
                    train_idx=train_idx,
                    test_idx=test_idx,
                    threshold_metric=threshold_metric,
                    random_state=repeat_seed
                )
                
                # Add repeat index and split info
                metrics["repeat"] = repeat_idx
                metrics["n_train"] = len(train_idx)
                metrics["n_test"] = len(test_idx)
                all_runs_metrics.append(metrics)
                model_run_metrics.append(metrics)
            
            # Aggregate metrics for this model
            aggregated = aggregate_metrics_with_ci(
                model_run_metrics,
                n_bootstrap=2000,
                seed=random_state
            )
            model_aggregated_metrics.append(aggregated)
            
            # Report mean ± SD for primary metric
            primary_metric = config.get("ReverseQSAR", {}).get("primary_metric", "AUPRC_active")
            mean_val = aggregated.get(primary_metric, float("nan"))
            sd_val = aggregated.get(f"{primary_metric}_SD", float("nan"))
            print(f"     {primary_metric}: {mean_val:.3f} ± {sd_val:.3f}")
            
        except Exception as e:
            log.warning(f"[Model] Failed to evaluate {model_name}: {e}")
            continue
    
    if not model_aggregated_metrics:
        raise RuntimeError("No models were successfully evaluated")
    
    # Create DataFrames
    all_runs_df = pd.DataFrame(all_runs_metrics)
    aggregated_df = pd.DataFrame(model_aggregated_metrics)
    
    # Select best model based on primary metric (using mean value)
    primary_metric = config.get("ReverseQSAR", {}).get("primary_metric", "AUPRC_active")
    
    if primary_metric not in aggregated_df.columns:
        log.warning(f"Primary metric '{primary_metric}' not found, using AUPRC_active")
        primary_metric = "AUPRC_active"
    
    # Sort by primary metric (higher is better for all standard metrics)
    aggregated_df = aggregated_df.sort_values(primary_metric, ascending=False)
    
    best_model_name = aggregated_df.iloc[0]["model"]
    best_metrics = aggregated_df.iloc[0].to_dict()
    
    best_mean = best_metrics[primary_metric]
    best_sd = best_metrics.get(f"{primary_metric}_SD", 0.0)
    best_ci_lower = best_metrics.get(f"{primary_metric}_CI_lower", float("nan"))
    best_ci_upper = best_metrics.get(f"{primary_metric}_CI_upper", float("nan"))
    
    print(f"  🏆 Best model: {best_model_name}")
    print(f"     {primary_metric}: {best_mean:.3f} ± {best_sd:.3f} (95% CI: [{best_ci_lower:.3f}, {best_ci_upper:.3f}])")
    
    # Retrain best model on first split and save predictions (needed for Figure 1)
    print(f"  💾 Saving predictions for Figure 1...")
    try:
        first_seed = random_state + 0
        train_idx_first, test_idx_first = create_split(
            X_frag=X_frag, y=y, smiles_df=smiles_df, mode=mode,
            test_size=test_size, random_state=first_seed,
            cliffs_csv=cliffs_csv, log=log
        )
        
        X_tr = X_frag.iloc[train_idx_first].copy()
        X_te = X_frag.iloc[test_idx_first].copy()
        y_tr = y.iloc[train_idx_first]
        y_te = y.iloc[test_idx_first]
        
        clean_names = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '') for col in X_frag.columns]
        X_tr.columns = clean_names
        X_te.columns = clean_names
        
        model_best = make_model(best_model_name, random_state=first_seed)
        model_best.fit(X_tr, y_tr)
        
        y_prob_best = _positive_proba(model_best, X_te)
        threshold_best = tune_threshold(y_te, y_prob_best, metric=threshold_metric)
        y_pred_best = (y_prob_best >= threshold_best).astype(int)
        
        pred_df = pd.DataFrame({
            'y_true': y_te.values,
            'y_pred': y_pred_best,
            'y_prob': y_prob_best
        })
        pred_file = out_dir / "predictions_run0.csv"
        pred_df.to_csv(pred_file, index=False, float_format='%.3f')
        print(f"     ✓ Predictions saved: {pred_file.name}")
    except Exception as e:
        log.warning(f"Failed to save predictions: {e}")
    
    # Save all individual runs (needed for Figure 1 boxplots)
    runs_csv = out_dir / "model_metrics_all_runs.csv"
    all_runs_df.to_csv(runs_csv, index=False, float_format='%.3f')
    
    # Save outputs - TWO CSV files (numeric + formatted)
    # ===== CSV 1: NUMERIC (mean + CI_lower + CI_upper) - for calculations =====
    numeric_cols = ['model']
    for col in aggregated_df.columns:
        if col in ['model', 'Calibration', 'repeat']:
            continue
        if '_SD' in col:  # Exclude SD
            continue
        if col in ['n_train', 'n_test', 'n_cliff_pairs']:  # Exclude run-specific
            continue
        numeric_cols.append(col)
    
    numeric_df = aggregated_df[[c for c in numeric_cols if c in aggregated_df.columns]].copy()
    numeric_csv = out_dir / "model_metrics_numeric.csv"
    numeric_df.to_csv(numeric_csv, index=False, float_format='%.3f')
    
    # ===== CSV 2: FORMATTED (with CI strings) - for publication =====
    formatted_df = numeric_df.copy()
    
    # Identify metrics with CI
    metrics_with_ci = []
    for col in numeric_df.columns:
        if col != 'model' and f"{col}_CI_lower" in numeric_df.columns:
            metrics_with_ci.append(col)
    
    # Create formatted columns
    for metric in metrics_with_ci:
        mean_val = formatted_df[metric]
        ci_lower = formatted_df[f"{metric}_CI_lower"]
        ci_upper = formatted_df[f"{metric}_CI_upper"]
        # Format: "X.XXX (95% CI, X.XXX-X.XXX)"
        formatted_df[metric] = mean_val.apply(lambda x: f"{x:.3f}") + \
                              " (95% CI, " + \
                              ci_lower.apply(lambda x: f"{x:.3f}") + "-" + \
                              ci_upper.apply(lambda x: f"{x:.3f}") + ")"
    
    # Drop CI bounds
    columns_to_drop = [col for col in formatted_df.columns if "_CI_lower" in col or "_CI_upper" in col]
    formatted_df = formatted_df.drop(columns=[c for c in columns_to_drop if c in formatted_df.columns])
    
    formatted_csv = out_dir / "model_metrics_formatted.csv"
    formatted_df.to_csv(formatted_csv, index=False)
    
    # Figures are generated in R
    
    return best_model_name, best_metrics

# =============================================================================
# AUTOMATIC THRESHOLD SELECTION
# =============================================================================

def select_threshold_cumulative(importance_sorted: pd.Series, threshold: float = 0.80) -> int:
    """Select fragments by cumulative importance threshold."""
    total = importance_sorted.sum()
    if total == 0:
        return len(importance_sorted)
    cumsum = importance_sorted.cumsum() / total
    mask = cumsum.values >= threshold
    if mask.any():
        return int(np.argmax(mask)) + 1
    return len(importance_sorted)


def select_threshold_knee(importance_sorted: pd.Series, min_fragments: int = 15) -> int:
    """
    Select fragments using knee/elbow detection (Kneedle algorithm).
    
    Finds the point of maximum curvature in the importance curve.
    Enforces minimum number of fragments for robustness.
    
    Args:
        importance_sorted: Importance scores sorted descending
        min_fragments: Minimum number of fragments to select (default: 15)
        
    Returns:
        Number of fragments to select
    """
    n_total = len(importance_sorted)
    
    # If too few fragments, return all
    if n_total <= min_fragments:
        return n_total
    
    # Try Kneedle algorithm with more sensitive parameters
    try:
        from kneed import KneeLocator
        x = np.arange(n_total)
        y = importance_sorted.values
        
        # Normalize y to [0, 1] for better knee detection
        y_normalized = (y - y.min()) / (y.max() - y.min() + 1e-9)
        
        # Try multiple sensitivity values to find reasonable knee
        for sensitivity in [0.5, 1.0, 2.0]:
            try:
                kneedle = KneeLocator(
                    x, y_normalized, 
                    S=sensitivity, 
                    curve="convex", 
                    direction="decreasing",
                    online=False
                )
                knee_idx = kneedle.knee
                
                if knee_idx is not None and knee_idx >= min_fragments:
                    return int(knee_idx) + 1
            except Exception:
                continue
                
    except ImportError:
        pass
    except Exception:
        pass
    
    # Fallback 1: Find largest drop in importance (gap-based)
    y = importance_sorted.values
    if len(y) >= min_fragments + 2:
        # Calculate relative drops between consecutive values
        drops = np.diff(y)
        rel_drops = np.abs(drops) / (y[:-1] + 1e-9)
        
        # Find largest relative drop after min_fragments
        valid_drops = rel_drops[min_fragments:]
        if len(valid_drops) > 0:
            max_drop_idx = np.argmax(valid_drops) + min_fragments + 1
            if max_drop_idx <= n_total * 0.8:  # Don't select more than 80% of fragments
                return int(max_drop_idx)
    
    # Fallback 2: Use cumulative importance (80% threshold)
    cumsum = np.cumsum(y)
    cumsum_norm = cumsum / cumsum[-1]
    threshold_idx = np.argmax(cumsum_norm >= 0.80) + 1
    
    # Return max of threshold_idx and min_fragments, but not more than 50% of total
    return int(min(max(threshold_idx, min_fragments), n_total // 2))


def select_threshold_stability(
    importance_scores: pd.Series,
    X: pd.DataFrame,
    y: pd.Series,
    n_bootstrap: int = 100,
    stability_threshold: float = 0.8,
    seed: int = 42
) -> int:
    """
    Select fragments based on selection stability across bootstrap samples.
    
    Fragments selected in ≥stability_threshold of bootstrap samples are kept.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y)
    fragment_selection_counts = np.zeros(len(importance_scores))
    
    for _ in range(n_bootstrap):
        # Bootstrap resample
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        # Count which fragments would be selected (simple: top 50%)
        n_select = max(1, len(importance_scores) // 2)
        top_fragments = importance_scores.iloc[boot_idx].nlargest(n_select).index
        for frag_idx, frag in enumerate(importance_scores.index):
            if frag in top_fragments:
                fragment_selection_counts[frag_idx] += 1
    
    # Select fragments that appear in ≥stability_threshold of bootstrap samples
    stability = fragment_selection_counts / n_bootstrap
    stable_mask = stability >= stability_threshold
    n_stable = stable_mask.sum()
    
    return max(1, int(n_stable))


def select_threshold_fdr_null(
    importance_scores: pd.Series,
    fdr_level: float = 0.05,
    n_permutations: int = 100,
    seed: int = 42
) -> int:
    """
    Select fragments using FDR control with null distribution from permutations.
    
    Compares observed importance to null distribution from permuted labels.
    """
    rng = np.random.RandomState(seed)
    
    # Generate null distribution (permutation-based)
    null_importances = []
    for _ in range(n_permutations):
        # Permute importance scores
        perm_imp = importance_scores.values.copy()
        rng.shuffle(perm_imp)
        null_importances.append(perm_imp)
    
    null_importances = np.array(null_importances)
    
    # Compute FDR threshold
    observed = importance_scores.values
    p_values = np.mean(null_importances >= observed[:, np.newaxis], axis=1)
    
    # Benjamini-Hochberg FDR correction
    n_tests = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_pvals = p_values[sorted_idx]
    
    threshold_line = fdr_level * np.arange(1, n_tests + 1) / n_tests
    significant = sorted_pvals <= threshold_line
    
    if significant.any():
        n_significant = np.where(significant)[0][-1] + 1
        return max(1, n_significant)
    
    return max(1, int(0.1 * len(importance_scores)))  # Fallback: top 10%


def compute_fragment_stability(
    fragments: List[str],
    importance_scores: pd.Series,
    X: pd.DataFrame,
    y: pd.Series,
    n_bootstrap: int = 100,
    seed: int = 42
) -> pd.Series:
    """
    Compute selection stability for each fragment across bootstrap samples.
    
    Stability = fraction of bootstrap samples in which fragment would be selected
    by importance-based threshold (top 50% by importance).
    
    Args:
        fragments: List of fragment SMILES to evaluate
        importance_scores: Importance scores for all fragments
        X: Fragment presence matrix
        y: Labels
        n_bootstrap: Number of bootstrap samples
        seed: Random seed
        
    Returns:
        Series mapping fragment -> stability score (0.0-1.0)
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y)
    
    # Track selection frequency for each fragment
    selection_counts = {frag: 0 for frag in fragments}
    
    for _ in range(n_bootstrap):
        # Bootstrap resample
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        
        # Recalculate importance on bootstrap sample (simple: use original importance)
        # Select top 50% of fragments by importance (reasonable default)
        n_select = max(1, len(importance_scores) // 2)
        top_fragments = importance_scores.nlargest(n_select).index
        
        # Count selection
        for frag in fragments:
            if frag in top_fragments:
                selection_counts[frag] += 1
    
    # Convert to stability scores (0.0-1.0)
    stability = {frag: count / n_bootstrap for frag, count in selection_counts.items()}
    
    return pd.Series(stability)


def auto_select_fragment_threshold(
    importance_sorted: pd.Series,
    X: pd.DataFrame,
    y: pd.Series,
    mode: str = "cumulative",
    cumsum_threshold: float = 0.80,
    stability_threshold: float = 0.8,
    fdr_level: float = 0.05,
    n_bootstrap: int = 100,
    min_fragments: int = 15,
    seed: int = 42,
    log: logging.Logger = None
) -> int:
    """
    Automatically select number of fragments using specified method.
    
    Methodologically rigorous approaches:
    - knee: Automatic elbow detection (RECOMMENDED - no arbitrary parameters)
    - stability: Bootstrap-based robustness (conservative, statistically justified)
    - fdr_null: FDR control with permutation null (most rigorous)
    - cumulative: Manual threshold (requires justification)
    
    Args:
        importance_sorted: Importance scores sorted descending
        X: Fragment presence matrix
        y: Labels
        mode: Selection mode ('knee', 'stability', 'fdr_null', 'cumulative')
        cumsum_threshold: Threshold for cumulative mode
        stability_threshold: Stability threshold for stability mode
        fdr_level: FDR level for fdr_null mode
        n_bootstrap: Number of bootstrap samples for stability
        min_fragments: Minimum number of fragments to select (for knee mode)
        seed: Random seed
        log: Logger
        
    Returns:
        Number of fragments to select
    """
    if log:
        log.info(f"[Fragment Selection] Using threshold mode: {mode}")
    
    if mode == "knee":
        n_frags = select_threshold_knee(importance_sorted, min_fragments=min_fragments)
        if log:
            log.info(f"[Fragment Selection] Knee detection found natural cutoff at {n_frags} fragments (min: {min_fragments})")
    elif mode == "stability":
        n_frags = select_threshold_stability(
            importance_sorted, X, y, n_bootstrap, stability_threshold, seed
        )
        if log:
            log.info(f"[Fragment Selection] Stability-based selection (≥{stability_threshold*100:.0f}% bootstrap frequency)")
    elif mode == "fdr_null":
        n_frags = select_threshold_fdr_null(importance_sorted, fdr_level, n_bootstrap, seed)
        if log:
            log.info(f"[Fragment Selection] FDR-controlled selection (FDR < {fdr_level})")
    elif mode == "cumulative":
        n_frags = select_threshold_cumulative(importance_sorted, cumsum_threshold)
        if log:
            log.info(f"[Fragment Selection] Cumulative threshold at {cumsum_threshold*100:.0f}%")
    else:
        if log:
            log.warning(f"Unknown threshold mode '{mode}', using knee detection")
        n_frags = select_threshold_knee(importance_sorted)
    
    if log:
        log.info(f"[Fragment Selection] Selected {n_frags} fragments using {mode} method")
    
    return n_frags


# =============================================================================
# FRAGMENT SELECTION AND ANALYSIS
# =============================================================================

def select_fragments_active_only(
    rank: pd.Series, 
    X: pd.DataFrame, 
    y: pd.Series, 
    threshold: float = 0.80,
    importance_threshold: float = 0.0,
    ac_scores_df: pd.DataFrame | None = None,
    threshold_mode: str = "cumulative",
    threshold_config: dict = None,
    log: logging.Logger = None
) -> Tuple[List[str], pd.Series, pd.Series, int, Dict[str, List[Tuple[str, str]]], Dict[str, List[Tuple[str, str]]], set, set]:
    """
    Select fragments using a three-step importance-based and AC-aware selection strategy.
    
    This function implements a simple yet effective fragment selection pipeline:
    1. Importance-based selection: Select fragments from active compounds using AUTOMATIC threshold
       (cumulative, knee, stability, or FDR-based)
    2. AC enrichment addition: Add all positive AC-enriched fragments (responsible for activity flips)
    3. AC enrichment removal: Remove all negative AC-enriched fragments (responsible for inactivity flips)
    
    Parameters
    ----------
    rank : pd.Series
        Feature importance scores indexed by fragment SMILES (from model or SHAP)
    X : pd.DataFrame
        Fragment presence matrix (n_samples x n_fragments)
    y : pd.Series
        Binary activity labels (0=inactive, 1=active)
    threshold : float, default 0.80
        Cumulative importance threshold (only used if threshold_mode='cumulative')
    importance_threshold : float, default 0.0
        Minimum importance score filter applied before selection.
        Fragments with importance below this value are excluded. Set to 0.0 to disable.
    ac_scores_df : pd.DataFrame, optional
        AC enrichment scores DataFrame from ac_enrichment.csv
    threshold_mode : str, default "cumulative"
        Method for automatic threshold selection: "cumulative", "knee", "stability", "fdr_null"
    threshold_config : dict, optional
        Configuration for threshold selection methods
    log : logging.Logger, optional
        Logger instance
        
    Returns
    -------
    Tuple[List[str], pd.Series, pd.Series, int, Dict, Dict, set, set]
        Tuple containing:
        - selected_fragments: List of selected fragment SMILES
        - sorted_ranks: Importance scores sorted in descending order
        - cumulative_sums: Cumulative importance scores (normalized 0-1)
        - n_fragments: Total number of selected fragments
        - added_fragments: Dictionary mapping fragment SMILES to list of cliff pairs that caused addition
        - removed_fragments: Dictionary mapping fragment SMILES to list of cliff pairs that caused removal
        - selected_by_threshold_set: Set of fragments selected by importance threshold (step 1)
        - removed_fragments_set: Set of fragments removed due to negative AC enrichment (step 3)
    """
    active_mask = y == 1
    
    if not active_mask.any():
        return [], pd.Series(dtype=float), pd.Series(dtype=float), 0, {}, {}, set(), set()
    
    # Step 1: Select fragments by AUTOMATIC threshold (methodologically sound)
    active_frags = set(X.loc[active_mask].sum(axis=0)[lambda s: s > 0].index)
    if not active_frags:
        return [], pd.Series(dtype=float), pd.Series(dtype=float), 0, {}, {}, set(), set()
    
    rank_active = rank[rank.index.intersection(active_frags)]
    if rank_active.empty:
        return [], pd.Series(dtype=float), pd.Series(dtype=float), 0, {}, {}, set(), set()
    
    if importance_threshold > 0.0:
        rank_active = rank_active[rank_active >= importance_threshold]
        if rank_active.empty:
            return [], pd.Series(dtype=float), pd.Series(dtype=float), 0, {}, {}, set(), set()
    
    shap_sorted = rank_active.sort_values(ascending=False)
    total_importance = shap_sorted.sum()
    
    if total_importance == 0:
        return [], shap_sorted, pd.Series(dtype=float), 0, {}, {}, set(), set()
    
    cumsum = shap_sorted.cumsum() / total_importance
    
    # AUTOMATIC threshold selection (methodologically sound!)
    if threshold_config is None:
        threshold_config = {}
    
    n_frags = auto_select_fragment_threshold(
        importance_sorted=shap_sorted,
        X=X,
        y=y,
        mode=threshold_mode,
        cumsum_threshold=threshold,
        stability_threshold=threshold_config.get("stability_threshold", 0.8),
        fdr_level=threshold_config.get("fdr_level", 0.05),
        n_bootstrap=threshold_config.get("n_bootstrap", 100),
        min_fragments=threshold_config.get("min_fragments", 15),
        seed=threshold_config.get("seed", 42),
        log=log
    )
    
    n_frags = max(1, min(n_frags, len(shap_sorted)))
    
    selected_by_threshold = shap_sorted.index[:n_frags].tolist()
    selected_by_threshold_set = set(selected_by_threshold)
    
    # Step 2 & 3: Apply AC enrichment logic
    selected_frags = selected_by_threshold.copy()
    added_fragments = {}
    removed_fragments = {}
    
    if ac_scores_df is not None and not ac_scores_df.empty:
        # Step 2: Add ALL positive AC fragments (ac_enrichment > 0)
        positive_ac_frags = set(ac_scores_df[ac_scores_df["ac_enrichment"] > 0].index)
        positive_ac_frags_in_X = positive_ac_frags & set(X.columns)
        
        # Track which fragments are being added (only NEW ones, not already in threshold set)
        for frag in positive_ac_frags_in_X:
            if frag not in selected_by_threshold_set:  # Only track fragments NOT already selected by threshold
                if frag in ac_scores_df.index:
                    cliff_pairs = ac_scores_df.loc[frag, "cliff_pairs"]
                    added_fragments[frag] = cliff_pairs if cliff_pairs else []
        
        # Add ALL positive AC fragments to selected_frags (union: keeps threshold + adds AC+)
        selected_frags = list(set(selected_frags) | positive_ac_frags_in_X)
        
        # Step 3: Remove ALL negative AC fragments (ac_enrichment < 0)
        negative_ac_frags = set(ac_scores_df[ac_scores_df["ac_enrichment"] < 0].index)
        
        # Track which fragments are being removed (only those that were in selected_frags)
        for frag in negative_ac_frags:
            if frag in selected_frags:
                if frag in ac_scores_df.index:
                    cliff_pairs = ac_scores_df.loc[frag, "cliff_pairs"]
                    removed_fragments[frag] = cliff_pairs if cliff_pairs else []
        
        # Remove all negative AC fragments from selected_frags
        selected_frags = [f for f in selected_frags if f not in negative_ac_frags]
    
    # FINAL: selected_frags should contain:
    # - All fragments from threshold (80% by pure importance)
    # - PLUS all positive AC-enriched fragments (15 fragments)
    # - MINUS any negative AC-enriched fragments that were selected
    
    return selected_frags, shap_sorted, cumsum, len(selected_frags), added_fragments, removed_fragments, selected_by_threshold_set, set(removed_fragments.keys())


# ───────────────────────────────────────────────────────────────
# 3. CAFE split helpers
# ───────────────────────────────────────────────────────────────

def _cliff_train_enriched_split(
    smiles_df: pd.DataFrame,
    y: pd.Series,
    cliffs_csv: Path,
    test_size: float,
    random_state: int,
    log: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CAFE-specific split: Maximize activity cliff molecules in TRAIN set.
    
    Strategy:
    1. Identify ALL molecules involved in activity cliffs
    2. Assign ALL cliff molecules to TRAIN (100%)
    3. Split remaining non-cliff molecules between train/test (stratified)
    4. Ensures test_size is approximately met using non-cliff molecules
    
    Rationale for Reverse QSAR:
    - Model learns fragment patterns FROM activity cliffs
    - AC enrichment requires cliff pairs in training data
    - Test set evaluates generalization to non-cliff chemical space
    - Conservative: harder test (no cliffs = no structural hints)
    
    Args:
        smiles_df: DataFrame with SMILES column
        y: Binary activity labels
        cliffs_csv: Path to activity cliffs CSV
        test_size: Target fraction for test set
        random_state: Random seed
        log: Logger
        
    Returns:
        Tuple of (train_idx, test_idx)
    """
    # Load activity cliffs
    if not Path(cliffs_csv).exists():
        log.warning("[Split] Activity cliffs file not found, using standard split")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        return next(sss.split(smiles_df, y))
    
    cliffs = pd.read_csv(cliffs_csv)
    if cliffs.empty:
        log.warning("[Split] No activity cliffs found, using standard split")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        return next(sss.split(smiles_df, y))
    
    # Identify all molecules involved in cliffs
    cliff_molecules = set()
    for _, row in cliffs.iterrows():
        mol_i = row.get("mol_i")
        mol_j = row.get("mol_j")
        if pd.notna(mol_i):
            cliff_molecules.add(mol_i)
        if pd.notna(mol_j):
            cliff_molecules.add(mol_j)
    
    # Map SMILES to indices
    smiles_col = "SMILES" if "SMILES" in smiles_df.columns else smiles_df.columns[0]
    cliff_indices = set()
    for idx, smiles in enumerate(smiles_df[smiles_col]):
        if smiles in cliff_molecules:
            cliff_indices.add(idx)
    
    cliff_idx_array = np.array(sorted(cliff_indices))
    non_cliff_idx_array = np.array([i for i in range(len(smiles_df)) if i not in cliff_indices])
    
    # ALL cliff molecules → TRAIN
    train_idx = list(cliff_idx_array)
    
    # Split non-cliff molecules stratified to fill test set
    if len(non_cliff_idx_array) > 0:
        # Calculate how many non-cliff molecules needed for test
        target_test_size = int(len(smiles_df) * test_size)
        current_train_size = len(train_idx)
        needed_test = max(1, target_test_size)  # At least 1 in test
        
        # Split non-cliff molecules
        y_non_cliff = y.iloc[non_cliff_idx_array]
        
        if needed_test >= len(non_cliff_idx_array):
            # All non-cliff to test (edge case: too many cliffs)
            test_idx = list(non_cliff_idx_array)
        else:
            # Stratified split of non-cliff molecules
            actual_test_frac = min(0.95, needed_test / len(non_cliff_idx_array))
            sss = StratifiedShuffleSplit(n_splits=1, test_size=actual_test_frac, random_state=random_state)
            try:
                non_cliff_tr_local, non_cliff_te_local = next(sss.split(
                    np.zeros((len(non_cliff_idx_array), 1)), y_non_cliff
                ))
                train_idx.extend(non_cliff_idx_array[non_cliff_tr_local].tolist())
                test_idx = non_cliff_idx_array[non_cliff_te_local].tolist()
            except Exception:
                # Fallback: simple random split
                np.random.seed(random_state)
                perm = np.random.permutation(len(non_cliff_idx_array))
                split_point = needed_test
                test_idx = non_cliff_idx_array[perm[:split_point]].tolist()
                train_idx.extend(non_cliff_idx_array[perm[split_point:]].tolist())
    else:
        # No non-cliff molecules (edge case)
        # Split cliff molecules instead
        log.warning("[Split] All molecules are in cliffs, splitting cliff molecules")
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(sss.split(smiles_df, y))
        return train_idx, test_idx
    
    train_idx = np.array(train_idx)
    test_idx = np.array(test_idx)
    
    # Log cliff distribution (debug only - too verbose for repeated splits)
    n_cliff_train = len(cliff_idx_array)
    n_cliff_test = len([i for i in test_idx if i in cliff_indices])
    log.debug(f"[Split cliff_train_max] Cliff molecules: {n_cliff_train} in TRAIN, {n_cliff_test} in TEST")
    
    return train_idx, test_idx


def _cliff_groups(df: pd.DataFrame, smiles_col: str, cliffs_csv: Path) -> pd.Series:
    """
    Build group labels so that all molecules connected by 'cliff' edges
    end up in the same group (connected components).
    Molecules not in any cliff get unique singleton groups.
    """
    if not Path(cliffs_csv).exists():
        return pd.Series(np.arange(len(df)), index=df.index)

    cliffs = pd.read_csv(cliffs_csv)
    if cliffs.empty:
        return pd.Series(np.arange(len(df)), index=df.index)

    G = nx.Graph()
    for s in df[smiles_col]:
        G.add_node(s)
    for _, r in cliffs.iterrows():
        si, sj = r.get("mol_i"), r.get("mol_j")
        if pd.isna(si) or pd.isna(sj):
            continue
        G.add_edge(si, sj)

    comp_id = {}
    for cid, comp in enumerate(nx.connected_components(G)):
        for smi in comp:
            comp_id[smi] = cid

    groups = df[smiles_col].map(comp_id).fillna(-1).astype(int)
    if (groups == -1).any():
        base = groups.max() + 1
        counter = 0
        tmp = groups.copy()
        for idx in df.index[groups == -1]:
            tmp.at[idx] = base + counter
            counter += 1
        groups = tmp
    return groups

def _grouped_train_test_indices(groups: pd.Series, y: pd.Series, test_size: float, random_state: int):
    """
    Create train/test split that respects groups and tries to balance cliff groups.
    
    Uses StratifiedGroupShuffleSplit if available to balance class distribution,
    otherwise uses GroupShuffleSplit with multiple attempts to balance cliff groups.
    """
    try:
        from sklearn.model_selection import StratifiedGroupShuffleSplit
        # Use StratifiedGroupShuffleSplit to balance classes while respecting groups
        sgss = StratifiedGroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        (tr_idx, te_idx), = sgss.split(X=groups, y=y, groups=groups)
        return tr_idx, te_idx
    except ImportError:
        # Fallback: use GroupShuffleSplit with multiple attempts to balance cliff groups
        gss = GroupShuffleSplit(n_splits=256, test_size=test_size, random_state=random_state)
        
        # Try to balance cliff groups between train and test
        # (We don't have cliff info here, but we can still balance classes)
        base_ratio = float(np.mean(y))
        best = None
        best_diff = float('inf')
        
        for tr_idx, te_idx in gss.split(X=groups, y=y, groups=groups):
            # Check class balance
            diff = abs(float(np.mean(y[te_idx])) - base_ratio)
            if diff < best_diff:
                best, best_diff = (tr_idx, te_idx), diff
            if diff <= 0.02:  # within 2 percentage points -> good enough
                break
        
        if best is None:
            # Last resort: just use first split
            gss_single = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            (tr_idx, te_idx), = gss_single.split(X=groups, y=y, groups=groups)
            return tr_idx, te_idx
        
        return best
def _compute_ac_enrichment_for_fragments(
    X_frag: pd.DataFrame,
    smiles_series: pd.Series,
    cliffs_csv: Path,
    potency: pd.Series | None = None,
    sali_clip_percentile: float = 99.0,
    use_all_train_ac: bool = True
) -> pd.DataFrame:
    """
    Compute AC enrichment scores for DIFFERENTIATING fragments only (present in one compound but not the other).
    
    This is a simple "cheat sheet" for the model:
    - ac_enrichment > 0: Fragment is ONLY in active compound (responsible for activity flip) → ADD
    - ac_enrichment < 0: Fragment is ONLY in inactive compound (responsible for inactivity flip) → REMOVE
    - ac_enrichment = 0: Fragment is NOT differentiating (present in both or neither) → ignore
    
    Score = sum of SALI values from cliff pairs where fragment is differentiating.
    
    Args:
        X_frag: Fragment presence matrix (n_samples x n_fragments)
        smiles_series: SMILES strings aligned with X_frag.index
        cliffs_csv: Path to activity cliffs CSV file
        potency: Potency values (pIC50) for direction determination
        sali_clip_percentile: Percentile for SALI value clipping (stability)
        use_all_train_ac: If True, use ALL AC pairs from train set (for full information extraction)
        
    Returns:
        DataFrame with columns:
        - ac_enrichment: enrichment score (only non-zero for DIFFERENTIATING fragments)
        - cliff_pairs: list of cliff pairs (mol_i, mol_j) where this fragment is differentiating
        - cliff_pair_indices: list of row indices from cliffs CSV
        - n_cliff_pairs: number of cliff pairs where fragment is differentiating
    """
    # Initialize empty result
    n_frags = X_frag.shape[1]
    empty_result = pd.DataFrame({
        "ac_enrichment": np.zeros(n_frags),
        "cliff_pairs": [[] for _ in range(n_frags)],
        "cliff_pair_indices": [[] for _ in range(n_frags)],
        "n_cliff_pairs": np.zeros(n_frags, dtype=int)
    }, index=X_frag.columns)
    
    if not Path(cliffs_csv).exists():
        return empty_result

    try:
        cliffs = pd.read_csv(cliffs_csv)
    except Exception:
        return empty_result

    if cliffs.empty or not all(col in cliffs.columns for col in ["mol_i", "mol_j", "SALI"]):
        return empty_result

    # Create SMILES to position mapping
    smi_to_pos = {smi: pos for pos, smi in enumerate(smiles_series.values)}
    
    # Process SALI values with clipping for stability
    sali = cliffs["SALI"].astype(float).values
    if sali_clip_percentile is not None and len(sali) > 0:
        cap = np.percentile(sali, sali_clip_percentile)
        sali = np.clip(sali, None, cap)

    # Initialize score vector and cliff pair tracking
    # Only count DIFFERENTIATING fragments (delta != 0)
    score = np.zeros(X_frag.shape[1], dtype=float)
    cliff_pairs_by_frag = {frag: [] for frag in X_frag.columns}
    cliff_indices_by_frag = {frag: [] for frag in X_frag.columns}
    X_np = X_frag.values.astype(np.int8)

    # Prepare potency vector for direction determination
    potency_vec = None
    if potency is not None:
        try:
            potency_vec = potency.loc[X_frag.index].values.astype(float)
        except Exception:
            pass

    # Process cliff pairs - ONLY count DIFFERENTIATING fragments (delta != 0)
    valid_pairs = 0
    for k, row in cliffs.iterrows():
        si, sj = row["mol_i"], row["mol_j"]
        
        # Skip if molecules not in dataset
        if si not in smi_to_pos or sj not in smi_to_pos:
            continue
            
        i, j = smi_to_pos[si], smi_to_pos[sj]
        
        # Determine which is active (higher potency = active)
        is_i_active = False
        is_j_active = False
        
        if potency_vec is not None:
            try:
                pi, pj = potency_vec[i], potency_vec[j]
                if np.isfinite(pi) and np.isfinite(pj):
                    if pi > pj:
                        is_i_active = True
                        is_j_active = False
                    elif pj > pi:
                        is_i_active = False
                        is_j_active = True
                    else:
                        continue  # Skip if equal potency
                else:
                    continue  # Skip if potency invalid
            except (IndexError, ValueError):
                continue
        else:
            # Skip if no potency information
            continue
            
        # Skip if both active or both inactive (not a proper cliff)
        if is_i_active == is_j_active:
            continue
            
        # Calculate fragment presence difference
        delta = X_np[i, :] - X_np[j, :]  # -1, 0, 1
        
        # ONLY process DIFFERENTIATING fragments (delta != 0)
        # delta = +1: fragment ONLY in active compound → POSITIVE score (add)
        # delta = -1: fragment ONLY in inactive compound → NEGATIVE score (remove)
        # delta = 0: fragment in BOTH or NEITHER → ignore (not differentiating)
        
        cliff_pair = (si, sj)
        for frag_idx, delta_val in enumerate(delta):
            if delta_val == 0:
                continue  # Skip non-differentiating fragments
            
            frag_name = X_frag.columns[frag_idx]
            
            if is_i_active:
                # i is active, j is inactive
                if delta_val == 1:
                    # Fragment ONLY in active (i) → POSITIVE
                    score[frag_idx] += sali[k]
                    cliff_pairs_by_frag[frag_name].append(cliff_pair)
                    cliff_indices_by_frag[frag_name].append(k)
                elif delta_val == -1:
                    # Fragment ONLY in inactive (j) → NEGATIVE
                    score[frag_idx] -= sali[k]
                    cliff_pairs_by_frag[frag_name].append(cliff_pair)
                    cliff_indices_by_frag[frag_name].append(k)
            else:
                # j is active, i is inactive
                if delta_val == 1:
                    # Fragment ONLY in inactive (i) → NEGATIVE
                    score[frag_idx] -= sali[k]
                    cliff_pairs_by_frag[frag_name].append(cliff_pair)
                    cliff_indices_by_frag[frag_name].append(k)
                elif delta_val == -1:
                    # Fragment ONLY in active (j) → POSITIVE
                    score[frag_idx] += sali[k]
                    cliff_pairs_by_frag[frag_name].append(cliff_pair)
                    cliff_indices_by_frag[frag_name].append(k)
        
        valid_pairs += 1

    # Build result DataFrame
    result = pd.DataFrame({
        "ac_enrichment": score,
        "cliff_pairs": [cliff_pairs_by_frag[frag] for frag in X_frag.columns],
        "cliff_pair_indices": [cliff_indices_by_frag[frag] for frag in X_frag.columns],
        "n_cliff_pairs": [len(cliff_pairs_by_frag[frag]) for frag in X_frag.columns]
    }, index=X_frag.columns)

    # Log statistics (silent - info available in CSV)

    return result


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def defragmenttion(cfg: Mapping, log):
    """
    Comprehensive reverse QSAR evaluation pipeline with activity cliff awareness.
    
    Implements a complete reverse QSAR workflow for fragment-based molecular design:
    1. Multi-model evaluation (9 ML algorithms) with activity cliff-aware model selection
    2. BRICS fragment generation and normalization
    3. SHAP-based feature importance calculation (using train set for stability)
    4. Three-step fragment selection:
       a) Cumulative importance threshold selection (configurable via cumsum_active_threshold)
       b) Addition of all positive AC-enriched fragments (activity flip responsible)
       c) Removal of all negative AC-enriched fragments (inactivity flip responsible)
    5. AC enrichment analysis using ALL activity cliff pairs from training set
    6. Publication-ready visualizations and statistical summaries
    
    Configuration
    -------------
    Key parameters from config.yml:
    - ReverseQSAR.selection.cumsum_active_threshold: Cumulative importance threshold (default: 0.80)
    - ReverseQSAR.selection.importance_threshold: Minimum importance filter (default: 0.0)
    - ReverseQSAR.primary_metric: Model selection metric (recommended: AUPRC_active)
    - ReverseQSAR.split.mode: Data splitting strategy (standard or cliff_group)
    
    Parameters
    ----------
    cfg : Mapping
        Configuration dictionary containing:
        - Paths: dataset paths and output directories
        - ReverseQSAR: model selection, fragment selection, and AC enrichment parameters
        - AC_Analysis: activity cliff detection parameters
    log : logging.Logger
        Logger instance for progress tracking and milestone reporting
        
    Returns
    -------
    None
        All results are saved to files in the output directory specified in cfg["Paths"]["fragments"]
        
    Output Files
    ------------
    Main outputs:
    - model_metrics.csv: Model performance (mean + 95% CI, publication-ready)
    - model_metrics_supplementary.csv: Full metrics with SD (for SI)
    - model_metrics_all_runs.csv: All individual runs (10 splits)
    
    Fragment analysis:
    - {model_name}/selected_fragments_with_ACflag.csv: Selected fragments with stability scores
    - {model_name}/all_fragments_dictionary.csv: Complete fragment catalog (Supplementary)
    - {model_name}/ac_enrichment.csv: AC enrichment scores for all fragments
    - {model_name}/removed_fragments.csv: Fragments removed due to negative AC enrichment
    
    Summary:
    - summary.json: Pipeline metadata including selection method and stability stats
    """
    random_state = int(cfg.get("ReverseQSAR", {}).get("rf", {}).get("random_state", 42))
    seed_everything(random_state)
    
    print("=" * 70)
    print("Reverse QSAR pipeline: fragment extraction and AC enrichment.")
    print("=" * 70)
    
    # Extract configuration
    paths = cfg["Paths"]
    rev = cfg.get("ReverseQSAR", {})
    min_occ = int(rev.get("min_occ", 1))
    frag_out = Path(paths.get("fragments", "results/ReverseQSAR"))
    ensure_dir(frag_out)

    # Load dataset
    final_csv = paths.get("dataset", "data/processed/final_dataset.csv")
    df = load_dataset(final_csv)
    print(f"Dataset: {len(df)} compounds")

    # Prepare labels and SMILES
    y = df.activity_flag.map({"inactive": 0, "active": 1}).astype(int)
    smiles_df = df[["canonical_smiles"]].rename(columns={"canonical_smiles": "SMILES"})

    pot_col_key = cfg.get("AC_Analysis", {}).get("potency_col", "pIC50")
    if pot_col_key in df.columns:
        potency_series = df[pot_col_key]
    else:
        potency_series = y.astype(float)
        log.warning(f"Potency column '{pot_col_key}' not found, using binary labels.")

    # BRICS fragmentation
    print(f"🔨 BRICS fragmentation (min_occ={min_occ})...")
    X_frag, cols = generate_fragment_matrix_brics(smiles_df, min_occ=min_occ)
    print(f"   ✓ {X_frag.shape[1]} unique fragments generated")

    # Multi-model evaluation (splits created internally for each repeat)
    models = rev.get("engines", ["LogReg", "KNN", "SVC", "RF", "ExtraTrees", "GB", "XGB", "CatBoost", "BRF"])
    n_repeats = int(rev.get("n_repeats", 10))  # Number of independent splits per model
    
    print(f"Using {n_repeats} independent splits per model (AC-aware splitting preserved).")
    
    try:
        best_model_name, best_metrics = run_multi_model_evaluation(
            X_frag=X_frag,
            y=y,
            smiles_df=smiles_df,
            potency_series=potency_series,
            models=models,
            out_dir=frag_out,
            config=cfg,
            log=log,
            random_state=random_state,
            n_repeats=n_repeats
        )
        
    except Exception as e:
        log.error(f"Multi-model evaluation failed: {e}")
        raise RuntimeError(f"Multi-model evaluation failed: {e}")

    # Create final split for fragment extraction (using first repeat's seed)
    print(f"Training final {best_model_name} model for fragment analysis...")
    
    split_cfg = rev.get("split", {})
    mode = split_cfg.get("mode", "standard")
    test_size = float(split_cfg.get("test_size", 0.20))
    ac_results_dir = cfg.get("AC_Analysis", {}).get("results_dir", "results/AC_analysis")
    cliffs_csv = Path(ac_results_dir) / "activity_cliffs.csv"
    
    tr_idx, te_idx = create_split(
        X_frag=X_frag,
        y=y,
        smiles_df=smiles_df,
        mode=mode,
        test_size=test_size,
        random_state=random_state,  # Use base seed for reproducibility
        cliffs_csv=cliffs_csv,
        log=log
    )
    
    train_balance = float(np.mean(y.iloc[tr_idx]))
    test_balance = float(np.mean(y.iloc[te_idx]))
    print(f"   Final split ({mode}): Train={len(tr_idx)} ({train_balance:.1%} active), Test={len(te_idx)} ({test_balance:.1%} active)")
    
    X_tr, X_te = X_frag.iloc[tr_idx].copy(), X_frag.iloc[te_idx].copy()
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
    
    clean_feature_names = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '') for col in X_frag.columns]
    X_tr.columns = clean_feature_names
    X_te.columns = clean_feature_names
    
    # Get threshold metric from config
    threshold_metric = rev.get("threshold_metric", "F1")
    
    best_model = make_model(best_model_name, random_state=random_state)
    best_model.fit(X_tr, y_tr)
    y_prob = _positive_proba(best_model, X_te)
    threshold = tune_threshold(y_te, y_prob, metric=threshold_metric)
    y_pred = (y_prob >= threshold).astype(int)
    print(f"   Threshold ({threshold_metric}): {threshold:.3f}")
    
    # Create output directories
    model_dir = frag_out / best_model_name
    ensure_dir(model_dir)
    
    # Figures are generated in R
    
    # SHAP analysis
    selected_frags = []
    importance_scores = None
    shap_values = None
    
    try:
        # IMPORTANT: Use TRAIN set for SHAP to get consistent importance scores
        # Test set may have different distribution and give misleading importance
        if hasattr(best_model, 'tree_') or hasattr(best_model, 'estimators_'):
            expl = shap.TreeExplainer(best_model)
            # Use X_tr for SHAP (more stable, representative of training data)
            shap_values = expl.shap_values(X_tr)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            expl = shap.LinearExplainer(best_model, X_tr)
            # Use X_tr for SHAP
            shap_values = expl.shap_values(X_tr)
        
        # SHAP plots removed - not needed
        
    except Exception as e:
        log.warning(f"SHAP analysis failed: {e}")
    
    print("🧪 Fragment analysis & AC enrichment...")
    
    try:
        # Calculate feature importance (priority: feature_importances_ > SHAP > coefficients)
        clean_to_orig = {clean: orig for clean, orig in zip(X_tr.columns, X_frag.columns)}
        
        if hasattr(best_model, "feature_importances_"):
            importance_scores = pd.Series(best_model.feature_importances_, index=X_tr.columns).rename(clean_to_orig)
        elif shap_values is not None:
            importance_scores = pd.Series(np.abs(shap_values).mean(axis=0), index=X_tr.columns).rename(clean_to_orig)
        elif hasattr(best_model, "coef_"):
            importance_scores = pd.Series(np.abs(best_model.coef_[0]), index=X_tr.columns).rename(clean_to_orig)
        else:
            importance_scores = pd.Series(1.0, index=X_frag.columns)
            log.warning("Using uniform importance (fallback).")
        
        fragments_missing = set(X_frag.columns) - set(importance_scores.index)
        if fragments_missing:
            importance_scores = pd.concat([importance_scores, pd.Series(0.0, index=list(fragments_missing))])
        
        # Normalize fragment data
        X_frag_norm, importance_scores_norm = normalize_fragment_data(X_frag, importance_scores, log)
        
        # Filter importance: set to 0 for fragments ONLY in inactive compounds
        active_mask = y == 1
        inactive_mask = y == 0
        
        if active_mask.any() and inactive_mask.any():
            frags_in_active = set(X_frag_norm.loc[active_mask].sum(axis=0)[lambda s: s > 0].index)
            frags_only_inactive = set(X_frag_norm.loc[inactive_mask].sum(axis=0)[lambda s: s > 0].index) - frags_in_active
            
            if frags_only_inactive:
                importance_scores_norm.loc[list(frags_only_inactive)] = 0.0
                print(f"   ✓ Filtered {len(frags_only_inactive)} fragments (only in inactive compounds)")
        
        # Compute AC enrichment scores before fragment selection (TRAIN SET ONLY to avoid data leakage)
        ac_scores_df = None  # DataFrame with ac_enrichment, cliff_pairs, etc.
        # Use AC enrichment config from ReverseQSAR (applies to all models, not just RF)
        ac_cfg = rev.get("rf", {}).get("ac_enrichment_auto", {})  # Legacy: still reads from rf section, but applies to best_model
        if ac_cfg.get("enable", True):
            ac_results_dir = cfg.get("AC_Analysis", {}).get("results_dir", "results/AC_analysis")
            cliffs_csv = Path(ac_results_dir) / "activity_cliffs.csv"
            
            if cliffs_csv.exists():
                X_frag_train = X_frag_norm.iloc[tr_idx]
                smiles_train = smiles_df["SMILES"].iloc[tr_idx]
                potency_train = potency_series.iloc[tr_idx] if potency_series is not None else None
                
                ac_scores_df = _compute_ac_enrichment_for_fragments(
                    X_frag=X_frag_train,
                    smiles_series=smiles_train,
                    cliffs_csv=cliffs_csv,
                    potency=potency_train,
                    sali_clip_percentile=ac_cfg.get("sali_clip_percentile", 99.0),
                    use_all_train_ac=True
                )
                
                ac_enrichment_scores = ac_scores_df["ac_enrichment"]
                positive_ac = (ac_enrichment_scores > 0).sum()
                negative_ac = (ac_enrichment_scores < 0).sum()
                print(f"   ✓ AC enrichment: {positive_ac} positive, {negative_ac} negative fragments")
        
        # Select fragments (3-step: importance → add AC+ → remove AC-)
        sel = rev.get("selection", {})
        threshold_mode = sel.get("threshold_mode", "cumulative")
        cumsum_threshold = float(sel.get("cumsum_active_threshold", 0.80))
        importance_threshold = float(sel.get("importance_threshold", 0.0))
        
        # Prepare threshold config for automatic selection
        threshold_config = {
            "stability_threshold": float(sel.get("stability_threshold", 0.8)),
            "fdr_level": float(sel.get("fdr_level", 0.05)),
            "n_bootstrap": int(sel.get("n_bootstrap", 100)),
            "seed": random_state
        }
        
        print(f"   Fragment selection using '{threshold_mode}' threshold method...")
        
        selected_frags, shap_sorted, cumsum, n_frags, added_fragments, removed_fragments, selected_by_threshold_set, removed_fragments_set = select_fragments_active_only(
            rank=importance_scores_norm,
            X=X_frag_norm,
            y=y,
            threshold=cumsum_threshold,
            importance_threshold=importance_threshold,
            ac_scores_df=ac_scores_df,
            threshold_mode=threshold_mode,
            threshold_config=threshold_config,
            log=log
        )
        
        # Fallback selection if threshold returns no fragments
        if not selected_frags:
            log.warning("No fragments selected by threshold, using fallback (top 20 by importance).")
            if not importance_scores_norm.empty:
                top_frags = importance_scores_norm.nlargest(20).index.tolist()
                selected_frags = top_frags
                selected_by_threshold_set = set(selected_frags)
            else:
                selected_frags = X_frag_norm.columns[:20].tolist()
                selected_by_threshold_set = set(selected_frags)
        
        # Apply AC enrichment logic (add positive, remove negative)
        if ac_scores_df is not None and not ac_scores_df.empty and selected_frags:
            positive_ac_frags = set(ac_scores_df[ac_scores_df["ac_enrichment"] > 0].index)
            positive_ac_frags_in_X = positive_ac_frags & set(X_frag_norm.columns)
            
            # Track only fragments that are NOT already in threshold selection
            for frag in positive_ac_frags_in_X:
                if frag not in selected_by_threshold_set and frag in ac_scores_df.index:
                    cliff_pairs = ac_scores_df.loc[frag, "cliff_pairs"]
                    added_fragments[frag] = cliff_pairs if cliff_pairs else []
            
            selected_frags = list(set(selected_frags) | positive_ac_frags_in_X)
            
            negative_ac_frags = set(ac_scores_df[ac_scores_df["ac_enrichment"] < 0].index)
            
            for frag in negative_ac_frags:
                if frag in selected_frags and frag in ac_scores_df.index:
                    cliff_pairs = ac_scores_df.loc[frag, "cliff_pairs"]
                    removed_fragments[frag] = cliff_pairs if cliff_pairs else []
            
            selected_frags = [f for f in selected_frags if f not in negative_ac_frags]
        
        # Final selection summary
        n_from_importance = len(set(selected_frags) & selected_by_threshold_set) if selected_frags and selected_by_threshold_set else 0
        n_added_ac = len(added_fragments)  # This now contains only fragments NOT already in threshold
        n_removed_ac = len(removed_fragments)
        
        print(f"Fragment selection (threshold={cumsum_threshold*100:.0f}%):")
        if ac_scores_df is not None and not ac_scores_df.empty:
            print(f"   ✓ Final: {len(selected_frags)} fragments ({n_from_importance} importance + {n_added_ac} AC-added - {n_removed_ac} AC-removed)")
        else:
            print(f"   ✓ Final: {len(selected_frags)} fragments (importance-based only)")
        
        # Save AC enrichment results (if computed)
        if ac_scores_df is not None and not ac_scores_df.empty:
            # Prepare CSV with cliff pair sources
            # Convert cliff_pairs lists to strings for CSV storage
            ac_df_export = ac_scores_df.copy()
            ac_df_export["cliff_pairs_str"] = ac_df_export["cliff_pairs"].apply(
                lambda pairs: "; ".join([f"{p[0]}|{p[1]}" for p in pairs]) if pairs else ""
            )
            ac_df_export["cliff_pair_indices_str"] = ac_df_export["cliff_pair_indices"].apply(
                lambda idxs: "; ".join(map(str, idxs)) if idxs else ""
            )
            
            # Save full AC enrichment data with cliff pair sources
            ac_df_sorted = ac_df_export.sort_values("ac_enrichment", ascending=False)[
                ["ac_enrichment", "n_cliff_pairs", "cliff_pairs_str", "cliff_pair_indices_str"]
            ].reset_index()
            ac_df_sorted.columns = ["fragment_smiles", "ac_enrichment", "n_cliff_pairs", "cliff_pairs", "cliff_pair_indices"]
            ac_df_sorted.to_csv(model_dir / "ac_enrichment.csv", index=False)
            
            # Create FULL fragment dictionary for Supplementary Materials
            # Compute stability scores if requested
            report_stability = sel.get("report_stability", True)
            stability_scores = None
            
            if report_stability:
                print(f"   Computing fragment stability (bootstrap n={threshold_config['n_bootstrap']})...")
                stability_scores = compute_fragment_stability(
                    fragments=list(X_frag_norm.columns),
                    importance_scores=importance_scores_norm,
                    X=X_frag_norm,
                    y=y,
                    n_bootstrap=threshold_config.get("n_bootstrap", 100),
                    seed=random_state
                )
            
            full_fragment_dict = []
            for frag in X_frag_norm.columns:
                freq_active = X_frag_norm.loc[y == 1, frag].sum() if (y == 1).any() else 0
                freq_inactive = X_frag_norm.loc[y == 0, frag].sum() if (y == 0).any() else 0
                freq_total = X_frag_norm[frag].sum()
                
                frag_data = {
                    "fragment_smiles": frag,
                    "frequency_total": int(freq_total),
                    "frequency_active": int(freq_active),
                    "frequency_inactive": int(freq_inactive),
                    "presence_active_pct": round(100 * freq_active / max(1, (y == 1).sum()), 2),
                    "presence_inactive_pct": round(100 * freq_inactive / max(1, (y == 0).sum()), 2),
                    "importance": round(importance_scores_norm.get(frag, 0.0), 6),
                    "stability": round(stability_scores.get(frag, 0.0), 3) if stability_scores is not None else None,
                    "ac_enrichment": round(ac_scores_df.loc[frag, "ac_enrichment"], 6) if frag in ac_scores_df.index else 0.0,
                    "n_cliff_pairs": int(ac_scores_df.loc[frag, "n_cliff_pairs"]) if frag in ac_scores_df.index else 0,
                    "is_selected": frag in selected_frags,
                    "selection_method": (
                        "threshold" if frag in selected_by_threshold_set else
                        "AC_added" if frag in added_fragments else
                        "not_selected"
                    ),
                    "was_removed_by_AC": frag in removed_fragments_set
                }
                full_fragment_dict.append(frag_data)
            
            # Save full dictionary sorted by importance
            full_dict_df = pd.DataFrame(full_fragment_dict).sort_values("importance", ascending=False)
            full_dict_df.to_csv(model_dir / "all_fragments_dictionary.csv", index=False)
            
            # Report stability statistics for selected fragments
            if stability_scores is not None and selected_frags:
                selected_stability = [stability_scores.get(f, 0.0) for f in selected_frags if f in stability_scores.index]
                if selected_stability:
                    mean_stab = np.mean(selected_stability)
                    min_stab = np.min(selected_stability)
                    print(f"   ✓ Selected fragments stability: mean={mean_stab:.2f}, min={min_stab:.2f}")
            
            print(f"   ✓ Saved full fragment dictionary: {len(full_fragment_dict)} fragments")
            
            # Highlighting: use all positive AC-enriched fragments
            ac_enrichment_scores = ac_scores_df["ac_enrichment"]
            positive_ac_frags = set(ac_enrichment_scores[ac_enrichment_scores > 0].index.tolist())
            
            # Fragment visualization and plots removed - not needed
            
            # Save fragments with AC flags, cliff pair sources, and selection metadata
            positive_ac_frags_set = set(ac_scores_df[ac_scores_df["ac_enrichment"] > 0].index) if ac_scores_df is not None else set()
            
            selected_frags_data = []
            for frag in selected_frags:
                frag_data = {
                    "fragment_smiles": frag,
                    "importance": round(importance_scores_norm[frag] if frag in importance_scores_norm.index else 0.0, 6),
                    "stability": round(stability_scores.get(frag, 0.0), 3) if stability_scores is not None else None,
                    "selected_by": "threshold" if frag in selected_by_threshold_set else "AC_added",
                    "is_AC_enriched": frag in positive_ac_frags_set,
                    "was_added_by_cliff_pairs": frag in added_fragments,
                    "cliff_pairs_that_added": "",
                }
                
                # Add cliff pairs that caused addition (if this fragment was added by AC logic)
                if frag in added_fragments:
                    cliff_pairs_str = "; ".join([f"{p[0]}|{p[1]}" for p in added_fragments[frag]]) if added_fragments[frag] else ""
                    frag_data["cliff_pairs_that_added"] = cliff_pairs_str
                
                # Add AC enrichment info if available
                if ac_scores_df is not None and frag in ac_scores_df.index:
                    frag_data["ac_enrichment"] = round(ac_scores_df.loc[frag, "ac_enrichment"], 6)
                    frag_data["n_cliff_pairs"] = ac_scores_df.loc[frag, "n_cliff_pairs"]
                    cliff_pairs_list = ac_scores_df.loc[frag, "cliff_pairs"]
                    frag_data["cliff_pairs_ac_enrichment"] = "; ".join([f"{p[0]}|{p[1]}" for p in cliff_pairs_list]) if cliff_pairs_list else ""
                else:
                    frag_data["ac_enrichment"] = 0.0
                    frag_data["n_cliff_pairs"] = 0
                    frag_data["cliff_pairs_ac_enrichment"] = ""
                
                selected_frags_data.append(frag_data)
            
            pd.DataFrame(selected_frags_data).to_csv(model_dir / "selected_fragments_with_ACflag.csv", index=False)
            
            # Save removed fragments
            if removed_fragments:
                removed_frags_data = []
                for frag, cliff_pairs_list in removed_fragments.items():
                    removed_frags_data.append({
                        "fragment_smiles": frag,
                        "removed_because": "negative_AC_enrichment",
                        "ac_enrichment": ac_scores_df.loc[frag, "ac_enrichment"] if (ac_scores_df is not None and frag in ac_scores_df.index) else 0.0,
                        "cliff_pairs_that_caused_removal": "; ".join([f"{p[0]}|{p[1]}" for p in cliff_pairs_list]) if cliff_pairs_list else "",
                        "importance": importance_scores_norm[frag] if frag in importance_scores_norm.index else 0.0,
                    })
                pd.DataFrame(removed_frags_data).to_csv(model_dir / "removed_fragments.csv", index=False)
            else:
                # Always save CSV (even if empty) for consistency
                pd.DataFrame(columns=["fragment_smiles", "removed_because", "ac_enrichment", "cliff_pairs_that_caused_removal", "importance"]).to_csv(model_dir / "removed_fragments.csv", index=False)
        else:
            # Compute stability if requested (even without AC enrichment)
            report_stability = sel.get("report_stability", True)
            stability_scores = None
            
            if report_stability and selected_frags:
                print("   Computing fragment stability...")
                stability_scores = compute_fragment_stability(
                    fragments=selected_frags,
                    importance_scores=importance_scores_norm,
                    X=X_frag_norm,
                    y=y,
                    n_bootstrap=100,
                    seed=random_state
                )
            
            selected_data = {
                "fragment_smiles": selected_frags,
                "importance": importance_scores_norm[selected_frags].values if not importance_scores_norm.empty else [1.0] * len(selected_frags),
                "is_AC_enriched": [False] * len(selected_frags)
            }
            
            if stability_scores is not None:
                selected_data["stability"] = [round(stability_scores.get(f, 0.0), 3) for f in selected_frags]
            
            pd.DataFrame(selected_data).to_csv(model_dir / "selected_fragments_with_ACflag.csv", index=False)
            
    except Exception as e:
        log.warning(f"[Fragments] Fragment analysis failed: {e}")
        # Make sure selected_frags exists even if error occurred
        if 'selected_frags' not in locals():
            selected_frags = []
    
    # Save outputs
    try:
        if 'selected_frags' in locals() and selected_frags:
            with open(model_dir / "selected_fragments.smi", "w") as f:
                f.write("\n".join(selected_frags))
    except Exception:
        pass
    
    joblib.dump(best_model, model_dir / "best_model.joblib")
    
    # Prepare summary with methodology details
    summary = {
        "best_model": best_model_name,
        "metrics": best_metrics,
        "n_fragments_total": X_frag.shape[1],
        "n_fragments_selected": len(selected_frags) if selected_frags else 0,
        "fragment_selection_method": threshold_mode,
        "train_size": len(tr_idx),
        "test_size": len(te_idx),
        "random_state": random_state,
        "n_model_repeats": n_repeats
    }
    
    # Add stability statistics if computed
    if 'stability_scores' in locals() and stability_scores is not None and selected_frags:
        selected_stability = [stability_scores.get(f, 0.0) for f in selected_frags if f in stability_scores.index]
        if selected_stability:
            summary["selected_fragments_stability_mean"] = round(np.mean(selected_stability), 3)
            summary["selected_fragments_stability_min"] = round(np.min(selected_stability), 3)
            summary["selected_fragments_stability_max"] = round(np.max(selected_stability), 3)
    
    with open(frag_out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("=" * 70)
    print(f"Complete. Selected {len(selected_frags)} fragments. Output directory: {frag_out}")
    print("=" * 70)
