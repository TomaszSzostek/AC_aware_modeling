"""
QSAR evaluation module with activity cliff-aware metrics.

This module builds features for multiple molecular backbones, trains a panel of
classical machine learning models, and aggregates their performance using both
standard and activity cliff-specific metrics.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pathlib
import random
import shutil
import sys
import warnings
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import shap
import yaml
from rdkit import Chem, DataStructs
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC

# Import from new modular structure
from predictive_model.activity_cliff_utils import (
    build_cliff_pairs_for_subset,
    compute_fps_for_graph,
    rmse_cliff_nodewise,
)
from predictive_model.data_splitting import make_cliff_groups, make_groups_butina_scaffold
from predictive_model.descriptor_backbone import _build_X_descriptors_trainfit
from predictive_model.feature_utils import _align_feature_names, _as_named_df, _safe_row_slice
from predictive_model.fingerprint_backbones import _build_X_fp_trainfit
from predictive_model.metrics_plots import (
    bootstrap_auc,
    compute_metrics,
    shap_plots,
    _positive_proba,
    _write_best_overall_itxt,
)
from predictive_model.model_utils import (
    _fit_model,
    _validate_dataset,
    repeated_cv_auc,
    safe_permutation_importance,
    tune_threshold,
    y_scramble_auc,
)

try:
    from imblearn.ensemble import BalancedRandomForestClassifier as BRF
    IMB_OK = True
except Exception:
    IMB_OK = False

try:
    import xgboost as xgb
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    import catboost as cb
    CAT_OK = True
except Exception:
    CAT_OK = False

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

# Cheminformatics gold standard parameters
DEFAULT_RANDOM_STATE = 42
DEFAULT_TOP_K = 128

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.calibration")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def ensure_dir(p: pathlib.Path):
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)


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


def aggregate_metrics_with_ci(metrics_list: List[dict], n_bootstrap: int = 2000, seed: int = 42) -> dict:
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


def seed_everything(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def nm_to_pIC50(nm: pd.Series) -> pd.Series:
    """Convert IC50 in nM to pIC50."""
    M = nm.astype(float) * 1e-9
    with np.errstate(divide='ignore'):
        return -np.log10(M)


# =============================================================================
# FEATURE BUILDING (uses imported modules)
# =============================================================================

def _build_X_for_backbone(
        backbone: str,
        df: pd.DataFrame,
        smiles_col: str,
        y: np.ndarray,
        fit_idx: np.ndarray,
        results_root: pathlib.Path,
        log: logging.Logger,
        config: dict,
        seed: int
) -> Tuple[np.ndarray, List[str]]:
    """
    Router for feature backbones (fingerprints vs descriptor-based).
    
    Args:
        backbone: Backbone name ("ecfp1024", "ecfp2048", "maccs", "descriptors")
        df: DataFrame containing molecular data
        smiles_col: Name of column containing SMILES strings
        y: Target labels
        fit_idx: Training indices
        results_root: Directory for saving artifacts
        log: Logger instance
        config: Configuration dictionary
        seed: Random seed
        
    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    b = backbone.strip().lower()

    if b in {"ecfp1024", "ecfp2048", "maccs"}:
        return _build_X_fp_trainfit(
            smiles_series=df[smiles_col],
            backbone=backbone,
            fit_idx=fit_idx,
            results_root=results_root,
            log=log
        )

    if b in {"descriptors", "descriptors_boruta"}:
        return _build_X_descriptors_trainfit(
            smiles_series=df[smiles_col],
            y=y,
            fit_idx=fit_idx,
            results_root=results_root,
            log=log,
            random_state=seed,
            include_mordred=True,
            boruta_max_iter=200,
            top_k=int(os.environ.get("DESC_TOP_K", str(config.get("Cheminformatics", {}).get("default_top_k", DEFAULT_TOP_K)))),
        )

    raise ValueError(f"Unknown backbone: {backbone}")


# =============================================================================
# MODEL FACTORY AND HELPERS
# =============================================================================

def _create_model_factory(qcfg, seed):
    """
    Create model factory function for ML engines.
    
    Args:
        qcfg: QSAR evaluation configuration
        seed: Random seed
        
    Returns:
        Function that creates model instances
    """
    def make_model(name: str):
        """Factory for classical ML engines."""
        name = name.strip()
        if name == "LogReg":
            return LogisticRegression(C=0.5, penalty="l2", solver="liblinear",
                                      class_weight="balanced", max_iter=4000, random_state=seed)
        if name == "Ridge":
            return RidgeClassifier(alpha=1.0, class_weight="balanced", random_state=seed)
        if name == "SVC":
            C_cfg = float(qcfg.get("svc_C", 2.0))
            return SVC(kernel="rbf", C=C_cfg, gamma="scale", probability=True,
                       class_weight="balanced", random_state=seed)
        if name == "RF":
            return RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=10, min_samples_split=20, max_features="sqrt",
                                          class_weight="balanced_subsample", n_jobs=-1, random_state=seed)
        if name == "BRF":
            if not IMB_OK:
                raise RuntimeError("imblearn not available for BalancedRandomForest.")
            return BRF(n_estimators=300, max_depth=4, min_samples_leaf=10, min_samples_split=20, max_features="sqrt",
                       sampling_strategy="all", replacement=True, bootstrap=False,
                       n_jobs=-1, random_state=seed)
        if name == "ExtraTrees":
            return ExtraTreesClassifier(n_estimators=300, max_depth=4, min_samples_leaf=10, min_samples_split=20, max_features="sqrt",
                                        class_weight="balanced", n_jobs=-1, random_state=seed)
        if name == "GB":
            return GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, min_samples_leaf=10,
                                              subsample=0.8, max_features="sqrt", random_state=seed)
        if name == "LinSVM":
            return LinearSVC(C=float(qcfg.get("linear_svm_C", 1.0)),
                             class_weight="balanced", random_state=seed)
        if name == "KNN":
            return KNeighborsClassifier(n_neighbors=int(qcfg.get("knn_k", 15)),
                                        weights="uniform", metric="minkowski", p=2)
        if name == "XGB":
            if not XGB_OK:
                raise RuntimeError("xgboost not installed.")
            return xgb.XGBClassifier(
                n_estimators=300, max_depth=3, learning_rate=0.1, subsample=0.8,
                colsample_bytree=0.8, min_child_weight=5.0, reg_alpha=1.0, reg_lambda=2.0,
                eval_metric="auc", n_jobs=-1, random_state=seed, tree_method="auto"
            )
        if name == "CatBoost":
            if not CAT_OK:
                raise RuntimeError("catboost not installed.")
            return cb.CatBoostClassifier(
                iterations=500, learning_rate=0.1, depth=4, l2_leaf_reg=5.0, min_data_in_leaf=10,
                loss_function="Logloss", eval_metric="Logloss", random_seed=seed, verbose=False
            )
        raise ValueError(f"Unknown engine in config: {name}")
    return make_model


def _rank_models(df_metrics: pd.DataFrame, primary: str) -> pd.DataFrame:
    """
    Rank models by primary metric STABILITY (CI width or SD) with tie-breakers.
    
    Ranking logic:
    1. Primary metric CI width (narrower = more stable = better)
    2. Primary metric mean (lower for RMSE, higher for AUC)
    3. AUPRC_active (higher = better)
    4. AUROC (higher = better)
    5. Brier (lower = better)
    
    Args:
        df_metrics: DataFrame with model metrics
        primary: Primary metric name
        
    Returns:
        Sorted DataFrame (most stable model first)
    """
    prim_in = str(primary)
    prim = "Cliff_RMSE" if prim_in.lower() in {"rmse_cliff", "cliff_rmse"} else prim_in
    dfc = df_metrics.copy()

    for col in ["AUROC", "AUPRC_active", "F1", "MCC", "ACC", "Brier", "ECE", "RMSE", "Cliff_RMSE", prim]:
        if col not in dfc.columns:
            dfc[col] = np.nan
    
    # Compute CI width for primary metric (stability measure)
    ci_lower_col = f"{prim}_CI_lower"
    ci_upper_col = f"{prim}_CI_upper"
    if ci_lower_col in dfc.columns and ci_upper_col in dfc.columns:
        dfc["_ci_width"] = (dfc[ci_upper_col] - dfc[ci_lower_col]).astype(float)
    else:
        # Fallback to SD if CI not available
        sd_col = f"{prim}_SD"
        if sd_col in dfc.columns:
            dfc["_ci_width"] = dfc[sd_col].astype(float)
        else:
            dfc["_ci_width"] = 0.0
    
    minimize = prim.lower() in {"brier", "ece", "rmse", "cliff_rmse", "rmse_cliff"}
    
    # Ranking keys: STABILITY FIRST, then mean
    dfc["_k1"] = dfc["_ci_width"]                                           # 1. CI width (narrower = better)
    dfc["_k2"] = (dfc[prim].astype(float)) if minimize else -(dfc[prim].astype(float))  # 2. Mean
    dfc["_k3"] = -dfc["AUPRC_active"].astype(float)                        # 3. AUPRC
    dfc["_k4"] = -dfc["AUROC"].astype(float)                               # 4. AUROC
    dfc["_k5"] = dfc["Brier"].astype(float)                                 # 5. Brier
    
    return dfc.sort_values(["_k1", "_k2", "_k3", "_k4", "_k5"]).drop(columns=["_k1", "_k2", "_k3", "_k4", "_k5", "_ci_width"])


# =============================================================================
# MODEL EVALUATION HELPERS
# =============================================================================

def create_split_qsar(
    df: pd.DataFrame,
    y: np.ndarray,
    smiles_col: str,
    mode: str,
    test_size: float,
    random_state: int,
    cliffs_csv: pathlib.Path,
    sphere_tanimoto: float,
    log: logging.Logger
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create train/test split using specified strategy (cliff_group or butina).
    
    Args:
        df: Full dataset DataFrame
        y: Binary labels
        smiles_col: Name of SMILES column
        mode: Split mode ("cliff_group" or "butina")
        test_size: Fraction for test set
        random_state: Random seed for this split
        cliffs_csv: Path to activity cliffs CSV
        sphere_tanimoto: Tanimoto threshold for Butina clustering
        log: Logger instance
        
    Returns:
        Tuple of (train_idx, test_idx)
    """
    if mode == "cliff_group":
        from predictive_model.data_splitting import make_cliff_groups
        groups = make_cliff_groups(df, smiles_col, cliffs_csv)
    else:  # butina
        from predictive_model.data_splitting import make_groups_butina_scaffold
        groups = make_groups_butina_scaffold(
            df[smiles_col].astype(str).tolist(),
            radius=2, n_bits=2048, tn_thr=sphere_tanimoto
        )
    
    # StratifiedGroupShuffleSplit
    try:
        from sklearn.model_selection import StratifiedGroupShuffleSplit
        sgss = StratifiedGroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        (train_idx, test_idx), = sgss.split(np.zeros((len(y), 1)), y, groups)
    except Exception:
        # Fallback to GroupShuffleSplit (silent, works equally well)
        from sklearn.model_selection import GroupShuffleSplit
        base_ratio = float(np.mean(y))
        best = None
        best_diff = 1e9
        gss = GroupShuffleSplit(n_splits=256, test_size=test_size, random_state=random_state)
        for tr_idx, te_idx in gss.split(np.zeros((len(y), 1)), y, groups):
            diff = abs(float(np.mean(y[te_idx])) - base_ratio)
            if diff < best_diff:
                best, best_diff = (tr_idx, te_idx), diff
            if diff <= 0.02:
                break
        if best is None:
            gss_single = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
            (train_idx, test_idx), = gss_single.split(np.zeros((len(y), 1)), y, groups)
        else:
            train_idx, test_idx = best
    
    return train_idx, test_idx


def evaluate_model_once(
    name: str,
    make_model: Callable,
    Xtr_df: pd.DataFrame,
    Xte_df: pd.DataFrame,
    ytr: np.ndarray,
    yte: np.ndarray,
    test_idx: np.ndarray,
    pairs_cliff_te: List[Tuple[int, int]],
    backbone: str,
    th_cfg: dict,
    seed: int,
    log: logging.Logger
) -> dict:
    """
    Evaluate a single model once (single train/test run).
    
    Args:
        name: Model name
        make_model: Model factory function
        Xtr_df: Training features
        Xte_df: Test features
        ytr: Training labels
        yte: Test labels
        test_idx: Test indices
        pairs_cliff_te: Activity cliff pairs in test set
        backbone: Feature backbone name
        th_cfg: Threshold optimization config
        seed: Random seed
        log: Logger
        
    Returns:
        Dictionary of metrics
    """
    from predictive_model.model_utils import _fit_model, tune_threshold
    from predictive_model.feature_utils import _safe_row_slice
    from predictive_model.metrics_plots import compute_metrics
    
    # Train model
    model = make_model(name)
    model_fitted = _fit_model(model, np.asarray(Xtr_df), np.asarray(ytr))
    
    # Threshold optimization
    tuned_thr: float = 0.5
    if bool(th_cfg.get("enable", True)):
        if len(np.unique(ytr)) < 2:
            tuned_thr = 0.5
        else:
            from sklearn.model_selection import StratifiedShuffleSplit
            sss = StratifiedShuffleSplit(
                n_splits=1,
                test_size=float(th_cfg.get("val_size", 0.2)),
                random_state=int(th_cfg.get("random_state", seed))
            )
            (tr_sub, val_sub), = sss.split(np.asarray(Xtr_df), ytr)
            
            if len(np.unique(ytr[val_sub])) < 2:
                tuned_thr = 0.5
            else:
                sub_model = make_model(name)
                sub_fitted = _fit_model(sub_model, np.asarray(_safe_row_slice(Xtr_df, tr_sub)), np.asarray(ytr[tr_sub]))
                if hasattr(sub_fitted, "predict_proba"):
                    p_val = sub_fitted.predict_proba(np.asarray(_safe_row_slice(Xtr_df, val_sub)))[:, 1]
                else:
                    s = sub_fitted.decision_function(np.asarray(_safe_row_slice(Xtr_df, val_sub)))
                    s = np.asarray(s, float).ravel()
                    p_val = (s - s.min()) / (s.max() - s.min() + 1e-9)
                
                tuned_thr = tune_threshold(ytr[val_sub], p_val, str(th_cfg.get("metric", "MCC")))
    
    # TEST predictions
    Xte_np = np.asarray(Xte_df, dtype=float)
    if hasattr(model_fitted, "predict_proba"):
        proba = model_fitted.predict_proba(Xte_np)[:, 1]
    else:
        s = model_fitted.decision_function(Xte_np)
        s = np.asarray(s, float).ravel()
        proba = (s - s.min()) / (s.max() - s.min() + 1e-9)
    
    if np.any(proba < 0) or np.any(proba > 1):
        proba = np.clip(proba, 0, 1)
    
    # Compute in_cliff_nodes for metrics
    nodes_cliff_set = set()
    for i, j in pairs_cliff_te:
        nodes_cliff_set.add(i)
        nodes_cliff_set.add(j)
    in_cliff_nodes = np.zeros(len(test_idx), dtype=bool)
    for local_idx, global_idx in enumerate(test_idx):
        if local_idx in nodes_cliff_set:
            in_cliff_nodes[local_idx] = True
    
    # Compute metrics
    try:
        auroc = float(roc_auc_score(yte, proba)) if len(np.unique(yte)) == 2 else float("nan")
    except Exception:
        auroc = float("nan")
    
    m = compute_metrics(yte, proba, pairs_cliff_te, in_cliff_nodes, threshold=tuned_thr)
    m.update({
        "model": name,
        "backbone": backbone,
        "AUROC": auroc,
        "Threshold": float(tuned_thr),
        "Calibration": "none",
    })
    
    return m


# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_qsar_evaluation(config: dict, log: logging.Logger):
    """
    Comprehensive QSAR evaluation pipeline with activity cliff awareness.
    
    This function orchestrates the complete QSAR evaluation workflow.
    """
    print("Starting QSAR evaluation pipeline...")
    
    # ==================== Paths, seeds, config ====================
    try:
        paths = config["Paths"]
        artifacts = config["Artifacts"]
        results_root = pathlib.Path(paths["results_root"])
        ensure_dir(results_root)
        qcfg = config.get("QSAR_Eval", {})
        
        seed = int(config.get("ReverseQSAR", {}).get("rf", {}).get("random_state", DEFAULT_RANDOM_STATE))
        seed_everything(seed)
        print(f"Configuration loaded. Seed: {seed}")
    except KeyError as e:
        log.error(f"[QSAR] Missing required configuration key: {e}")
        raise ValueError(f"Missing required configuration key: {e}")
    except Exception as e:
        log.error(f"[QSAR] Configuration error: {e}")
        raise ValueError(f"Configuration error: {e}")

    RESUME_ENABLE = bool(qcfg.get("resume", {}).get("enable", True))
    RESUME_OVERWRITE = bool(qcfg.get("resume", {}).get("overwrite", False))
    REBUILD_FEATURES = bool(qcfg.get("rebuild_features", False))

    print("Loading dataset...")
    # ==================== Load dataset ====================
    try:
        final_csv = pathlib.Path(paths.get("dataset"))
        if not final_csv.exists():
            log.error(f"Missing dataset: {final_csv}")
            raise FileNotFoundError(f"Missing dataset: {final_csv}")
        
        df = pd.read_csv(final_csv)
        print(f"Dataset loaded: {df.shape[0]} compounds, {df.shape[1]} features")
    except FileNotFoundError as e:
        log.error(f"[QSAR] Dataset file not found: {e}")
        raise
    except Exception as e:
        log.error(f"[QSAR] Error loading dataset: {e}")
        raise RuntimeError(f"Error loading dataset: {e}")

    smiles_col = config.get("AC_Analysis", {}).get("smiles_col", "canonical_smiles")
    if "pIC50" not in df.columns and "IC50_nM" in df.columns:
        df["pIC50"] = nm_to_pIC50(df["IC50_nM"])
    thr_nm = float(config.get("THRESHOLD_NM", 10000))
    thr_pic50 = 9.0 - math.log10(thr_nm)
    df["y_active"] = (df["pIC50"] >= thr_pic50).astype(int)
    y = df["y_active"].values

    # Validate dataset
    try:
        if not _validate_dataset(df, smiles_col, log):
            log.error("Dataset validation failed. Aborting evaluation.")
            raise ValueError("Dataset validation failed")
    except Exception as e:
        log.error(f"[QSAR] Dataset validation error: {e}")
        raise ValueError(f"Dataset validation error: {e}")

    print("Creating train/test split...")
    # ==================== Grouped Split (Cliff-group or Butina, stratified 80/20) ====================
    df = df.reset_index(drop=True)
    y = np.asarray(df["y_active"].values).astype(int)

    split_cfg = config.get("QSAR_Eval", {}).get("split", {}) if isinstance(config, dict) else {}
    mode = split_cfg.get("mode", "butina")
    test_size = float(split_cfg.get("test_size", 0.2))
    seed = int(split_cfg.get("random_state", 42))

    if mode == "cliff_group":
        cliffs_csv = Path(config["Paths"]["ac_analysis"]) / "activity_cliffs.csv"
        groups = make_cliff_groups(df, smiles_col, cliffs_csv)
        print(f"Using cliff-group split from {cliffs_csv}")
    else:
        _tn = float(split_cfg.get("sphere_tanimoto", 0.7))
        groups = make_groups_butina_scaffold(
            df[smiles_col].astype(str).tolist(),
            radius=2, n_bits=2048, tn_thr=_tn
        )
        print(f"Using Butina scaffold clustering (Tanimoto ≥ {_tn})")

    unique_groups = len(np.unique(groups))
    if len(groups) != len(y):
        log.error(f"[Split] Length mismatch: y={len(y)}, groups={len(groups)}")
        raise ValueError(f"Length mismatch: y={len(y)}, groups={len(groups)}")

    # --- StratifiedGroupShuffleSplit (with safe fallback for older sklearn) ---
    try:
        from sklearn.model_selection import StratifiedGroupShuffleSplit
        sgss = StratifiedGroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        (train_idx, test_idx), = sgss.split(np.zeros((len(y), 1)), y, groups)
    except Exception as e:
        log.warning(f"[Split] StratifiedGroupShuffleSplit failed: {e}. Using fallback method.")
        from sklearn.model_selection import GroupShuffleSplit
        base_ratio = float(np.mean(y))
        best = None
        best_diff = 1e9
        gss = GroupShuffleSplit(n_splits=256, test_size=test_size, random_state=seed)
        for tr_idx, te_idx in gss.split(np.zeros((len(y), 1)), y, groups):
            diff = abs(float(np.mean(y[te_idx])) - base_ratio)
            if diff < best_diff:
                best, best_diff = (tr_idx, te_idx), diff
            if diff <= 0.02:
                break
        if best is None:
            raise RuntimeError("Failed to create valid train/test split")
        train_idx, test_idx = best

    ytr, yte = y[train_idx], y[test_idx]
    train_balance = float(np.mean(ytr))
    test_balance = float(np.mean(yte))
    print(
        f"Split complete: Train={len(train_idx)} ({train_balance:.1%} active) | "
        f"Test={len(test_idx)} ({test_balance:.1%} active)"
    )

    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        raise RuntimeError("Both splits must contain both active and inactive compounds")

    # ==================== Cliff pairs (TEST only) ====================
    ac_cfg = config.get("AC_Analysis", {})
    sim_thr = float(ac_cfg.get("similarity_threshold", 0.85))
    dp_thr = float(ac_cfg.get("potency_diff", 1.0))
    fps = compute_fps_for_graph(df[smiles_col].tolist(), radius=2, n_bits=2048)
    pairs_cliff_te = build_cliff_pairs_for_subset(
        fps, np.asarray(test_idx), df["pIC50"].values, sim_thr, dp_thr
    )
    print(f"Found {len(pairs_cliff_te)} test cliff pairs")

    # ----------------------------- Model factory -------------------------------
    engines_cfg = qcfg.get("engines", None)
    if not engines_cfg or not isinstance(engines_cfg, (list, tuple)) or len(engines_cfg) == 0:
        log.error("QSAR_Eval.engines is missing or empty in config.")
        return

    make_model = _create_model_factory(qcfg, seed)
    shap_cfg = qcfg.get("shap", {})
    th_cfg = qcfg.get("threshold_opt", {})

    backbones_req = qcfg.get("backbones", ["descriptors", "ecfp1024", "ecfp2048", "maccs"])

    def _canon_backbone(name: str) -> str:
        n = str(name).strip().lower()
        if n in {"desc", "descriptor", "descriptors", "rdkit", "rdkit_desc", "rdkit_descriptors", "mordred"}:
            return "descriptors"
        if n in {"ecfp", "ecfp2048"}:
            return "ecfp2048"
        if n in {"ecfp1024"}:
            return "ecfp1024"
        if n in {"maccs"}:
            return "maccs"
        return n

    backbones = [
        b for b in map(_canon_backbone, backbones_req)
        if b in {"descriptors", "ecfp1024", "ecfp2048", "maccs"}
    ]
    print(f"Running backbones: {', '.join(backbones)}")

    global_rows = []

    # ========================= Main loop: per backbone ==========================
    for backbone in backbones:
        print(f"\n{'='*60}")
        print(f"Processing backbone: {backbone}")
        print(f"{'='*60}")
        RR = results_root / backbone
        ensure_dir(RR)

        # ---- Build or load features (fit transforms on TRAIN only) ----
        x_blob_path = RR / artifacts.get("X_full", "X_full.pkl")
        X_full, feat_names = None, None
        if RESUME_ENABLE and not RESUME_OVERWRITE and x_blob_path.exists() and not REBUILD_FEATURES:
            try:
                blob = joblib.load(x_blob_path)
                X_full = blob.get("X", None)
                feat_names = blob.get("features", None)
                if X_full is not None and feat_names is not None:
                        print(f"  Loaded cached features for {backbone}")
            except Exception as e:
                log.warning(f"[Features] Could not load cached features ({e}); rebuilding.")
        if X_full is None or feat_names is None:
            print(f"  🔨 Building features for {backbone}...")
            X_full, feat_names = _build_X_for_backbone(
                backbone=backbone, df=df, smiles_col=smiles_col,
                y=y, fit_idx=train_idx, results_root=RR, log=log, config=config, seed=seed
            )
            joblib.dump({"X": X_full, "features": feat_names}, x_blob_path)

        X_full = np.asarray(X_full, dtype=np.float32)
        feat_names = _align_feature_names(feat_names, X_full.shape[1], prefix=backbone)
        joblib.dump({"X": X_full, "features": feat_names}, x_blob_path)

        Xtr, Xte = X_full[train_idx], X_full[test_idx]
        Xtr_df = _as_named_df(Xtr, feat_names, log=log, ctx="train")
        Xte_df = _as_named_df(Xte, feat_names, log=log, ctx="test")
        feat_names_used = list(getattr(Xtr_df, "columns", feat_names))
        
        # Remove zero-variance features
        try:
            Xtr_arr = np.asarray(Xtr_df, dtype=float)
            var_mask = (np.nanstd(Xtr_arr, axis=0) > 0.0)
            if not bool(np.all(var_mask)):
                kept_names = [feat_names_used[i] for i, k in enumerate(var_mask) if bool(k)]
                Xtr_df = _as_named_df(Xtr_arr[:, var_mask], kept_names, log=log, ctx="train_var")
                Xte_df = _as_named_df(np.asarray(Xte_df, dtype=float)[:, var_mask], kept_names, log=log, ctx="test_var")
                feat_names_used = kept_names
                
                # Save selected feature names after variance filtering (for generator scorer)
                selected_features_path = RR / "selected_features.txt"
                with open(selected_features_path, "w") as f:
                    f.write("\n".join(kept_names))
                log.info(f"[Features] Saved {len(kept_names)} selected features (after var-filtering) to {selected_features_path.name}")
        except Exception:
            pass

        # ------------------------ Per-engine training loop -----------------------
        metrics_dir = RR / "model_metrics"
        ensure_dir(metrics_dir)
        models_dir = RR / "models"
        ensure_dir(models_dir)
        plots_dir = RR / "plots"
        ensure_dir(plots_dir)
        pred_dir = RR / artifacts.get("per_model_pred_csv_dir", "predictions")
        ensure_dir(pred_dir)
        pi_dir = RR / artifacts.get("per_model_pi_csv_dir", "permutation_importance")
        ensure_dir(pi_dir)

        per_model_rows = []  # For backward compatibility (first run only)
        all_runs_metrics = []  # All runs from all models
        aggregated_metrics = []  # Aggregated metrics per model
        
        # Get number of repeats from config
        n_repeats = int(qcfg.get("n_repeats", 10))
        print(f"  Running {len(engines_cfg)} models with {n_repeats} independent splits each...")
        
        for name in engines_cfg:
            label = f"{name}"
            model_path = models_dir / f"{label}.joblib"
            metrics_path = metrics_dir / f"metrics_{label}.json"
            pred_path = pred_dir / f"predictions_{label}.csv"
            
            print(f"  Training {name} ({n_repeats} independent splits)...")
            
            model_run_metrics = []  # Store metrics for this model's repeats
            
            try:
                # Run model multiple times with DIFFERENT SPLITS
                for repeat_idx in range(n_repeats):
                    repeat_seed = seed + repeat_idx
                    
                    # Create NEW split for this repeat (preserving split logic)
                    repeat_train_idx, repeat_test_idx = create_split_qsar(
                        df=df,
                        y=y,
                        smiles_col=smiles_col,
                        mode=mode,
                        test_size=test_size,
                        random_state=repeat_seed,
                        cliffs_csv=Path(config["Paths"]["ac_analysis"]) / "activity_cliffs.csv",
                        sphere_tanimoto=float(split_cfg.get("sphere_tanimoto", 0.7)),
                        log=log
                    )
                    
                    # Prepare data for this split
                    repeat_Xtr, repeat_Xte = X_full[repeat_train_idx], X_full[repeat_test_idx]
                    repeat_ytr, repeat_yte = y[repeat_train_idx], y[repeat_test_idx]
                    repeat_Xtr_df = _as_named_df(repeat_Xtr, feat_names_used, log=log, ctx=f"train_r{repeat_idx}")
                    repeat_Xte_df = _as_named_df(repeat_Xte, feat_names_used, log=log, ctx=f"test_r{repeat_idx}")
                    
                    # Compute cliff pairs for this split's test set
                    repeat_pairs_cliff_te = build_cliff_pairs_for_subset(fps, np.asarray(repeat_test_idx), df["pIC50"].values, sim_thr, dp_thr)
                    
                    # Evaluate model on this split
                    m = evaluate_model_once(
                        name=name,
                        make_model=make_model,
                        Xtr_df=repeat_Xtr_df,
                        Xte_df=repeat_Xte_df,
                        ytr=repeat_ytr,
                        yte=repeat_yte,
                        test_idx=repeat_test_idx,
                        pairs_cliff_te=repeat_pairs_cliff_te,
                        backbone=backbone,
                        th_cfg=th_cfg,
                        seed=repeat_seed,
                        log=log
                    )
                    
                    # Add repeat index and split info
                    m["repeat"] = repeat_idx
                    m["n_train"] = len(repeat_train_idx)
                    m["n_test"] = len(repeat_test_idx)
                    m["n_cliff_pairs"] = len(repeat_pairs_cliff_te)
                    model_run_metrics.append(m)
                    all_runs_metrics.append(m)
                
                # Aggregate metrics for this model
                aggregated = aggregate_metrics_with_ci(
                    model_run_metrics,
                    n_bootstrap=2000,
                    seed=seed
                )
                aggregated_metrics.append(aggregated)
                
                # Report mean ± SD for primary metric
                primary_metric = str(config.get("QSAR_Eval", {}).get("primary_metric", "Cliff_RMSE"))
                prim_key = "Cliff_RMSE" if primary_metric.lower() in {"rmse_cliff", "cliff_rmse"} else primary_metric
                mean_val = aggregated.get(prim_key, float("nan"))
                sd_val = aggregated.get(f"{prim_key}_SD", float("nan"))
                ci_lower = aggregated.get(f"{prim_key}_CI_lower", float("nan"))
                ci_upper = aggregated.get(f"{prim_key}_CI_upper", float("nan"))
                print(f"     ✓ {prim_key}: {mean_val:.3f} ± {sd_val:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
                
                # Save first run for backward compatibility
                if model_run_metrics:
                    first_run = model_run_metrics[0].copy()
                    if "repeat" in first_run:
                        del first_run["repeat"]
                    per_model_rows.append(first_run)
                    
                    # Save first run metrics (backward compatibility)
                    try:
                        with open(metrics_path, "w") as fh:
                            json.dump(first_run, fh, indent=2)
                    except Exception as e:
                        log.warning(f"Could not save metrics for {label}: {e}")
                    
                    # Train and save first model for downstream use
                    if RESUME_ENABLE and not RESUME_OVERWRITE and model_path.exists():
                        pass  # Keep existing model
                    else:
                        try:
                            model = make_model(name)
                            model_fitted = _fit_model(model, np.asarray(Xtr_df), np.asarray(ytr))
                            joblib.dump(model_fitted, model_path)
                        except Exception as e:
                            log.warning(f"Could not save model {label}: {e}")
                
            except Exception as e:
                log.warning(f"[Model] Failed to evaluate {name}: {e}")
                continue

        # ----------------- Persist per-backbone metrics & summary ----------------
        # Save CSV files: numeric (for calculations), formatted (for publication) and all_runs (for diagnostics)
        print(f"  Saving metrics CSVs (all_runs: {len(all_runs_metrics)}, aggregated: {len(aggregated_metrics)})")
        
        # Save all_runs CSV
        all_runs_csv = RR / "model_metrics_all_runs.csv"
        if all_runs_metrics:
            all_runs_df = pd.DataFrame(all_runs_metrics)
            if 'backbone' not in all_runs_df.columns:
                all_runs_df['backbone'] = backbone
            if 'model' not in all_runs_df.columns:
                all_runs_df['model'] = all_runs_df.get('name', '')
            all_runs_df.to_csv(all_runs_csv, index=False, float_format='%.6f')
        
        if all_runs_metrics:
            if aggregated_metrics:
                aggregated_df = pd.DataFrame(aggregated_metrics)
                
                # Add backbone column if missing
                if 'backbone' not in aggregated_df.columns:
                    aggregated_df['backbone'] = backbone
                
                # ===== CSV 1: NUMERIC (mean + CI_lower + CI_upper) - for calculations/ranking =====
                numeric_cols = ['model', 'backbone']
                for col in aggregated_df.columns:
                    if col in ['model', 'backbone', 'Calibration', 'repeat']:
                        continue
                    if '_SD' in col:  # Exclude SD
                        continue
                    if col in ['n_train', 'n_test', 'n_cliff_pairs']:  # Exclude run-specific
                        continue
                    numeric_cols.append(col)
                
                numeric_df = aggregated_df[[c for c in numeric_cols if c in aggregated_df.columns]].copy()
                numeric_csv = RR / "models_metrics_numeric.csv"
                numeric_df.to_csv(numeric_csv, index=False, float_format='%.3f')
                
                # For global ranking
                global_rows.append(numeric_df)
                
                # ===== CSV 2: FORMATTED (with CI strings) - for publication =====
                formatted_df = numeric_df.copy()
                
                # Identify metrics with CI
                metrics_with_ci = []
                for col in numeric_df.columns:
                    if col not in ['model', 'backbone'] and f"{col}_CI_lower" in numeric_df.columns:
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
                
                # Drop CI bounds (now included in formatted strings)
                columns_to_drop = [col for col in formatted_df.columns if "_CI_lower" in col or "_CI_upper" in col]
                formatted_df = formatted_df.drop(columns=[c for c in columns_to_drop if c in formatted_df.columns])
                
                formatted_csv = RR / "models_metrics_formatted.csv"
                formatted_df.to_csv(formatted_csv, index=False)
                
                # Figures are generated in R

        # ------------------------ Select & refit best model ----------------------
        if not aggregated_metrics:
            continue
        primary_metric = str(config.get("QSAR_Eval", {}).get("primary_metric", "RMSE_cliff"))
        # Use aggregated metrics (mean over all repeats) instead of first repeat only
        ranked = _rank_models(pd.DataFrame(aggregated_metrics), primary_metric)
        best_row = ranked.iloc[0]
        best_name = str(best_row['model'])

        print(f"  🏆 Best model for {backbone}: {best_name}")
        mdl_best = make_model(best_name)
        mdl_best_fitted = _fit_model(mdl_best, np.asarray(Xtr_df), np.asarray(ytr))
        joblib.dump(mdl_best_fitted, models_dir / f"{best_name}.joblib")
        canonical = RR / artifacts.get("qsar_best_model", "best_model.joblib")
        shutil.copyfile(models_dir / f"{best_name}.joblib", canonical)

        # Winner diagnostics + SHAP/PI
        Xte_np_for_best = np.asarray(Xte_df, dtype=float)
        if hasattr(mdl_best_fitted, "predict_proba"):
            best_prob = mdl_best_fitted.predict_proba(Xte_np_for_best)[:, 1]
        else:
            s = mdl_best_fitted.decision_function(Xte_np_for_best)
            s = np.asarray(s, float).ravel()
            best_prob = (s - s.min()) / (s.max() - s.min() + 1e-9)

        best_thr = float(best_row.get("Threshold", 0.5)) if np.isfinite(best_row.get("Threshold", np.nan)) else 0.5

        best_out_dir = RR
        ensure_dir(best_out_dir)

        # SHAP plots only (figures are generated in R)
        try:
            shap_plots(mdl_best_fitted, Xte_df, feat_names_used, plots_dir, prefix=best_name, log=log, shap_cfg=shap_cfg)
        except Exception as e:
            log.warning(f"SHAP failed for winner ({backbone}/{best_name}): {e}")

        pi_cfg = qcfg.get("permutation_importance", {})
        if bool(pi_cfg.get("enable", True)):
            pi_path = pi_dir / f"permutation_importance_{best_name}.csv"
            if RESUME_OVERWRITE or not pi_path.exists():
                try:
                    pi = safe_permutation_importance(
                        mdl_best_fitted, np.asarray(Xte_df), np.asarray(yte), feat_names_used,
                        n_repeats=int(pi_cfg.get("n_repeats", 10)), random_state=seed,
                        scoring="roc_auc", top_n=int(pi_cfg.get("top_n", 30)), log=log
                    )
                    pi.to_csv(pi_path, index=False)
                except Exception as e:
                    log.warning(f"Permutation importance failed for winner ({backbone}/{best_name}): {e}")

        # ======================= EXTRAS (bootstrap / y-scramble / CV) =======================
        extras_cfg = qcfg.get("extras", {})
        if bool(extras_cfg.get("enable", True)):
            extras_dir = RR / artifacts.get("extras_dir", "extras")
            ensure_dir(extras_dir)

            extras_txt = extras_dir / f"extras_{best_name}.txt"
            need_extras = RESUME_OVERWRITE or (not extras_txt.exists())

            if need_extras:
                try:
                    b_n = int(extras_cfg.get("bootstrap_n", 1000))
                    b_med, b_lo, b_hi = bootstrap_auc(yte, best_prob, n=b_n, seed=seed)

                    ys_n = int(extras_cfg.get("yscramble_n", 30))
                    ys_mean, ys_std = y_scramble_auc(
                        lambda: make_model(best_name),
                        np.asarray(Xtr_df), np.asarray(ytr),
                        n=ys_n, seed=seed
                    )

                    cv_splits = int(extras_cfg.get("cv_splits", 5))
                    cv_repeats = int(extras_cfg.get("cv_repeats", 10))
                    cv_scores = repeated_cv_auc(
                        lambda: make_model(best_name),
                        np.asarray(Xtr_df), np.asarray(ytr),
                        splits=cv_splits, repeats=cv_repeats, seed=seed
                    )

                    with open(extras_txt, "w") as fh:
                        fh.write(f"Model: {best_name}\n")
                        fh.write(f"Backbone: {backbone}\n")
                        fh.write(f"Bootstrap_AUROC_median: {b_med}\n")
                        fh.write(f"Bootstrap_AUROC_CI95: [{b_lo}, {b_hi}]\n")
                        fh.write(f"Yscramble_AUROC_mean: {ys_mean}\n")
                        fh.write(f"Yscramble_AUROC_std: {ys_std}\n")
                        if cv_scores is not None and len(cv_scores) > 0:
                            fh.write(f"CV_AUROC_mean: {float(np.mean(cv_scores))}\n")
                            fh.write(f"CV_AUROC_std: {float(np.std(cv_scores))}\n")
                            fh.write(f"CV_splits: {cv_splits}, CV_repeats: {cv_repeats}\n")
                        else:
                            fh.write("CV_AUROC_mean: nan\nCV_AUROC_std: nan\n")

                    try:
                        if cv_scores is not None and len(cv_scores) > 0:
                            pd.DataFrame({"cv_auroc": cv_scores}).to_csv(
                                extras_dir / f"cv_auroc_{best_name}.csv", index=False
                            )
                    except Exception as e_csv:
                        log.warning(f"[EXTRAS] Could not save CV CSV for {backbone}/{best_name}: {e_csv}")

                except Exception as e:
                    log.warning(f"[EXTRAS] Failed to compute extras for {backbone}/{best_name}: {e}")

    # ================= Global aggregation & BEST OVERALL ==============
    print(f"\n{'='*60}")
    print("Aggregating results across all backbones...")
    print(f"{'='*60}")
    non_empty = [d for d in global_rows if d is not None and not d.empty]
    
    if non_empty:
        # Concatenate all numeric tables (for ranking)
        all_rows_numeric = pd.concat(non_empty, ignore_index=True)
        
        # Rank using NUMERIC columns (stability-based)
        primary_metric = str(qcfg.get("primary_metric", "RMSE_cliff"))
        ranked_overall = _rank_models(all_rows_numeric, primary_metric)
        best_overall = ranked_overall.iloc[0]
        
        # ===== ALL BACKBONES CSV 1: NUMERIC =====
        all_numeric_csv = results_root / "all_backbones_metrics_numeric.csv"
        all_rows_numeric.to_csv(all_numeric_csv, index=False, float_format='%.3f')
        
        # ===== ALL BACKBONES CSV 2: FORMATTED =====
        formatted_all = all_rows_numeric.copy()
        
        # Identify metrics with CI
        metrics_with_ci = []
        for col in formatted_all.columns:
            if col not in ['model', 'backbone'] and f"{col}_CI_lower" in formatted_all.columns:
                metrics_with_ci.append(col)
        
        # Format metrics with CI strings
        for metric in metrics_with_ci:
            mean_val = formatted_all[metric]
            ci_lower = formatted_all[f"{metric}_CI_lower"]
            ci_upper = formatted_all[f"{metric}_CI_upper"]
            formatted_all[metric] = mean_val.apply(lambda x: f"{x:.3f}") + \
                                   " (95% CI, " + \
                                   ci_lower.apply(lambda x: f"{x:.3f}") + "-" + \
                                   ci_upper.apply(lambda x: f"{x:.3f}") + ")"
        
        # Drop CI bounds
        columns_to_drop = [col for col in formatted_all.columns if "_CI_lower" in col or "_CI_upper" in col]
        formatted_all = formatted_all.drop(columns=[c for c in columns_to_drop if c in formatted_all.columns])
        
        all_formatted_csv = results_root / "all_backbones_metrics_formatted.csv"
        formatted_all.to_csv(all_formatted_csv, index=False)

        b_tag = str(best_overall["backbone"])
        m_name = str(best_overall["model"])
        src = results_root / b_tag / artifacts.get("qsar_best_model", "best_model.joblib")
        dst = results_root / artifacts.get("qsar_best_model_overall", "best_overall_model.joblib")
        try:
            shutil.copyfile(src, dst)
            print(f"🏆 Best OVERALL model → {m_name} [{b_tag}]")
        except Exception as e:
            log.warning(f"Failed to copy best overall model from {src}: {e}")

        try:
            best_row_report = {
                "backbone": b_tag,
                "engine": m_name,
                "primary_metric_name": primary_metric,
                "primary_metric_value": float(best_overall.get(
                    "Cliff_RMSE" if primary_metric.lower() in {"rmse_cliff", "cliff_rmse"} else primary_metric,
                    np.nan)),
                "AUROC": float(best_overall.get("AUROC", np.nan)),
                "AUPRC_active": float(best_overall.get("AUPRC_active", np.nan)),
                "Brier": float(best_overall.get("Brier", np.nan)),
                "ECE": float(best_overall.get("ECE", np.nan)),
                "Cliff_RMSE": float(best_overall.get("Cliff_RMSE", np.nan)),
                "threshold": float(best_overall.get("Threshold", 0.5)),
            }
            out_file = results_root / "best_overall.itxt"
            _write_best_overall_itxt(best_row_report, out_file)
        except Exception as e:
            log.warning(f"[EXTRAS] Could not write best_overall.itxt: {e}")
    else:
        log.warning("[Summary] No per-backbone metrics. Skipping overall CSV and selection.")

    print("\nQSAR evaluation pipeline complete.")


# =============================================================================
# CLI
# =============================================================================

def _build_logger():
    """Build logger for QSAR evaluation."""
    log = logging.getLogger("qsar_eval")
    if not log.handlers:
        log.setLevel(logging.INFO)
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        log.addHandler(h)
    return log


def main(config_path: str = "config.yaml"):
    """Main entry point for QSAR evaluation pipeline."""
    log = _build_logger()
    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh)
    if not cfg.get("QSAR_Eval", {}).get("enable", True):
        print("QSAR_Eval.enable is False — skipping evaluation.")
        return
    run_qsar_evaluation(cfg, log)


if __name__ == "__main__":
    cp = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    main(cp)
