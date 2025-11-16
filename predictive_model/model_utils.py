"""
Utility functions for QSAR model training and validation.

Includes helpers for model fitting, threshold tuning, permutation importance,
cross-validation, y-scrambling and basic dataset validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_score,
)

from predictive_model.metrics_plots import _positive_proba

DEFAULT_RANDOM_STATE = 42


def _fit_model(model, X, y):
    """
    Simple model fitting without calibration.
    
    Args:
        model: Model instance to fit
        X: Feature matrix
        y: Target labels
        
    Returns:
        Fitted model
    """
    X_np = np.asarray(X)
    y_np = np.asarray(y).astype(int)
    return model.fit(X_np, y_np)


def tune_threshold(y_true_val, p_val, metric: str) -> float:
    """
    Tune decision threshold to maximize a chosen metric on a validation split.
    
    Args:
        y_true_val: True validation labels
        p_val: Predicted probabilities
        metric: Metric to optimize ("MCC", "F1", "YOUDEN", "ACC")
        
    Returns:
        Optimal threshold value
    """
    metric = (metric or "MCC").upper()
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
        elif metric == "YOUDEN":
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


def safe_permutation_importance(
        model,
        X_te,
        y_te,
        feature_names,
        n_repeats: int = 10,
        random_state: int = 42,
        scoring: Optional[Union[str, Callable]] = "roc_auc",
        top_n: Optional[int] = 30,
        log: Optional[logging.Logger] = None,
):
    """
    Permutation importance with robust defaults and global imports.
    
    Args:
        model: Fitted model
        X_te: Test features
        y_te: Test labels
        feature_names: Feature names
        n_repeats: Number of permutation repeats
        random_state: Random seed
        scoring: Scoring function or string
        top_n: Number of top features to return
        log: Logger instance
        
    Returns:
        DataFrame with permutation importance scores
    """
    X_sub = np.asarray(X_te)
    y_te = np.asarray(y_te).astype(int)

    # Align / generate feature names and ALWAYS wrap into DataFrame
    n_feat = X_sub.shape[1]
    names = []
    try:
        if feature_names is not None:
            names = [str(x) for x in feature_names]
    except Exception:
        names = []
    if len(names) != n_feat:
        if log:
            log.info(f"[PI] Align feature names: {len(names)} -> {n_feat}")
        if len(names) < n_feat:
            names = list(names) + [f"f{i}" for i in range(len(names), n_feat)]
        else:
            names = names[:n_feat]
    X_in = pd.DataFrame(X_sub, columns=names)

    # Decide on scorer
    def _robust_scorer(estimator, X, y):
        p = _positive_proba(estimator, X)
        if np.unique(y).size < 2:
            return -float(brier_score_loss(y, p))
        try:
            return float(roc_auc_score(y, p))
        except Exception:
            return -float(brier_score_loss(y, p))

    use_scoring = _robust_scorer

    r = permutation_importance(
        model, X_in, y_te,
        n_repeats=int(n_repeats),
        random_state=int(random_state),
        scoring=use_scoring,
        n_jobs=-1,
    )

    imp = pd.DataFrame(
        {"feature": names, "importance_mean": r.importances_mean, "importance_std": r.importances_std}
    ).sort_values("importance_mean", ascending=False)

    if top_n is not None:
        imp = imp.head(int(top_n)).reset_index(drop=True)

    imp["feature"] = [n[6:] if isinstance(n, str) and n.startswith("DESC::") else n for n in imp["feature"]]
    return imp


def y_scramble_auc(model_factory, X, y, n=30, seed=42):
    """
    Y-scrambling sanity check: fit on permuted labels and compute AUROC.
    
    Returns mean and std of AUROC across `n` permutations.
    
    Args:
        model_factory: Function that returns a new model instance
        X: Feature matrix
        y: True labels
        n: Number of permutations
        seed: Random seed
        
    Returns:
        Tuple of (mean_auc, std_auc)
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    scores = []
    for _ in range(int(n)):
        y_perm = rng.permutation(y)
        try:
            mdl = model_factory()
            mdl.fit(X, y_perm)
            if hasattr(mdl, 'predict_proba'):
                pr = np.asarray(mdl.predict_proba(X))
                pr = pr[:, 1] if pr.ndim == 2 and pr.shape[1] == 2 else pr.ravel()
            else:
                s = np.asarray(mdl.decision_function(X)).ravel()
                # min-max to [0,1] preserves ranking for AUROC
                pr = (s - s.min()) / (s.max() - s.min() + 1e-9)
            scores.append(roc_auc_score(y_perm, pr))
        except Exception:
            # ignore failed fits in degenerate cases
            continue
    if not scores:
        return float("nan"), float("nan")
    return float(np.mean(scores)), float(np.std(scores))


def repeated_cv_auc(model_factory, X, y, splits=5, repeats=10, seed=42):
    """
    Repeated stratified CV AUROC for a given model factory.
    
    Returns the vector of AUROC scores (length = splits * repeats).
    
    Args:
        model_factory: Function that returns a new model instance
        X: Feature matrix
        y: Target labels
        splits: Number of CV folds
        repeats: Number of CV repeats
        seed: Random seed
        
    Returns:
        Array of AUROC scores
    """
    X = np.asarray(X)
    y = np.asarray(y).astype(int)
    cv = RepeatedStratifiedKFold(n_splits=int(splits), n_repeats=int(repeats), random_state=int(seed))
    return cross_val_score(model_factory(), X, y, cv=cv, scoring="roc_auc", n_jobs=-1)


def _validate_dataset(df: pd.DataFrame, smiles_col: str, log: logging.Logger) -> bool:
    """
    Validate dataset for QSAR evaluation.
    
    Args:
        df: DataFrame to validate
        smiles_col: Name of SMILES column
        log: Logger instance
        
    Returns:
        True if validation passes, False otherwise
    """
    
    # Check required columns
    required_cols = [smiles_col, "pIC50"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        log.error(f"[Validation] Missing required columns: {missing_cols}")
        return False
    
    # Check for empty dataset
    if len(df) == 0:
        log.error("[Validation] Dataset is empty")
        return False
    
    # Check for valid SMILES
    valid_smiles = df[smiles_col].notna() & (df[smiles_col] != "")
    if not valid_smiles.all():
        invalid_count = (~valid_smiles).sum()
        log.warning(f"[Validation] {invalid_count} invalid SMILES found")
    
    # Check for valid pIC50 values
    valid_pic50 = df["pIC50"].notna() & np.isfinite(df["pIC50"])
    if not valid_pic50.all():
        invalid_count = (~valid_pic50).sum()
        log.warning(f"[Validation] {invalid_count} invalid pIC50 values found")
    
    # Check class balance
    y_active = (df["pIC50"] >= 9.0 - np.log10(10000)).astype(int)
    active_ratio = float(np.mean(y_active))
    log.info(f"[Validation] Class balance: {active_ratio:.3f} active, {1-active_ratio:.3f} inactive")
    
    if active_ratio < 0.05 or active_ratio > 0.95:
        log.warning(f"[Validation] Extreme class imbalance detected: {active_ratio:.3f} active")
    
    if len(np.unique(y_active)) < 2:
        log.error("[Validation] Only one class present in dataset")
        return False
    
    log.info("[Validation] Dataset validation passed")
    return True

