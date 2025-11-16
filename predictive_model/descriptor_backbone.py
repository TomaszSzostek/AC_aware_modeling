"""
Descriptor Backbone Module for QSAR Evaluation

This module provides descriptor-based feature engineering for molecular QSAR models.
It implements RDKit 2D descriptors and Mordred descriptors with advanced feature
selection using the Boruta algorithm optimized for small cheminformatics datasets.

Features:
- RDKit 2D descriptors (~200 descriptors)
- Mordred 2D descriptors (optional, ~1800 descriptors)
- Boruta feature selection optimized for small datasets (≤350 compounds)
- Mutual information fallback when Boruta fails
- Correlation filtering to remove redundant features
- Adaptive feature selection based on dataset size

"""

from __future__ import annotations

import logging
import pathlib
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Cheminformatics gold standard parameters
DEFAULT_TOP_K = 128
DEFAULT_CORRELATION_THRESHOLD = 0.95
DEFAULT_RANDOM_STATE = 42

# Mordred import (optional)
try:
    from mordred import Calculator, descriptors as md
    _MORDRED_OK = True
except Exception:
    _MORDRED_OK = False

# Boruta import (optional)
try:
    from boruta import BorutaPy
    _BORUTA_OK = True
except Exception:
    _BORUTA_OK = False


def ensure_dir(p: pathlib.Path):
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def smiles_to_mol(s: str):
    """Convert SMILES string to RDKit molecule object."""
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


def _compute_rdkit2d(smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute classical RDKit 2D descriptors (numeric only).
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Tuple of (descriptor_matrix, descriptor_names)
    """
    funcs = [(n, f) for (n, f) in Descriptors.descList]  # ~200 descriptors
    names = [n for n, _ in funcs]
    X = np.full((len(smiles_list), len(funcs)), np.nan, dtype=np.float64)
    for i, s in enumerate(smiles_list):
        m = smiles_to_mol(s)
        if m is None:
            continue
        row = []
        for _, f in funcs:
            try:
                v = f(m)
                row.append(float(v))
            except Exception:
                row.append(np.nan)
        X[i, :] = np.array(row, dtype=np.float64)
    return X, names


def _compute_mordred2d(smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Compute Mordred 2D descriptors (ignore_3D=True); keep numeric, non-constant columns only.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Tuple of (descriptor_matrix, descriptor_names)
    """
    if not _MORDRED_OK:
        return np.empty((len(smiles_list), 0), dtype=np.float64), []
    calc = Calculator(md, ignore_3D=True)
    mols = [smiles_to_mol(s) for s in smiles_list]
    df = calc.pandas(mols).apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    nunique = df.nunique(dropna=True)
    df = df.loc[:, nunique > 1]
    return df.values.astype(np.float64), df.columns.astype(str).tolist()


def _stack_descriptors(smiles_list: List[str], include_mordred: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Concatenate RDKit2D (+ optional Mordred2D).
    
    Args:
        smiles_list: List of SMILES strings
        include_mordred: Whether to include Mordred descriptors (default: True)
        
    Returns:
        Tuple of (combined_descriptor_matrix, combined_descriptor_names)
    """
    X_rd, n_rd = _compute_rdkit2d(smiles_list)
    if include_mordred:
        X_md, n_md = _compute_mordred2d(smiles_list)
    else:
        X_md, n_md = np.empty((len(smiles_list), 0), dtype=np.float64), []
    if X_md.shape[1] == 0:
        return X_rd, n_rd
    return np.hstack([X_rd, X_md]), (n_rd + n_md)


def _boruta_select(
        X_tr: np.ndarray,
        y_tr: np.ndarray,
        feat_names: List[str],
        log: logging.Logger,
        random_state: int = DEFAULT_RANDOM_STATE,
        max_iter: int = 200,
        top_k: int = DEFAULT_TOP_K
) -> List[str]:
    """
    Cheminformatics Gold Standard Boruta feature selection for small datasets.
    
    This function implements an optimized Boruta algorithm specifically designed for
    cheminformatics datasets with ~350 compounds. It includes adaptive parameters,
    correlation filtering, and robust fallback mechanisms.
    
    Args:
        X_tr: Training features (n_samples, n_features)
        y_tr: Training labels (n_samples,)
        feat_names: List of feature names
        log: Logger instance for progress tracking
        random_state: Random state for reproducibility
        max_iter: Maximum Boruta iterations
        top_k: Maximum number of features to select
        
    Returns:
        List of selected feature names
        
    Note:
        Falls back to mutual information selection if Boruta fails
    """
    # Median imputation only for fitting RF/Boruta/MI (no leakage to TEST)
    imp = SimpleImputer(strategy="median")
    X_tr_imp = imp.fit_transform(X_tr)
    
    n_samples, n_features = X_tr_imp.shape
    log.info(f"[Boruta] Starting feature selection: {n_samples} samples, {n_features} features")
    
    # Adaptive top_k based on dataset size (cheminformatics gold standard)
    if n_samples < 100:
        top_k = min(64, n_features // 4)  # Very conservative for small datasets
    elif n_samples < 200:
        top_k = min(96, n_features // 3)  # Conservative for medium datasets
    elif n_samples < 350:
        top_k = min(DEFAULT_TOP_K, n_features // 2)  # Standard for 350 compounds
    else:
        top_k = min(256, n_features // 2)  # More features for larger datasets
    
    log.info(f"[Boruta] Using top_k={top_k} for {n_samples} samples")

    if _BORUTA_OK:
        # Conservative RF for small datasets to prevent overfitting
        rf = RandomForestClassifier(
            n_estimators=min(500, n_samples * 2),  # Adaptive n_estimators
            max_depth=min(6, int(np.log2(n_samples))),  # Conservative depth
            min_samples_leaf=max(2, n_samples // 50),  # Prevent overfitting
            min_samples_split=max(5, n_samples // 25),
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
        
        # Boruta with conservative settings
        bor = BorutaPy(
            rf,
            n_estimators="auto",
            max_iter=max_iter,
            verbose=1,  # Show progress for debugging
            random_state=random_state,
        )
        
        try:
            bor.fit(X_tr_imp, y_tr)
            mask = getattr(bor, "support_", None)
            
            if mask is None or not np.any(mask):
                log.warning("[Boruta] No features selected by Boruta, falling back to MI")
                return _fallback_mi_selection(X_tr_imp, y_tr, feat_names, top_k, random_state, log)
            
            selected = [n for n, keep in zip(feat_names, mask) if keep]
            log.info(f"[Boruta] Selected {len(selected)} features")
            
            # If too many features, use ranking to select top_k
            if len(selected) > top_k:
                try:
                    rnk = getattr(bor, "ranking_", None)
                    if rnk is not None:
                        order = np.argsort(rnk)  # 1 is best
                        idxs = [i for i in order if mask[i]][:top_k]
                        selected = [feat_names[i] for i in idxs]
                        log.info(f"[Boruta] Limited to top {top_k} features using ranking")
                    else:
                        selected = selected[:top_k]
                        log.info(f"[Boruta] Limited to top {top_k} features (no ranking available)")
                except Exception as e:
                    log.warning(f"[Boruta] Error using ranking: {e}, using first {top_k}")
                    selected = selected[:top_k]
            
            # Stability check: remove highly correlated features
            selected = _remove_correlated_features(X_tr_imp, feat_names, selected, log)
            log.info(f"[Boruta] After correlation filtering: {len(selected)} features")
            
            return selected
            
        except Exception as e:
            log.warning(f"[Boruta] Boruta failed: {e}, falling back to MI")
            try:
                return _fallback_mi_selection(X_tr_imp, y_tr, feat_names, top_k, random_state, log)
            except Exception as mi_error:
                log.error(f"[Boruta] MI fallback also failed: {mi_error}")
                # Last resort: return top features by variance
                variances = np.var(X_tr_imp, axis=0)
                top_indices = np.argsort(variances)[-top_k:]
                return [feat_names[i] for i in top_indices if i < len(feat_names)]
    else:
        log.info("[Boruta] Boruta not available, using MI selection")
        return _fallback_mi_selection(X_tr_imp, y_tr, feat_names, top_k, random_state, log)


def _fallback_mi_selection(X_tr_imp, y_tr, feat_names, top_k, random_state=DEFAULT_RANDOM_STATE, log=None):
    """
    Fallback mutual information feature selection with correlation filtering.
    
    This function serves as a robust fallback when Boruta fails, using mutual
    information to select the most informative features while removing highly
    correlated ones.
    
    Args:
        X_tr_imp: Imputed training features
        y_tr: Training labels
        feat_names: List of feature names
        top_k: Maximum number of features to select
        random_state: Random state for reproducibility
        log: Logger instance
        
    Returns:
        List of selected feature names
    """
    try:
        mi = mutual_info_classif(X_tr_imp, y_tr, random_state=random_state, discrete_features=False)
        order = np.argsort(mi)[::-1]
        
        # Select features with positive MI
        idxs = [i for i in order if np.isfinite(mi[i]) and mi[i] > 0]
        
        if not idxs:
            if log:
                log.warning("[MI] No features with positive MI, using top features by variance")
            # Fallback to variance-based selection
            var_scores = np.var(X_tr_imp, axis=0)
            order = np.argsort(var_scores)[::-1]
            idxs = order[:top_k]
        else:
            idxs = idxs[:top_k]
    except Exception as e:
        if log:
            log.error(f"[MI] Mutual information calculation failed: {e}")
        # Last resort: variance-based selection
        var_scores = np.var(X_tr_imp, axis=0)
        order = np.argsort(var_scores)[::-1]
        idxs = order[:top_k]
    
    selected = [feat_names[i] for i in idxs]
    if log:
        log.info(f"[MI] Selected {len(selected)} features using mutual information")
    
    # Remove highly correlated features
    selected = _remove_correlated_features(X_tr_imp, feat_names, selected, log)
    if log:
        log.info(f"[MI] After correlation filtering: {len(selected)} features")
    
    return selected


def _remove_correlated_features(X_tr_imp, feat_names, selected, log, threshold=DEFAULT_CORRELATION_THRESHOLD):
    """
    Remove highly correlated features to reduce redundancy.
    
    This function identifies and removes features that are highly correlated
    with other selected features, helping to reduce multicollinearity and
    improve model stability.
    
    Args:
        X_tr_imp: Imputed training features
        feat_names: List of all feature names
        selected: List of currently selected feature names
        log: Logger instance
        threshold: Correlation threshold for removal (default: 0.95)
        
    Returns:
        List of selected features after correlation filtering
    """
    if len(selected) <= 1:
        return selected
    
    # Get indices of selected features
    selected_idx = [feat_names.index(name) for name in selected if name in feat_names]
    
    if len(selected_idx) <= 1:
        return selected
    
    # Calculate correlation matrix for selected features
    X_selected = X_tr_imp[:, selected_idx]
    corr_matrix = np.corrcoef(X_selected.T)
    
    # Find highly correlated pairs
    to_remove = set()
    for i in range(len(selected_idx)):
        for j in range(i + 1, len(selected_idx)):
            if abs(corr_matrix[i, j]) > threshold:
                # Remove the feature with lower index (keep the first one)
                to_remove.add(j)
    
    # Return features that are not highly correlated
    filtered_selected = [selected[i] for i in range(len(selected)) if i not in to_remove]
    
    if len(filtered_selected) < len(selected) and log:
        log.info(f"[Correlation] Removed {len(selected) - len(filtered_selected)} highly correlated features")
    
    return filtered_selected


def _build_X_descriptors_trainfit(
        smiles_series: pd.Series,
        y: np.ndarray,
        fit_idx: np.ndarray,
        results_root: pathlib.Path,
        log: logging.Logger,
        random_state: int = 42,
        include_mordred: bool = True,
        boruta_max_iter: int = 200,
        top_k: int = 256,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build descriptor-based features with feature selection.
    
    Pipeline: RDKit2D (+ optional Mordred2D) → cleaning → feature selection (Boruta/MI) on TRAIN →
    TRAIN-only imputer + scaler → stable names with 'DESC::' prefix.
    
    Args:
        smiles_series: Series of SMILES strings
        y: Target labels for all samples
        fit_idx: Indices for training (used for feature selection)
        results_root: Directory to save feature artifacts
        log: Logger instance for progress tracking
        random_state: Random state for reproducibility
        include_mordred: Whether to include Mordred descriptors (default: True)
        boruta_max_iter: Maximum Boruta iterations (default: 200)
        top_k: Maximum number of features to select (default: 256)
        
    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    smiles_list = smiles_series.astype(str).tolist()
    X_all, names_all = _stack_descriptors(smiles_list, include_mordred=include_mordred)

    # DataFrame view + drop all-NaN / constant / duplicated columns
    df = pd.DataFrame(X_all, columns=names_all)
    df = df.dropna(axis=1, how="all")
    nunique = df.nunique(dropna=True)
    df = df.loc[:, nunique > 1]
    df = df.loc[:, ~df.columns.duplicated()]  # keep only first occurrence of duplicate names

    X_all = df.values.astype(np.float64)
    # --- Fix: sanitize to prevent overflow warnings ---
    X_all = np.nan_to_num(X_all, nan=np.nan, posinf=np.nan, neginf=np.nan)
    # Clip extreme descriptor values to safe range
    X_all = np.clip(X_all, -1e6, 1e6)
    # Replace any remaining inf or NaN with column medians
    col_medians = np.nanmedian(X_all, axis=0)
    inds = np.where(~np.isfinite(X_all))
    if inds[0].size > 0:
        X_all[inds] = np.take(col_medians, inds[1])

    names_all = df.columns.astype(str).tolist()

    # Selection on TRAIN only
    X_tr = X_all[fit_idx]
    y_tr = y[fit_idx].astype(int)
    selected_names = _boruta_select(
        X_tr=X_tr,
        y_tr=y_tr,
        feat_names=names_all,
        log=log,
        random_state=random_state,
        max_iter=boruta_max_iter,
        top_k=top_k,
    )

    # Deduplicate selected list while preserving order
    _seen = set()
    selected_names = [n for n in selected_names if not (n in _seen or _seen.add(n))]

    ensure_dir(results_root)

    # Build post-selection matrix in the original row order
    df_sel = df.loc[:, selected_names].copy()

    # Stable labels — add 'DESC::' prefix
    selected_names_prefixed = [f"DESC::{c}" for c in df_sel.columns]
    df_sel.columns = selected_names_prefixed

    # TRAIN-only: median imputation + z-score scaling
    imp = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_tr_imp = imp.fit_transform(df_sel.values[fit_idx])
    scaler.fit(X_tr_imp)

    X_all_imp = imp.transform(df_sel.values)
    X_scaled = scaler.transform(X_all_imp).astype(np.float32)

    # Persist artifacts for reproducibility
    try:
        joblib.dump(imp, results_root / "desc_imputer.joblib")
        np.savez(
            results_root / "desc_scaler.npz",
            mean_=scaler.mean_.astype(np.float64),
            scale_=scaler.scale_.astype(np.float64),
        )
    except Exception:
        pass

    with open(results_root / "selected_descriptors.txt", "w") as f:
        f.write("\n".join(selected_names_prefixed))
    with open(results_root / "features_used.txt", "w") as f:
        f.write("\n".join(selected_names_prefixed))
    log.info(f"[FS] Selected {len(selected_names_prefixed)} descriptors (Boruta/MI).")

    return X_scaled, selected_names_prefixed

