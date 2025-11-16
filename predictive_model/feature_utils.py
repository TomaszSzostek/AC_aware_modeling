"""
Feature Utilities Module for QSAR Evaluation

This module provides feature name alignment and data handling utilities including:
- Feature name alignment and padding
- DataFrame wrapping with proper column names
- Safe row slicing for arrays and DataFrames

"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
import pandas as pd


def _as_named_df(X, names, log=None, ctx=""):
    """
    Wrap X in a DataFrame with column names; regenerate safe defaults on mismatch.
    
    Args:
        X: Feature matrix (numpy array or DataFrame)
        names: List of feature names
        log: Logger instance (optional)
        ctx: Context string for logging
        
    Returns:
        DataFrame with proper column names
    """
    X = np.asarray(X)
    n_cols = X.shape[1]
    try:
        cols = list(names) if names is not None else None
    except Exception:
        cols = None
    if (cols is None) or (len(cols) != n_cols):
        cols = [f"f{i}" for i in range(n_cols)]
        if log is not None:
            try:
                log.debug(f"[names][{ctx}] regenerated feature names (len mismatch or missing)")
            except Exception:
                pass
    try:
        import pandas as _pd
        return _pd.DataFrame(X, columns=cols)
    except Exception:
        return X


def _align_feature_names(names, n_cols: int, prefix: str):
    """
    Ensure len(names)==n_cols; pad with sensible defaults and deduplicate.
    
    Args:
        names: List of feature names
        n_cols: Required number of columns
        prefix: Prefix for generated names
        
    Returns:
        List of aligned feature names (length = n_cols)
    """
    n_cols = int(n_cols)
    if names is None:
        names = []
    names = list(map(str, names))
    if len(names) < n_cols:
        # Preserve 'DESC::' prefix if present
        base = "DESC::f" if prefix.lower() in {"descriptors", "desc", "rdkit", "rdkit_descriptors"} else f"{prefix}_"
        pad = [f"{base}{i}" for i in range(len(names), n_cols)]
        names = names + pad
    elif len(names) > n_cols:
        names = names[:n_cols]
    # dedup
    out, seen = [], {}
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}__dup{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out


def _safe_row_slice(X, idx):
    """
    Return rows of X for indices idx, whether X is a numpy array or pandas DataFrame.
    
    Args:
        X: Feature matrix (numpy array or DataFrame)
        idx: Row indices
        
    Returns:
        Sliced feature matrix
    """
    try:
        if isinstance(X, pd.DataFrame):
            return X.iloc[idx]
    except Exception:
        pass
    return X[idx]

