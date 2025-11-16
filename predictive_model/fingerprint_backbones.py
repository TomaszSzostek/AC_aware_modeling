"""
Fingerprint Backbones Module for QSAR Evaluation

This module provides fingerprint-based feature engineering for molecular QSAR models.
It implements Extended Connectivity Fingerprints (ECFP) and MACCS structural keys
with optimized caching and efficient computation.

Supported Fingerprints:
- ECFP1024: Extended Connectivity Fingerprints with 1024 bits
- ECFP2048: Extended Connectivity Fingerprints with 2048 bits
- MACCS: MACCS structural keys (166 bits)

"""

from __future__ import annotations

import pathlib
from typing import List, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as rdFPGen


# Morgan (ECFP) generator cache (new RDKit API, no deprecation warnings)
_MORGAN_GEN_CACHE = {}


def get_morgan_generator(radius: int = 2, n_bits: int = 2048,
                         include_chirality: bool = False,
                         include_bond_types: bool = True,
                         include_features: bool = False):
    """
    Get or create a cached Morgan fingerprint generator.
    
    Args:
        radius: Fingerprint radius (default: 2)
        n_bits: Number of bits in fingerprint (default: 2048)
        include_chirality: Include chirality information
        include_bond_types: Include bond type information
        include_features: Use feature-based fingerprints
        
    Returns:
        Cached Morgan fingerprint generator
    """
    key = (radius, n_bits, include_chirality, include_bond_types, include_features)
    gen = _MORGAN_GEN_CACHE.get(key)
    if gen is None:
        # RDKit has multiple signatures across versions. Try modern kw set first,
        # then older variants, finally a positional fallback.
        try:
            # Newer: useBondTypes keyword; includeFeatures often not present
            gen = rdFPGen.GetMorganGenerator(
                radius=int(radius),
                fpSize=int(n_bits),
                includeChirality=bool(include_chirality),
                useBondTypes=bool(include_bond_types),
            )
        except TypeError:
            try:
                # Older signature requires countSimulation before others
                gen = rdFPGen.GetMorganGenerator(
                    radius=int(radius),
                    countSimulation=False,
                    includeChirality=bool(include_chirality),
                    useBondTypes=bool(include_bond_types),
                    fpSize=int(n_bits),
                )
            except TypeError:
                # Positional ultra-fallback (radius, countSimulation, includeChirality,
                # useBondTypes, onlyNonzeroInvariants, includeRingMembership, countBounds, fpSize)
                gen = rdFPGen.GetMorganGenerator(int(radius), False, bool(include_chirality), bool(include_bond_types),
                                                 False, True, None, int(n_bits))
        _MORGAN_GEN_CACHE[key] = gen
    return gen


def morgan_bitvect(mol, radius: int = 2, n_bits: int = 2048):
    """Generate Morgan fingerprint bit vector for a molecule."""
    gen = get_morgan_generator(radius=radius, n_bits=n_bits)
    return gen.GetFingerprint(mol)


def smiles_to_mol(s: str):
    """Convert SMILES string to RDKit molecule object."""
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


def ensure_dir(p: pathlib.Path):
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def _ecfp_matrix(smiles_list: List[str], n_bits: int = 2048, radius: int = 2) -> Tuple[np.ndarray, List[str]]:
    """
    Return a binary ECFP matrix (0/1) of shape [N, n_bits] and a list of feature names.
    
    Args:
        smiles_list: List of SMILES strings
        n_bits: Number of bits in fingerprint (default: 2048)
        radius: Fingerprint radius (default: 2)
        
    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    N = len(smiles_list)
    X = np.zeros((N, n_bits), dtype=np.float32)
    gen = get_morgan_generator(radius=radius, n_bits=n_bits)
    for i, s in enumerate(smiles_list):
        m = smiles_to_mol(s)
        if m is None:
            continue
        fp = gen.GetFingerprint(m)
        arr = np.zeros((n_bits,), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i, :] = arr.astype(np.float32)
    names = [f"FP::ECFP{n_bits}_{i:04d}" for i in range(n_bits)]
    return X, names


def _maccs_matrix(smiles_list: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Return a binary MACCS matrix (166 bits; bit #0 is dropped) and a list of feature names.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    N = len(smiles_list)
    X = np.zeros((N, 166), dtype=np.float32)
    for i, s in enumerate(smiles_list):
        m = smiles_to_mol(s)
        if m is None:
            continue
        fp = MACCSkeys.GenMACCSKeys(m)  # 167 bits (index 0..166)
        arr = np.zeros((fp.GetNumBits(),), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, arr)
        if arr.size >= 167:
            X[i, :] = arr[1:167].astype(np.float32)  # drop bit #0
        else:
            take = min(166, arr.size)
            X[i, :take] = arr[:take].astype(np.float32)
    names = [f"FP::MACCS_{i + 1:04d}" for i in range(166)]
    return X, names


def _build_X_fp_trainfit(
        smiles_series: pd.Series,
        backbone: str,
        fit_idx: np.ndarray,
        results_root: pathlib.Path,
        log
) -> Tuple[np.ndarray, List[str]]:
    """
    Build feature matrix for a given fingerprint backbone.
    
    This function computes fingerprint features for all molecules and saves
    the feature names to a file for reproducibility.
    
    Args:
        smiles_series: Series of SMILES strings
        backbone: Fingerprint backbone name ("ecfp1024", "ecfp2048", "maccs")
        fit_idx: Indices for training (currently unused but kept for API consistency)
        results_root: Directory to save feature names
        log: Logger instance for progress tracking
        
    Returns:
        Tuple of (feature_matrix, feature_names)
        
    Raises:
        ValueError: If backbone name is unknown
    """
    smiles_list = smiles_series.astype(str).tolist()
    b = backbone.strip().lower()

    if b == "ecfp1024":
        X, names = _ecfp_matrix(smiles_list, n_bits=1024, radius=2)
        X = X.astype(np.float32)  # 0/1; no scaling
        ensure_dir(results_root)
        with open(results_root / "features_used.txt", "w") as f:
            f.write("\n".join(names))
        log.info(f"[Features] Using backbone=ecfp1024 with {len(names)} binary features (no scaling).")
        return X, names

    if b == "ecfp2048":
        X, names = _ecfp_matrix(smiles_list, n_bits=2048, radius=2)
        X = X.astype(np.float32)
        ensure_dir(results_root)
        with open(results_root / "features_used.txt", "w") as f:
            f.write("\n".join(names))
        log.info(f"[Features] Using backbone=ecfp2048 with {len(names)} binary features (no scaling).")
        return X, names

    if b == "maccs":
        X, names = _maccs_matrix(smiles_list)
        X = X.astype(np.float32)
        ensure_dir(results_root)
        with open(results_root / "features_used.txt", "w") as f:
            f.write("\n".join(names))
        log.info(f"[Features] Using backbone=maccs with {len(names)} binary features (no scaling).")
        return X, names

    raise ValueError(f"Unknown fingerprint backbone: {backbone}")

