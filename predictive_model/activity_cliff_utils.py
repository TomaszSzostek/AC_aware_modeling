"""
Activity Cliff Utilities Module for QSAR Evaluation

This module provides activity cliff detection and evaluation utilities including:
- Fingerprint computation for similarity graphs
- Activity cliff pair detection
- Node-wise RMSE computation on activity cliffs

"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from rdkit import DataStructs

from predictive_model.fingerprint_backbones import get_morgan_generator
from predictive_model.fingerprint_backbones import smiles_to_mol


def compute_fps_for_graph(smiles_list: List[str], radius: int = 2, n_bits: int = 2048):
    """
    Compute Morgan/ECFP bitvectors for similarity graph construction.
    
    Returns None for invalid molecules.
    
    Args:
        smiles_list: List of SMILES strings
        radius: Fingerprint radius (default: 2)
        n_bits: Number of bits (default: 2048)
        
    Returns:
        List of fingerprint bitvectors (or None for invalid molecules)
    """
    fps = []
    gen = get_morgan_generator(radius=radius, n_bits=n_bits)
    for s in smiles_list:
        m = smiles_to_mol(s)
        if m is None:
            fps.append(None)
        else:
            try:
                fps.append(gen.GetFingerprint(m))  # ExplicitBitVect compatible with BulkTanimotoSimilarity
            except Exception:
                fps.append(None)
    return fps


def _build_subset_neighbors(fps_all, subset_idx: np.ndarray, sim_thr: float) -> dict:
    """
    Find neighbors (local indices 0..len(subset)-1) with Tanimoto ≥ sim_thr.
    
    Args:
        fps_all: All fingerprints
        subset_idx: Indices of subset
        sim_thr: Similarity threshold
        
    Returns:
        Dictionary mapping local index to list of (neighbor_index, similarity) tuples
    """
    subset_idx = np.asarray(subset_idx, dtype=int)
    fps_sub = [fps_all[g] for g in subset_idx]
    nbrs = {i: [] for i in range(len(subset_idx))}
    for i_local, fp_i in enumerate(fps_sub):
        if fp_i is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp_i, fps_sub)
        for j_local, s in enumerate(sims):
            if j_local == i_local or fps_sub[j_local] is None:
                continue
            if float(s) >= sim_thr:
                nbrs[i_local].append((j_local, float(s)))
    return nbrs


def build_cliff_pairs_for_subset(
        fps_all,
        subset_idx: np.ndarray,
        pIC50_full: np.ndarray,
        sim_thr: float,
        dp_thr: float
) -> List[Tuple[int, int]]:
    """
    Find LOCAL pairs (i,j) inside subset that satisfy MoleculeACE cliff definition.
    
    A cliff pair requires:
    - Tanimoto similarity ≥ sim_thr
    - Absolute pIC50 difference ≥ dp_thr
    
    Args:
        fps_all: All fingerprints
        subset_idx: Indices of the subset
        pIC50_full: Full array of pIC50 values
        sim_thr: Similarity threshold
        dp_thr: Potency difference threshold
        
    Returns:
        List of (local_i, local_j) tuples representing cliff pairs
    """
    subset_idx = np.asarray(subset_idx, dtype=int)
    nbrs_local = _build_subset_neighbors(fps_all, subset_idx, sim_thr=sim_thr)
    pairs = []
    for i, neigh in nbrs_local.items():
        gi = int(subset_idx[i])
        pi = float(pIC50_full[gi])
        if not np.isfinite(pi):
            continue
        for j, _ in neigh:
            if j <= i:  # unique i<j
                continue
            gj = int(subset_idx[j])
            pj = float(pIC50_full[gj])
            if not np.isfinite(pj):
                continue
            if abs(pi - pj) >= float(dp_thr):
                pairs.append((int(i), int(j)))
    return pairs


def rmse_cliff_nodewise(
        proba_subset: np.ndarray,
        y_subset: np.ndarray,
        cliff_pairs_local: List[Tuple[int, int]]
) -> Tuple[float, int]:
    """
    Compute RMSE_cliff (MoleculeACE-adapted for classification).
    
    RMSE calculated on nodes belonging to cliff pairs:
        sqrt( mean_{k in cliff_nodes} (p_k - y_k)^2 )
    
    Args:
        proba_subset: Predicted probabilities for subset
        y_subset: True labels for subset
        cliff_pairs_local: List of (i, j) cliff pairs (local indices)
        
    Returns:
        Tuple of (rmse, number_of_unique_nodes)
    """
    if not cliff_pairs_local:
        return float("nan"), 0
    nodes = sorted({i for pair in cliff_pairs_local for i in pair})
    p = np.asarray(proba_subset, float).ravel()[nodes]
    y = np.asarray(y_subset, int).ravel()[nodes]
    return float(np.sqrt(np.mean((p - y) ** 2))), int(len(nodes))

