"""
Data Splitting Module for Activity Cliff-Aware QSAR Evaluation

This module provides group-based data splitting strategies to prevent data leakage
in QSAR model evaluation. It implements two main approaches:
1. Cliff-group splitting: Groups molecules connected by activity cliff edges
2. Butina clustering: Groups molecules based on structural similarity (scaffold fingerprints)

"""

from __future__ import annotations

from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.ML.Cluster import Butina

# Try new sklearn ≥1.3 API; fallback for older versions
try:
    from sklearn.model_selection import StratifiedGroupShuffleSplit
except ImportError:
    from sklearn.model_selection import GroupShuffleSplit

    class StratifiedGroupShuffleSplit:
        """Fallback emulation for older sklearn: ensures class balance."""
        def __init__(self, n_splits=1, test_size=0.2, random_state=None, max_tries=256, tol=0.02):
            self.n_splits = n_splits
            self.test_size = test_size
            self.random_state = random_state
            self.max_tries = max_tries
            self.tol = tol

        def split(self, X, y, groups):
            import numpy as np
            y = np.asarray(y).astype(int)
            groups = np.asarray(groups).astype(int)
            base_ratio = y.mean()
            gss = GroupShuffleSplit(n_splits=self.max_tries, test_size=self.test_size, random_state=self.random_state)
            best = None
            best_diff = 1e9
            for tr_idx, te_idx in gss.split(X, y, groups):
                diff = abs(y[te_idx].mean() - base_ratio)
                if diff < best_diff:
                    best, best_diff = (tr_idx, te_idx), diff
                if diff <= self.tol:
                    break
            yield best

def smiles_to_mol(s: str):
    """Convert SMILES string to RDKit molecule object."""
    try:
        return Chem.MolFromSmiles(s)
    except Exception:
        return None


def make_cliff_groups(df: pd.DataFrame, smiles_col: str, cliffs_csv: Path) -> np.ndarray:
    """
    Build group labels so that molecules connected by cliff edges end up in the same group.
    
    This function creates groups based on activity cliff relationships to prevent data leakage
    during train/test splitting. Molecules that are connected by activity cliff edges will
    be assigned to the same group.
    
    Args:
        df: DataFrame containing molecular data
        smiles_col: Name of the column containing SMILES strings
        cliffs_csv: Path to CSV file containing activity cliff pairs
        
    Returns:
        Array of group labels for each molecule in df
    """
    if not Path(cliffs_csv).exists():
        return np.arange(len(df))
    cliffs = pd.read_csv(cliffs_csv)
    if cliffs.empty or "mol_i" not in cliffs.columns or "mol_j" not in cliffs.columns:
        return np.arange(len(df))

    G = nx.Graph()
    for s in df[smiles_col]:
        G.add_node(s)
    for _, row in cliffs.iterrows():
        si, sj = row.get("mol_i"), row.get("mol_j")
        if pd.notna(si) and pd.notna(sj):
            G.add_edge(si, sj)

    comp_id = {}
    for cid, comp in enumerate(nx.connected_components(G)):
        for smi in comp:
            comp_id[smi] = cid

    groups = df[smiles_col].map(comp_id).fillna(-1).astype(int)
    base = groups.max() + 1
    for i, g in enumerate(groups):
        if g < 0:
            groups.iat[i] = base + i
    return groups.values


def make_groups_butina_scaffold(smiles_list: List[str], radius: int = 2, n_bits: int = 2048, tn_thr: float = 0.7) -> np.ndarray:
    """
    Group indices from Butina clustering on scaffold ECFP fingerprints.
    
    This function uses Butina clustering to group molecules based on their structural
    similarity using ECFP fingerprints. Molecules with Tanimoto similarity above the
    threshold are grouped together.
    
    Args:
        smiles_list: List of SMILES strings
        radius: Fingerprint radius for ECFP generation (default: 2)
        n_bits: Number of bits in fingerprint (default: 2048)
        tn_thr: Tanimoto similarity threshold for clustering (default: 0.7)
        
    Returns:
        Array of group labels for each molecule
    """
    from predictive_model.fingerprint_backbones import get_morgan_generator
    
    mols = [smiles_to_mol(s) for s in smiles_list]
    fps = []
    gen = get_morgan_generator(radius=radius, n_bits=n_bits)
    for m in mols:
        try:
            fp = gen.GetFingerprint(m) if m else None
        except Exception:
            fp = None
        fps.append(fp)
    dists = []
    for i in range(1, len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1 - s for s in sims])
    clusters = Butina.ClusterData(dists, len(fps), 1.0 - tn_thr, isDistData=True)
    gid = np.full(len(fps), -1, dtype=int)
    for cid, members in enumerate(clusters):
        for idx in members:
            gid[idx] = cid
    # assign singletons
    next_gid = gid.max() + 1
    for i in np.where(gid == -1)[0]:
        gid[i] = next_gid
        next_gid += 1
    return gid


def stratified_group_split(X, y, groups, test_size=0.2, random_state=42):
    """
    Perform stratified group split with balanced classes.
    
    This function ensures that train/test splits maintain class balance while
    respecting group boundaries (no molecules from the same group in both splits).
    
    Args:
        X: Feature matrix (not used for splitting, only for shape)
        y: Target labels
        groups: Group labels for each sample
        test_size: Fraction of data for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (train_indices, test_indices) as numpy arrays
    """
    sgss = StratifiedGroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    (train_idx, test_idx), = sgss.split(X, y, groups)
    return np.asarray(train_idx), np.asarray(test_idx)

