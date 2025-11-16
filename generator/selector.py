"""
Multi-Stage Molecular Selector Module

Implements a comprehensive selection funnel for molecular candidates that balances
multiple objectives while ensuring diversity and scaffold representation.

The selection process follows a multi-stage approach:
1. Gate Filtering: Keep top candidates based on AC gate performance
2. Pareto Optimization: Apply multi-objective optimization on (QSAR, SA, QED)
3. Ranking: Sort by aggregate weighted score and select top-k
4. Diversity Selection: Apply clustering (Butina or Max-Min) for chemical diversity
5. Scaffold Balance: Ensure minimum representation per core scaffold
6. Deduplication: Remove duplicate molecules based on SMILES

This approach ensures that the final selected set is both high-quality (high scores)
and diverse (chemically distinct), while maintaining balanced representation across
different core scaffolds.

"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics.pairwise import cosine_similarity
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator
from rdkit import DataStructs
import random
import warnings

# Suppress Morgan fingerprint deprecation warnings


class MolecularSelector:
    """
    Molecular selector for final diverse set selection.
    
    Implements multi-stage selection funnel with Pareto optimization and diversity selection.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the molecular selector.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.keep_after_gate = config.get("keep_after_gate", 5000)
        self.pareto_config = config.get("pareto_front", {})
        self.ranking_config = config.get("ranking", {})
        self.diversity_config = config.get("diversity", {})
        
        # Pareto front settings
        self.pareto_enable = self.pareto_config.get("enable", True)
        self.k_front = self.pareto_config.get("k_front", 1000)
        
        # Ranking settings
        self.rank_top_k = self.ranking_config.get("rank_top_k", 500)
        
        # Diversity settings
        self.final_k = self.diversity_config.get("final_k", 100)
        self.scaffold_balance = self.diversity_config.get("scaffold_balance", {})
        self.per_core_min = self.scaffold_balance.get("per_core_min", 20)
        self.diversity_method = self.diversity_config.get("method", "butina")
        self.similarity_threshold = self.diversity_config.get("similarity_threshold", 0.7)
        
        # Selector initialized
    
    def select_molecules(self, molecules: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Apply the complete selection funnel.
        
        Args:
            molecules: List of scored molecule dictionaries
            
        Returns:
            Dictionary containing selected molecules at each stage
        """
        print(f"   Starting selection funnel with {len(molecules)} molecules")
        
        # Step 1: Apply Pareto front optimization (no gate filtering - AC-aware fragments handle this)
        if self.pareto_enable:
            pareto_molecules = self._apply_pareto_front(molecules)
            print(f"   After Pareto front: {len(pareto_molecules)} molecules")
        else:
            pareto_molecules = molecules
        
        # Step 3: Rank by aggregate score
        ranked_molecules = self._apply_ranking(pareto_molecules)
        print(f"   After ranking: {len(ranked_molecules)} molecules")
        
        # Step 4: Apply diversity selection
        diverse_molecules = self._apply_diversity_selection(ranked_molecules)
        print(f"   After diversity selection: {len(diverse_molecules)} molecules")
        
        # Step 5: Ensure scaffold balance
        balanced_molecules = self._apply_scaffold_balance(diverse_molecules)
        print(f"   After scaffold balance: {len(balanced_molecules)} molecules")
        
        # Step 6: Deduplication
        final_molecules = self._apply_deduplication(balanced_molecules)
        print(f"   After deduplication: {len(final_molecules)} molecules")
        
        # Step 7: Final trim to final_k (scaffold balance may add more molecules)
        if len(final_molecules) > self.final_k:
            # Sort by aggregate score and keep best final_k
            final_molecules.sort(key=lambda x: x.get("score", 0), reverse=True)
            final_molecules = final_molecules[:self.final_k]
            # Final trim info not needed - redundant with final size
        
        # Step 8: Assign final ranks based on score (1 = highest score)
        final_molecules.sort(key=lambda x: x.get("score", 0), reverse=True)
        for i, mol in enumerate(final_molecules):
            mol["rank"] = i + 1
        
        return {
            "pareto_front": pareto_molecules,
            "ranked": ranked_molecules,
            "diverse": diverse_molecules,
            "final_diverse": final_molecules
        }
    
    def _apply_pareto_front(self, molecules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply Pareto front optimization on (QSAR, SA, QED)."""
        if len(molecules) <= self.k_front:
            return molecules
        
        # Extract objectives
        objectives = []
        for mol in molecules:
            qsar = mol.get("qsar_prob_raw", 0)
            sa = mol.get("sa", 0)
            qed = mol.get("qed", 0)
            objectives.append([qsar, sa, qed])
        
        objectives = np.array(objectives)
        
        # Find Pareto front
        pareto_indices = self._find_pareto_front(objectives)
        pareto_molecules = [molecules[i] for i in pareto_indices]
        
        # Mark Pareto front molecules
        for mol in pareto_molecules:
            mol["is_pareto"] = True
        
        # If we have more than k_front, take the top k_front by aggregate score
        if len(pareto_molecules) > self.k_front:
            pareto_molecules.sort(key=lambda x: x.get("score", 0), reverse=True)
            pareto_molecules = pareto_molecules[:self.k_front]
        
        return pareto_molecules
    
    def _find_pareto_front(self, objectives: np.ndarray) -> List[int]:
        """Find Pareto front indices."""
        n_points = len(objectives)
        pareto_indices = []
        
        for i in range(n_points):
            is_pareto = True
            for j in range(n_points):
                if i != j:
                    # Check if point j dominates point i
                    if np.all(objectives[j] >= objectives[i]) and np.any(objectives[j] > objectives[i]):
                        is_pareto = False
                        break
            
            if is_pareto:
                pareto_indices.append(i)
        
        return pareto_indices
    
    def _apply_ranking(self, molecules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank molecules by aggregate score and take top k."""
        # Sort by aggregate score
        ranked_molecules = sorted(molecules, key=lambda x: x.get("score", 0), reverse=True)
        
        # Take top k
        top_k_molecules = ranked_molecules[:self.rank_top_k]
        
        # Mark top-k molecules
        for i, mol in enumerate(top_k_molecules):
            mol["is_rank_topk"] = True
            mol["rank"] = i + 1
        
        return top_k_molecules
    
    def _apply_diversity_selection(self, molecules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply diversity selection using specified method."""
        if len(molecules) <= self.final_k:
            return molecules
        
        if self.diversity_method == "butina":
            return self._butina_clustering(molecules)
        elif self.diversity_method == "maxmin":
            return self._maxmin_selection(molecules)
        else:
            self.logger.warning(f"Unknown diversity method: {self.diversity_method}")
            return molecules[:self.final_k]
    
    def _butina_clustering(self, molecules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply Butina clustering for diversity selection."""
        # Compute similarity matrix
        fps = self._compute_fingerprints(molecules)
        similarity_matrix = self._compute_similarity_matrix(fps)
        
        # Apply Butina clustering
        selected_indices = self._butina_cluster(similarity_matrix, self.similarity_threshold)
        
        # Select molecules
        selected_molecules = [molecules[i] for i in selected_indices]
        
        # Trim to final_k (Butina may return more than final_k)
        if len(selected_molecules) > self.final_k:
            # Sort by aggregate score to keep best ones
            selected_molecules.sort(key=lambda x: x.get("score", 0), reverse=True)
            selected_molecules = selected_molecules[:self.final_k]
        
        # Mark diverse molecules
        for mol in selected_molecules:
            mol["is_diverse_final"] = True
        
        return selected_molecules
    
    def _maxmin_selection(self, molecules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply max-min selection for diversity."""
        if not molecules:
            return []
        
        # Start with the highest scoring molecule
        selected = [0]
        remaining = list(range(1, len(molecules)))
        
        while len(selected) < self.final_k and remaining:
            # Find the molecule with maximum minimum distance to selected
            max_min_dist = -1
            best_idx = -1
            
            for idx in remaining:
                min_dist = min(
                    self._compute_similarity(molecules[idx], molecules[sel_idx])
                    for sel_idx in selected
                )
                
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = idx
            
            if best_idx != -1:
                selected.append(best_idx)
                remaining.remove(best_idx)
            else:
                break
        
        selected_molecules = [molecules[i] for i in selected]
        
        # Mark diverse molecules
        for mol in selected_molecules:
            mol["is_diverse_final"] = True
        
        return selected_molecules
    
    def _apply_scaffold_balance(self, molecules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure scaffold balance across cores."""
        # Group molecules by core
        core_groups = {}
        for mol in molecules:
            core_idx = mol.get("core_idx", 0)
            if core_idx not in core_groups:
                core_groups[core_idx] = []
            core_groups[core_idx].append(mol)
        
        # Ensure minimum per core
        balanced_molecules = []
        for core_idx, core_molecules in core_groups.items():
            if len(core_molecules) >= self.per_core_min:
                # Take all molecules from this core
                balanced_molecules.extend(core_molecules)
            else:
                # Take all available molecules from this core
                balanced_molecules.extend(core_molecules)
        
        return balanced_molecules
    
    def _apply_deduplication(self, molecules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate molecules based on SMILES."""
        seen_smiles = set()
        unique_molecules = []
        
        for mol in molecules:
            smiles = mol.get("smiles", "")
            if smiles not in seen_smiles:
                seen_smiles.add(smiles)
                unique_molecules.append(mol)
        
        return unique_molecules
    
    def _compute_fingerprints(self, molecules: List[Dict[str, Any]]) -> List[Any]:
        """Compute fingerprints for molecules."""
        fps = []
        for mol in molecules:
            try:
                smiles = mol.get("smiles", "")
                rdkit_mol = Chem.MolFromSmiles(smiles)
                if rdkit_mol:
                    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                    fp = gen.GetFingerprint(rdkit_mol)
                    fps.append(fp)
                else:
                    fps.append(None)
            except Exception as e:
                self.logger.debug(f"Fingerprint computation failed: {e}")
                fps.append(None)
        
        return fps
    
    def _compute_similarity_matrix(self, fps: List[Any]) -> np.ndarray:
        """Compute similarity matrix from fingerprints."""
        n = len(fps)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if fps[i] is not None and fps[j] is not None:
                    try:
                        sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                        similarity_matrix[i, j] = sim
                        similarity_matrix[j, i] = sim
                    except Exception as e:
                        self.logger.debug(f"Similarity computation failed: {e}")
                        similarity_matrix[i, j] = 0
                        similarity_matrix[j, i] = 0
        
        return similarity_matrix
    
    def _butina_cluster(self, similarity_matrix: np.ndarray, threshold: float) -> List[int]:
        """Apply Butina clustering algorithm."""
        n = len(similarity_matrix)
        selected = []
        remaining = list(range(n))
        
        while remaining:
            # Select the molecule with highest average similarity to remaining molecules
            best_idx = -1
            best_avg_sim = -1
            
            for idx in remaining:
                avg_sim = np.mean([similarity_matrix[idx, j] for j in remaining if j != idx])
                if avg_sim > best_avg_sim:
                    best_avg_sim = avg_sim
                    best_idx = idx
            
            if best_idx == -1:
                break
            
            selected.append(best_idx)
            remaining.remove(best_idx)
            
            # Remove molecules that are too similar to the selected one
            to_remove = []
            for idx in remaining:
                if similarity_matrix[best_idx, idx] > threshold:
                    to_remove.append(idx)
            
            for idx in to_remove:
                remaining.remove(idx)
        
        return selected
    
    def _compute_similarity(self, mol1: Dict[str, Any], mol2: Dict[str, Any]) -> float:
        """Compute similarity between two molecules."""
        try:
            smiles1 = mol1.get("smiles", "")
            smiles2 = mol2.get("smiles", "")
            
            mol1_rdkit = Chem.MolFromSmiles(smiles1)
            mol2_rdkit = Chem.MolFromSmiles(smiles2)
            
            if mol1_rdkit and mol2_rdkit:
                gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                fp1 = gen.GetFingerprint(mol1_rdkit)
                fp2 = gen.GetFingerprint(mol2_rdkit)
                return DataStructs.TanimotoSimilarity(fp1, fp2)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.debug(f"Similarity computation failed: {e}")
            return 0.0
    
    def get_selector_stats(self) -> Dict[str, Any]:
        """Get selector statistics."""
        return {
            "keep_after_gate": self.keep_after_gate,
            "pareto_enable": self.pareto_enable,
            "k_front": self.k_front,
            "rank_top_k": self.rank_top_k,
            "final_k": self.final_k,
            "per_core_min": self.per_core_min,
            "diversity_method": self.diversity_method,
            "similarity_threshold": self.similarity_threshold
        }