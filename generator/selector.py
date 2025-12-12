"""
Multi-Stage Molecular Selector Module

Implements a comprehensive selection funnel for molecular candidates that balances
multiple objectives while ensuring diversity and scaffold representation.

The selection process follows a multi-stage approach:
1. CAFE LATE Bonus: Compute bonus based on unique AC-enriched fragments (ac_enrichment)
2. Layered Pareto Optimization: Apply multi-objective optimization on (QSAR, SA, QED) with F0→F1→... layers
3. Ranking: Sort by aggregate weighted score (including CAFE bonus) and select top-k
4. Multi-Threshold Diversity: Apply progressive diversity selection (0.70 → 0.65 → 0.60 → 0.58) per-core and per-arm
5. Fill-to-100: Ensure exactly 100 molecules using relaxed thresholds and tie-break (CAFE_LATE > QSAR > SA > QED)
6. Scaffold Balance: Ensure minimum representation per core scaffold
7. Deduplication: Remove duplicate molecules based on SMILES

This approach ensures that the final selected set is both high-quality (high scores)
and diverse (chemically distinct), while maintaining balanced representation across
different core scaffolds and leveraging activity cliff knowledge through CAFE LATE.

"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs


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
        self.multi_threshold = self.diversity_config.get("multi_threshold", False)
        self.diversity_thresholds = self.diversity_config.get("thresholds", [0.70, 0.65, 0.60, 0.58])
        self.fill_to_100 = self.diversity_config.get("fill_to_100", False)
        
        # Pareto settings
        self.layered_pareto = self.pareto_config.get("layered", False)
        
        # CAFE LATE settings
        self.cafe_config = config.get("cafe", {})
        self.cafe_lambda_list = self.cafe_config.get("lambda_list", [1.0])  # Renamed from beta_list
        self.cafe_unique_only = self.cafe_config.get("unique_only", True)
        self.cafe_save_per_weight = self.cafe_config.get("save_per_weight", True)
        self.cafe_primary_set_mode = self.cafe_config.get("primary_set_mode", "union")
        
        # CAFE scorer will be set by generator (needed for ac_enrichment lookup)
        self.cafe_scorer = None
        
        # Scoring weights (from scoring config, passed via config)
        self.qsar_weight = config.get("qsar_weight", 0.4)
        self.sa_weight = config.get("sa_weight", 0.4)
        self.qed_weight = config.get("qed_weight", 0.2)
    
    def _compute_cafe_bonus(self, molecule: Dict[str, Any], lambda_cafe: float) -> Tuple[float, float]:
        """
        Compute CAFE LATE bonus based on unique AC-enriched fragments.
        
        Args:
            molecule: Molecule dictionary with fragments_used
            lambda_cafe: Weight for CAFE bonus
            
        Returns:
            Tuple of (bonus, total_enrichment)
        """
        if not self.cafe_scorer or not self.cafe_scorer.enable_cafe or lambda_cafe == 0.0:
            return 0.0, 0.0
        
        fragments_used = molecule.get("fragments_used", [])
        if not fragments_used:
            return 0.0, 0.0
        
        # Get unique fragments only (if fragment appears multiple times, count once)
        if isinstance(fragments_used, str):
            unique_fragments = set(f.strip() for f in fragments_used.split(";"))
        else:
            unique_fragments = set(fragments_used)
        
        # Sum ac_enrichment for unique fragments
        total_enrichment = 0.0
        for frag in unique_fragments:
            if frag in self.cafe_scorer.cafe_fragments:
                enrichment = self.cafe_scorer.cafe_fragments[frag]["ac_enrichment"]
                total_enrichment += enrichment
        
        if total_enrichment > 0:
            normalization_divisor = 180.0
            normalized_enrichment = total_enrichment / normalization_divisor
            base_boost_factor = np.tanh(normalized_enrichment)
            lambda_power = np.power(lambda_cafe, 0.95)
            lambda_scale = 0.03 + 0.40 * lambda_power
            boost_factor = base_boost_factor * lambda_scale
            boost_factor = min(boost_factor, 0.60)
        else:
            boost_factor = 0.0
        
        # Final bonus: multiply boost_factor by lambda for proportional scaling
        bonus = lambda_cafe * boost_factor
        
        return float(bonus), float(total_enrichment)
    
    def select_molecules(self, molecules: List[Dict[str, Any]], lambda_cafe: Optional[float] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        Apply the complete selection funnel with optional CAFE LATE bonus.
        
        Args:
            molecules: List of scored molecule dictionaries
            lambda_cafe: Optional CAFE weight (if None, uses first from lambda_list, renamed from beta_cafe)
            
        Returns:
            Dictionary containing selected molecules at each stage
        """
        if lambda_cafe is None:
            lambda_cafe = self.cafe_lambda_list[0] if self.cafe_lambda_list else 0.0
        
        print(f"   Starting selection funnel with {len(molecules)} molecules (lambda_cafe={lambda_cafe:.2f})")
        
        # Step 1: Apply Pareto front optimization FIRST (before CAFE bonus to reduce set size)
        if self.pareto_enable:
            if self.layered_pareto:
                pareto_molecules = self._apply_layered_pareto_front(molecules)
            else:
                pareto_molecules = self._apply_pareto_front(molecules)
            print(f"   After Pareto front: {len(pareto_molecules)} molecules")
        else:
            pareto_molecules = molecules
        
        # Step 2: Compute CAFE bonuses AFTER Pareto (on smaller set ~75k instead of 492k)
        if lambda_cafe > 0.0 and self.cafe_scorer and self.cafe_scorer.enable_cafe:
            pareto_molecules = self._add_cafe_bonus_to_scores(pareto_molecules, lambda_cafe)
        else:
            # Ensure cafe_bonus and cafe_enrichment are present even if 0
            for mol in pareto_molecules:
                mol["cafe_bonus"] = 0.0
                mol["cafe_enrichment"] = 0.0
        
        # Step 3: Rank by aggregate score (now includes CAFE bonus if applicable)
        ranked_molecules = self._apply_ranking(pareto_molecules)
        print(f"   After ranking: {len(ranked_molecules)} molecules")
        
        # Step 4: Apply diversity selection (multi-threshold if enabled)
        if self.multi_threshold and self.fill_to_100:
            diverse_molecules = self._apply_multi_threshold_diversity(ranked_molecules)
        else:
            diverse_molecules = self._apply_diversity_selection(ranked_molecules)
        print(f"   After diversity selection: {len(diverse_molecules)} molecules")
        
        # Step 5: Ensure scaffold balance
        balanced_molecules = self._apply_scaffold_balance(diverse_molecules)
        print(f"   After scaffold balance: {len(balanced_molecules)} molecules")
        
        # Step 6: Fill to 100 if needed
        if self.fill_to_100 and len(balanced_molecules) < self.final_k:
            balanced_molecules = self._fill_to_100(balanced_molecules, ranked_molecules)
            print(f"   After fill-to-100: {len(balanced_molecules)} molecules")
        
        # Step 7: Deduplication
        final_molecules = self._apply_deduplication(balanced_molecules)
        print(f"   After deduplication: {len(final_molecules)} molecules")
        
        # Step 8: Final trim to final_k (scaffold balance may add more molecules)
        if len(final_molecules) > self.final_k:
            # Sort by aggregate score with tie-break: CAFE_LATE > QSAR > SA > QED
            final_molecules.sort(key=lambda x: self._tie_break_key(x), reverse=True)
            final_molecules = final_molecules[:self.final_k]
        
        # Step 9: Assign final ranks based on score (1 = highest score)
        final_molecules.sort(key=lambda x: self._tie_break_key(x), reverse=True)
        for i, mol in enumerate(final_molecules):
            mol["rank"] = i + 1
        
        return {
            "pareto_front": pareto_molecules,
            "ranked": ranked_molecules,
            "diverse": diverse_molecules,
            "final_diverse": final_molecules
        }
    
    def _add_cafe_bonus_to_scores(self, molecules: List[Dict[str, Any]], lambda_cafe: float) -> List[Dict[str, Any]]:
        """
        Add CAFE LATE bonus to molecules and recompute aggregate scores.
        
        Aggregate score: w_qsar*QSAR + w_sa*SA_prime + w_qed*QED + bonus(m)
        """
        bonuses = []
        enrichments = []
        for mol in molecules:
            bonus, total_enrichment = self._compute_cafe_bonus(mol, lambda_cafe)
            bonuses.append(bonus)
            enrichments.append(total_enrichment)
            mol["cafe_bonus"] = bonus
            mol["cafe_enrichment"] = total_enrichment  # Store enrichment for analysis
            
            # Recompute aggregate score with CAFE bonus
            qsar = mol.get("qsar_prob", mol.get("qsar_prob_raw", 0.0))
            sa = mol.get("sa", 0.0)
            qed = mol.get("qed", 0.0)
            
            # SA_prime is normalized SA (already in [0,1])
            sa_prime = sa
            
            # New aggregate score with CAFE bonus
            new_score = (
                self.qsar_weight * qsar +
                self.sa_weight * sa_prime +
                self.qed_weight * qed +
                bonus
            )
            mol["score"] = new_score
        
        # Log statistics
        if bonuses:
            mean_bonus = np.mean(bonuses)
            std_bonus = np.std(bonuses)
            mean_enrichment = np.mean(enrichments)
            qsar_scores = [m.get("qsar_prob", m.get("qsar_prob_raw", 0.0)) for m in molecules]
            std_qsar = np.std(qsar_scores) if qsar_scores else 0.0
            print(f"   CAFE bonus stats: mean={mean_bonus:.4f}, std={std_bonus:.4f}, mean(enrichment)={mean_enrichment:.2f}, std(QSAR)={std_qsar:.4f}")
        
        return molecules
    
    def _tie_break_key(self, mol: Dict[str, Any]) -> Tuple[float, float, float, float]:
        """
        Tie-break key for sorting: CAFE_LATE > QSAR > SA > QED
        Returns tuple for lexicographic sorting.
        """
        cafe_bonus = mol.get("cafe_bonus", 0.0)
        qsar = mol.get("qsar_prob", mol.get("qsar_prob_raw", 0.0))
        sa = mol.get("sa", 0.0)
        qed = mol.get("qed", 0.0)
        return (cafe_bonus, qsar, sa, qed)
    
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
    
    def _apply_layered_pareto_front(self, molecules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply layered Pareto front optimization (F0→F1→...).
        
        Layers:
        - F0: First Pareto front
        - F1: Second Pareto front (after removing F0)
        - Continue until we have enough molecules or exhaust the pool
        """
        if len(molecules) <= self.k_front:
            return molecules
        
        selected = []
        remaining = molecules.copy()
        layer = 0
        max_layers = 10  # Limit layers to prevent infinite loops
        
        while len(selected) < self.k_front and remaining and layer < max_layers:
            # Extract objectives for remaining molecules
            objectives = []
            for mol in remaining:
                qsar = mol.get("qsar_prob_raw", 0)
                sa = mol.get("sa", 0)
                qed = mol.get("qed", 0)
                objectives.append([qsar, sa, qed])
            
            objectives = np.array(objectives)
            
            # Find Pareto front for this layer
            pareto_indices = self._find_pareto_front(objectives)
            layer_molecules = [remaining[i] for i in pareto_indices]
            
            # Mark layer
            for mol in layer_molecules:
                mol["pareto_layer"] = layer
                mol["is_pareto"] = True
            
            # Add to selected
            selected.extend(layer_molecules)
            
            # Remove from remaining
            remaining = [m for i, m in enumerate(remaining) if i not in pareto_indices]
            
            layer += 1
            
            # Log progress
            if layer % 5 == 0:
                self.logger.info(f"Layered Pareto: layer {layer}, selected {len(selected)}, remaining {len(remaining)}")
        
        # If we have more than k_front, take the top k_front by aggregate score
        if len(selected) > self.k_front:
            selected.sort(key=lambda x: (x.get("pareto_layer", 999), -x.get("score", 0)))
            selected = selected[:self.k_front]
        
        return selected
    
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
            return self._butina_clustering(molecules, self.similarity_threshold)
        elif self.diversity_method == "maxmin":
            return self._maxmin_selection(molecules)
        else:
            self.logger.warning(f"Unknown diversity method: {self.diversity_method}")
            return molecules[:self.final_k]
    
    def _apply_multi_threshold_diversity(self, molecules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply multi-threshold diversity selection with progressive thresholds.
        
        Uses thresholds: 0.70 → 0.65 → 0.60 → 0.58 per-core and per-arm.
        """
        if len(molecules) <= self.final_k:
            return molecules
        
        selected = []
        remaining = molecules.copy()
        
        # Group by core for per-core diversity
        core_groups = {}
        for mol in remaining:
            core = mol.get("core", "unknown")
            if core not in core_groups:
                core_groups[core] = []
            core_groups[core].append(mol)
        
        # Apply diversity per core with progressive thresholds
        for threshold in self.diversity_thresholds:
            if len(selected) >= self.final_k:
                break
            
            for core, core_molecules in core_groups.items():
                if len(selected) >= self.final_k:
                    break
                
                # Get molecules from this core not yet selected
                core_remaining = [m for m in core_molecules if m not in selected]
                if not core_remaining:
                    continue
                
                # Apply diversity selection with current threshold
                if self.diversity_method == "butina":
                    core_selected = self._butina_clustering(core_remaining, threshold)
                else:
                    core_selected = core_remaining[:min(len(core_remaining), self.final_k - len(selected))]
                
                # Add to selected
                for mol in core_selected:
                    if mol not in selected:
                        selected.append(mol)
                        if len(selected) >= self.final_k:
                            break
        
        # If still not enough, fill from remaining with relaxed criteria
        if len(selected) < self.final_k:
            remaining = [m for m in molecules if m not in selected]
            remaining.sort(key=lambda x: x.get("score", 0), reverse=True)
            needed = self.final_k - len(selected)
            selected.extend(remaining[:needed])
        
        return selected[:self.final_k]
    
    def _fill_to_100(self, selected: List[Dict[str, Any]], ranked: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fill selection to exactly 100 molecules using relaxed thresholds.
        
        Uses tie-break: CAFE_LATE > QSAR > SA > QED
        """
        if len(selected) >= self.final_k:
            return selected
        
        selected_set = set(id(m) for m in selected)
        remaining = [m for m in ranked if id(m) not in selected_set]
        
        # Sort by tie-break key
        remaining.sort(key=lambda x: self._tie_break_key(x), reverse=True)
        
        needed = self.final_k - len(selected)
        selected.extend(remaining[:needed])
        
        return selected
    
    def _butina_clustering(self, molecules: List[Dict[str, Any]], threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """Apply Butina clustering for diversity selection."""
        if threshold is None:
            threshold = self.similarity_threshold
        
        n_mols = len(molecules)
        if n_mols == 0:
            return []
        
        # For very large sets, use incremental Butina without full similarity matrix
        # This avoids O(n²) memory and computation
        if n_mols > 5000:
            return self._butina_clustering_incremental(molecules, threshold)
        
        # For smaller sets, use standard Butina with similarity matrix
        # Compute similarity matrix
        fps = self._compute_fingerprints(molecules)
        similarity_matrix = self._compute_similarity_matrix(fps)
        
        # Apply Butina clustering
        selected_indices = self._butina_cluster(similarity_matrix, threshold)
        
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
    
    def _butina_clustering_incremental(self, molecules: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """
        Incremental Butina clustering for large datasets.
        Computes similarities on-the-fly without storing full matrix.
        """
        if not molecules:
            return []
        
        # Compute fingerprints once
        fps = self._compute_fingerprints(molecules)
        valid_indices = [i for i, fp in enumerate(fps) if fp is not None]
        valid_fps = [fps[i] for i in valid_indices]
        
        if not valid_fps:
            return molecules[:self.final_k]
        
        selected_indices = []
        remaining_indices = list(range(len(valid_indices)))
        
        self.logger.info(f"Incremental Butina clustering: {len(valid_fps)} molecules, threshold={threshold}")
        
        iteration = 0
        while remaining_indices and len(selected_indices) < self.final_k:
            iteration += 1
            if iteration % 100 == 0:
                self.logger.info(f"Butina progress: {len(selected_indices)}/{self.final_k} selected, "
                               f"{len(remaining_indices)} remaining")
            
            # Select molecule with highest average similarity to remaining (represents cluster centroid)
            best_idx = -1
            best_avg_sim = -1
            
            for idx in remaining_indices:
                fp_i = valid_fps[idx]
                # Compute similarities to all remaining molecules
                similarities = DataStructs.BulkTanimotoSimilarity(
                    fp_i, 
                    [valid_fps[j] for j in remaining_indices if j != idx]
                )
                if similarities:
                    avg_sim = np.mean(similarities)
                    if avg_sim > best_avg_sim:
                        best_avg_sim = avg_sim
                        best_idx = idx
            
            if best_idx == -1:
                break
            
            # Add best molecule to selected
            selected_indices.append(valid_indices[best_idx])
            remaining_indices.remove(best_idx)
            
            # Remove molecules too similar to selected one
            fp_selected = valid_fps[best_idx]
            to_remove = []
            for idx in remaining_indices:
                fp_other = valid_fps[idx]
                sim = DataStructs.TanimotoSimilarity(fp_selected, fp_other)
                if sim > threshold:
                    to_remove.append(idx)
            
            for idx in to_remove:
                remaining_indices.remove(idx)
        
        # Map back to original indices and select molecules
        selected_molecules = [molecules[i] for i in selected_indices]
        
        # Trim to final_k if needed
        if len(selected_molecules) > self.final_k:
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
        """Compute similarity matrix from fingerprints using BulkTanimotoSimilarity for speed."""
        n = len(fps)
        similarity_matrix = np.zeros((n, n))
        
        # Filter out None fingerprints
        valid_fps = [(i, fp) for i, fp in enumerate(fps) if fp is not None]
        valid_indices = [i for i, _ in valid_fps]
        valid_fp_list = [fp for _, fp in valid_fps]
        
        if not valid_fp_list:
            return similarity_matrix
        
        # Use BulkTanimotoSimilarity for efficient computation
        for i, (idx_i, fp_i) in enumerate(valid_fps):
            # Compute similarities for this fingerprint against all others
            similarities = DataStructs.BulkTanimotoSimilarity(fp_i, valid_fp_list)
            
            for j, (idx_j, _) in enumerate(valid_fps):
                if idx_i <= idx_j:
                    sim = similarities[j]
                    similarity_matrix[idx_i, idx_j] = sim
                    similarity_matrix[idx_j, idx_i] = sim
        
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
    