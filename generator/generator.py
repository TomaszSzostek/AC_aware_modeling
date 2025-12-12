"""
AC-aware fragment-based molecular generation.

This module loads the labeled dataset and fragment library, generates molecules
around predefined cores, scores them with QSAR/SA/QED (with optional CAFE LATE
adjustment) and selects a diverse set of candidates written to CSV.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import time
import tqdm
import random

from .sampler import FragmentSampler
from .molecular_generator import MolecularGenerator
from .scorer import MolecularScorer
from .selector import MolecularSelector


class ACGenerator:
    """
    AC-aware molecular generator using fragments from Reverse QSAR with CAFE LATE scoring.
    
    Orchestrates fragment-based molecular generation pipeline:
    1. Load data and AC-enriched fragments from Reverse QSAR results
    2. Generate molecules using Island Algorithm (cores + fragments)
    3. Score molecules (QSAR with CAFE LATE adjustment, SA, QED)
    4. Select diverse final set
    5. Save publication-ready hits.csv with fragment AC flags and CAFE LATE metadata
    
    CAFE LATE (Late-stage Activity Cliff Fragment Enrichment) adjusts QSAR predictions
    for molecules containing AC_added fragments, leveraging activity cliff knowledge
    to improve predictions for cliff-like chemical space.
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None, vanilla_mode: bool = False):
        """
        Initialize the AC Generator.
        
        Args:
            config: Configuration dictionary
            logger: Optional logger instance
            vanilla_mode: If True, use vanilla fragments and disable CAFE scoring
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.vanilla_mode = vanilla_mode
        
        # Extract configuration sections
        self.gen_config = config.get("Generator", {})
        self.paths_config = config.get("Paths", {})
        
        # Initialize components
        self.sampler = None
        self.molecular_generator = None
        self.scorer = None
        self.selector = None
        
        # Data storage
        self.labeled_data = None
        self.fragment_library = None
        self.results = {}
        self.original_smiles_set = set()
        self.ac_enriched_fragments: set = set()  # Set of AC-enriched fragment SMILES
        
        # Statistics tracking
        self.stats = {
            "generated": 0,
            "dropped_floors": 0,
            "rejection_reasons": {},
            "final_size": 0,
            "start_time": None,
            "end_time": None
        }
    
    def load_data(self) -> None:
        """Load and preprocess the labeled dataset and fragment library."""
        print("Loading and preprocessing data...")
        
        # Load labeled dataset
        dataset_path = self.paths_config.get("dataset", "data/processed/final_dataset.csv")
        self.labeled_data = pd.read_csv(dataset_path)
        print(f"   Loaded {len(self.labeled_data)} labeled compounds")
        # Build exact-match set of original SMILES for deduplication
        if "canonical_smiles" in self.labeled_data.columns:
            self.original_smiles_set = set(self.labeled_data["canonical_smiles"].astype(str).tolist())
        else:
            self.original_smiles_set = set()
        
        # Load fragment library (vanilla or CAFE)
        if self.vanilla_mode:
            frag_root = self.paths_config.get("fragments", "results/reverse_QSAR")
            frag_path = Path(frag_root) / "reinvent_fragments_vanilla.csv"
            if not frag_path.exists():
                raise FileNotFoundError(f"Vanilla fragment library not found: {frag_path}. Run vanilla library generation first.")
            print(f"   [VANILLA MODE] Loading vanilla fragments from {frag_path}")
        else:
            frag_path = self.gen_config.get("fragment_library")
            if not frag_path:
                raise ValueError("fragment_library path not found in config!")
            print(f"   [CAFE MODE] Loading CAFE fragments from {frag_path}")
        
        self.fragment_library = pd.read_csv(frag_path)
        print(f"   Loaded {len(self.fragment_library)} fragments from {frag_path}")
        
        # Load AC-enriched fragments if available (skip in vanilla mode)
        if not self.vanilla_mode:
            self._load_ac_enriched_fragments()
        else:
            print("   [VANILLA MODE] Skipping AC-enriched fragments loading")
        
        # Initialize components with data
        self._initialize_components()
    
    def _load_ac_enriched_fragments(self) -> None:
        """Load AC-enriched fragments from Reverse QSAR results (only AC_added by CAFE)."""
        try:
            # Try to find AC-enriched fragments from Reverse QSAR
            frag_root = Path(self.paths_config.get("fragments", "results/ReverseQSAR"))
            summary_path = frag_root / "summary.json"
            
            if summary_path.exists():
                import json
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                best_model = summary.get("best_model")
                if best_model:
                    model_dir = frag_root / best_model
                    ac_csv = model_dir / "selected_fragments_with_ACflag.csv"
                    if ac_csv.exists():
                        ac_df = pd.read_csv(ac_csv)
                        # Get fragments that were added by CAFE (selected_by == "AC_added")
                        if "selected_by" in ac_df.columns:
                            ac_frags = ac_df[ac_df["selected_by"] == "AC_added"]
                            if "fragment_smiles" in ac_frags.columns:
                                self.ac_enriched_fragments = set(ac_frags["fragment_smiles"].astype(str).tolist())
                                print(f"   Loaded {len(self.ac_enriched_fragments)} AC-added fragments (CAFE only)")
                                return
            
            # Fallback: try any model directory
            if frag_root.exists():
                for model_dir in frag_root.iterdir():
                    if model_dir.is_dir():
                        ac_csv = model_dir / "selected_fragments_with_ACflag.csv"
                        if ac_csv.exists():
                            ac_df = pd.read_csv(ac_csv)
                            if "selected_by" in ac_df.columns:
                                ac_frags = ac_df[ac_df["selected_by"] == "AC_added"]
                                if "fragment_smiles" in ac_frags.columns:
                                    self.ac_enriched_fragments = set(ac_frags["fragment_smiles"].astype(str).tolist())
                                    print(f"   Loaded {len(self.ac_enriched_fragments)} AC-added fragments (CAFE only)")
                                    return
            
            print("   No AC-added fragments found, continuing without AC flags")
        except Exception as e:
            self.logger.warning(f"Could not load AC-added fragments: {e}")
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components."""
        print("Initializing pipeline components...")
        
        # Initialize sampler
        self.sampler = FragmentSampler(
            fragments=self.fragment_library,
            config=self.gen_config.get("seed_assembly", {}),
            logger=self.logger
        )
        
        # Initialize molecular generator
        self.molecular_generator = MolecularGenerator(
            cores=self.gen_config.get("cores", []),
            config=self.gen_config.get("generation", {}),
            logger=self.logger
        )
        
        # Set core count for sampler
        self.sampler.set_core_count(len(self.gen_config.get("cores", [])))
        
        # Initialize scorer (disable CAFE in vanilla mode)
        scoring_config = self.gen_config.get("scoring", {}).copy()
        if self.vanilla_mode:
            scoring_config["enable_cafe_scoring"] = False
            print("   [VANILLA MODE] CAFE LATE scoring disabled")
        
        self.scorer = MolecularScorer(
            qsar_model_path=scoring_config.get("qsar_model_path"),
            config=scoring_config,
            logger=self.logger
        )
        
        # Deduplication is handled by exact SMILES match in _passes_floors (no Tanimoto novelty)
        
        # Initialize selector (pass scoring weights for CAFE bonus calculation)
        selection_config = self.gen_config.get("selection", {}).copy()
        scoring_config = self.gen_config.get("scoring", {})
        selection_config["qsar_weight"] = scoring_config.get("qsar_weight", 0.4)
        selection_config["sa_weight"] = scoring_config.get("sa_weight", 0.4)
        selection_config["qed_weight"] = scoring_config.get("qed_weight", 0.2)
        selection_config["paths"] = self.paths_config
        selection_config["output"] = self.gen_config.get("output", {})
        
        self.selector = MolecularSelector(
            config=selection_config,
            logger=self.logger
        )
        
        # Pass CAFE scorer to selector for ac_enrichment lookup
        if hasattr(self.scorer, 'cafe_scorer'):
            self.selector.cafe_scorer = self.scorer.cafe_scorer
    
    def generate_molecules(self) -> Dict[str, Any]:
        """
        Main generation pipeline.
        
        Returns:
            Dictionary containing generation results and statistics
        """
        print("Starting molecular generation pipeline...")
        self.stats["start_time"] = datetime.now()
        
        try:
            # Step 1: Generate molecules
            print("Step 1: Generating molecules...")
            generated_molecules = self._generate_molecule_batch()
            
            # Step 2: Score molecules
            print("Step 2: Scoring molecules...")
            scored_molecules = self._score_molecules(generated_molecules)
            
            # Step 3: Select final diverse set
            print("Step 3: Selecting final diverse set...")
            final_molecules = self._select_final_molecules(scored_molecules)
            
            # Step 4: Save results
            print("Step 4: Saving results...")
            self._save_results(final_molecules, scored_molecules)
            
            self.stats["end_time"] = datetime.now()
            self._log_final_statistics()
            
            return {
                "success": True,
                "stats": self.stats,
                "results": self.results
            }
            
        except Exception as e:
            self.logger.error(f"Generation failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "stats": self.stats
            }
    
    def _generate_molecule_batch(self) -> List[Dict[str, Any]]:
        """Generate a batch of molecules using our custom molecular generator with global deduplication."""
        n_samples = self.gen_config.get("generation", {}).get("n_samples", 10000)
        batch_size = self.gen_config.get("generation", {}).get("batch_size", 1000)
        dedup_config = self.gen_config.get("generation", {}).get("deduplication", {})
        
        mode_str = "[VANILLA]" if self.vanilla_mode else "[CAFE]"
        print(f"{mode_str} Starting generation: target={n_samples}, batch_size={batch_size}, fragments={len(self.fragment_library)}")
        
        all_molecules = []
        global_seen_smiles = set()
        total_attempts = 0
        max_total_attempts = n_samples * 50 if self.vanilla_mode else n_samples * 20  # More attempts for vanilla
        consecutive_empty_batches = 0
        max_empty_batches = 50 if self.vanilla_mode else 10  # More tolerance for vanilla with fewer fragments
        
        import time
        generation_start_time = time.time()
        last_progress_time = time.time()
        max_no_progress_time = 3600 if self.vanilla_mode else 600  # 1 hour for vanilla, 10 min for CAFE
        
        while len(global_seen_smiles) < n_samples and total_attempts < max_total_attempts:
            actual_batch_size = batch_size
            if self.vanilla_mode and len(self.fragment_library) < batch_size:
                # For vanilla with limited fragments, use more aggressive sampling
                actual_batch_size = min(len(self.fragment_library) * 50, batch_size * 2)
            fragment_batch = self.sampler.sample_fragments(actual_batch_size)
            random.shuffle(fragment_batch)
            
            remaining_samples = n_samples - len(global_seen_smiles)
            batch_start = time.time()
            molecules = self.molecular_generator.generate_molecules(fragment_batch, max_molecules=remaining_samples) if remaining_samples > 0 else []
            batch_time = time.time() - batch_start
            
            if len(molecules) == 0:
                if len(global_seen_smiles) == 0:
                    print(f"{mode_str} WARNING: generate_molecules returned 0 molecules after {batch_time:.1f}s. Check fragment/core compatibility.")
                elif consecutive_empty_batches == 0:
                    print(f"{mode_str} Empty batch after {batch_time:.1f}s (total: {len(global_seen_smiles)} molecules)")
            
            unique_molecules = []
            duplicates_count = 0
            for mol in molecules:
                if mol["smiles"] not in global_seen_smiles:
                    unique_molecules.append(mol)
                    global_seen_smiles.add(mol["smiles"])
                else:
                    duplicates_count += 1
            
            self.stats["generated"] += len(unique_molecules)
            total_attempts += len(molecules)
            
            log_interval = self.gen_config.get("output", {}).get("log_interval", 1000)
            if len(global_seen_smiles) % log_interval == 0 or len(global_seen_smiles) == n_samples or (self.vanilla_mode and len(global_seen_smiles) % 100 == 0):
                progress_pct = (len(global_seen_smiles) / n_samples) * 100
                dup_rate = (duplicates_count / len(molecules) * 100) if len(molecules) > 0 else 0
                print(
                    f"{mode_str} Generated: {len(global_seen_smiles)}/{n_samples} "
                    f"({progress_pct:.1f}%) | Attempts: {total_attempts} | "
                    f"This batch: {len(unique_molecules)} unique, "
                    f"{duplicates_count} duplicates ({dup_rate:.1f}%)"
                )
            
            all_molecules.extend(unique_molecules)
            
            if len(unique_molecules) > 0:
                last_progress_time = time.time()
                consecutive_empty_batches = 0
            else:
                consecutive_empty_batches += 1
                if consecutive_empty_batches >= max_empty_batches:
                    print(
                        f"{mode_str} Chemical space exhausted. Generated "
                        f"{len(global_seen_smiles)} unique molecules (target was {n_samples})."
                    )
                    break
            
            if self.vanilla_mode:
                time_since_progress = time.time() - last_progress_time
                if time_since_progress > max_no_progress_time:
                    elapsed_minutes = (time.time() - generation_start_time) / 60
                    print(
                        f"{mode_str} No progress for {time_since_progress/60:.1f} minutes. "
                        f"Generated {len(global_seen_smiles)} molecules in {elapsed_minutes:.1f} minutes. Stopping."
                    )
                    break
        
        print(
            f"{mode_str} Total unique molecules generated: {len(global_seen_smiles)} "
            f"(target: {n_samples})"
        )
        return all_molecules
    
    
    def _score_molecules(self, molecules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score molecules using batch QSAR, SA, and QED scoring.
        
        Args:
            molecules: List of molecule dictionaries to score
            
        Returns:
            List of scored molecules that pass floor criteria
        """
        if not molecules:
            return []
        
        batch_size = 1000
        scored_molecules_all = self.scorer.score_molecules_batch(molecules, batch_size=batch_size)
        
        scored_molecules = []
        for mol in tqdm.tqdm(scored_molecules_all, desc="Filtering", unit="mol"):
            try:
                passed, reason = self._passes_floors(mol)
                if passed:
                    scored_molecules.append(mol)
                else:
                    self.stats["dropped_floors"] += 1
                    key = f"floor:{reason}"
                    self.stats["rejection_reasons"][key] = self.stats["rejection_reasons"].get(key, 0) + 1
            except Exception as e:
                self.logger.warning(f"Failed to filter molecule {mol.get('smiles', 'unknown')}: {e}")
                continue
        
        return scored_molecules
    
    def _passes_floors(self, mol: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if molecule passes all floor criteria and return reason."""
        scoring_cfg = self.gen_config.get("scoring", {})
        qed_floor = scoring_cfg.get("qed_floor", 0.1)
        
        # Check QED floor
        if mol.get("qed", 0) < qed_floor:
            return False, "qed"
        
        # Check novelty guard
        # Drop if molecule exists exactly in original dataset
        if mol.get("smiles") in self.original_smiles_set:
            return False, "in_dataset"
        
        return True, "ok"
    
    def _select_final_molecules(self, molecules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select final diverse set of molecules with multi-weight CAFE selection.
        
        Returns dictionary with:
        - final_diverse: Primary set (union of all weights or best weight)
        - per_weight: Dictionary of TOP-100 per beta_cafe weight
        """
        cafe_config = self.gen_config.get("selection", {}).get("cafe", {})
        lambda_list = cafe_config.get("lambda_list", [1.0])  # Renamed from beta_list
        save_per_weight = cafe_config.get("save_per_weight", True)
        primary_set_mode = cafe_config.get("primary_set_mode", "union")
        
        # If no CAFE or only one weight, use simple selection
        if not self.vanilla_mode and len(lambda_list) > 1:
            return self._select_multi_weight_cafe(molecules, lambda_list, save_per_weight, primary_set_mode)
        else:
            # Single weight selection
            lambda_cafe = lambda_list[0] if lambda_list else 0.0
            selected = self.selector.select_molecules(molecules, lambda_cafe=lambda_cafe)
            
            # Update statistics
            self._update_selection_stats(molecules, selected)
            
            return selected
    
    def _select_multi_weight_cafe(self, molecules: List[Dict[str, Any]], lambda_list: List[float], 
                                   save_per_weight: bool, primary_set_mode: str) -> Dict[str, Any]:
        """
        Run selection with multiple CAFE weights and create primary set.
        
        Args:
            molecules: List of scored molecules
            lambda_list: List of lambda_cafe weights to test (renamed from beta_list)
            save_per_weight: Whether to save TOP-100 per weight
            primary_set_mode: "union" or "best"
            
        Returns:
            Dictionary with final_diverse (primary set) and per_weight results
        """
        print(f"   Running multi-weight CAFE selection with {len(lambda_list)} weights")
        
        per_weight_results = {}
        all_selected_smiles = set()
        
        # Run selection for each weight
        for lambda_cafe in lambda_list:
            print(f"   Processing lambda_cafe={lambda_cafe:.2f}...")
            selected = self.selector.select_molecules(molecules, lambda_cafe=lambda_cafe)
            final_diverse = selected.get("final_diverse", [])
            
            per_weight_results[lambda_cafe] = {
                "final_diverse": final_diverse,
                "n_selected": len(final_diverse)
            }
            
            # Track all selected SMILES for union
            for mol in final_diverse:
                all_selected_smiles.add(mol.get("smiles", ""))
        
        # Create primary set
        if primary_set_mode == "union":
            # Union of all selected molecules from all weights
            primary_set = []
            seen_smiles = set()
            for lambda_cafe in lambda_list:
                for mol in per_weight_results[lambda_cafe]["final_diverse"]:
                    smiles = mol.get("smiles", "")
                    if smiles not in seen_smiles:
                        primary_set.append(mol)
                        seen_smiles.add(smiles)
            
            # Re-select from union using best weight (or first weight)
            best_lambda = lambda_list[-1]  # Use highest weight for primary set
            primary_selected = self.selector.select_molecules(primary_set, lambda_cafe=best_lambda)
            primary_final = primary_selected.get("final_diverse", [])
        else:
            # Use best weight's selection
            best_lambda = lambda_list[-1]
            primary_final = per_weight_results[best_lambda]["final_diverse"]
        
        # Update statistics
        self._update_selection_stats(molecules, {"final_diverse": primary_final})
        
        return {
            "final_diverse": primary_final,
            "per_weight": per_weight_results,
            "primary_set_mode": primary_set_mode
        }
    
    def _update_selection_stats(self, molecules: List[Dict[str, Any]], selected: Dict[str, Any]) -> None:
        """Update selection statistics."""
        self.stats["pareto_size"] = len(selected.get("pareto_front", []))
        self.stats["final_size"] = len(selected.get("final_diverse", []))
        # Rejection breakdown across selection funnel
        scored_n = len(molecules)  # Input to selection
        pareto_n = len(selected.get("pareto_front", []))
        ranked_n = len(selected.get("ranked", []))
        diverse_n = len(selected.get("diverse", []))
        final_n = len(selected.get("final_diverse", []))
        self.stats["rejection_breakdown"] = {
            "scored_molecules": scored_n,
            "pareto_removed": max(0, scored_n - pareto_n),
            "ranking_removed": max(0, pareto_n - ranked_n),
            "diversity_removed": max(0, ranked_n - diverse_n),
            "scaffold_removed": 0,  # balance step currently keeps all
            "dedup_removed": max(0, diverse_n - final_n)
        }
    
    def _save_results(self, results: Dict[str, Any], scored_molecules: List[Dict[str, Any]] = None) -> None:
        """Save all results to files."""
        output_config = self.gen_config.get("output", {})
        results_dir = Path(output_config.get("results_dir", "results/Generator"))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        files_config = output_config.get("files", {})
        use_parquet = output_config.get("use_parquet", True)
        
        # Save full pool of unique molecules to Parquet (generate-once, select-many)
        if scored_molecules and use_parquet:
            if self.vanilla_mode:
                pool_path = results_dir / "generated_pool_vanilla.parquet"
            else:
                pool_path = results_dir / "generated_pool_cafe.parquet"  # Explicit CAFE pool name
            self._save_generated_pool(scored_molecules, pool_path)
        
        # Save scored molecules CSV (optional, for backward compatibility)
        if scored_molecules and not use_parquet:
            scored_path = results_dir / files_config.get("post_score", "post_score.csv")
            self._save_scored_csv(scored_molecules, scored_path)
        
        # Save final diverse set with continuous ligand numbering
        if "final_diverse" in results:
            # Save as CSV file (publication-ready format)
            if self.vanilla_mode:
                final_csv_path = results_dir / files_config.get("final_dataset_vanilla", "hits_vanilla.csv")
                # Vanilla starts at ligand_id 1
                self._save_scored_csv(results["final_diverse"], final_csv_path, start_ligand_id=1)
            # NOTE: hits.csv (CAFE primary) removed - we have per-weight sets instead
        
        # Save per-weight results if available (with continuous numbering)
        if "per_weight" in results and not self.vanilla_mode:
            cafe_config = self.gen_config.get("selection", {}).get("cafe", {})
            if cafe_config.get("save_per_weight", True):
                # Start after vanilla (1-100) and primary CAFE (101-200)
                start_id = 201
                for lambda_cafe, weight_results in results["per_weight"].items():
                    weight_csv_path = results_dir / f"hits_lambda_{lambda_cafe:.2f}.csv"
                    self._save_scored_csv(weight_results["final_diverse"], weight_csv_path, start_ligand_id=start_id)
                    start_id += 100  # Each set has 100 molecules

        # Save summary
        if self.vanilla_mode:
            summary_path = results_dir / files_config.get("summary_vanilla", "generation_summary_vanilla.json")
        else:
            summary_path = results_dir / files_config.get("summary", "generation_summary.json")
        self._save_summary(summary_path)
        
        # CAFE Ez scores no longer needed (using old ac_enrichment logic)
        
        # Save additional monitoring files
        if "coverage_stats" in files_config:
            coverage_path = results_dir / files_config["coverage_stats"]
            self._save_coverage_stats(coverage_path)
        
    def create_hits_for_docking(self, results_dir: Path) -> None:
        """
        Create unified hits_for_docking.csv with unique ligands from:
        - Vanilla primary set
        - All CAFE per-weight sets
        
        Removes duplicates by SMILES to avoid docking the same molecule twice.
        Uses original ligand_id from datasets (no renumbering).
        """
        files_config = self.gen_config.get("output", {}).get("files", {})
        docking_path = results_dir / files_config.get("hits_for_docking", "hits_for_docking.csv")
        
        all_ligands = []
        seen_smiles = set()
        
        # Load vanilla set (if exists)
        vanilla_path = results_dir / files_config.get("final_dataset_vanilla", "hits_vanilla.csv")
        if vanilla_path.exists():
            vanilla_df = pd.read_csv(vanilla_path)
            for _, row in vanilla_df.iterrows():
                smiles = str(row.get("SMILES", ""))
                if smiles and smiles not in seen_smiles:
                    row_dict = row.to_dict()
                    original_ligand_id = row_dict.get("ligand_id")  # Use original ligand_id
                    row_dict["ligand_id"] = original_ligand_id  # Keep original ligand_id
                    row_dict["source"] = "vanilla"
                    all_ligands.append(row_dict)
                    seen_smiles.add(smiles)
        
        # Load all per-weight CAFE sets
        cafe_config = self.gen_config.get("selection", {}).get("cafe", {})
        lambda_list = cafe_config.get("lambda_list", [])
        for lambda_cafe in lambda_list:
            weight_path = results_dir / f"hits_lambda_{lambda_cafe:.2f}.csv"
            if weight_path.exists():
                weight_df = pd.read_csv(weight_path)
                for _, row in weight_df.iterrows():
                    smiles = str(row.get("SMILES", ""))
                    if smiles and smiles not in seen_smiles:
                        row_dict = row.to_dict()
                        original_ligand_id = row_dict.get("ligand_id")  # Use original ligand_id
                        row_dict["ligand_id"] = original_ligand_id  # Keep original ligand_id
                        row_dict["source"] = f"CAFE_lambda_{lambda_cafe:.2f}"
                        all_ligands.append(row_dict)
                        seen_smiles.add(smiles)
        
        # Save unified set
        if all_ligands:
            df = pd.DataFrame(all_ligands)
            # Ensure ligand_id is first column, then source, then rest
            priority_cols = ["ligand_id", "source"]
            other_cols = [c for c in df.columns if c not in priority_cols]
            cols = priority_cols + other_cols
            df = df[cols]
            df.to_csv(docking_path, index=False)
            print(f"   Created hits_for_docking.csv with {len(all_ligands)} unique ligands (no duplicates)")
        else:
            print("   Warning: No ligands found to create hits_for_docking.csv")
    
    def _save_generated_pool(self, molecules: List[Dict[str, Any]], filepath: Path) -> None:
        """
        Save full pool of unique molecules to Parquet with essential metadata only.
        
        Columns: smiles, core, arm, fragments_used, qsar_prob, sa, qed, score
        (Removed: qsar_prob_raw, cafe_enrichment, cafe_boost, attempts, generation_time, valid_flags)
        """
        rows = []
        for mol in molecules:
            row = {
                "smiles": mol.get("smiles", ""),
                "core": mol.get("core", ""),
                "arm": "vanilla" if self.vanilla_mode else "CAFE",
                "fragments_used": "; ".join(mol.get("fragments_used", [])),
                "qsar_prob": mol.get("qsar_prob", 0.0),
                "sa": mol.get("sa", 0.0),
                "qed": mol.get("qed", 0.0),
                "score": mol.get("score", 0.0)
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_parquet(filepath, index=False, engine='pyarrow')
        
        if self.logger.level <= logging.INFO:
            print(f"   Saved {len(rows)} molecules to {filepath}")
    
    def _save_scored_csv(self, molecules: List[Dict[str, Any]], filepath: Path, start_ligand_id: int = 1) -> None:
        """
        Save scored molecules to CSV file in publication-ready format.
        
        Args:
            molecules: List of molecule dictionaries
            filepath: Path to save CSV file
            start_ligand_id: Starting ligand ID for continuous numbering (default: 1)
        
        Output columns:
        - ligand_id: Unique ligand identifier (continuous numbering)
        - SMILES: Canonical SMILES string
        - core: Core scaffold SMARTS pattern
        - fragments: Fragment SMILES with (AC) flags for AC_added fragments
        - qsar: CAFE LATE-adjusted QSAR score (activity prediction)
        - qed: Quantitative Estimate of Drug-likeness
        - sa: Synthetic Accessibility score (normalized)
        - score: Aggregate multi-objective score
        - cafe_enrichment: Total SALI AC enrichment from AC_added fragments
        - cafe_boost: QSAR adjustment from CAFE LATE
        - rank: Final ranking (1 = best)
        """
        # Build rows with required format
        rows = []
        ligand_id = start_ligand_id
        for mol in molecules:
            smiles = mol.get("smiles", "")
            core = mol.get("core", "")
            fragments_used = mol.get("fragments_used", [])
            
            # Format fragments with AC flags: "fragment1 (AC)" or "fragment1"
            fragments_str = "; ".join([
                f"{frag} (AC)" if frag in self.ac_enriched_fragments else frag
                for frag in fragments_used
            ])
            
            # Get scores - use CAFE-adjusted qsar_prob (falls back to qsar_prob_raw if CAFE not used)
            qsar_raw = mol.get("qsar_prob_raw", 0.0)
            qsar_adjusted = mol.get("qsar_prob", qsar_raw)
            qed_val = mol.get("qed", 0.0)
            sa_val = mol.get("sa", 0.0)
            score_val = mol.get("score", 0.0)  # Aggregate score (CAFE-adjusted QSAR + SA + QED)
            rank_val = mol.get("rank", None)   # Rank from ranking step (1 = highest score)
            
            # Extract CAFE LATE metadata
            cafe_enrichment = mol.get("cafe_enrichment", 0.0)  # Total ac_enrichment from unique fragments
            cafe_boost = mol.get("cafe_bonus", 0.0)  # CAFE bonus = lambda_cafe * boost_factor
            
            row = {
                "ligand_id": ligand_id,
                "SMILES": smiles,
                "core": core,
                "fragments": fragments_str,
                "qsar": qsar_adjusted,
                "qed": qed_val,
                "sa": sa_val,
                "score": score_val,
                "cafe_enrichment": cafe_enrichment,  # Total SALI AC enrichment from AC_added fragments (CAFE LATE)
                "cafe_boost": cafe_boost,             # QSAR adjustment from CAFE LATE
                "rank": rank_val if rank_val is not None else ""
            }
            rows.append(row)
            ligand_id += 1
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        # File saved, no need for verbose logging
    
    def _save_summary(self, filepath: Path) -> None:
        """Save generation summary."""
        # Convert datetime objects to strings for JSON serialization
        stats_copy = self.stats.copy()
        if stats_copy.get("start_time"):
            stats_copy["start_time"] = stats_copy["start_time"].isoformat()
        if stats_copy.get("end_time"):
            stats_copy["end_time"] = stats_copy["end_time"].isoformat()
        
        summary = {
            "generation_stats": stats_copy,
            "config": self.gen_config,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _save_coverage_stats(self, filepath: Path) -> None:
        """Save coverage statistics."""
        if hasattr(self, 'sampler') and self.sampler:
            coverage_stats = self.sampler.get_coverage_stats()
        else:
            coverage_stats = {}
        
        with open(filepath, 'w') as f:
            json.dump(coverage_stats, f, indent=2)
    
    
    def _log_final_statistics(self) -> None:
        """Log final generation statistics."""
        duration = (self.stats["end_time"] - self.stats["start_time"]).total_seconds()
        
        print("Generation complete.")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Generated: {self.stats['generated']}")
        print(f"Dropped by floors: {self.stats['dropped_floors']}")
        print(f"Final size: {self.stats['final_size']}")
        # Log breakdown and reasons
        if self.stats.get("rejection_reasons"):
            print("Rejection breakdown:")
            for reason, count in sorted(self.stats["rejection_reasons"].items(), key=lambda x: -x[1]):
                print(f"   {reason}: {count}")


def run_generator(config: Dict[str, Any], logger: Optional[logging.Logger] = None, vanilla_mode: bool = False) -> Dict[str, Any]:
    """
    Run the molecular generation pipeline.
    
    Args:
        config: Configuration dictionary
        logger: Optional logger instance
        vanilla_mode: If True, use vanilla fragments and disable CAFE scoring
        
    Returns:
        Dictionary containing generation results
    """
    generator = ACGenerator(config, logger, vanilla_mode=vanilla_mode)
    generator.load_data()
    return generator.generate_molecules()