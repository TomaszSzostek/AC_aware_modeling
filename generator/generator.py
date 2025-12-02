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
import yaml
import json
from datetime import datetime
import time
from multiprocessing import Pool, cpu_count
from functools import partial
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
        
        # Initialize selector
        self.selector = MolecularSelector(
            config=self.gen_config.get("selection", {}),
            logger=self.logger
        )
    
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
        """Score molecules using QSAR, SA, and QED with multiprocessing."""
        if not molecules:
            return []
        
        # Get multiprocessing configuration
        mp_config = self.gen_config.get("generation", {}).get("multiprocessing", {})
        enable_mp = mp_config.get("enable", True)
        max_cores = mp_config.get("max_cores", 8)
        threshold = mp_config.get("scoring_threshold", 500)
        
        # Use multiprocessing for large batches
        n_molecules = len(molecules)
        n_cores = min(cpu_count(), max_cores) if max_cores > 0 else cpu_count()  # Use all cores if max_cores is 0 or negative
        
        if enable_mp and n_molecules > threshold and n_cores > 1:
            print(f"   Using {n_cores} cores for scoring ({n_molecules} molecules)")
            return self._score_molecules_parallel(molecules, n_cores)
        else:
            return self._score_molecules_sequential(molecules)
    
    def _score_molecules_sequential(self, molecules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score molecules sequentially (for small batches)."""
        scored_molecules = []
        
        for mol in tqdm.tqdm(molecules, desc="Scoring", unit="mol"):
            try:
                # Pass fragments_used to enable CAFE scoring
                fragments_used = mol.get("fragments_used", [])
                scores = self.scorer.score_molecule(mol["smiles"], fragments_used=fragments_used)
                mol.update(scores)
                
                # Apply floors and filters
                passed, reason = self._passes_floors(mol)
                if passed:
                    scored_molecules.append(mol)
                else:
                    self.stats["dropped_floors"] += 1
                    key = f"floor:{reason}"
                    self.stats["rejection_reasons"][key] = self.stats["rejection_reasons"].get(key, 0) + 1
                    
            except Exception as e:
                self.logger.warning(f"Failed to score molecule {mol.get('smiles', 'unknown')}: {e}")
                continue
        
        return scored_molecules
    
    def _score_molecules_parallel(self, molecules: List[Dict[str, Any]], n_cores: int) -> List[Dict[str, Any]]:
        """Score molecules in parallel using multiprocessing."""
        # Split molecules into chunks
        chunk_size = max(1, len(molecules) // n_cores)
        molecule_chunks = [molecules[i:i + chunk_size] for i in range(0, len(molecules), chunk_size)]
        
        # Create partial function with config (scorer will be recreated in each worker)
        score_chunk = partial(self._score_molecule_chunk, self.gen_config, self.original_smiles_set)
        
        # Process chunks in parallel
        with Pool(n_cores) as pool:
            results = list(tqdm.tqdm(
                pool.imap(score_chunk, molecule_chunks),
                total=len(molecule_chunks),
                desc="Scoring (Parallel)",
                unit="chunk"
            ))
        
        # Combine results
        scored_molecules = []
        for chunk_results in results:
            for mol, passed_floors, reason in chunk_results:
                if passed_floors:
                    scored_molecules.append(mol)
                else:
                    self.stats["dropped_floors"] += 1
                    key = f"floor:{reason}"
                    self.stats["rejection_reasons"][key] = self.stats["rejection_reasons"].get(key, 0) + 1
        
        return scored_molecules
    
    @staticmethod
    def _score_molecule_chunk(config, original_smiles_set: set, molecule_chunk: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], bool, str]]:
        """
        Score a chunk of molecules (for multiprocessing).
        
        Note: Creates scorer instance within worker to avoid pickling issues with CAFE LATE.
        """
        # Import here to avoid issues
        from .scorer import MolecularScorer
        import logging
        
        # Create scorer in worker process (avoids pickle issues)
        scoring_cfg = config.get("scoring", {})
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.ERROR)  # Reduce verbosity in workers
        
        scorer = MolecularScorer(
            qsar_model_path=scoring_cfg.get("qsar_model_path"),
            config=scoring_cfg,
            logger=logger
        )
        
        results = []
        qed_floor = scoring_cfg.get("qed_floor", 0.1)
        # novelty disabled – exact-match only against original dataset
        
        for mol in molecule_chunk:
            try:
                # Pass fragments_used to enable CAFE LATE scoring
                fragments_used = mol.get("fragments_used", [])
                scores = scorer.score_molecule(mol["smiles"], fragments_used=fragments_used)
                mol.update(scores)
                
                # Apply floors and filters
                if mol.get("qed", 0) < qed_floor:
                    results.append((mol, False, "qed"))
                    continue
                # Drop if molecule exists exactly in original dataset
                if mol.get("smiles") in original_smiles_set:
                    results.append((mol, False, "in_dataset"))
                    continue
                results.append((mol, True, "ok"))
                    
            except Exception as e:
                # Skip failed molecules
                continue
        
        return results
    
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
    
    def _select_final_molecules(self, molecules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select final diverse set of molecules."""
        # Apply selection funnel
        selected = self.selector.select_molecules(molecules)
        
        # Update statistics
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
        
        return selected
    
    def _save_results(self, results: Dict[str, Any], scored_molecules: List[Dict[str, Any]] = None) -> None:
        """Save all results to files."""
        output_config = self.gen_config.get("output", {})
        results_dir = Path(output_config.get("results_dir", "results/Generator"))
        results_dir.mkdir(parents=True, exist_ok=True)
        
        files_config = output_config.get("files", {})
        
        # Save scored molecules CSV (optional)
        if scored_molecules:
            scored_path = results_dir / files_config.get("post_score", "post_score.csv")
            self._save_scored_csv(scored_molecules, scored_path)
        
        # Save final diverse set
        if "final_diverse" in results:
            # Save as CSV file (publication-ready format)
            if self.vanilla_mode:
                final_csv_path = results_dir / files_config.get("final_dataset_vanilla", "hits_vanilla.csv")
            else:
                final_csv_path = results_dir / files_config.get("final_dataset", "hits.csv")
            self._save_scored_csv(results["final_diverse"], final_csv_path)

        # Save summary
        if self.vanilla_mode:
            summary_path = results_dir / files_config.get("summary_vanilla", "generation_summary_vanilla.json")
        else:
            summary_path = results_dir / files_config.get("summary", "generation_summary.json")
        self._save_summary(summary_path)
        
        # Save additional monitoring files
        if "coverage_stats" in files_config:
            coverage_path = results_dir / files_config["coverage_stats"]
            self._save_coverage_stats(coverage_path)
        
        if "fragment_usage" in files_config:
            fragment_path = results_dir / files_config["fragment_usage"]
            self._save_fragment_usage(fragment_path)
        
        if "core_usage" in files_config:
            core_path = results_dir / files_config["core_usage"]
            self._save_core_usage(core_path)
    
    def _save_scored_csv(self, molecules: List[Dict[str, Any]], filepath: Path) -> None:
        """
        Save scored molecules to CSV file in publication-ready format.
        
        Output columns:
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
            cafe_enrichment = 0.0
            cafe_boost = 0.0
            if "cafe" in mol and mol["cafe"].get("cafe_enabled"):
                cafe_data = mol["cafe"]
                cafe_enrichment = cafe_data.get("total_enrichment", 0.0)
                cafe_boost = cafe_data.get("adjustment", 0.0)
            
            row = {
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
    
    def _save_fragment_usage(self, filepath: Path) -> None:
        """Save fragment usage statistics."""
        if hasattr(self, 'sampler') and self.sampler:
            fragment_attempts = self.sampler.fragment_attempts
            fragment_list = self.sampler.fragment_list
            
            usage_data = []
            for i, (fragment, attempts) in enumerate(zip(fragment_list, fragment_attempts)):
                usage_data.append({
                    "fragment_index": i,
                    "fragment_smiles": fragment,
                    "attempts": int(attempts),
                    "usage_percentage": float(attempts / max(fragment_attempts) * 100) if max(fragment_attempts) > 0 else 0
                })
            
            df = pd.DataFrame(usage_data)
            df.to_csv(filepath, index=False)
    
    def _save_core_usage(self, filepath: Path) -> None:
        """Save core usage statistics."""
        if hasattr(self, 'molecular_generator') and self.molecular_generator:
            cores = self.molecular_generator.cores
            core_usage = self.molecular_generator.generation_stats.get("core_usage", {})
            
            core_data = []
            for i, core in enumerate(cores):
                core_data.append({
                    "core_index": i,
                    "core_smarts": core,
                    "usage_count": core_usage.get(i, 0),
                    "usage_percentage": float(core_usage.get(i, 0) / max(core_usage.values()) * 100) if max(core_usage.values()) > 0 else 0
                })
            
            df = pd.DataFrame(core_data)
            df.to_csv(filepath, index=False)
    
    
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