"""
CAFE LATE (Late-stage Activity Cliff Fragment Enrichment) Scoring Module

This module implements CAFE LATE - a post-generation scoring adjustment that leverages
activity cliff knowledge to correct QSAR predictions for molecules containing AC-enriched fragments.

CAFE LATE Concept:
- QSAR provides baseline activity prediction (may not recognize AC patterns)
- CAFE LATE identifies AC_added fragments in the generated molecule
- CAFE LATE adjusts QSAR score based on fragment AC enrichment (SALI scores)
- Stronger AC enrichment = higher confidence boost

CAFE LATE operates only on fragments with selected_by == "AC_added" from CAFE analysis.
These are fragments that were specifically added by CAFE due to their activity cliff enrichment,
not fragments that vanilla QSAR would have selected anyway (selected_by == "threshold").

Key Principle:
CAFE LATE rewards only the novel knowledge from activity cliffs, not the baseline QSAR knowledge.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from rdkit import Chem


class CAFEScorer:
    """
    CAFE LATE scorer - adjusts QSAR predictions based on AC-enriched fragments.
    
    CAFE LATE (Late-stage Activity Cliff Fragment Enrichment) is a post-generation
    scoring correction that leverages activity cliff knowledge to improve QSAR predictions.
    
    The scorer analyzes which CAFE-added fragments (selected_by == "AC_added") are present
    in a molecule and adjusts the QSAR prediction based on the strength of AC enrichment.
    
    Philosophy:
    - QSAR may not recognize subtle activity cliff patterns
    - CAFE LATE provides targeted corrections for molecules with AC_added fragments
    - Only rewards novel AC knowledge, not baseline QSAR knowledge
    """
    
    def __init__(self, config: Dict, logger: Optional[logging.Logger] = None):
        """
        Initialize CAFE LATE scorer.
        
        Args:
            config: Configuration dictionary with CAFE scoring parameters
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # CAFE parameters from config
        self.enable_cafe = config.get("enable_cafe_scoring", True)
        self.cafe_weight = config.get("cafe_weight", 0.3)  # Max 30% adjustment
        self.enrichment_threshold = config.get("enrichment_threshold", 1.0)
        
        # Load CAFE fragment data
        self.cafe_fragments: Dict[str, Dict] = {}
        self._load_cafe_fragments(config)
    
    def _load_cafe_fragments(self, config: Dict) -> None:
        """
        Load CAFE LATE fragment data from selected_fragments_with_ACflag.csv.
        
        Loads only fragments with selected_by == "AC_added" and their AC enrichment data.
        These are fragments specifically added by CAFE analysis, not by vanilla QSAR.
        """
        try:
            # Get fragment file path from config
            paths_config = config.get("paths", {})
            frag_root = Path(paths_config.get("fragments", "results/reverse_QSAR"))
            
            # Try to find the selected_fragments_with_ACflag.csv
            ac_csv = None
            summary_path = frag_root / "summary.json"
            
            if summary_path.exists():
                import json
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                best_model = summary.get("best_model")
                if best_model:
                    ac_csv = frag_root / best_model / "selected_fragments_with_ACflag.csv"
            
            # Fallback: try any model directory
            if not ac_csv or not ac_csv.exists():
                for model_dir in frag_root.iterdir():
                    if model_dir.is_dir():
                        potential_csv = model_dir / "selected_fragments_with_ACflag.csv"
                        if potential_csv.exists():
                            ac_csv = potential_csv
                            break
            
            if not ac_csv or not ac_csv.exists():
                self.logger.warning("CAFE fragments file not found - CAFE scoring disabled.")
                self.enable_cafe = False
                return
            
            # Load fragment data
            if self.logger.level <= logging.INFO:
                self.logger.info(f"Loading CAFE fragments from {ac_csv}")
            df = pd.read_csv(ac_csv)
            
            # Filter for AC_added fragments only
            if "selected_by" not in df.columns:
                if self.logger.level <= logging.WARNING:
                    self.logger.warning("'selected_by' column not found - CAFE scoring disabled.")
                self.enable_cafe = False
                return
            
            ac_added = df[df["selected_by"] == "AC_added"].copy()
            
            # Build fragment lookup dictionary
            for _, row in ac_added.iterrows():
                frag_smiles = row["fragment_smiles"]
                
                # Store fragment metadata
                self.cafe_fragments[frag_smiles] = {
                    "ac_enrichment": row.get("ac_enrichment", 0.0),
                    "n_cliff_pairs": row.get("n_cliff_pairs", 0),
                    "cliff_pairs_ac_enrichment": row.get("cliff_pairs_ac_enrichment", ""),
                    "importance": row.get("importance", 0.0)
                }
            
            if self.logger.level <= logging.INFO:
                self.logger.info(f"   Loaded {len(self.cafe_fragments)} CAFE-added fragments.")
            if len(self.cafe_fragments) == 0:
                self.enable_cafe = False
                if self.logger.level <= logging.WARNING:
                    self.logger.warning("No AC_added fragments found - CAFE scoring disabled.")
            
        except Exception as e:
            self.logger.error(f"Failed to load CAFE fragments: {e}")
            self.enable_cafe = False
    
    def calculate_cafe_boost(self, fragments_used: List[str]) -> Tuple[float, Dict]:
        """
        Calculate CAFE LATE boost factor based on AC-enriched fragments present.
        
        Uses ac_enrichment values directly from CAFE analysis (SALI-based scores).
        ac_enrichment is already a computed score that doesn't need additional scaling.
        
        CAFE LATE only considers fragments with selected_by == "AC_added" - fragments
        that were specifically added due to activity cliff enrichment, not those that
        vanilla QSAR would have selected anyway.
        
        Args:
            fragments_used: List of fragment SMILES used in the molecule
            
        Returns:
            Tuple of (boost_factor, cafe_metadata)
            - boost_factor: Value between 0 and 1 representing CAFE LATE confidence
            - cafe_metadata: Dictionary with CAFE LATE analysis details
        """
        if not self.enable_cafe or not fragments_used:
            return 0.0, {"cafe_enabled": False}
        
        # Identify which CAFE fragments (AC_added only) are present
        cafe_fragments_present = []
        total_enrichment = 0.0
        
        for frag in fragments_used:
            if frag in self.cafe_fragments:
                frag_data = self.cafe_fragments[frag]
                enrichment = frag_data["ac_enrichment"]
                n_pairs = frag_data["n_cliff_pairs"]
                
                # Use ac_enrichment directly - it's already a computed SALI-based score
                # No additional scaling needed - CAFE already computed the optimal score
                contribution = enrichment
                
                cafe_fragments_present.append({
                    "fragment": frag,
                    "ac_enrichment": enrichment,
                    "n_cliff_pairs": n_pairs,
                    "contribution": contribution
                })
                
                total_enrichment += contribution
        
        # Normalize to 0-1 range
        # Use sigmoid-like function to map enrichment to boost factor
        if total_enrichment > 0:
            # Normalize by typical enrichment values for AC_added fragments
            # AC_added fragments typically have enrichment in range 5-50
            normalized_enrichment = total_enrichment / 30.0
            
            # Apply sigmoid to get boost factor in (0, 1)
            boost_factor = 1.0 - np.exp(-normalized_enrichment)
            
            # Cap at max weight
            boost_factor = min(boost_factor, 1.0)
        else:
            boost_factor = 0.0
        
        # Prepare metadata
        metadata = {
            "cafe_enabled": True,
            "n_cafe_fragments": len(cafe_fragments_present),
            "cafe_fragments": cafe_fragments_present,
            "total_enrichment": float(total_enrichment),
            "boost_factor": float(boost_factor)
        }
        
        return boost_factor, metadata
    
    def adjust_qsar_score(self, qsar_score: float, fragments_used: List[str]) -> Tuple[float, Dict]:
        """
        Adjust QSAR score using CAFE LATE knowledge.
        
        CAFE LATE Strategy:
        - If QSAR predicts low activity but molecule has strong AC_added fragments,
          boost the score (CAFE LATE knows these fragments increase activity in cliffs)
        - If QSAR predicts high activity and has AC_added fragments,
          give a smaller boost (confirmation of activity)
        
        Rationale:
        QSAR may not recognize activity cliff patterns because it was trained on
        diverse data. CAFE LATE provides targeted corrections based on AC knowledge.
        
        Args:
            qsar_score: Base QSAR prediction (0-1, probability of being active)
            fragments_used: List of fragment SMILES used in the molecule
            
        Returns:
            Tuple of (adjusted_score, cafe_late_metadata)
        """
        if not self.enable_cafe:
            return qsar_score, {"cafe_enabled": False}
        
        # Calculate CAFE boost
        boost_factor, metadata = self.calculate_cafe_boost(fragments_used)
        
        if boost_factor == 0.0:
            # No CAFE fragments present
            metadata["adjusted_score"] = qsar_score
            metadata["adjustment"] = 0.0
            return qsar_score, metadata
        
        # Calculate adjustment based on current QSAR score
        # Key insight: CAFE boost is more powerful for low QSAR scores
        # because CAFE fragments come from activity cliffs where small changes
        # cause large activity differences
        
        initial_score = qsar_score

        if qsar_score < 0.5:
            # Low QSAR prediction - strong CAFE boost
            # Pull score towards activity threshold (0.5-0.8 range)
            cafe_target = 0.75  # Target score for strong CAFE fragments (increased from 0.65)
            weight = self.cafe_weight * boost_factor
        else:
            # High QSAR prediction - moderate CAFE boost (confirmation)
            cafe_target = 0.85  # Boost towards high confidence
            weight = self.cafe_weight * 0.5 * boost_factor  # Reduced weight
        
        # Weighted average between QSAR and CAFE target
        adjusted_score = (1 - weight) * qsar_score + weight * cafe_target
        
        # Ensure score stays in [0, 1]
        adjusted_score = np.clip(adjusted_score, 0.0, 1.0)

        # CAFE should never penalize the score – clamp to original if adjustment is negative
        if adjusted_score < initial_score:
            adjusted_score = initial_score
            weight = 0.0
        
        # Add adjustment info to metadata
        adjustment = adjusted_score - qsar_score
        metadata["qsar_score"] = float(qsar_score)
        metadata["adjusted_score"] = float(adjusted_score)
        metadata["adjustment"] = float(adjustment)
        metadata["weight_used"] = float(weight)
        
        return adjusted_score, metadata
    
    def get_fragment_info(self, fragment_smiles: str) -> Optional[Dict]:
        """
        Get CAFE LATE information for a specific fragment.
        
        Args:
            fragment_smiles: Fragment SMILES string
            
        Returns:
            Dictionary with fragment CAFE data or None if not an AC_added fragment
        """
        return self.cafe_fragments.get(fragment_smiles)
    
    def is_cafe_fragment(self, fragment_smiles: str) -> bool:
        """Check if a fragment is AC_added by CAFE."""
        return fragment_smiles in self.cafe_fragments
    
    def get_stats(self) -> Dict:
        """Get CAFE LATE scorer statistics."""
        return {
            "enabled": self.enable_cafe,
            "n_cafe_fragments": len(self.cafe_fragments),
            "cafe_weight": self.cafe_weight,
            "enrichment_threshold": self.enrichment_threshold
        }

