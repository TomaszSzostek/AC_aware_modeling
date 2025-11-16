"""
Multi-Objective Molecular Scorer Module

Implements comprehensive molecular scoring using multiple complementary metrics:
- QSAR: Quantitative Structure-Activity Relationship prediction using pre-trained models
- SA: Synthetic Accessibility score for synthetic feasibility assessment  
- QED: Quantitative Estimate of Drug-likeness for drug-likeness evaluation

The scorer integrates these metrics into a weighted composite score that balances
activity potential, synthetic feasibility, and drug-likeness properties. It uses
pre-trained QSAR models (CatBoost with ECFP fingerprints) for accurate activity prediction.

Scoring Pipeline:
1. QSAR: predict_proba_active(smiles) from best evaluation model
2. SA: Invert/normalize SA with clamp to [0,1] using sa_max
3. QED: Direct score in [0,1] range
4. Apply floors (qed_floor) and novelty guard (drop if sim ≥ novelty_guard)
5. Aggregate: score = w_qsar * QSAR_prob + w_sa * SA + w_qed * QED

"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import joblib
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, rdFingerprintGenerator
from rdkit import DataStructs
import pickle
from pathlib import Path
import warnings
import os

# Import CAFE LATE scorer
from .cafe_scorer import CAFEScorer

# Suppress Morgan fingerprint deprecation warnings


class MolecularScorer:
    """
    Molecular scorer for post-gate evaluation.
    
    Implements scoring using QSAR, SA, and QED with aggregation and filtering.
    """
    
    def __init__(self, qsar_model_path: str, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the molecular scorer.
        
        Args:
            qsar_model_path: Path to the trained QSAR model
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.qsar_model_path = qsar_model_path
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.qsar_weight = config.get("qsar_weight", 0.5)
        self.sa_weight = config.get("sa_weight", 0.3)
        self.qed_weight = config.get("qed_weight", 0.2)
        self.sa_max = config.get("sa_max", 10.0)
        self.qed_floor = config.get("qed_floor", 0.1)
        
        # Load QSAR model
        self.qsar_model = None
        self.qsar_scaler = None
        self.qsar_imputer = None
        self.qsar_feature_names = None
        self._load_qsar_model()
        
        # Initialize CAFE LATE scorer
        self.cafe_scorer = CAFEScorer(config, logger)
        # Only print if logger level is INFO or lower (not in worker processes)
        if logger.level <= logging.INFO:
            if self.cafe_scorer.enable_cafe:
                print(f"   CAFE LATE scoring enabled with {len(self.cafe_scorer.cafe_fragments)} AC-added fragments")
            else:
                print("   CAFE LATE scoring disabled")
        
        # Scorer initialized
    
    def _load_qsar_model(self) -> None:
        """Load the trained QSAR model."""
        try:
            # Load the main model
            self.qsar_model = joblib.load(self.qsar_model_path)
            if self.logger.level <= logging.INFO:
                print(f"   Loaded QSAR model from {self.qsar_model_path}")
            
            # Determine model type from best_overall.itxt
            self.model_type = self._determine_model_type()
            if self.logger.level <= logging.INFO:
                print(f"   Model type: {self.model_type}")
            
            # Load selected features (after variance filtering)
            self.selected_features = self._load_selected_features()
            self.selected_indices = self._parse_feature_indices()
            if self.selected_indices and self.logger.level <= logging.INFO:
                print(f"   Selected features: {len(self.selected_indices)} / {self._get_full_fp_size()} bits")
                
        except Exception as e:
            self.logger.error(f"Failed to load QSAR model: {e}")
            self.qsar_model = None
            self.model_type = None
            self.selected_features = None
            self.selected_indices = None
    
    def _determine_model_type(self) -> str:
        """Determine model type from best_overall.itxt file."""
        try:
            info_path = Path("results/predictive_model/best_overall.itxt")
            if info_path.exists():
                with open(info_path, 'r') as f:
                    content = f.read()
                    for line in content.split('\n'):
                        if line.startswith('Backbone:'):
                            backbone = line.split(':', 1)[1].strip()
                            return backbone
        except Exception as e:
            self.logger.debug(f"Could not determine model type: {e}")
        
        # Default to ecfp1024 if we can't determine
        return "ecfp1024"
    
    def _load_selected_features(self) -> Optional[List[str]]:
        """Load selected features after variance filtering."""
        try:
            if not self.model_type:
                return None
            
            selected_path = Path(f"results/predictive_model/{self.model_type}/selected_features.txt")
            if selected_path.exists():
                with open(selected_path, 'r') as f:
                    features = [line.strip() for line in f if line.strip()]
                return features
            else:
                self.logger.warning(f"Selected features file not found: {selected_path}")
                return None
        except Exception as e:
            self.logger.debug(f"Could not load selected features: {e}")
            return None
    
    def _parse_feature_indices(self) -> Optional[List[int]]:
        """
        Parse feature indices from feature names.
        
        Examples:
            FP::MACCS_0042 -> 42 (but we need index 41 for 0-based array after removing bit 0)
            FP::ECFP1024_0123 -> 123
        """
        if not self.selected_features:
            return None
        
        try:
            indices = []
            for feat_name in self.selected_features:
                if 'MACCS' in feat_name:
                    # FP::MACCS_0001 -> bit 1, but in array after removing bit 0, it's index 0
                    bit_num = int(feat_name.split('_')[-1])
                    indices.append(bit_num - 1)  # Convert to 0-based after removing bit 0
                elif 'ECFP' in feat_name:
                    # FP::ECFP1024_0123 -> index 123
                    idx = int(feat_name.split('_')[-1])
                    indices.append(idx)
                else:
                    # Unknown format, skip
                    continue
            
            return indices if indices else None
        except Exception as e:
            self.logger.error(f"Failed to parse feature indices: {e}")
            return None
    
    def _get_full_fp_size(self) -> int:
        """Get full fingerprint size based on model type."""
        if not self.model_type:
            return 1024
        
        if 'ecfp1024' in self.model_type.lower():
            return 1024
        elif 'ecfp2048' in self.model_type.lower():
            return 2048
        elif 'maccs' in self.model_type.lower():
            return 166  # MACCS after removing bit 0
        else:
            return 1024  # Default
    
    def _load_descriptors_list(self) -> List[str]:
        """Load selected descriptors list for descriptors model."""
        try:
            # Use selected_features.txt (unified format for all backbones)
            # For descriptors, it contains DESC:: prefixed names
            if self.selected_features:
                # Filter to only DESC:: prefixed features
                descriptors = [f for f in self.selected_features if f.startswith("DESC::")]
                if descriptors:
                    return descriptors
            
            # Fallback: try old selected_descriptors.txt (backward compatibility)
            descriptors_path = "results/predictive_model/descriptors/selected_descriptors.txt"
            if os.path.exists(descriptors_path):
                with open(descriptors_path, 'r') as f:
                    descriptors = [line.strip() for line in f if line.strip()]
                return descriptors
            else:
                self.logger.warning(f"Descriptors not found in selected_features.txt and {descriptors_path} not found")
                return []
        except Exception as e:
            self.logger.error(f"Failed to load descriptors: {e}")
            return []
    
    def _compute_descriptors(self, mol: Chem.Mol, descriptors: List[str]) -> np.ndarray:
        """Compute selected descriptors for a molecule."""
        try:
            features = []
            for desc_name in descriptors:
                try:
                    # Extract descriptor name (remove DESC:: prefix if present)
                    clean_name = desc_name.replace("DESC::", "")
                    
                    # Get descriptor function from RDKit
                    desc_func = getattr(Descriptors, clean_name, None)
                    if desc_func:
                        value = desc_func(mol)
                        features.append(float(value) if not np.isnan(value) else 0.0)
                    else:
                        self.logger.warning(f"Descriptor {clean_name} not found in RDKit")
                        features.append(0.0)
                except Exception as e:
                    self.logger.debug(f"Failed to compute descriptor {desc_name}: {e}")
                    features.append(0.0)
            
            return np.array(features).reshape(1, -1)
        except Exception as e:
            self.logger.error(f"Failed to compute descriptors: {e}")
            # Return zero vector as fallback
            return np.zeros((1, len(descriptors)))
    
    
    
    def score_molecule(self, smiles: str, fragments_used: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Score a single molecule with optional CAFE LATE adjustment.
        
        Args:
            smiles: SMILES string of the molecule
            fragments_used: Optional list of fragment SMILES used to build this molecule
            
        Returns:
            Dictionary containing all scores and CAFE LATE metadata
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return self._create_error_result("invalid_molecule")
            
            # Compute individual scores
            qsar_prob_raw = self._compute_qsar_score(mol)
            sa_score = self._compute_sa_score(mol)
            qed_score = self._compute_qed_score(mol)
            
            # Apply CAFE LATE adjustment to QSAR score if fragments are provided
            qsar_prob = qsar_prob_raw
            cafe_metadata = {"cafe_enabled": False}
            
            if fragments_used is not None and self.cafe_scorer.enable_cafe:
                qsar_prob, cafe_metadata = self.cafe_scorer.adjust_qsar_score(
                    qsar_prob_raw, 
                    fragments_used
                )
            
            # Compute aggregate score (using CAFE LATE-adjusted QSAR if applicable)
            aggregate_score = (
                self.qsar_weight * qsar_prob +
                self.sa_weight * sa_score +
                self.qed_weight * qed_score
            )
            
            result = {
                "qsar_prob_raw": qsar_prob_raw,  # Original QSAR prediction
                "qsar_prob": qsar_prob,  # CAFE LATE-adjusted QSAR (may be same as raw)
                "sa": sa_score,
                "qed": qed_score,
                "score": aggregate_score,
                "valid_flag": True
            }
            
            # Add CAFE LATE metadata if available
            if cafe_metadata.get("cafe_enabled"):
                result["cafe"] = cafe_metadata
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Scoring failed for {smiles}: {e}")
            return self._create_error_result("scoring_error")
    
    def _compute_qsar_score(self, mol: Chem.Mol) -> float:
        """Compute QSAR probability score."""
        if self.qsar_model is None:
            return 0.5  # Default neutral score
        
        try:
            # Compute features based on detected model type
            if self.model_type == "ecfp1024":
                gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
                fp = gen.GetFingerprint(mol)
                features = np.array(fp).reshape(1, -1)
            elif self.model_type == "ecfp2048":
                gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
                fp = gen.GetFingerprint(mol)
                features = np.array(fp).reshape(1, -1)
            elif self.model_type == "maccs":
                fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
                # MACCS returns 167 bits (0-166), but we need to drop bit 0
                arr = np.zeros((fp.GetNumBits(),), dtype=np.int32)
                DataStructs.ConvertToNumpyArray(fp, arr)
                features = arr[1:].reshape(1, -1).astype(np.float32)  # Drop bit 0, keep 1-166
            elif self.model_type == "descriptors":
                # Compute selected descriptors for descriptors model
                descriptors = self._load_descriptors_list()
                if descriptors:
                    features = self._compute_descriptors(mol, descriptors)
                    # For descriptors, features are already in correct order (no index filtering needed)
                    # selected_features.txt contains the exact list of descriptors to use
                else:
                    self.logger.warning("No descriptors loaded, falling back to ECFP1024")
                    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
                    fp = gen.GetFingerprint(mol)
                    features = np.array(fp).reshape(1, -1)
            else:
                # Default to ECFP1024
                gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
                fp = gen.GetFingerprint(mol)
                features = np.array(fp).reshape(1, -1)
            
            # Apply feature selection (variance filtering) if available
            # NOTE: For descriptors, this is skipped because features are already computed
            # in the correct order from selected_features.txt
            if self.model_type != "descriptors" and self.selected_indices is not None and len(self.selected_indices) > 0:
                # Filter to selected features only (for fingerprints: MACCS, ECFP)
                features = features[:, self.selected_indices]
            
            # Predict probability directly (model is already trained and ready)
            prob = self.qsar_model.predict_proba(features)[0, 1]
            return float(prob)
            
        except Exception as e:
            self.logger.debug(f"QSAR scoring failed: {e}")
            return 0.5
    
    
    def _compute_sa_score(self, mol: Chem.Mol) -> float:
        """Compute synthetic accessibility score using RDKit SA_Score."""
        try:
            # Import the SA_Score module
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from SA_score_folder.sascorer import calculateScore
            
            # Calculate SA score (lower is better, range typically 1-10)
            sa_score = calculateScore(mol)
            
            if sa_score is None:
                return 0.5  # Default if calculation fails
            
            # Normalize to [0, 1] and invert (lower SA is better)
            # SA scores typically range from 1 (easy to synthesize) to 10 (very difficult)
            normalized_sa = 1.0 - min((sa_score - 1.0) / 9.0, 1.0)  # Map 1-10 to 1-0
            
            return float(normalized_sa)
            
        except Exception as e:
            self.logger.debug(f"SA scoring failed: {e}")
            return 0.5
    
    def _compute_qed_score(self, mol: Chem.Mol) -> float:
        """Compute QED (Quantitative Estimate of Drug-likeness) score."""
        try:
            qed_score = Descriptors.qed(mol)
            return float(qed_score)
            
        except Exception as e:
            self.logger.debug(f"QED scoring failed: {e}")
            return 0.5
    
    
    def _create_error_result(self, error_type: str) -> Dict[str, Any]:
        """Create an error result."""
        return {
            "qsar_prob_raw": 0.0,
            "sa": 0.0,
            "qed": 0.0,
            "score": 0.0,
            "valid_flag": False,
            "error": error_type
        }
    
    def get_scorer_stats(self) -> Dict[str, Any]:
        """Get scorer statistics."""
        stats = {
            "qsar_weight": self.qsar_weight,
            "sa_weight": self.sa_weight,
            "qed_weight": self.qed_weight,
            "sa_max": self.sa_max,
            "qed_floor": self.qed_floor,
            "qsar_model_loaded": self.qsar_model is not None,
            "model_type": getattr(self, 'model_type', None)
        }
        
        # Add CAFE LATE stats
        if hasattr(self, 'cafe_scorer'):
            stats["cafe_late"] = self.cafe_scorer.get_stats()
        
        return stats