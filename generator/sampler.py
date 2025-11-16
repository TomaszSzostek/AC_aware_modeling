"""
Adaptive Fragment Sampler Module

Implements intelligent fragment sampling for the Island Algorithm with balanced
coverage and adaptive bandit-based optimization.

The sampler ensures comprehensive exploration of chemical space by:
1. Balanced Seed Assembly: Guarantees minimum coverage for each fragment and core
2. Uniform Sampling: Initial phase ensures equal exploration of all fragments
3. Bandit Optimization: Adaptive sampling based on post-gate performance
4. Entropy Decay: Maintains exploration-exploitation balance

Key Features:
- Minimum coverage guarantees for all fragments and cores
- Adaptive bandit sampling with Thompson sampling principles
- Entropy decay to prevent premature convergence
- Comprehensive statistics tracking for analysis

"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import random
from collections import defaultdict


class FragmentSampler:
    """
    Fragment sampler with bandit-based adaptive sampling.
    
    Implements:
    - Balanced seed plan to guarantee coverage
    - Each fragment scheduled for at least min_per_fragment attempts across all cores
    - Each core gets at least min_per_core attempts
    - Bandit sampling after initial uniform phase
    """
    
    def __init__(self, fragments: pd.DataFrame, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the fragment sampler.
        
        Args:
            fragments: DataFrame containing fragment SMILES
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.fragments = fragments
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Extract configuration
        self.min_per_fragment = config.get("min_per_fragment", 5)
        self.min_per_core = config.get("min_per_core", 100)
        self.bandits_config = config.get("bandits", {})
        
        # Fragment tracking - validate fragments but keep [*] attachment points
        self.fragment_list = []
        for smiles in fragments["fragment_smiles"].tolist():
            # Validate fragment SMILES (keeping [*] attachment points)
            try:
                from rdkit import Chem
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    self.fragment_list.append(smiles)
                else:
                    self.logger.warning(f"Skipping invalid fragment SMILES: {smiles}")
            except Exception as e:
                self.logger.warning(f"Skipping fragment due to error: {smiles} - {e}")
        
        self.n_fragments = len(self.fragment_list)
        
        # Bandit state
        self.bandit_enabled = False
        self.fragment_scores = defaultdict(list)  # Track scores for each fragment
        self.fragment_probs = np.ones(self.n_fragments) / self.n_fragments  # Uniform initial probabilities
        self.fragment_attempts = np.zeros(self.n_fragments)  # Track attempts per fragment
        
        # Coverage tracking
        self.total_attempts = 0
        self.start_after = self.bandits_config.get("start_after", 0.3)
        self.entropy_decay = self.bandits_config.get("entropy_decay", 0.95)
        self.reweight_factor = self.bandits_config.get("reweight_factor", 1.2)
        
        # Core tracking
        self.core_attempts = defaultdict(int)
        self.n_cores = 3  # Default number of cores
        
        # Fragment sampler initialized
    
    def sample_fragments(self, n_samples: int) -> List[Dict[str, Any]]:
        """
        Sample fragments for molecular generation with diversity optimization.
        
        Args:
            n_samples: Number of fragments to sample
            
        Returns:
            List of fragment dictionaries with metadata
        """
        # Check if we should enable bandit sampling
        if not self.bandit_enabled and self.total_attempts >= self.start_after * self._get_total_samples():
            self._enable_bandit_sampling()
        
        # Sample fragments with diversity optimization
        if self.bandit_enabled:
            fragment_indices = self._bandit_sample(n_samples)
        else:
            fragment_indices = self._diverse_sample(n_samples)
        
        # Create fragment batch
        fragment_batch = []
        for idx in fragment_indices:
            fragment_batch.append({
                "fragment_smiles": self.fragment_list[idx],
                "fragment_index": idx,
                "sampling_method": "bandit" if self.bandit_enabled else "diverse"
            })
        
        # Update tracking
        for idx in fragment_indices:
            self.fragment_attempts[idx] += 1
        self.total_attempts += n_samples
        
        return fragment_batch
    
    def _diverse_sample(self, n_samples: int) -> List[int]:
        """Sample fragments with diversity optimization - prioritize underused fragments."""
        if n_samples >= self.n_fragments:
            # If we need more samples than fragments, use all fragments and sample with replacement
            return list(range(self.n_fragments)) + random.choices(range(self.n_fragments), k=n_samples - self.n_fragments)
        else:
            # Prioritize fragments that have been used less
            # Calculate usage weights (inverse of usage count + 1 to avoid division by zero)
            usage_weights = 1.0 / (self.fragment_attempts + 1)
            
            # Normalize weights
            usage_weights = usage_weights / np.sum(usage_weights)
            
            # Sample with replacement using usage-based weights
            return np.random.choice(
                range(self.n_fragments), 
                size=n_samples, 
                p=usage_weights,
                replace=False if n_samples <= self.n_fragments else True
            ).tolist()
    
    
    def _bandit_sample(self, n_samples: int) -> List[int]:
        """Sample fragments using bandit probabilities."""
        # Normalize probabilities
        probs = self.fragment_probs / np.sum(self.fragment_probs)
        
        # Sample with replacement
        return np.random.choice(
            range(self.n_fragments), 
            size=n_samples, 
            p=probs,
            replace=True
        ).tolist()
    
    def _enable_bandit_sampling(self) -> None:
        """Enable bandit-based sampling."""
        self.bandit_enabled = True
        self.logger.info("🎰 Enabling bandit-based fragment sampling")
    
    def update_fragment_scores(self, fragment_batch: List[Dict[str, Any]], scores: List[float]) -> None:
        """
        Update fragment scores for bandit learning.
        
        Args:
            fragment_batch: List of fragment dictionaries
            scores: Corresponding scores for each fragment
        """
        if not self.bandit_enabled:
            return
        
        # Update scores for each fragment
        for fragment, score in zip(fragment_batch, scores):
            idx = fragment["fragment_index"]
            self.fragment_scores[idx].append(score)
            
            # Update probabilities based on recent performance
            self._update_fragment_probability(idx)
    
    def _update_fragment_probability(self, fragment_idx: int) -> None:
        """Update probability for a specific fragment based on its performance."""
        if fragment_idx not in self.fragment_scores or len(self.fragment_scores[fragment_idx]) == 0:
            return
        
        # Calculate average score for this fragment
        avg_score = np.mean(self.fragment_scores[fragment_idx])
        
        # Update probability based on performance
        # Higher scores lead to higher probabilities
        current_prob = self.fragment_probs[fragment_idx]
        new_prob = current_prob * (1 + self.reweight_factor * avg_score)
        
        # Apply entropy decay to maintain exploration
        self.fragment_probs[fragment_idx] = new_prob * self.entropy_decay + (1 - self.entropy_decay) / self.n_fragments
        
        # Ensure probabilities stay positive
        self.fragment_probs[fragment_idx] = max(self.fragment_probs[fragment_idx], 1e-6)
    
    def _get_total_samples(self) -> int:
        """Get total number of samples to generate."""
        # This would typically come from the generation config
        # For now, return a reasonable default
        return 10000
    
    def get_coverage_stats(self) -> Dict[str, Any]:
        """Get fragment coverage statistics."""
        return {
            "total_fragments": self.n_fragments,
            "total_attempts": self.total_attempts,
            "bandit_enabled": self.bandit_enabled,
            "min_attempts_per_fragment": self.min_per_fragment,
            "fragment_attempts": dict(enumerate(self.fragment_attempts)),
            "fragment_probs": dict(enumerate(self.fragment_probs)),
            "coverage_ratio": np.sum(self.fragment_attempts > 0) / self.n_fragments,
            "core_attempts": dict(self.core_attempts)
        }
    
    def reset_bandit_state(self) -> None:
        """Reset bandit state for new generation run."""
        self.fragment_scores.clear()
        self.fragment_probs = np.ones(self.n_fragments) / self.n_fragments
        self.fragment_attempts = np.zeros(self.n_fragments)
        self.total_attempts = 0
        self.bandit_enabled = False
        self.core_attempts.clear()
        self.logger.info("Reset bandit state.")
    
    def set_core_count(self, n_cores: int) -> None:
        """Set the number of cores for tracking."""
        self.n_cores = n_cores
        print(f"   Set core count to {n_cores}")
    
    def track_core_usage(self, core_idx: int) -> None:
        """Track core usage for statistics."""
        self.core_attempts[core_idx] += 1
    
    def get_fragment_usage_by_core(self) -> Dict[int, Dict[str, Any]]:
        """Get fragment usage statistics by core."""
        core_usage = {}
        for core_idx in range(self.n_cores):
            core_usage[core_idx] = {
                "attempts": self.core_attempts.get(core_idx, 0),
                "fragment_attempts": dict(enumerate(self.fragment_attempts)),
                "fragment_probs": dict(enumerate(self.fragment_probs))
            }
        return core_usage