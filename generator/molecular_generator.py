"""
AC-Aware Molecular Generator Module

Implements an island algorithm for fragment-based molecular generation with activity cliff awareness.
The algorithm generates molecules by systematically exploring chemical space around predefined core scaffolds
using fragment libraries, with integrated activity cliff gating and multi-objective optimization.

Key Features:
- Island Algorithm: Generates molecules around each core scaffold independently
- Fragment Attachment: Uses [*] attachment points for proper fragment-core connections
- Activity Cliff Gating: Pre-filters candidates based on similarity to known active/inactive compounds
- Multi-objective Optimization: Balances QSAR activity, synthetic accessibility, and drug-likeness

"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Generator
import random
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdMolDescriptors
import time
import os
import tempfile
import json
from pathlib import Path


class MolecularGenerator:
    """
    AC-Aware Molecular Generator implementing the Island Algorithm.
    
    This class implements a systematic approach to molecular generation that explores
    chemical space around predefined core scaffolds using fragment libraries. The algorithm
    ensures comprehensive coverage of chemical space while maintaining chemical validity
    and enabling activity cliff-aware filtering.
    
    The Island Algorithm works by:
    1. Treating each core scaffold as an independent "island" in chemical space
    2. Systematically attaching fragments to each core to explore its neighborhood
    3. Ensuring all attachment points ([*]) are properly resolved
    4. Generating chemically valid molecules ready for downstream analysis
    
    Attributes:
        cores (List[str]): List of core SMARTS patterns with [*] attachment points
        config (Dict[str, Any]): Configuration parameters for generation
        logger (logging.Logger): Logger instance for debugging and monitoring
        generation_stats (Dict[str, Any]): Statistics tracking generation performance
    """
    
    def __init__(self, cores: List[str], config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize the molecular generator.
        
        Args:
            cores: List of core SMARTS patterns with [*] attachment points
            config: Configuration dictionary
            logger: Optional logger instance
        """
        self.cores = cores
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration - all parameters from config.yml
        self.max_attempts = config.get("max_attempts", 50)
        self.timeout_seconds = config.get("timeout_seconds", 300)
        self.batch_size = config.get("batch_size", 100)
        self.max_mol_weight = config.get("max_mol_weight", 600)
        self.min_mol_weight = config.get("min_mol_weight", 100)
        
        # Core patterns
        self.core_patterns = [Chem.MolFromSmarts(core) for core in cores]
        self.n_cores = len(cores)
        
        # Statistics
        self.generation_stats = {
            "total_attempts": 0,
            "successful_generations": 0,
            "failed_generations": 0,
            "timeout_generations": 0,
            "core_usage": {i: 0 for i in range(self.n_cores)},
            "fragment_usage": {}
        }
        
        # Molecular generator initialized
    
    def generate_molecules(self, fragment_batch: List[Dict[str, Any]], max_molecules: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate molecules using the Island Algorithm with deduplication.
        
        This method implements the core of the Island Algorithm by systematically
        exploring chemical space around each core scaffold. For each core (island),
        it attempts to attach all available fragments, ensuring comprehensive
        coverage of the chemical neighborhood around each scaffold.
        
        The algorithm guarantees that:
        - Each core is explored with all available fragments
        - No attachment points ([*]) remain in final molecules
        - Generated molecules are chemically valid
        - Duplicates are removed to ensure diversity
        - Statistics are properly tracked for analysis
        
        Args:
            fragment_batch (List[Dict[str, Any]]): List of fragment dictionaries
                containing fragment SMILES and metadata
        
        Returns:
            List[Dict[str, Any]]: List of generated molecule dictionaries with
                SMILES, core information, and generation metadata
        """
        molecules = []
        seen_smiles = set()  # Track unique SMILES to avoid duplicates
        
        # Island Algorithm: Generate around each core
        target_molecules = max_molecules if max_molecules is not None else len(fragment_batch)
        
        for core_idx in range(self.n_cores):
            if len(molecules) >= target_molecules:
                break
                
            core_smarts = self.cores[core_idx]
            self.logger.debug(f"Generating molecules around core {core_idx}: {core_smarts}")
            
            # Generate molecules for this core using available fragments
            for fragment in fragment_batch:
                if len(molecules) >= target_molecules:
                    break
                    
                try:
                    mol_data = self._generate_single_molecule(
                        fragment["fragment_smiles"], 
                        core_smarts, 
                        core_idx,
                        fragment_batch  # Pass all fragments for iterative building
                    )
                    
                    if mol_data and mol_data["smiles"] not in seen_smiles:
                        molecules.append(mol_data)
                        seen_smiles.add(mol_data["smiles"])
                        self.generation_stats["core_usage"][core_idx] += 1
                        self.generation_stats["fragment_usage"][fragment["fragment_smiles"]] = \
                            self.generation_stats["fragment_usage"].get(fragment["fragment_smiles"], 0) + 1
                    elif mol_data and mol_data["smiles"] in seen_smiles:
                        self.logger.debug(f"Duplicate molecule skipped: {mol_data['smiles']}")
                        
                except Exception as e:
                    self.logger.debug(f"Failed to generate molecule from fragment {fragment.get('fragment_smiles', 'unknown')} on core {core_idx}: {e}")
                    continue
        
        # Continue generating until we have enough molecules or exhaust all possibilities
        max_attempts = max(len(fragment_batch) * self.n_cores * 50, target_molecules * 2)  # Scale with target
        attempt = 0
        stall_counter = 0
        last_count = len(molecules)
        
        while len(molecules) < target_molecules and attempt < max_attempts:
            attempt += 1
            
            # Randomly select core and fragment for additional generation
            core_idx = random.randint(0, self.n_cores - 1)
            fragment = random.choice(fragment_batch)
            core_smarts = self.cores[core_idx]
            
            try:
                mol_data = self._generate_single_molecule(
                    fragment["fragment_smiles"], 
                    core_smarts, 
                    core_idx,
                    fragment_batch
                )
                
                if mol_data and mol_data["smiles"] not in seen_smiles:
                    molecules.append(mol_data)
                    seen_smiles.add(mol_data["smiles"])
                    self.generation_stats["core_usage"][core_idx] += 1
                    self.generation_stats["fragment_usage"][fragment["fragment_smiles"]] = \
                        self.generation_stats["fragment_usage"].get(fragment["fragment_smiles"], 0) + 1
                    
            except Exception as e:
                self.logger.debug(f"Additional generation attempt {attempt} failed: {e}")
                continue
            
            # Check if we're making progress every 500 attempts (increased from 100)
            if attempt % 500 == 0:
                if len(molecules) == last_count:
                    stall_counter += 1
                    if stall_counter >= 10:  # No progress in 5000 attempts (increased from 5)
                        self.logger.debug(f"No progress after {attempt} attempts, stopping (have {len(molecules)} molecules)")
                        break
                else:
                    stall_counter = 0  # Reset if we made progress
                last_count = len(molecules)
        
        self.logger.debug(f"Generated {len(molecules)} unique molecules from {len(fragment_batch)} fragments across {self.n_cores} cores")
        return molecules
    
    def _generate_single_molecule(self, fragment_smiles: str, core_smarts: str, core_idx: int, fragment_batch: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Generate a single molecule from fragment and core using iterative fragment attachment.
        
        Args:
            fragment_smiles: SMILES string of the first fragment (with [*] attachment points)
            core_smarts: SMARTS pattern of the core (with [*] attachment points)
            core_idx: Index of the core
            fragment_batch: List of all available fragments for iterative building
            
        Returns:
            Dictionary containing molecule data or None if generation failed
        """
        start_time = time.time()
        
        for attempt in range(self.max_attempts):
            # Check timeout
            if time.time() - start_time > self.timeout_seconds:
                self.generation_stats["timeout_generations"] += 1
                break
            
            try:
                # Generate molecule by iteratively attaching fragments to core
                generated_smiles, fragments_used = self._build_molecule_iteratively(fragment_smiles, core_smarts, fragment_batch)
                
                if generated_smiles and self._validate_molecule(generated_smiles):
                    self.generation_stats["successful_generations"] += 1
                    return {
                        "smiles": generated_smiles,
                        "core": core_smarts,
                        "fragments_used": fragments_used,
                        "generation_time": time.time() - start_time,
                        "attempts": attempt + 1,
                        "generation_method": "iterative_attachment"
                    }
                    
            except Exception as e:
                self.logger.debug(f"Generation attempt {attempt + 1} failed: {e}")
                continue
        
        self.generation_stats["failed_generations"] += 1
        return None
    
    def _build_molecule_iteratively(self, first_fragment: str, core_smarts: str, fragment_batch: List[Dict[str, Any]]) -> Tuple[Optional[str], List[str]]:
        """
        Build molecule iteratively by attaching fragments until no [*] remain.
        Uses proper chemical bonding like in the reference code.
        
        Args:
            first_fragment: SMILES string of the first fragment to attach
            core_smarts: SMARTS pattern of the core (with [*] attachment points)
            fragment_batch: List of all available fragments for iterative building
            
        Returns:
            Tuple of (Complete SMILES string or None, List of fragment SMILES used)
        """
        fragments_used = []
        try:
            # Start with core molecule
            core_mol = Chem.MolFromSmiles(core_smarts, sanitize=False)
            if not core_mol:
                return None, []
            try:
                Chem.SanitizeMol(core_mol)
            except:
                return None, []
            
            # Attach first fragment
            first_mol = Chem.MolFromSmiles(first_fragment, sanitize=False)
            if not first_mol:
                return None, []
            try:
                Chem.SanitizeMol(first_mol)
            except:
                return None, []
            
            # Track first fragment
            fragments_used.append(first_fragment)
            
            # Connect core and first fragment
            current_mol = self._connect_pair(core_mol, first_mol)
            if not current_mol:
                return None, []
            
            # Continue attaching fragments with [*] until no [*] remain
            iteration = 0
            max_iterations = 50  # Safety limit to prevent infinite loops
            
            while self._has_stars(current_mol) and iteration < max_iterations:
                iteration += 1
                
                # Get all fragments with [*] attachment points
                fragments_with_stars = [f for f in fragment_batch if "[*]" in f["fragment_smiles"]]
                
                if not fragments_with_stars:
                    break  # No more fragments with attachment points
                
                # Randomly shuffle fragments for diversity
                random.shuffle(fragments_with_stars)
                
                # Try to attach another fragment from the batch
                fragment_attached = False
                for fragment in fragments_with_stars:
                    fragment_smiles = fragment["fragment_smiles"]
                    frag_mol = Chem.MolFromSmiles(fragment_smiles, sanitize=False)
                    if not frag_mol:
                        continue
                    try:
                        Chem.SanitizeMol(frag_mol)
                    except:
                        continue
                    
                    # Try to connect this fragment
                    new_mol = self._connect_pair(current_mol, frag_mol)
                    if new_mol:
                        current_mol = new_mol
                        fragment_attached = True
                        fragments_used.append(fragment_smiles)  # Track used fragment
                        break
                
                # If no fragment could be attached, break
                if not fragment_attached:
                    break
            
            # If there are still [*] in the molecule, convert them to hydrogen
            if self._has_stars(current_mol):
                current_mol = self._star_to_h(current_mol)
                if not current_mol:
                    return None, []
            
            # Convert to SMILES
            smiles = Chem.MolToSmiles(current_mol, canonical=True)
            
            # Early molecular weight check - before full validation
            if smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mol_weight = Descriptors.MolWt(mol)
                    if mol_weight < self.min_mol_weight or mol_weight > self.max_mol_weight:
                        self.logger.debug(f"Molecule weight out of range: {mol_weight} (range: {self.min_mol_weight}-{self.max_mol_weight})")
                        return None, []
            
            if self._is_valid_smiles(smiles):
                return smiles, fragments_used
            
            return None, []
            
        except Exception as e:
            self.logger.debug(f"Iterative building failed: {e}")
            return None, []
    
    def _has_stars(self, mol) -> bool:
        """Check if molecule has [*] attachment points (atomic number 0)."""
        if not mol:
            return False
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 0:  # [*] has atomic number 0
                return True
        return False
    
    def _find_stars(self, mol) -> List[int]:
        """Find atom indices with [*] attachment points."""
        if not mol:
            return []
        return [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    
    def _star_to_h(self, mol):
        """Convert remaining [*] to hydrogen atoms."""
        if not mol:
            return None
        rw = Chem.RWMol(mol)
        for idx in self._find_stars(rw):
            rw.GetAtomWithIdx(idx).SetAtomicNum(1)  # Set to hydrogen
        try:
            Chem.SanitizeMol(rw)
            return rw.GetMol()
        except:
            return None
    
    def _connect_pair(self, mol1, mol2, tries=20):
        """Connect two molecules through [*] attachment points."""
        if not mol1 or not mol2:
            return None
        
        n1 = mol1.GetNumAtoms()
        for _ in range(tries):
            w1, w2 = self._find_stars(mol1), self._find_stars(mol2)
            if not w1 or not w2:
                return None
            
            # Choose random attachment points
            i1, i2 = w1[0], w2[0]  # Use first available for simplicity
            
            # Get neighbors of attachment points
            nei1 = mol1.GetAtomWithIdx(i1).GetNeighbors()
            nei2 = mol2.GetAtomWithIdx(i2).GetNeighbors()
            if not nei1 or not nei2:
                continue
            
            nei1_idx = nei1[0].GetIdx()
            nei2_idx = nei2[0].GetIdx()
            
            # Combine molecules
            combo = Chem.CombineMols(mol1, mol2)
            rw = Chem.RWMol(combo)
            
            # Add bond between neighbors
            rw.AddBond(nei1_idx, nei2_idx + n1, Chem.BondType.SINGLE)
            
            # Remove attachment point atoms
            for idx in sorted([i2 + n1, i1], reverse=True):
                rw.RemoveAtom(idx)
            
            try:
                Chem.SanitizeMol(rw)
                return rw.GetMol()
            except:
                continue
        
        return None
    
    
    
    
    
    
    def _is_valid_smiles(self, smiles: str) -> bool:
        """Check if SMILES string is chemically valid."""
        try:
            from rdkit import Chem
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False
            
            # Check for basic chemical validity
            # Remove any empty parentheses
            if "()" in smiles:
                return False
            
            # Check for basic structure
            if len(smiles) < 3:
                return False
                
            return True
            
        except Exception:
            return False
    
    def _validate_molecule(self, smiles: str) -> bool:
        """
        Validate generated molecule.
        
        Args:
            smiles: SMILES string to validate
            
        Returns:
            True if molecule is valid, False otherwise
        """
        try:
            # First check if SMILES is parseable
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                self.logger.debug(f"Invalid SMILES: {smiles}")
                return False
            
            # Check for attachment points - final molecules should not have [*]
            if "[*]" in smiles:
                self.logger.debug(f"Molecule still has attachment points: {smiles}")
                return False
            
            # Sanitize the molecule to check for chemical validity
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                self.logger.debug(f"Molecule failed sanitization: {smiles} - {e}")
                return False
            
            # Basic validation checks
            if mol.GetNumAtoms() < 3:  # Too small
                self.logger.debug(f"Molecule too small: {smiles}")
                return False
            
            if mol.GetNumAtoms() > 200:  # Too large
                self.logger.debug(f"Molecule too large: {smiles}")
                return False
            
            # Check molecular weight
            mol_weight = Descriptors.MolWt(mol)
            if mol_weight < self.min_mol_weight or mol_weight > self.max_mol_weight:
                self.logger.debug(f"Molecule weight out of range: {mol_weight}")
                return False
            
            
            # Sanitize the molecule
            try:
                Chem.SanitizeMol(mol)
            except Exception as e:
                self.logger.debug(f"Molecule sanitization failed: {smiles}, error: {e}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Molecule validation failed: {smiles}, error: {e}")
            return False
    
    
    def stream_generate(self, fragment_batch: List[Dict[str, Any]]) -> Generator[Dict[str, Any], None, None]:
        """
        Stream generate molecules (generator function).
        
        Args:
            fragment_batch: List of fragment dictionaries
            
        Yields:
            Generated molecule dictionaries
        """
        for fragment in fragment_batch:
            try:
                # Generate molecule around random core
                core_idx = random.randint(0, self.n_cores - 1)
                core_smarts = self.cores[core_idx]
                
                # Generate molecule
                mol_data = self._generate_single_molecule(
                    fragment["fragment_smiles"], 
                    core_smarts, 
                    core_idx
                )
                
                if mol_data:
                    yield mol_data
                    
            except Exception as e:
                self.logger.warning(f"Stream generation failed for fragment {fragment.get('fragment_smiles', 'unknown')}: {e}")
                continue
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        return {
            "n_cores": self.n_cores,
            "cores": self.cores,
            "max_attempts": self.max_attempts,
            "timeout_seconds": self.timeout_seconds,
            "max_mol_weight": self.max_mol_weight,
            "min_mol_weight": self.min_mol_weight,
            "stats": self.generation_stats
        }