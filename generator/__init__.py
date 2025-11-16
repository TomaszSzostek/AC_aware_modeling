"""
AC-Aware Molecular Generator Package

A config-driven Python package for AC-aware molecular generation using fragments from Reverse QSAR.
Generates molecules around core scaffolds using AC-enriched fragments, scores with QSAR/SA/QED,
and selects a diverse final set. Outputs publication-ready hits.csv with fragment AC flags.

"""

import warnings
import os

# Suppress all RDKit warnings globally
warnings.filterwarnings("ignore", message=".*not removing hydrogen atom.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rdkit.*")
warnings.filterwarnings("ignore", category=UserWarning, module="rdkit.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy.*")
warnings.filterwarnings("ignore", message=".*Mean of empty slice.*")
warnings.filterwarnings("ignore", message=".*invalid value encountered.*")

# Set RDKit to suppress warnings
os.environ["RDKIT_QUIET"] = "1"

from .generator import ACGenerator, run_generator
from .sampler import FragmentSampler
from .scorer import MolecularScorer
from .selector import MolecularSelector
from .molecular_generator import MolecularGenerator

__all__ = [
    'ACGenerator',
    'run_generator',
    'FragmentSampler',
    'MolecularScorer',
    'MolecularSelector',
    'MolecularGenerator'
]

__version__ = "1.0.0"
