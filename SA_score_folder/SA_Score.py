"""
Thin wrapper for the RDKit-based synthetic accessibility (SA) score.

The actual scoring logic is implemented in `sascorer.calculateScore`; this
module exists for backward compatibility within the AC-aware modeling pipeline.
"""

from .sascorer import calculateScore  # re-export for convenience

