"""
Dataset preparation utilities for activity cliff-aware QSAR modeling.

Functions in this module fetch ChEMBL data, merge it with chemical space files,
standardize compounds, derive activity labels and remove tautomer duplicates.
"""

from __future__ import annotations

import csv
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import requests
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from tqdm import tqdm


# =============================================================================
# ChEMBL Data Fetching
# =============================================================================

def fetch_page(TARGET_ID: str, PAGE_LIMIT: int, offset: int) -> list[dict]:
    """
    Fetch a single page of activity data from ChEMBL API.
    
    Args:
        TARGET_ID: ChEMBL target ID (e.g., "CHEMBL392")
        PAGE_LIMIT: Maximum number of records per page
        offset: Offset for pagination
        
    Returns:
        List of activity dictionaries with keys: ChEMBL_ID, canonical_smiles,
        standard_type, standard_value, standard_units, assay_description
    """
    url = (f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
           f"?target_chembl_id={TARGET_ID}&limit={PAGE_LIMIT}&offset={offset}")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return [{
            "ChEMBL_ID": a.get("molecule_chembl_id"),
            "canonical_smiles": a.get("canonical_smiles"),
            "standard_type": a.get("standard_type"),
            "standard_value": a.get("standard_value"),
            "standard_units": a.get("standard_units"),
            "assay_description": a.get("assay_description")
        } for a in r.json().get("activities", [])]
    except Exception as exc:
        print(f"fetch_page error: {exc}")
        return []


def total_records(TARGET_ID: str) -> int:
    """
    Get total number of records available for a ChEMBL target.
    
    Args:
        TARGET_ID: ChEMBL target ID
        
    Returns:
        Total number of activity records
    """
    url = (f"https://www.ebi.ac.uk/chembl/api/data/activity.json"
           f"?target_chembl_id={TARGET_ID}&limit=1")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()["page_meta"]["total_count"]
    except Exception:
        return 0


# =============================================================================
# Dataset Merging and Cleaning
# =============================================================================

def merge_with_chemical_space(
    all_target_compounds: str, 
    chemical_space_csv: str, 
    merged_csv: str
) -> pd.DataFrame:
    """
    Merge ChEMBL activity data with chemical space data.
    
    Args:
        all_target_compounds: Path to ChEMBL compounds CSV
        chemical_space_csv: Path to chemical space data CSV
        merged_csv: Path to save merged dataset
        
    Returns:
        Merged DataFrame with combined activity and chemical space data
    """
    df_act = pd.read_csv(all_target_compounds, on_bad_lines="skip")
    df_thia = pd.read_csv(chemical_space_csv, sep=";", on_bad_lines="skip")
    
    # Normalize column names
    df_act = df_act.rename(columns=lambda c: c.strip().replace(" ", "_"))
    df_thia = df_thia.rename(columns=lambda c: c.strip().replace(" ", "_"))
    
    # Merge on ChEMBL_ID
    merged = pd.merge(df_thia, df_act, on="ChEMBL_ID", how="inner")
    merged.dropna(axis=1, how="all").to_csv(merged_csv, index=False)
    return merged


def convert_to_nM(row: pd.Series) -> float:
    """
    Convert activity values to nM units.
    
    Handles conversion from ug.mL-1 to nM using molecular weight.
    
    Args:
        row: DataFrame row containing standard_value, standard_units, and MolWt
        
    Returns:
        Activity value in nM
    """
    if row["standard_units"] == "ug.mL-1":
        mw = row.get("MolWt")
        if mw and pd.notna(mw):
            return float(row["standard_value"] / mw * 1e6)
        return None
    return row["standard_value"]


def add_flag(df: pd.DataFrame, THRESHOLD_NM: float) -> pd.DataFrame:
    """
    Add activity flag based on threshold.
    
    Args:
        df: DataFrame with standard_value column
        THRESHOLD_NM: Activity threshold in nM
        
    Returns:
        DataFrame with added activity_flag column ("active" or "inactive")
    """
    df = df.copy()
    df["activity_flag"] = np.where(
        df["standard_value"] < THRESHOLD_NM, "active", "inactive"
    )
    return df


def clean_dataset(df: pd.DataFrame, THRESHOLD_NM: float) -> pd.DataFrame:
    """
    Clean and standardize dataset with molecular filters.
    
    This function:
    - Validates SMILES and converts to RDKit molecules
    - Filters by molecular weight (< 900 Da) and LogP (< 8)
    - Normalizes activity values to nM
    - Adds activity flags and pIC50 values
    - Removes duplicates
    
    Args:
        df: Raw DataFrame with canonical_smiles and standard_value
        THRESHOLD_NM: Activity threshold in nM for flagging
        
    Returns:
        Cleaned DataFrame with columns: ChEMBL_ID, canonical_smiles,
        activity_flag, standard_value, pIC50
    """
    df = df.copy()
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["canonical_smiles", "standard_value"])
    
    # Convert SMILES to molecules
    df["Mol"] = df["canonical_smiles"].map(Chem.MolFromSmiles)
    df["MolWt"] = df["Mol"].map(Descriptors.MolWt)
    df["LogP"] = df["Mol"].map(Descriptors.MolLogP)
    df = df.dropna(subset=["Mol", "MolWt", "LogP"])
    
    # Apply drug-like filters
    df = df[(df["MolWt"] < 900) & (df["LogP"] < 8)]
    
    # Normalize to nM
    df["standard_value"] = df.apply(convert_to_nM, axis=1)
    df = df.dropna(subset=["standard_value"])
    df["standard_units"] = "nM"
    
    # Add activity flag
    df = add_flag(df, THRESHOLD_NM).drop_duplicates(subset="canonical_smiles")
    
    # Add pIC50 (convert from nM to M, then take -log10)
    df["pIC50"] = -np.log10(df["standard_value"] * 1e-9)
    df = df[df["pIC50"].notna()]
    
    return df[["ChEMBL_ID", "canonical_smiles", "activity_flag", "standard_value", "pIC50"]]


# =============================================================================
# Custom Data Resources
# =============================================================================

def load_custom(path: str) -> pd.DataFrame:
    """
    Load custom compounds from CSV file and standardize columns.

    Supports optional activity values provided in either nM or µM units.

    Expected columns (case insensitive, subset accepted):
        - ID / ChEMBL_ID (optional)
        - canonical_smiles / SMILES
        - activity_flag
        - standard_value (numeric potency)
        - standard_units or standard_value_unit ("nM", "uM", "micromolar", ... optional)

    Returns a DataFrame compatible with the cleaned dataset schema:
        ChEMBL_ID, canonical_smiles, activity_flag, standard_value, pIC50
    """

    if not os.path.isfile(path):
        print(f"Custom compounds file not found: {path} – skipping.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, sep=';', on_bad_lines="skip")
        has_header = set(map(str.lower, df.columns)) & {"smiles", "canonical_smiles"}
    except pd.errors.ParserError:
        has_header = False

    if not has_header:
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        df = pd.DataFrame(rows, columns=["ID", "canonical_smiles", "activity_flag", "standard_value"])

    df = df.rename(columns={
        "Smiles": "canonical_smiles",
        "SMILES": "canonical_smiles",
        "smiles": "canonical_smiles",
        "chembl_id": "ID",
        "standard_value_unit": "standard_units",
        "Standard_value": "standard_value",
        "STANDARD_VALUE": "standard_value"
    })

    # Ensure required columns exist
    for col in ["ID", "canonical_smiles", "activity_flag", "standard_value", "standard_units"]:
        if col not in df.columns:
            df[col] = np.nan

    if "pIC50" not in df.columns:
        df["pIC50"] = np.nan

    df = df[["ID", "canonical_smiles", "activity_flag", "standard_value", "standard_units", "pIC50"]]
    df = df.dropna(subset=["canonical_smiles"])

    # Activity flag normalization
    df["activity_flag"] = df["activity_flag"].astype(str).str.strip().str.lower()

    # Standard value normalization (default: already in nM)
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df["standard_units"] = df["standard_units"].astype(str).str.strip().str.lower()

    micromolar_terms = {"um", "µm", "micromolar", "micromole", "micromoles", "micromol"}
    mask_uM = df["standard_units"].isin(micromolar_terms)
    df.loc[mask_uM, "standard_value"] = df.loc[mask_uM, "standard_value"] * 1_000.0  # µM → nM
    df.loc[mask_uM, "standard_units"] = "nm"

    # Default to nM if units missing
    df.loc[df["standard_units"].isna() | (df["standard_units"] == "nan"), "standard_units"] = "nm"

    # Compute pIC50 when potency available and positive
    valid_mask = df["standard_value"].astype(float) > 0
    df.loc[valid_mask, "pIC50"] = -np.log10(df.loc[valid_mask, "standard_value"].astype(float) * 1e-9)

    # Carry custom IDs into ChEMBL_ID column (prefixed with OWN_)
    df["ChEMBL_ID"] = df["ID"].astype(str).str.strip()
    df.loc[df["ChEMBL_ID"].isin(["", "nan", "None"]), "ChEMBL_ID"] = np.nan
    df["ChEMBL_ID"] = df["ChEMBL_ID"].apply(lambda x: f"OWN_{x}" if pd.notna(x) else x)

    return df[["ChEMBL_ID", "canonical_smiles", "activity_flag", "standard_value", "pIC50"]]


def drop_tautomer_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove tautomer duplicates using canonical tautomer representation.
    
    For each group of tautomers:
    - Prefers active compounds over inactive
    - Among actives, selects the most potent (lowest standard_value)
    - Falls back to first compound if no standard_value available
    
    Args:
        df: DataFrame with canonical_smiles and activity_flag columns
        
    Returns:
        Deduplicated DataFrame with one representative per tautomer
    """
    enum = rdMolStandardize.TautomerEnumerator()
    
    def canon_taut(s: str) -> str | None:
        """Canonicalize SMILES via tautomer enumeration."""
        try:
            m = Chem.MolFromSmiles(str(s))
            if m:
                return Chem.MolToSmiles(
                    enum.Canonicalize(m), canonical=True
                )
        except Exception:
            pass
        return None
    
    df = df.copy()
    df["taut_can"] = df["canonical_smiles"].apply(canon_taut)
    df = df.dropna(subset=["taut_can"])
    
    def pick_one(group: pd.DataFrame) -> pd.Series:
        """Select best representative from tautomer group."""
        # Prefer active compounds
        active = group[group["activity_flag"] == "active"]
        sub = active if not active.empty else group
        
        # Among selected, prefer most potent (lowest standard_value)
        if "standard_value" in sub.columns and sub["standard_value"].notna().any():
            sub = sub.loc[sub["standard_value"].astype(float).idxmin()]
        else:
            sub = sub.iloc[0]
        
        return sub
    
    deduped = (
        df.groupby("taut_can", group_keys=False)
        .apply(pick_one, include_groups=False)
    )
    
    return deduped


# =============================================================================
# Main Pipeline Function
# =============================================================================

def create_final_dataset(cfg: Mapping, log) -> None:
    """
    Main dataset preparation pipeline.
    
    Orchestrates the complete dataset preparation workflow:
    1. Fetch ChEMBL data (if not cached)
    2. Merge with chemical space data
    3. Clean and standardize compounds
    4. Append custom compounds (if available)
    5. Deduplicate tautomers
    6. Save final dataset
    
    All parameters are read from YAML configuration file.
    
    Args:
        cfg: Configuration dictionary from YAML file
        log: Logger instance for progress tracking
        
    Returns:
        None (saves final dataset to CSV file)
    """
    # Extract configuration parameters
    target_id = cfg["TARGET_ID"]
    page_limit = cfg["PAGE_LIMIT"]
    threshold_nm = cfg["THRESHOLD_NM"]
    paths = cfg["Paths"]
    
    # Define file paths
    all_target_compounds_csv = Path(paths["target_path"])
    merged_csv = Path(paths["merged_path"])
    final_csv = Path(paths["dataset"])
    custom_csv = Path(paths["own_resources_path"])
    chemical_space_csv = Path(paths["chemical_space_path"])
    
    # Create directories if needed
    for p in [all_target_compounds_csv.parent, merged_csv.parent, final_csv.parent]:
        p.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Fetch data from ChEMBL if not already fetched
    if not all_target_compounds_csv.exists():
        log.info("[1/5] Fetching ChEMBL data for %s", target_id)
        total = total_records(target_id)
        offsets = range(0, total, page_limit)
        with Pool(cpu_count()) as pool:
            pages = list(tqdm(
                pool.imap_unordered(
                    lambda o: fetch_page(target_id, page_limit, o), offsets
                ),
                desc="Downloading", total=len(list(offsets))
            ))
        df_raw = pd.DataFrame([r for page in pages for r in page])
        df_raw.to_csv(all_target_compounds_csv, index=False)
        log.info("Saved %d records to: %s", len(df_raw), all_target_compounds_csv)
    else:
        log.info("[1/5] ChEMBL data already present: %s", all_target_compounds_csv)
        df_raw = pd.read_csv(all_target_compounds_csv)
    
    # Step 2: Merge with chemical space
    if not merged_csv.exists():
        log.info("[2/5] Merging with chemical_space...")
        df_merged = merge_with_chemical_space(
            all_target_compounds_csv, chemical_space_csv, merged_csv
        )
        df_merged.to_csv(merged_csv, index=False)
        log.info("Merged data saved: %s (%d records)", merged_csv, len(df_merged))
    else:
        log.info("[2/5] Merged CSV already exists.")
        df_merged = pd.read_csv(merged_csv)
    
    # Step 3: Clean and flag dataset
    log.info("[3/5] Cleaning and flagging dataset...")
    df_ic50 = df_merged[df_merged["standard_type"] == "IC50"]
    df_clean = clean_dataset(df_ic50, threshold_nm)
    log.info("Cleaned dataset: %d compounds", len(df_clean))
    
    # Step 4: Append custom compounds if present
    if custom_csv.exists():
        log.info("[4/5] Appending custom compounds...")
        df_custom = load_custom(custom_csv)
        if not df_custom.empty:
            df_combined = pd.concat([df_clean, df_custom], ignore_index=True)
        else:
            df_combined = df_clean
    else:
        log.info("[4/5] No custom compounds file found, skipping")
        df_combined = df_clean
    
    # Step 5: Tautomer deduplication and save final dataset
    log.info("[5/5] Deduplicating tautomers and saving final dataset...")
    df_final = drop_tautomer_duplicates(df_combined)
    
    df_final = df_final[[
        "ChEMBL_ID", "canonical_smiles", "activity_flag", "standard_value", "pIC50"
    ]]
    df_final.to_csv(final_csv, index=False)
    
    log.info("Final dataset saved: %s (%d unique compounds)", final_csv, len(df_final))
