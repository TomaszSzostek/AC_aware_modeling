from __future__ import annotations
from pathlib import Path
import csv
import logging
import pandas as pd

# Convenience fragments (often useful for builders/decoration)
HYDROGEN_FRAGMENT  = "[H][*]"

def _load_selected_fragments(frag_root: Path) -> list[str]:
    """
    Load selected fragments from the best model directory.
    Reads summary.json to find the best model, then loads fragments from that model's directory.
    Prefers .smi, falls back to CSV if needed.
    """
    # First, try to read summary.json to find the best model
    summary_path = frag_root / "summary.json"
    if summary_path.exists():
        try:
            import json
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            best_model = summary.get("best_model")
            if best_model:
                model_dir = frag_root / best_model
                print(f"[REINVENT] Found best model: {best_model}")
                
                # Try .smi file first
                smi_path = model_dir / "selected_fragments.smi"
                if smi_path.exists():
                    return [ln.strip() for ln in smi_path.read_text().splitlines() if ln.strip()]
                
                # Try CSV file
                csv_path = model_dir / "selected_fragments_with_ACflag.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    col = "fragment_smiles" if "fragment_smiles" in df.columns else df.columns[0]
                    return df[col].dropna().astype(str).tolist()
        except Exception as e:
            print(f"[REINVENT] Error reading summary.json: {e}")
    
    # Fallback: try to find any model directory with fragments
    model_dirs = [d for d in frag_root.iterdir() if d.is_dir() and d.name in 
                  ["LogReg", "KNN", "SVC", "RF", "ExtraTrees", "GB", "XGB", "CatBoost", "BRF"]]
    
    for model_dir in model_dirs:
        print(f"[REINVENT] Trying model directory: {model_dir.name}")
        # Try .smi file first
        smi_path = model_dir / "selected_fragments.smi"
        if smi_path.exists():
            print(f"[REINVENT] Found fragments in {model_dir.name}")
            return [ln.strip() for ln in smi_path.read_text().splitlines() if ln.strip()]
        
        # Try CSV file
        csv_path = model_dir / "selected_fragments_with_ACflag.csv"
        if csv_path.exists():
            print(f"[REINVENT] Found fragments in {model_dir.name}")
            df = pd.read_csv(csv_path)
            col = "fragment_smiles" if "fragment_smiles" in df.columns else df.columns[0]
            return df[col].dropna().astype(str).tolist()

    raise FileNotFoundError(
        f"Cannot find selected fragments in {frag_root} "
        "(expected selected_fragments.smi or selected_fragments_with_ACflag.csv in model directories)."
    )

def _load_vanilla_fragments(frag_root: Path) -> list[str]:
    """
    Load vanilla fragments (selection_method == "threshold") from all_fragments_dictionary.csv.
    These are ALL fragments selected by importance threshold only, including those removed by CAFE.
    We use all_fragments_dictionary.csv instead of selected_fragments_with_ACflag.csv to ensure
    we get ALL threshold fragments, even if CAFE removed them.
    """
    # Handle both "reverse_QSAR" and "ReverseQSAR" naming conventions
    if not frag_root.exists():
        alt_frag_root = Path(str(frag_root).replace("reverse_QSAR", "ReverseQSAR"))
        if alt_frag_root.exists():
            frag_root = alt_frag_root
    
    # First, try to read summary.json to find the best model
    summary_path = frag_root / "summary.json"
    if summary_path.exists():
        try:
            import json
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            best_model = summary.get("best_model")
            if best_model:
                model_dir = frag_root / best_model
                # Try all_fragments_dictionary.csv first (contains ALL threshold fragments, even removed ones)
                all_frags_path = model_dir / "all_fragments_dictionary.csv"
                if all_frags_path.exists():
                    df = pd.read_csv(all_frags_path)
                    if "fragment_smiles" in df.columns and "selection_method" in df.columns:
                        vanilla_frags = df[df["selection_method"] == "threshold"]["fragment_smiles"].dropna().astype(str).tolist()
                        print(f"[VANILLA] Found {len(vanilla_frags)} vanilla fragments in {best_model} (from all_fragments_dictionary.csv, includes removed by CAFE)")
                        return vanilla_frags
                
                # Fallback: use selected_fragments_with_ACflag.csv (may miss fragments removed by CAFE)
                csv_path = model_dir / "selected_fragments_with_ACflag.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    if "fragment_smiles" in df.columns and "selected_by" in df.columns:
                        vanilla_frags = df[df["selected_by"] == "threshold"]["fragment_smiles"].dropna().astype(str).tolist()
                        print(f"[VANILLA] Found {len(vanilla_frags)} vanilla fragments in {best_model} (from selected_fragments_with_ACflag.csv)")
                        return vanilla_frags
        except Exception as e:
            print(f"[VANILLA] Error reading summary.json: {e}")
    
    # Fallback: try to find any model directory with fragments
    model_dirs = [d for d in frag_root.iterdir() if d.is_dir() and d.name in 
                  ["LogReg", "KNN", "SVC", "RF", "ExtraTrees", "GB", "XGB", "CatBoost", "BRF"]]
    
    for model_dir in model_dirs:
        # Try all_fragments_dictionary.csv first
        all_frags_path = model_dir / "all_fragments_dictionary.csv"
        if all_frags_path.exists():
            df = pd.read_csv(all_frags_path)
            if "fragment_smiles" in df.columns and "selection_method" in df.columns:
                vanilla_frags = df[df["selection_method"] == "threshold"]["fragment_smiles"].dropna().astype(str).tolist()
                print(f"[VANILLA] Found {len(vanilla_frags)} vanilla fragments in {model_dir.name} (from all_fragments_dictionary.csv, includes removed by CAFE)")
                return vanilla_frags
        
        # Fallback: use selected_fragments_with_ACflag.csv
        csv_path = model_dir / "selected_fragments_with_ACflag.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "fragment_smiles" in df.columns and "selected_by" in df.columns:
                vanilla_frags = df[df["selected_by"] == "threshold"]["fragment_smiles"].dropna().astype(str).tolist()
                print(f"[VANILLA] Found {len(vanilla_frags)} vanilla fragments in {model_dir.name} (from selected_fragments_with_ACflag.csv)")
                return vanilla_frags

    raise FileNotFoundError(
        f"Cannot find vanilla fragments in {frag_root} "
        "(expected all_fragments_dictionary.csv or selected_fragments_with_ACflag.csv)."
    )


def run_reinvent_library(cfg: dict) -> list[str]:
    """
    Simple REINVENT export:
      1) Load selected fragments from Reverse-QSAR (already normalized and deduplicated).
      2) Add convenience fragments (H only).
      3) Export ONE CSV: reinvent_fragments_all.csv

    Returns the list of exported fragment SMILES.
    """
    frag_root = Path(cfg["Paths"].get("fragments", "results/ReverseQSAR"))
    out_csv   = frag_root / "reinvent_fragments_all.csv"

    print(f"[REINVENT] Looking for fragments in: {frag_root}")
    print(f"[REINVENT] Fragments directory exists: {frag_root.exists()}")
    if frag_root.exists():
        print(f"[REINVENT] Directory contents: {list(frag_root.iterdir())}")

    # Load selected fragments (already normalized and deduplicated in defragmentation.py)
    fragments = _load_selected_fragments(frag_root)
    print(f"[REINVENT] Loaded {len(fragments)} fragments from best model")

    # Convert to set and add convenience fragments
    fragment_set = set(fragments)
    fragment_set.add(HYDROGEN_FRAGMENT)

    # Sort for consistent output
    final_fragments = sorted(fragment_set)
    print(f"[REINVENT] Final library size: {len(final_fragments)} fragments (including convenience fragments)")

    # Export
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fragment_smiles"])
        for s in final_fragments:
            w.writerow([s])

    logging.info(f"REINVENT library exported: {len(final_fragments)} fragments → {out_csv}")
    return final_fragments


def run_vanilla_library(cfg: dict) -> list[str]:
    """
    Vanilla fragment library export (threshold fragments only, no CAFE):
      1) Load vanilla fragments (selected_by == "threshold") from Reverse-QSAR.
      2) Add convenience fragments (H only).
      3) Export CSV: reinvent_fragments_vanilla.csv

    Returns the list of exported fragment SMILES.
    """
    frag_root = Path(cfg["Paths"].get("fragments", "results/reverse_QSAR"))
    # Handle both "reverse_QSAR" and "ReverseQSAR" naming conventions
    if not frag_root.exists():
        frag_root = Path(cfg["Paths"].get("fragments", "results/ReverseQSAR"))
    out_csv = frag_root / "reinvent_fragments_vanilla.csv"

    print(f"[VANILLA] Looking for vanilla fragments in: {frag_root}")

    # Load vanilla fragments (threshold only)
    fragments = _load_vanilla_fragments(frag_root)
    print(f"[VANILLA] Loaded {len(fragments)} vanilla fragments (threshold only)")

    # Convert to set and add convenience fragments
    fragment_set = set(fragments)
    fragment_set.add(HYDROGEN_FRAGMENT)

    # Sort for consistent output
    final_fragments = sorted(fragment_set)
    print(f"[VANILLA] Final library size: {len(final_fragments)} fragments (including convenience fragments)")

    # Export
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["fragment_smiles"])
        for s in final_fragments:
            w.writerow([s])

    logging.info(f"Vanilla library exported: {len(final_fragments)} fragments → {out_csv}")
    return final_fragments
