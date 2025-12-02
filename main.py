from pathlib import Path
import logging
import yaml
import pandas as pd
import warnings
import os

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*Loky-backed parallel loops.*')

# Prevent nested parallelism issues
os.environ['LOKY_MAX_CPU_COUNT'] = '4'

from data_preparation import create_final_dataset
from AC_analysis.ac_analysis import run_ac_analysis
from reverse_qsar.defragmentation import defragmenttion
from reverse_qsar.reinvent_library import run_reinvent_library
from predictive_model.prediction import run_qsar_evaluation
from generator import run_generator


def load_config(path: str):
    return yaml.safe_load(open(path, "r"))


def setup_logger():
    """Configure and return the main pipeline logger."""
    logger = logging.getLogger("ac_aware_pipeline")
    if not logger.hasHandlers():
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S"
        )
    return logger


def main():
    log = setup_logger()
    config = load_config("config.yml")

    paths = config["Paths"]
    final_csv = Path(paths["dataset"])

    # ─────────────────────────────────────────────────────────────
    # 1) Dataset preparation
    # ─────────────────────────────────────────────────────────────
    force_dataset = bool(config.get("REBUILD_DATASET", False))
    if force_dataset or not final_csv.exists():
        print("Building dataset (final_dataset.csv)...")
        create_final_dataset(config, log)
    else:
        print("final_dataset.csv already exists — skipping dataset build.")

    # Always load (downstream steps may need it)
    if not final_csv.exists():
        log.error("final_dataset.csv not found after preparation. Aborting.")
        return
    df = pd.read_csv(final_csv)

    # ─────────────────────────────────────────────────────────────
    # 2) Activity Cliffs Analysis
    # ─────────────────────────────────────────────────────────────
    ac_cfg = config.get("AC_Analysis", {})
    if ac_cfg.get("enable", True):
        ac_dir = Path(ac_cfg.get("results_dir", "results/AC_analysis"))
        ac_csv = ac_dir / "activity_cliffs.csv"
        force_ac = bool(ac_cfg.get("force_rerun", False))

        smiles_col = ac_cfg.get("smiles_col", "canonical_smiles")
        potency_col = ac_cfg.get("potency_col", "pIC50")

        if force_ac or not ac_csv.exists():
            if potency_col not in df.columns:
                log.error(f"'{potency_col}' column missing in dataset — cannot run AC analysis.")
            else:
                print("Running Activity Cliff analysis...")
                run_ac_analysis(
                    df,
                    smiles_col=smiles_col,
                    potency_col=potency_col,
                    results_dir=str(ac_dir),
                    config=config
                )
        else:
            print(f"AC analysis outputs found ({ac_csv}). Skipping AC step.")
    else:
        print("AC analysis disabled in config.")

    # ─────────────────────────────────────────────────────────────
    # 3) Reverse-QSAR (RF + SHAP + AC-enrichment)
    # ─────────────────────────────────────────────────────────────
    rqs_cfg = config.get("ReverseQSAR", {})
    if rqs_cfg.get("enable", True):
        frag_root = Path(paths.get("fragments", "results/ReverseQSAR"))
        force_rqsar = bool(rqs_cfg.get("force_rerun", False))
        
        # Check if ReverseQSAR folder exists and has results
        if frag_root.exists():
            # Check for the best model directory (LogReg, RandomForest, etc.)
            best_model = None
            for model_dir in frag_root.iterdir():
                if model_dir.is_dir() and (model_dir / "selected_fragments_with_ACflag.csv").exists():
                    best_model = model_dir
                    break
            
            if best_model:
                sel_csv = best_model / "selected_fragments_with_ACflag.csv"
            else:
                # Fallback to LogReg if no model directory found
                sel_csv = frag_root / "LogReg" / "selected_fragments_with_ACflag.csv"
        else:
            # Folder doesn't exist, will need to run defragmentation
            sel_csv = frag_root / "LogReg" / "selected_fragments_with_ACflag.csv"

        if force_rqsar or not sel_csv.exists():
            print("Running reverse-QSAR (defragmentation + SHAP + AC-enrichment)...")
            defragmenttion(config, log)
        else:
            print(f"Reverse-QSAR outputs found ({sel_csv}). Skipping Reverse-QSAR step.")
    else:
        print("⏭️  Reverse-QSAR disabled in config.")

    # ─────────────────────────────────────────────────────────────
    # 4) REINVENT fragment library
    # ─────────────────────────────────────────────────────────────
    lib_cfg = config.get("ReverseQSAR", {}).get("library", {})
    if lib_cfg.get("enable", True):
        frag_root = Path(paths.get("fragments", "results/reverse_QSAR"))
        all_csv = frag_root / "reinvent_fragments_all.csv"
        force_lib = bool(lib_cfg.get("force_rerun", False))
        
        if force_lib or not all_csv.exists():
            print("Building REINVENT plain-star fragment library from selected fragments...")
            from reverse_qsar.reinvent_library import run_reinvent_library
            frags = run_reinvent_library(config)
            print(f"Exported {len(frags)} fragments to {all_csv}")
        else:
            print(f"REINVENT library already present ({all_csv}). Skipping.")
        
        # Also create vanilla fragment library (for validation)
        vanilla_csv = frag_root / "reinvent_fragments_vanilla.csv"
        if force_lib or not vanilla_csv.exists():
            print("Building vanilla fragment library (threshold fragments only)...")
            from reverse_qsar.reinvent_library import run_vanilla_library
            vanilla_frags = run_vanilla_library(config)
            print(f"Exported {len(vanilla_frags)} vanilla fragments to {vanilla_csv}")
        else:
            print(f"Vanilla library already present ({vanilla_csv}). Skipping.")
    else:
        print("REINVENT library export disabled in config.")

    # ─────────────────────────────────────────────────────────────
    # 4) QSAR evaluation (resume-aware; do not block partial rebuilds)
    # ─────────────────────────────────────────────────────────────
    qsar_cfg = config.get("QSAR_Eval", {})
    if qsar_cfg.get("enable", True):
        paths_cfg = config.get("Paths", {})
        arts_cfg = config.get("Artifacts", {})
        eval_root = Path(paths_cfg.get("results_root", "results/predictive_model"))
        force_qsar = bool(qsar_cfg.get("force_rerun", False))
        resume_cfg = qsar_cfg.get("resume", {})
        resume_enabled = bool(resume_cfg.get("enable", True))
        overwrite = bool(resume_cfg.get("overwrite", False))

        # Report what's missing (for info only); do not block execution when resume is on
        required_keys = [
            "X_full",
            "metrics_summary"
        ]
        required_filenames = [arts_cfg[k] for k in required_keys if arts_cfg.get(k)]

        found_paths = {}
        missing_filenames = []
        if eval_root.exists():
            for fname in required_filenames:
                match = next(eval_root.rglob(fname), None)
                if match is not None and match.exists():
                    found_paths[fname] = match
                else:
                    missing_filenames.append(fname)
        else:
            missing_filenames = required_filenames[:]

        if force_qsar:
            print("QSAR evaluation: forced recomputation (force_rerun=true).")
            run_qsar_evaluation(config, log)
        else:
            if resume_enabled and not overwrite:
                # Always run; evaluator will skip existing artifacts and rebuild gaps
                if missing_filenames:
                    print(f"🧪 QSAR evaluation (resume): some artifacts are missing → will rebuild only gaps: "
                             f"{', '.join(missing_filenames)}")
                else:
                    print("🧪 QSAR evaluation (resume): artifacts detected — will skip existing and refresh summaries.")
                run_qsar_evaluation(config, log)
            else:
                # Resume disabled or overwrite requested — decide based on presence
                if missing_filenames:
                    print("QSAR evaluation: artifacts missing — running full evaluation.")
                    run_qsar_evaluation(config, log)
                else:
                    print("QSAR evaluation artifacts detected and resume disabled; skipping evaluation.")
    else:
        print("QSAR evaluation disabled in config.")

    # ─────────────────────────────────────────────────────────────
    # 5) Molecular Generation (AC-aware)
    # ─────────────────────────────────────────────────────────────
    gen_cfg = config.get("Generator", {})
    if gen_cfg.get("enable", True):
        gen_dir = Path(gen_cfg.get("output", {}).get("results_dir", "results/generation"))
        summary_json = gen_dir / gen_cfg.get("output", {}).get("files", {}).get("summary", "generation_summary.json")
        force_gen = bool(gen_cfg.get("force_rerun", False))
        
        # Check if vanilla-only mode is requested
        vanilla_only = bool(gen_cfg.get("vanilla_only", False))
        
        # Run CAFE generation (unless vanilla_only is True)
        if not vanilla_only:
            if force_gen or not summary_json.exists():
                print("Running AC-aware molecular generation (CAFE/CAFE LATE)...")
                result = run_generator(config, log, vanilla_mode=False)
                if result["success"]:
                    print("CAFE molecular generation completed successfully.")
                else:
                    log.error(f"CAFE molecular generation failed: {result.get('error', 'Unknown error')}")
            else:
                print(f"CAFE molecular generation outputs found ({summary_json}). Skipping CAFE generation step.")
        
        # Run vanilla generation (always run for validation, or if vanilla_only is True)
        vanilla_summary_json = gen_dir / "generation_summary_vanilla.json"
        if vanilla_only or force_gen or not vanilla_summary_json.exists():
            print("Running vanilla molecular generation (validation mode, no CAFE/CAFE LATE)...")
            
            # Vanilla fragment library should already exist from step 4, but check
            vanilla_csv = Path(paths.get("fragments", "results/reverse_QSAR")) / "reinvent_fragments_vanilla.csv"
            if not vanilla_csv.exists():
                log.warning(f"Vanilla fragment library not found at {vanilla_csv}. Creating it now...")
                from reverse_qsar.reinvent_library import run_vanilla_library
                try:
                    vanilla_frags = run_vanilla_library(config)
                    print(f"Vanilla fragment library prepared: {len(vanilla_frags)} fragments")
                except Exception as e:
                    log.error(f"Failed to create vanilla fragment library: {e}")
                    if not vanilla_only:
                        print("Skipping vanilla generation due to library creation failure.")
                    else:
                        raise
            
            result = run_generator(config, log, vanilla_mode=True)
            if result["success"]:
                print("Vanilla molecular generation completed successfully.")
            else:
                log.error(f"Vanilla molecular generation failed: {result.get('error', 'Unknown error')}")
        else:
            print(f"Vanilla molecular generation outputs found ({vanilla_summary_json}). Skipping vanilla generation step.")
    else:
        print("Molecular generation disabled in config.")

if __name__ == "__main__":
    main()