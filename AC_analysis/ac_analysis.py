"""
Activity cliff analysis utilities.

This module detects activity cliffs in QSAR datasets, computes SALI scores and
produces PCA/t-SNE-based visualizations stored under `results/AC_analysis/`.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
from sklearn.manifold import TSNE
import networkx as nx

# ======================================================
# Fingerprint generation
# ======================================================

def compute_fingerprints(smiles_list: list[str], radius: int = 2, nbits: int = 2048):
    """
    Generate ECFP (Morgan) fingerprints for all molecules.
    
    Args:
        smiles_list: List of SMILES strings
        radius: Fingerprint radius (default: 2, equivalent to ECFP4)
        nbits: Number of bits in fingerprint (default: 2048)
        
    Returns:
        Tuple of (fingerprints_list, molecules_list) where None indicates invalid molecules
    """
    fps = []
    mols = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
            fp = gen.GetFingerprint(mol)
            fps.append(fp)
            mols.append(mol)
        else:
            fps.append(None)
            mols.append(None)
    return fps, mols


# ======================================================
# Detect activity cliffs
# ======================================================

def detect_cliffs(
    df: pd.DataFrame,
    smiles_col: str,
    potency_col: str,
    sim_threshold: float,
    potency_diff: float
) -> pd.DataFrame:
    """
    Detect activity cliffs in dataset based on similarity and potency difference.
    
    Activity cliff definition:
    - Tanimoto similarity >= sim_threshold (typically 0.8)
    - Absolute potency difference >= potency_diff (typically 1.0 log units, i.e., 10-fold)
    
    Also computes SALI (Structure-Activity Landscape Index) for each cliff pair:
        SALI = ΔpIC50 / (1 - Tanimoto_similarity)
    
    Args:
        df: DataFrame containing molecular data
        smiles_col: Name of column containing SMILES strings
        potency_col: Name of column containing potency values (typically pIC50)
        sim_threshold: Tanimoto similarity threshold (0.0-1.0)
        potency_diff: Minimum potency difference in log units
        
    Returns:
        DataFrame with columns: mol_i, mol_j, sim, ∆p, SALI
    """
    fps, _ = compute_fingerprints(df[smiles_col].tolist())
    cliff_data = []
    n = len(fps)

    for i in range(n):
        for j in range(i + 1, n):
            if fps[i] and fps[j]:
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                if sim >= sim_threshold:
                    act_i, act_j = df.iloc[i][potency_col], df.iloc[j][potency_col]
                    diff = abs(act_i - act_j)
                    if diff >= potency_diff:
                        sali = diff / (1 - sim + 1e-6)
                        cliff_data.append({
                            "mol_i": df.iloc[i][smiles_col],
                            "mol_j": df.iloc[j][smiles_col],
                            "sim": sim,
                            "∆p": diff,
                            "SALI": sali
                        })

    return pd.DataFrame(cliff_data)


# ======================================================
# Visualization functions
# ======================================================

def plot_sali_scatter(
    cliffs_df: pd.DataFrame,
    df: pd.DataFrame,
    smiles_col: str,
    potency_col: str,
    outpath: str,
    config: dict
) -> None:
    """
    Create SALI scatterplot visualization.
    
    Generates a scatterplot of Tanimoto similarity vs. ΔpIC50 with:
    - All molecular pairs shown in light grey
    - Activity cliff pairs highlighted with SALI coloring (viridis colormap)
    - Threshold lines indicating cliff definition boundaries
    
    Also saves all pair data (similarity, ΔpIC50, SALI, is_cliff) to CSV.
    
    Args:
        cliffs_df: DataFrame of detected activity cliffs
        df: Full dataset DataFrame
        smiles_col: Name of SMILES column
        potency_col: Name of potency column
        outpath: Output path for plot (PNG)
        config: Configuration dictionary with Cheminformatics thresholds
    """
    sim_thr = config["Cheminformatics"]["similarity_threshold"]
    potency_diff = config["Cheminformatics"]["potency_diff_threshold"]

    fps, _ = compute_fingerprints(df[smiles_col].tolist())
    all_pairs = []
    n = len(fps)

    for i in range(n):
        for j in range(i + 1, n):
            if fps[i] and fps[j]:
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                act_i, act_j = df.iloc[i][potency_col], df.iloc[j][potency_col]
                diff = abs(act_i - act_j)
                sali = diff / (1 - sim + 1e-6)
                is_cliff = (sim >= sim_thr) and (diff >= potency_diff)
                all_pairs.append([sim, diff, sali, is_cliff])

    sali_df = pd.DataFrame(all_pairs, columns=["sim", "∆p", "SALI", "is_cliff"])
    sali_df.to_csv(os.path.join(os.path.dirname(outpath), "sali_all.csv"), index=False)

    # ---- plotting ----
    plt.figure(figsize=(8, 7))

    # non-cliffs
    non_cliffs = sali_df[~sali_df["is_cliff"]]
    plt.scatter(
        non_cliffs["sim"], non_cliffs["∆p"],
        c="lightgrey", alpha=0.4, s=20, label="Non-cliff pairs"
    )

    # cliffs
    cliffs = sali_df[sali_df["is_cliff"]]
    scatter = plt.scatter(
        cliffs["sim"], cliffs["∆p"],
        c=cliffs["SALI"], cmap="viridis",
        edgecolor="k", linewidth=0.3, s=50, alpha=0.9,
        label="Cliff pairs"
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label("SALI", fontsize=12)

    # thresholds
    plt.axvline(x=sim_thr, color="red", linestyle="--", alpha=0.7, label=f"Similarity ≥ {sim_thr}")
    plt.axhline(y=potency_diff, color="blue", linestyle="--", alpha=0.7, label=f"ΔpIC50 ≥ {potency_diff}")

    plt.xlabel("Tanimoto similarity", fontsize=12)
    plt.ylabel("ΔpIC50 (log units)", fontsize=12)
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()



def plot_tsne(
    df: pd.DataFrame,
    smiles_col: str,
    potency_col: str,
    cliffs_df: pd.DataFrame,
    outpath: str,
    perplexity: int = 30,
    random_state: int = 42
) -> None:
    """
    Create t-SNE embedding visualization with activity cliff highlighting.
    
    Generates a 2D t-SNE embedding of molecular fingerprints with:
    - Color-coding by potency (coolwarm colormap)
    - Highlighting of molecules involved in activity cliffs
    
    Args:
        df: Dataset DataFrame
        smiles_col: Name of SMILES column
        potency_col: Name of potency column
        cliffs_df: DataFrame of detected activity cliffs
        outpath: Output path for plot (PNG)
        perplexity: t-SNE perplexity parameter (default: 30)
        random_state: Random seed for reproducibility (default: 42)
    """
    fps, _ = compute_fingerprints(df[smiles_col].tolist())
    fps_array = np.array([list(fp.ToBitString()) for fp in fps if fp])
    fps_array = fps_array.astype(np.float32)

    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    coords = tsne.fit_transform(fps_array)

    df_plot = df.copy()
    df_plot["x"] = coords[:, 0]
    df_plot["y"] = coords[:, 1]

    # mark cliff molecules
    cliff_mols = set(cliffs_df["mol_i"]).union(set(cliffs_df["mol_j"]))
    df_plot["is_cliff"] = df_plot[smiles_col].isin(cliff_mols)

    plt.figure(figsize=(8, 7))
    scatter = plt.scatter(
        df_plot["x"], df_plot["y"],
        c=df_plot[potency_col],
        cmap="coolwarm",  # pastel-friendly continuous colormap
        alpha=0.6, s=40, edgecolor="none"
    )
    plt.colorbar(scatter, label=potency_col)

    # overlay cliffs
    cliff_points = df_plot[df_plot["is_cliff"]]
    plt.scatter(
        cliff_points["x"], cliff_points["y"],
        c="green", s=60, marker="o", edgecolor="white", linewidth=0.6,
        label="Cliff molecules"
    )

    plt.legend()
    plt.xlabel("t-SNE 1", fontsize=12)
    plt.ylabel("t-SNE 2", fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_combined_ac_figures(
    cliffs_df: pd.DataFrame,
    df: pd.DataFrame,
    smiles_col: str,
    potency_col: str,
    outpath: str,
    config: dict,
    perplexity: int = 30,
    random_state: int = 42
) -> None:
    """
    Create combined publication-quality figure with SALI and t-SNE plots side by side.
    
    Generates a two-panel figure:
    - Panel A: SALI scatterplot (similarity vs. ΔpIC50)
    - Panel B: t-SNE embedding with cliff highlighting
    
    Args:
        cliffs_df: DataFrame of detected activity cliffs
        df: Full dataset DataFrame
        smiles_col: Name of SMILES column
        potency_col: Name of potency column
        outpath: Output path for combined plot (PNG)
        config: Configuration dictionary with Cheminformatics thresholds
        perplexity: t-SNE perplexity parameter (default: 30)
        random_state: Random seed for reproducibility (default: 42)
    """
    sim_thr = config["Cheminformatics"]["similarity_threshold"]
    potency_diff = config["Cheminformatics"]["potency_diff_threshold"]
    
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # ============================================================
    # PANEL A: t-SNE Embedding
    # ============================================================
    ax_tsne = axes[0]
    
    # Compute t-SNE
    fps_tsne, _ = compute_fingerprints(df[smiles_col].tolist())
    fps_array = np.array([list(fp.ToBitString()) for fp in fps_tsne if fp])
    fps_array = fps_array.astype(np.float32)
    
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity)
    coords = tsne.fit_transform(fps_array)
    
    df_plot = df.copy()
    df_plot["x"] = coords[:, 0]
    df_plot["y"] = coords[:, 1]
    
    # Mark cliff molecules
    cliff_mols = set(cliffs_df["mol_i"]).union(set(cliffs_df["mol_j"]))
    df_plot["is_cliff"] = df_plot[smiles_col].isin(cliff_mols)
    
    # Plot all points colored by potency
    scatter_tsne = ax_tsne.scatter(
        df_plot["x"], df_plot["y"],
        c=df_plot[potency_col],
        cmap="coolwarm",
        alpha=0.6, s=40, edgecolor="none"
    )
    cbar_tsne = plt.colorbar(scatter_tsne, ax=ax_tsne)
    cbar_tsne.set_label(potency_col, fontsize=16)
    cbar_tsne.ax.tick_params(labelsize=12)
    
    # Overlay cliff molecules
    cliff_points = df_plot[df_plot["is_cliff"]]
    ax_tsne.scatter(
        cliff_points["x"], cliff_points["y"],
        c="green", s=80, marker="o", edgecolor="white", linewidth=1.0,
        label="Cliff molecules", alpha=0.9
    )
    
    ax_tsne.set_xlabel("t-SNE 1", fontsize=16)
    ax_tsne.set_ylabel("t-SNE 2", fontsize=16)
    ax_tsne.tick_params(axis='both', labelsize=12)
    ax_tsne.legend(fontsize=12, frameon=True, loc='upper right')
    ax_tsne.grid(True, alpha=0.2)
    
    # Panel label A (top-left corner)
    ax_tsne.text(0.02, 0.98, 'A.', transform=ax_tsne.transAxes,
                fontsize=20, fontweight='bold', va='top', ha='left')
    
    # ============================================================
    # PANEL B: SALI Scatterplot
    # ============================================================
    ax_sali = axes[1]
    
    # Compute all pairs for SALI plot
    fps, _ = compute_fingerprints(df[smiles_col].tolist())
    all_pairs = []
    n = len(fps)
    
    for i in range(n):
        for j in range(i + 1, n):
            if fps[i] and fps[j]:
                sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                act_i, act_j = df.iloc[i][potency_col], df.iloc[j][potency_col]
                diff = abs(act_i - act_j)
                sali = diff / (1 - sim + 1e-6)
                is_cliff = (sim >= sim_thr) and (diff >= potency_diff)
                all_pairs.append([sim, diff, sali, is_cliff])
    
    sali_df = pd.DataFrame(all_pairs, columns=["sim", "∆p", "SALI", "is_cliff"])
    
    # Plot non-cliffs
    non_cliffs = sali_df[~sali_df["is_cliff"]]
    ax_sali.scatter(
        non_cliffs["sim"], non_cliffs["∆p"],
        c="lightgrey", s=10, alpha=0.3, label="Non-cliffs"
    )
    
    # Plot cliffs with SALI coloring
    cliff_pairs = sali_df[sali_df["is_cliff"]]
    if not cliff_pairs.empty:
        scatter = ax_sali.scatter(
            cliff_pairs["sim"], cliff_pairs["∆p"],
            c=cliff_pairs["SALI"], cmap="viridis",
            s=60, alpha=0.9, edgecolor="black", linewidth=0.5,
            label="Activity cliffs"
        )
        cbar = plt.colorbar(scatter, ax=ax_sali)
        cbar.set_label("SALI", fontsize=16)
        cbar.ax.tick_params(labelsize=12)
    
    # Threshold lines
    ax_sali.axvline(sim_thr, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax_sali.axhline(potency_diff, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    
    ax_sali.set_xlabel("Tanimoto Similarity", fontsize=16)
    ax_sali.set_ylabel("ΔpIC₅₀", fontsize=16)
    ax_sali.tick_params(axis='both', labelsize=12)
    ax_sali.legend(fontsize=12, frameon=True, loc='upper right')
    ax_sali.grid(True, alpha=0.2)
    
    # Panel label B (top-left corner)
    ax_sali.text(0.02, 0.98, 'B.', transform=ax_sali.transAxes,
                fontsize=20, fontweight='bold', va='top', ha='left')
    
    plt.tight_layout()
    plt.savefig(outpath, dpi=500, bbox_inches='tight')
    plt.close()


# ======================================================
# Main function for pipeline integration
# ======================================================

def run_ac_analysis(
    df: pd.DataFrame,
    smiles_col: str,
    potency_col: str,
    results_dir: str,
    config: dict
) -> pd.DataFrame:
    """
    Main function for activity cliffs analysis pipeline.
    
    Orchestrates the complete AC analysis workflow:
    1. Detect activity cliffs based on similarity and potency thresholds
    2. Generate SALI scatterplot visualization
    3. Generate t-SNE embedding with cliff highlighting (if enabled)
    4. Save results to CSV
    
    Args:
        df: Dataset DataFrame with molecular and activity data
        smiles_col: Name of SMILES column
        potency_col: Name of potency column (typically "pIC50")
        results_dir: Directory to save results
        config: Configuration dictionary with thresholds and parameters
        
    Returns:
        DataFrame of detected activity cliff pairs
    """
    os.makedirs(results_dir, exist_ok=True)
    logging.info("Starting Activity Cliffs analysis...")

    sim_thr = config["Cheminformatics"]["similarity_threshold"]
    potency_diff = config["Cheminformatics"]["potency_diff_threshold"]
    embedding = config["AC_Analysis"].get("embedding", "tsne")
    tsne_params = config["AC_Analysis"].get("tsne", {})
    tsne_params["random_state"] = config["Cheminformatics"]["random_state"]

    cliffs_df = detect_cliffs(
        df,
        smiles_col=smiles_col,
        potency_col=potency_col,
        sim_threshold=sim_thr,
        potency_diff=potency_diff
    )
    cliffs_df.to_csv(os.path.join(results_dir, "activity_cliffs.csv"), index=False)
    logging.info(f"Detected {len(cliffs_df)} activity cliff pairs")

    if not cliffs_df.empty:
        # Generate individual plots (for backward compatibility)
        plot_sali_scatter(
            cliffs_df, df, smiles_col, potency_col,
            os.path.join(results_dir, "sali_scatter.png"),
            config
        )

        if embedding == "tsne":
            plot_tsne(
                df, smiles_col, potency_col, cliffs_df,
                os.path.join(results_dir, "tsne_cliffs.png"),
                **tsne_params
            )
        
        # Generate combined publication-quality figure
        plot_combined_ac_figures(
            cliffs_df, df, smiles_col, potency_col,
            os.path.join(results_dir, "combined_ac_figure.png"),
            config,
            **tsne_params
        )
        logging.info("Generated combined AC figure (SALI + t-SNE)")
    
    logging.info("Activity Cliffs analysis completed.")
    return cliffs_df


