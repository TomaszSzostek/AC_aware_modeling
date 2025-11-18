"""
PCA Projections for Chemical Space Visualization

This module provides PCA-based visualization of chemical space using molecular
fingerprints, colored by activity classes (active/inactive), with activity cliffs (AC) 
highlighted as triangles.

Key Features:
- ECFP fingerprint generation for molecular representation
- PCA dimensionality reduction for 2D visualization
- Color coding by activity classes (active/inactive)
- Activity cliff highlighting (triangles)
- Publication-ready visualizations

Results are saved in `results/AC_analysis/pca_projection_(Figure_S2).png`.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.decomposition import PCA

# Import AC analysis functions for fingerprint computation
from AC_analysis.ac_analysis import compute_fingerprints

# ======================================================
# PCA Projection Visualization
# ======================================================

def create_pca_projection(
    df: pd.DataFrame,
    smiles_col: str,
    potency_col: str,
    cliffs_df: pd.DataFrame,
    outpath: str,
    n_components: int = 2,
    radius: int = 2,
    nbits: int = 2048,
    random_state: int = 42,
    activity_threshold: float = 5.0
) -> None:
    """
    Create PCA projection of chemical space with activity class coloring and AC highlighting.
    
    Args:
        df: DataFrame containing molecular data with SMILES and pIC50
        smiles_col: Name of column containing SMILES strings
        potency_col: Name of column containing potency values (pIC50)
        cliffs_df: DataFrame with activity cliffs (columns: mol_i, mol_j, ...)
        outpath: Output path for PCA plot (PNG)
        n_components: Number of PCA components (default: 2)
        radius: Fingerprint radius (default: 2, equivalent to ECFP4)
        nbits: Number of bits in fingerprint (default: 2048)
        random_state: Random seed for reproducibility (default: 42)
        activity_threshold: pIC50 threshold for active/inactive classification (default: 5.0)
    """
    print(f"Computing ECFP fingerprints for {len(df)} molecules...")
    
    # Compute fingerprints
    fps, mols = compute_fingerprints(df[smiles_col].tolist(), radius=radius, nbits=nbits)
    
    # Filter out invalid molecules
    valid_indices = [i for i, fp in enumerate(fps) if fp is not None]
    fps_valid = [fps[i] for i in valid_indices]
    df_valid = df.iloc[valid_indices].copy()
    
    print(f"   Valid molecules: {len(fps_valid)}")
    
    # Convert fingerprints to numpy array
    fps_array = np.array([list(fp.ToBitString()) for fp in fps_valid])
    fps_array = fps_array.astype(np.float32)
    
    print("Computing PCA projection...")
    
    # Compute PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    coords = pca.fit_transform(fps_array)
    
    # Calculate variance explained
    var_explained = pca.explained_variance_ratio_
    total_var = sum(var_explained)
    
    print(f"   PCA explained variance: PC1={var_explained[0]:.1%}, PC2={var_explained[1]:.1%}, Total={total_var:.1%}")
    
    # Prepare plot data
    df_plot = df_valid.copy()
    df_plot["PC1"] = coords[:, 0]
    df_plot["PC2"] = coords[:, 1]
    
    # Identify AC molecules
    if cliffs_df is not None and not cliffs_df.empty:
        cliff_mols = set(cliffs_df["mol_i"]).union(set(cliffs_df["mol_j"]))
        df_plot["is_ac"] = df_plot[smiles_col].isin(cliff_mols)
        n_ac = df_plot["is_ac"].sum()
        print(f"   Activity cliffs identified: {n_ac} molecules")
    else:
        df_plot["is_ac"] = False
        n_ac = 0
        print("   No activity cliffs found.")
    
    # Classify molecules as active/inactive based on pIC50 threshold
    df_plot["activity_class"] = df_plot[potency_col].apply(
        lambda x: "Active" if x >= activity_threshold else "Inactive"
    )
    
    # Separate AC and non-AC molecules
    non_ac_df = df_plot[~df_plot["is_ac"]]
    ac_df = df_plot[df_plot["is_ac"]]
    
    # Define pastel colors for activity classes
    activity_colors = {
        "Active": "#a5d6a7",    # Pastel green
        "Inactive": "#ef9a9a"    # Pastel pink/red
    }
    ac_color = "#90caf9"  # Pastel blue for AC molecules
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot all non-AC molecules colored by activity class (larger points)
    if len(non_ac_df) > 0:
        for activity_class, color in activity_colors.items():
            class_data = non_ac_df[non_ac_df["activity_class"] == activity_class]
            if len(class_data) > 0:
                ax.scatter(
                    class_data["PC1"], class_data["PC2"],
                    c=color,
                    alpha=0.7,
                    s=120,  # Larger points
                    edgecolor="white",
                    linewidths=0.5,
                    zorder=1,
                    label=activity_class
                )
    
    # Overlay AC molecules as triangles in pastel blue (regardless of activity class)
    if len(ac_df) > 0:
        ax.scatter(
            ac_df["PC1"], ac_df["PC2"],
            c=ac_color,
            marker="^",  # Triangle marker
            s=180,  # Larger triangles
            alpha=0.85,
            edgecolor="white",
            linewidths=1.2,
            zorder=2,
            label=f"Activity Cliffs (n={n_ac})"
        )
    
    # Create legend for activity classes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=activity_colors["Active"], label="Active", edgecolor="white", linewidth=0.5),
        Patch(facecolor=activity_colors["Inactive"], label="Inactive", edgecolor="white", linewidth=0.5)
    ]
    if n_ac > 0:
        legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', 
                                         markerfacecolor=ac_color, markersize=12, 
                                         markeredgecolor='white', markeredgewidth=1.2,
                                         label=f'Activity Cliffs (n={n_ac})', linestyle='None'))
    ax.legend(handles=legend_elements, loc="upper right", fontsize=14, framealpha=0.95)
    
    # Labels with larger font
    ax.set_xlabel(
        f"PC1 ({var_explained[0]:.1%} variance explained)",
        fontsize=18,
        fontweight="bold"
    )
    ax.set_ylabel(
        f"PC2 ({var_explained[1]:.1%} variance explained)",
        fontsize=18,
        fontweight="bold"
    )
    
    # Increase tick label sizes
    ax.tick_params(axis='both', which='major', labelsize=14)
    
    # No grid
    ax.grid(False)
    
    # Clean style
    sns.despine(ax=ax, trim=True)
    
    plt.tight_layout()
    
    # Save figure
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"PCA projection saved to: {outpath}")


# ======================================================
# Main execution
# ======================================================

def main():
    """Main function to run PCA projection analysis."""
    import yaml
    
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Get paths from config
    dataset_path = Path(config["Paths"]["dataset"])
    ac_results_dir = Path(config["Paths"]["ac_analysis"])
    cliffs_path = ac_results_dir / "activity_cliffs.csv"
    output_path = ac_results_dir / "pca_projection_(Figure_S2).png"
    
    # AC analysis config
    ac_config = config.get("AC_Analysis", {})
    smiles_col = ac_config.get("smiles_col", "canonical_smiles")
    potency_col = ac_config.get("potency_col", "pIC50")
    
    # Calculate activity threshold from THRESHOLD_NM
    threshold_nm = config.get("THRESHOLD_NM", 10000)
    activity_threshold = -np.log10(threshold_nm * 1e-9)  # Convert nM to pIC50
    
    print("=" * 70)
    print("PCA Projection of Chemical Space")
    print("=" * 70)
    print(f"Dataset: {dataset_path}")
    print(f"Activity cliffs: {cliffs_path}")
    print(f"Output: {output_path}")
    print(f"Activity threshold: pIC50 >= {activity_threshold:.2f} (IC50 <= {threshold_nm} nM)")
    print()
    
    # Load dataset
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    print(f"Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"   Loaded {len(df)} molecules")
    
    # Load activity cliffs (if available)
    cliffs_df = None
    if cliffs_path.exists():
        print(f"Loading activity cliffs...")
        cliffs_df = pd.read_csv(cliffs_path)
        print(f"   Loaded {len(cliffs_df)} activity cliff pairs")
    else:
        print(f"   Activity cliffs file not found: {cliffs_path}")
        print("   Running without AC highlighting.")
    
    # Create PCA projection
    create_pca_projection(
        df=df,
        smiles_col=smiles_col,
        potency_col=potency_col,
        cliffs_df=cliffs_df,
        outpath=str(output_path),
        random_state=config["Cheminformatics"]["random_state"],
        activity_threshold=activity_threshold
    )
    
    print()
    print("=" * 70)
    print("PCA projection analysis complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
