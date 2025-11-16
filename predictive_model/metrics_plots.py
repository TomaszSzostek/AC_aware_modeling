"""
Metrics and Plots Module for QSAR Evaluation

This module provides comprehensive metrics computation and publication-ready
visualization functions for QSAR model evaluation. It includes activity cliff-aware
metrics, calibration analysis, and statistical validation plots.

Features:
- Activity cliff-aware metrics (Cliff_RMSE, Cliff_F1, etc.)
- Standard classification metrics (AUROC, AUPRC, F1, MCC, etc.)
- Calibration metrics (Brier score, ECE)
- Publication-ready visualizations (ROC, PR, confusion matrices, etc.)
- Bootstrap confidence intervals
- Model comparison plots

"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import pathlib
import sys
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


# Publication settings
PUB_DPI = 500
PALETTE = {
    "blue": "#8ecae6",      # pastel blue
    "green": "#a8ddb5",     # pastel green
    "purple": "#c7b9ff",    # pastel purple
    "orange": "#fdbf6f",    # pastel orange
    "red": "#f4aaaa",       # pastel red
    "gray": "#cfcfcf",      # pastel gray
    "ink": "#3a3a3a",       # dark labels
}

# Soft colormap for confusion matrices
CMAP_SOFT = ListedColormap(["#f0f4ff", "#d7e3ff", "#bcd4ff", "#a1c6ff", "#85b7ff", "#6aa8ff"])

# Configure matplotlib for publication-quality plots
plt.rcParams.update({
    "figure.dpi": PUB_DPI,
    "savefig.dpi": PUB_DPI,
    "axes.edgecolor": "#aaaaaa",
    "axes.labelcolor": PALETTE["ink"],
    "xtick.color": PALETTE["ink"],
    "ytick.color": PALETTE["ink"],
    "font.size": 10,
})


def ensure_dir(p: pathlib.Path):
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def logits_from_proba(p, eps=1e-6):
    """Convert probabilities to logits."""
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def ece_score(y_true, y_prob, n_bins=15):
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve (default: 15)
        
    Returns:
        ECE score as float
    """
    pt, pp = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    # ECE = mean absolute calibration error across uniform bins
    return float(np.mean(np.abs(pt - pp)))


def compute_metrics(y_true, y_prob, pairs_cliff_test, in_cliff_nodes, threshold: float = 0.5):
    """
    Compute comprehensive metrics including activity cliff-aware metrics.
    
    This function computes standard classification metrics as well as
    activity cliff-specific metrics for evaluating QSAR models on activity cliffs.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        pairs_cliff_test: List of activity cliff pairs (unused, kept for API consistency)
        in_cliff_nodes: Boolean array indicating which nodes are in activity cliffs
        threshold: Decision threshold for binary predictions (default: 0.5)
        
    Returns:
        Dictionary of computed metrics
    """
    y_pred = (y_prob >= float(threshold)).astype(int)
    out = {}
    out["AUROC"] = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float("nan")
    out["AUPRC_active"] = average_precision_score(y_true, y_prob)
    out["F1"] = f1_score(y_true, y_pred, zero_division=0)
    out["MCC"] = matthews_corrcoef(y_true, y_pred) if (
                len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1) else 0.0
    out["ACC"] = accuracy_score(y_true, y_pred)
    out["Brier"] = brier_score_loss(y_true, y_prob)
    out["ECE"] = ece_score(y_true, y_prob, n_bins=15)
    out["RMSE"] = float(np.sqrt(np.mean((y_prob - y_true) ** 2)))
    if in_cliff_nodes.sum() > 0:
        yc = y_true[in_cliff_nodes]
        pc = y_prob[in_cliff_nodes]
        ypc = (pc >= float(threshold)).astype(int)
        out["Cliff_Brier"] = brier_score_loss(yc, pc)
        out["Cliff_ECE"] = ece_score(yc, pc, n_bins=15)
        out["Cliff_RMSE"] = float(np.sqrt(np.mean((pc - yc) ** 2)))
        out["Cliff_F1"] = f1_score(yc, ypc, zero_division=0)
        # vs RMSE global
        out["Delta_RMSE"] = out["Cliff_RMSE"] - out["RMSE"]

        # PR-AUC on cliffs
        out["AUPRC_cliff"] = average_precision_score(yc, pc)
    else:
        out["Cliff_Brier"] = out["Cliff_ECE"] = out["Cliff_RMSE"] = out["Delta_RMSE"] = float("nan")
        out["AUPRC_cliff"] = float("nan")

    out["Cliff_NodeShare"] = float(in_cliff_nodes.mean())

    return out


@contextlib.contextmanager
def _suppress_output(enabled: bool = True):
    """Context manager to suppress stdout/stderr."""
    if not enabled:
        yield
        return
    new_out, new_err = io.StringIO(), io.StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def plot_confusion(y_true, y_prob, out_png: pathlib.Path, threshold: float = 0.5):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        out_png: Output path for PNG file
        threshold: Decision threshold (default: 0.5)
    """
    classes = ["inactive", "active"]
    y_pred = (y_prob >= float(threshold)).astype(int)
    cmx = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cmx, interpolation='nearest', cmap=CMAP_SOFT, vmin=0)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(classes)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for (i, j), v in np.ndenumerate(cmx):
        ax.text(j, i, str(int(v)), ha='center', va='center', color="#222222")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    ensure_dir(pathlib.Path(out_png).parent)
    plt.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _ci_from_bootstrap(y_true, y_prob, what="roc", it=400, seed=42, points=50):
    """
    Compute bootstrap confidence intervals for ROC or PR curves.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        what: "roc" or "pr" (default: "roc")
        it: Number of bootstrap iterations (default: 400)
        seed: Random seed (default: 42)
        points: Number of points for interpolation (default: 50)
        
    Returns:
        Tuple of (xs, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    xs = np.linspace(0, 1, points)
    curves = []
    for _ in range(it):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, yp = y_true[idx], y_prob[idx]
        if what == "roc":
            disp = RocCurveDisplay.from_predictions(yt, yp)
            fpr, tpr = disp.fpr, disp.tpr
            curves.append(np.interp(xs, fpr, tpr, left=0, right=1))
        else:
            disp = PrecisionRecallDisplay.from_predictions(yt, yp)
            recall, precision = disp.recall, disp.precision
            curves.append(np.interp(xs, recall, precision, left=precision.min(), right=precision.max()))
    arr = np.vstack(curves)
    return xs, np.percentile(arr, 2.5, axis=0), np.percentile(arr, 97.5, axis=0)


def plot_roc_pr_hist_rel(y_true, y_prob, out_dir: pathlib.Path, prefix: str):
    """
    Plot ROC, PR, reliability, and histogram plots.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        out_dir: Output directory for plots
        prefix: Prefix for output filenames
    """
    base = f"{prefix}_" if prefix else ""
    # ROC
    fig, ax = plt.subplots(figsize=(4, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    for l in ax.lines:
        l.set_color(PALETTE["blue"])
        l.set_linewidth(2)
    xs, lo, hi = _ci_from_bootstrap(y_true, y_prob, "roc")
    ax.fill_between(xs, lo, hi, alpha=0.2)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    plt.tight_layout()
    fig.savefig(out_dir / f"{base}roc.png")
    plt.close(fig)
    # PR
    fig, ax = plt.subplots(figsize=(4, 4))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, ax=ax)
    for l in ax.lines:
        l.set_color(PALETTE["green"])
        l.set_linewidth(2)
    xs, lo, hi = _ci_from_bootstrap(y_true, y_prob, "pr")
    ax.fill_between(xs, lo, hi, alpha=0.2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    plt.tight_layout()
    fig.savefig(out_dir / f"{base}pr.png")
    plt.close(fig)
    # reliability
    bins = np.linspace(0, 1, 11)
    centers = 0.5 * (bins[:-1] + bins[1:])
    pred = (y_prob >= 0.5).astype(int)
    err = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (y_prob >= lo) & (y_prob < hi)
        err.append(np.nan if not m.any() else 1 - accuracy_score(y_true[m], pred[m]))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(centers, err, 's-', linewidth=2, color=PALETTE["purple"])
    ax.set_xlabel("Confidence bin center")
    ax.set_ylabel("Error (1-ACC)")
    plt.tight_layout()
    fig.savefig(out_dir / f"{base}reliability.png")
    plt.close(fig)
    # histogram
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.hist(y_prob, bins=20, color=PALETTE["orange"], edgecolor="#ffffff")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    plt.tight_layout()
    fig.savefig(out_dir / f"{base}proba_hist.png")
    plt.close(fig)

    # --- extras.json: AUROC CI, AUPRC, Brier, ECE, RMSE ---
    try:
        ensure_dir(out_dir)
        try:
            auc_med, auc_lo, auc_hi = bootstrap_auc(np.asarray(y_true), np.asarray(y_prob), n=500, seed=42)
        except Exception:
            auc_med = auc_lo = auc_hi = float("nan")
        extras = {
            "AUROC": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) == 2 else float("nan"),
            "AUROC_bootstrap": {"median": float(auc_med), "low": float(auc_lo), "high": float(auc_hi), "n": 500},
            "AUPRC_active": float(average_precision_score(y_true, y_prob)),
            "Brier": float(brier_score_loss(y_true, y_prob)),
            "ECE": float(ece_score(np.asarray(y_true), np.asarray(y_prob), n_bins=15)),
            "RMSE": float(np.sqrt(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))),
        }
        with open(out_dir / f"{base}extras.json", "w") as f:
            json.dump(extras, f, indent=2)
    except Exception as e:
        logging.getLogger(__name__).warning(f"[Extras] writing extras.json failed: {e}")


def plot_metrics_summary(csv_path: pathlib.Path, out_png: pathlib.Path):
    """
    Plot summary of model metrics as bar chart.
    
    Args:
        csv_path: Path to CSV file containing metrics
        out_png: Output path for PNG file
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
    if df is None or df.empty:
        return

    for k in ["model", "AUROC", "AUPRC_active", "Cliff_RMSE"]:
        if k not in df.columns:
            df[k] = np.nan

    long = df.melt(id_vars=["model"],
                   value_vars=["AUROC", "AUPRC_active", "Cliff_RMSE"],
                   var_name="Metric", value_name="Score")

    fig, ax = plt.subplots(figsize=(10, 4), dpi=PUB_DPI)
    sns.barplot(data=long, x="model", y="Score", hue="Metric", palette="pastel", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.legend(title="Metric", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)
    for tick in ax.get_xticklabels():
        tick.set_rotation(35)
        tick.set_ha("right")
        tick.set_fontsize(8)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    ensure_dir(out_png.parent)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def _prepare_backbone_comparison_df(all_metrics_csv: pathlib.Path, log: logging.Logger) -> pd.DataFrame:
    """
    Prepare DataFrame for backbone comparison plot.
    
    Args:
        all_metrics_csv: Path to CSV file with all metrics
        log: Logger instance
        
    Returns:
        Prepared DataFrame
    """
    if not all_metrics_csv.exists():
        log.warning(f"Backbone-comparison source not found: {all_metrics_csv}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(all_metrics_csv)
    except Exception as e:
        log.warning(f"Could not read {all_metrics_csv}: {e}")
        return pd.DataFrame()

    # Required
    for col in ["model", "backbone"]:
        if col not in df.columns:
            log.warning(f"Missing column '{col}' in metrics CSV; cannot compare backbones.")
            return pd.DataFrame()

    df = df.drop_duplicates(subset=["model", "backbone", "Threshold"], keep="first").copy()
    df["model"] = df["model"].astype(str)
    df["backbone"] = df["backbone"].astype(str)
    return df


def plot_backbone_comparison(
        all_metrics_csv: pathlib.Path,
        out_png: pathlib.Path,
        metric: str,
        log: logging.Logger,
        order_models: Optional[List[str]] = None,
):
    """
    Plot backbone comparison as grouped bar chart.
    
    Args:
        all_metrics_csv: Path to CSV file with all metrics
        out_png: Output path for PNG file
        metric: Metric name to plot
        log: Logger instance
        order_models: Optional list of model names in desired order
    """
    df = _prepare_backbone_comparison_df(all_metrics_csv, log)
    if df.empty:
        log.warning("No data for backbone comparison plot.")
        return

    metric_map = {
        "auroc": "AUROC",
        "auc": "AUROC",
        "auprc": "AUPRC_active",
        "auprc_active": "AUPRC_active",
        "brier": "Brier",
        "ece": "ECE",
        "rmse": "RMSE",
        "rmse_cliff": "Cliff_RMSE",
        "cliff_rmse": "Cliff_RMSE",
        "acc": "ACC",
        "accuracy": "ACC",
        "f1": "F1",
        "mcc": "MCC",
    }
    key = metric_map.get(str(metric).lower(), "Cliff_RMSE")
    if key not in df.columns:
        df[key] = np.nan

    df_plot = df[["model", "backbone", key]].copy().rename(columns={key: "score"})

    if order_models is None:
        best_per_model = df_plot.groupby("model")["score"].max().sort_values(ascending=False)
        order_models = best_per_model.index.tolist()
    df_plot["model"] = pd.Categorical(df_plot["model"], categories=order_models, ordered=True)

    fig, ax = plt.subplots(figsize=(12, 4), dpi=PUB_DPI)
    sns.set_style("whitegrid")
    sns.barplot(data=df_plot, x="model", y="score", hue="backbone", ax=ax, palette="pastel")
    ax.set_xlabel("")
    ax.set_ylabel(key)
    ax.tick_params(axis='x', rotation=35)
    ax.legend(title="Backbone", loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5, frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    ensure_dir(out_png.parent)
    fig.savefig(out_png, bbox_inches="tight")
    try:
        sns.set_style("white")
    except Exception:
        pass
    plt.close(fig)


def _positive_proba(model, X):
    """
    Return P(y=1) robustly for calibrated/sklearn-like models.
    
    Args:
        model: Trained model
        X: Feature matrix
        
    Returns:
        Array of positive class probabilities
    """
    X = np.asarray(X, dtype=float)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 1:
            return proba
        if proba.shape[1] == 2:
            return proba[:, 1]
        if hasattr(model, "classes_"):
            try:
                pos_idx = int(np.argmax(model.classes_))
                return proba[:, pos_idx]
            except Exception:
                pass
        return proba.ravel()
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        s = np.asarray(s)
        if s.ndim > 1:
            s = s.ravel()
        return 1.0 / (1.0 + np.exp(-s))
    yhat = model.predict(X)
    return np.asarray(yhat, float).ravel()


def bootstrap_auc(y, p, n=1000, seed=42):
    """
    Bootstrap the AUROC on (y, p) with replacement.
    Robust to degenerate class cases (returns NaNs if AUROC undefined).
    
    Args:
        y: True binary labels
        p: Predicted probabilities
        n: Number of bootstrap iterations (default: 1000)
        seed: Random seed (default: 42)
        
    Returns:
        Tuple of (median, lower_ci, upper_ci)
    """
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    N = len(y)
    out = []
    for _ in range(int(n)):
        idx = rng.integers(0, N, N)
        try:
            out.append(roc_auc_score(y[idx], p[idx]))
        except Exception:
            # skip rare folds where AUROC may be undefined
            continue
    if not out:
        return float("nan"), float("nan"), float("nan")
    return float(np.median(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def shap_plots(model, X_sample, feat_names, out_dir: pathlib.Path, prefix: str, log, shap_cfg: dict = None):
    """
    Publication-grade SHAP explanations for sklearn-like models.
    
    Generates SHAP bar plot and beeswarm plot with cleaned feature names
    (removes 'DESC::' prefix for display).
    
    Args:
        model: Trained model to explain
        X_sample: Feature matrix for explanation
        feat_names: Feature names
        out_dir: Output directory
        prefix: Model prefix for output files
        log: Logger instance
        shap_cfg: SHAP configuration dictionary
    """
    try:
        import shap
    except ImportError:
        if log:
            log.warning("[SHAP] SHAP library not available. Skipping SHAP plots.")
        return
    
    shap_cfg = shap_cfg or {}

    # Use model directly (no calibration)
    target_model = model

    X_arr = np.asarray(X_sample, dtype=float)
    n_rows, n_feat = X_arr.shape

    def _clean_names(names):
        if names is None:
            return []
        out = [str(x) for x in names]
        return [x.replace("/", "_").replace("\\", "_").strip() for x in out]

    raw_names = _clean_names(feat_names)
    # align length
    if len(raw_names) != n_feat:
        if log:
            log.info(f"[SHAP] Align feature names: {len(raw_names)} -> {n_feat}")
        if len(raw_names) < n_feat:
            names_aligned = list(raw_names) + [f"DESC::f{i}" for i in range(len(raw_names), n_feat)]
        else:
            names_aligned = raw_names[:n_feat]
    else:
        names_aligned = raw_names
    # never empty names
    names_aligned = [nm if str(nm).strip() != "" else f"DESC::f{i}" for i, nm in enumerate(names_aligned)]

    # --- display labels without 'DESC::' (only for plotting) ---
    names_display = [n[6:] if n.startswith("DESC::") else n for n in names_aligned]

    # pandas frames
    try:
        X_df_full = pd.DataFrame(X_arr, columns=names_aligned)
    except Exception:
        X_df_full = X_arr

    # sampling
    bg_size = int(shap_cfg.get("background_size", 200))
    sample_sz = int(shap_cfg.get("sample_size", min(400, n_rows)))
    maxeval_mul = float(shap_cfg.get("permutation_max_evals_multiplier", 2.2))
    skip_heavy = bool(shap_cfg.get("skip_heavy_models", True))

    bg_size = max(20, min(bg_size, n_rows))
    sample_sz = max(20, min(sample_sz, n_rows))
    sm_idx = np.random.choice(n_rows, size=sample_sz, replace=False)
    bg_idx = np.random.choice(n_rows, size=bg_size, replace=False)

    try:
        Xs_df = pd.DataFrame(X_arr[sm_idx].astype(np.float32), columns=names_aligned)
        Xbg_df = pd.DataFrame(X_arr[bg_idx].astype(np.float32), columns=names_aligned)
    except Exception:
        Xs_df = X_arr[sm_idx].astype(np.float32)
        Xbg_df = X_arr[bg_idx].astype(np.float32)

    model_dir = out_dir / prefix
    ensure_dir(model_dir)

    def _is_tree(m):
        name = m.__class__.__name__
        return (
                hasattr(m, "estimators_") or
                "Forest" in name or "GradientBoosting" in name or
                "XGB" in name or "CatBoost" in name or
                "RandomForest" in name or "ExtraTrees" in name or "DecisionTree" in name
        )

    def _is_logreg(m):
        return m.__class__.__name__ == "LogisticRegression"

    # skip pathological heavy case
    if skip_heavy and model.__class__.__name__ == "SVC":
        if log:
            log.info("[SHAP] Skipping SHAP for SVC (set shap.skip_heavy_models=false to force).")
        return

    try:
        # choose explainer
        if _is_tree(target_model):
            expl = shap.TreeExplainer(target_model, model_output="raw", feature_perturbation="tree_path_dependent")
            sv = expl.shap_values(Xs_df)
            sv = sv[1] if isinstance(sv, list) and len(sv) > 1 else sv
        elif _is_logreg(target_model):
            expl = shap.LinearExplainer(target_model, Xbg_df)
            sv = expl(Xs_df)
        else:
            need_evals = max(10, int(maxeval_mul * Xs_df.shape[1]))
            f = lambda X: _positive_proba(target_model, X)
            expl = shap.PermutationExplainer(f, Xbg_df, max_evals=need_evals, silent=True)
            sv = expl(Xs_df)

        # plots (with cleaned labels, max 30 features for readability)
        max_display = int(shap_cfg.get("max_display_features", 30))
        
        plt.figure(figsize=(7, 5))
        shap.summary_plot(sv, Xs_df, feature_names=names_display, show=False, plot_type="bar", max_display=max_display)
        for ax in plt.gcf().axes:
            ax.grid(False)
        plt.tight_layout()
        plt.savefig(model_dir / "shap_bar.png")
        plt.close("all")

        plt.figure(figsize=(7, 5))
        shap.summary_plot(sv, Xs_df, feature_names=names_display, show=False, max_display=max_display)
        for ax in plt.gcf().axes:
            ax.grid(False)
        plt.tight_layout()
        plt.savefig(model_dir / "shap_beeswarm.png")
        plt.close("all")

        if log:
            log.info(f"SHAP saved for {prefix}")
    except Exception as e:
        if log:
            log.warning(f"SHAP failed for {prefix}: {e}")


def _write_best_overall_itxt(best_row: dict, out_path):
    """
    Write concise summary of the best overall model.
    
    Args:
        best_row: Dictionary with model metrics and metadata
        out_path: Output file path
    """
    lines = [
        "# Best overall model (QSAR Evaluation)\n",
        f"Backbone: {best_row.get('backbone', '?')}\n",
        f"Engine:   {best_row.get('engine', '?')}\n",
        "\n",
        "# Model selection: Composite score balancing mean performance and stability\n",
        f"Composite score: {best_row.get('composite_score', '?'):.3f}\n",
        f"  = {best_row.get('primary_metric_name', '?')} ({best_row.get('primary_metric_value', '?'):.3f})\n",
        f"  + stability_weight ({best_row.get('stability_weight', '?')}) × CI_width ({best_row.get('CI_width', '?'):.3f})\n",
        f"CI: [{best_row.get('CI_lower', '?'):.3f}, {best_row.get('CI_upper', '?'):.3f}]\n",
        "\n",
        "# Performance metrics\n",
        f"AUROC:    {best_row.get('AUROC', '?'):.3f}\n",
        f"AUPRC:    {best_row.get('AUPRC_active', '?'):.3f}\n",
        f"Brier:    {best_row.get('Brier', '?'):.3f}\n",
        f"ECE:      {best_row.get('ECE', '?'):.3f}\n",
        f"Threshold:{best_row.get('threshold', '?'):.3f}\n",
    ]
    with open(out_path, 'w') as f:
        f.writelines(lines)

