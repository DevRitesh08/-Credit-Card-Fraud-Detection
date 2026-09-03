"""Evaluation metrics, plots, and threshold selection utilities."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute classification metrics for fraud class (class 1).

    Args:
        y_true: True labels
        y_pred: Predicted labels (0 or 1)
        y_prob: Predicted probabilities for class 1 (optional)

    Returns:
        Dictionary of metrics
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["pr_auc"] = average_precision_score(y_true, y_prob)

    return metrics


def compute_metrics_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: list[float],
) -> pd.DataFrame:
    """Compute metrics at multiple probability thresholds.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities for class 1
        thresholds: List of thresholds to evaluate

    Returns:
        DataFrame with metrics at each threshold
    """
    rows = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics["threshold"] = thresh
        rows.append(metrics)

    return pd.DataFrame(rows)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: str | Path | None = None,
) -> None:
    """Plot precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="#d04a4a", linewidth=2, label=f"PR-AUC = {pr_auc:.4f}")
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve",
    save_path: str | Path | None = None,
) -> None:
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#d04a4a", linewidth=2, label=f"ROC-AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
    normalize: bool = False,
) -> None:
    """Plot confusion matrix with labels."""
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        title += " (Normalized)"
    else:
        fmt = "d"

    labels = ["Legitimate", "Fraud"]

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Proportion" if normalize else "Count"},
        annot_kws={"size": 14},
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("Actual", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_threshold_analysis(
    threshold_df: pd.DataFrame,
    title: str = "Threshold Analysis: Precision, Recall, F1",
    save_path: str | Path | None = None,
) -> None:
    """Plot precision, recall, F1 across thresholds."""
    plt.figure(figsize=(10, 6))
    plt.plot(
        threshold_df["threshold"],
        threshold_df["precision"],
        label="Precision",
        color="#2e86de",
        linewidth=2,
    )
    plt.plot(
        threshold_df["threshold"],
        threshold_df["recall"],
        label="Recall",
        color="#d04a4a",
        linewidth=2,
    )
    plt.plot(
        threshold_df["threshold"],
        threshold_df["f1"],
        label="F1",
        color="#27ae60",
        linewidth=2,
    )
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    top_n: int = 15,
    title: str = "Feature Importance",
    save_path: str | Path | None = None,
) -> None:
    """Plot top-N feature importances."""
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    top_features = importance_df.head(top_n)

    plt.figure(figsize=(10, 6))
    bars = plt.barh(
        range(len(top_features)),
        top_features["importance"],
        color="#d04a4a",
        alpha=0.8,
    )
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Importance", fontsize=12)
    plt.title(title, fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(alpha=0.25, axis="x")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_amount_distribution(
    df: pd.DataFrame,
    save_path: str | Path | None = None,
) -> None:
    """Plot Amount distribution by class (log scale)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Log-scaled histogram
    fraud_amounts = df[df["Class"] == 1]["Amount"]
    legit_amounts = df[df["Class"] == 0]["Amount"]

    bins = np.logspace(np.log10(0.1), np.log10(df["Amount"].max()), 50)

    axes[0].hist(
        legit_amounts,
        bins=bins,
        alpha=0.5,
        label=f"Legitimate (n={len(legit_amounts):,})",
        color="#2e86de",
        density=True,
    )
    axes[0].hist(
        fraud_amounts,
        bins=bins,
        alpha=0.7,
        label=f"Fraud (n={len(fraud_amounts):,})",
        color="#d04a4a",
        density=True,
    )
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Amount (log scale)", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].set_title("Amount Distribution by Class", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.25)

    # Box plot
    data_to_plot = [legit_amounts, fraud_amounts]
    axes[1].boxplot(data_to_plot, labels=["Legitimate", "Fraud"], showfliers=False)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Amount (log scale)", fontsize=12)
    axes[1].set_title("Amount Box Plot by Class (Outliers Hidden)", fontsize=14)
    axes[1].grid(alpha=0.25)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_fraud_rate_over_time(
    df: pd.DataFrame,
    n_bins: int = 20,
    save_path: str | Path | None = None,
) -> None:
    """Plot fraud rate across time bins."""
    df = df.copy()
    df["time_bin"] = pd.cut(df["Time"], bins=n_bins)
    fraud_rate = df.groupby("time_bin")["Class"].mean()
    bin_centers = [interval.mid for interval in fraud_rate.index]

    plt.figure(figsize=(10, 5))
    plt.plot(bin_centers, fraud_rate.values, "o-", color="#d04a4a", linewidth=2, markersize=5)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Fraud Rate", fontsize=12)
    plt.title("Fraud Rate Across Time Bins", fontsize=14)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: list[str] | None = None,
    save_path: str | Path | None = None,
) -> None:
    """Plot correlation heatmap for selected features."""
    if features is None:
        features = ["Amount", "Time", "Class"] + [f"V{i}" for i in range(1, 11)]

    available = [f for f in features if f in df.columns]
    corr = df[available].corr()

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        cbar_kws={"label": "Correlation"},
        annot_kws={"size": 8},
    )
    plt.title("Feature Correlation Heatmap (Upper Triangle Masked)", fontsize=14)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str] = ("Legitimate", "Fraud"),
) -> None:
    """Print formatted classification report."""
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    print("=" * 60)