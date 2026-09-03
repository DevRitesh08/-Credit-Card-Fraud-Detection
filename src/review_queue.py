"""Fraud review queue: approve, manual review, block decisions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DecisionBands:
    """Thresholds for approve/review/block decisions.

    Approve: probability < approve_threshold
    Review: approve_threshold <= probability < block_threshold
    Block: probability >= block_threshold
    """
    approve_threshold: float = 0.25
    block_threshold: float = 0.70

    def __post_init__(self):
        if not (0 <= self.approve_threshold < self.block_threshold <= 1):
            raise ValueError(
                f"Invalid thresholds: 0 <= approve({self.approve_threshold}) "
                f"< block({self.block_threshold}) <= 1"
            )

    def decide(self, probability: float) -> str:
        """Return decision for a single probability."""
        if probability < self.approve_threshold:
            return "Approve"
        elif probability < self.block_threshold:
            return "Manual Review"
        else:
            return "Block"

    def decide_batch(self, probabilities: np.ndarray) -> np.ndarray:
        """Return decisions for an array of probabilities."""
        return np.array([self.decide(p) for p in probabilities])


def create_review_queue(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray,
    decision_bands: DecisionBands | None = None,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Create a fraud review queue DataFrame.

    Args:
        X_test: Test features
        y_test: True labels
        y_prob: Predicted probabilities for fraud class
        decision_bands: Decision thresholds (uses defaults if None)
        feature_cols: Which features to include in queue (default: Amount)

    Returns:
        DataFrame with review queue columns
    """
    if decision_bands is None:
        decision_bands = DecisionBands()

    if feature_cols is None:
        feature_cols = ["Amount"]

    # Ensure we have the requested feature columns
    available_cols = [c for c in feature_cols if c in X_test.columns]
    if not available_cols:
        available_cols = ["Amount"] if "Amount" in X_test.columns else [X_test.columns[0]]

    queue = pd.DataFrame(index=X_test.index)
    queue["transaction_index"] = X_test.index
    queue["predicted_fraud_probability"] = y_prob
    queue["decision"] = decision_bands.decide_batch(y_prob)
    queue["actual_label"] = y_test.values

    # Add requested features
    for col in available_cols:
        queue[col] = X_test[col].values

    # Add outcome (TP, FP, TN, FN) - evaluation only
    y_pred = (y_prob >= decision_bands.block_threshold).astype(int)
    queue["outcome"] = classify_outcomes(y_test.values, y_pred)

    # Sort by probability descending (highest risk first)
    queue = queue.sort_values("predicted_fraud_probability", ascending=False).reset_index(drop=True)

    return queue


def classify_outcomes(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Classify each prediction as TP, FP, TN, FN."""
    outcomes = np.empty_like(y_true, dtype=object)
    outcomes[(y_true == 1) & (y_pred == 1)] = "TP"  # Fraud caught
    outcomes[(y_true == 1) & (y_pred == 0)] = "FN"  # Fraud missed
    outcomes[(y_true == 0) & (y_pred == 1)] = "FP"  # Customer flagged
    outcomes[(y_true == 0) & (y_pred == 0)] = "TN"  # Legitimate approved
    return outcomes


def print_review_queue_summary(queue: pd.DataFrame) -> None:
    """Print summary statistics for the review queue."""
    print("\n" + "=" * 60)
    print("FRAUD REVIEW QUEUE SUMMARY")
    print("=" * 60)

    total = len(queue)
    print(f"\nTotal transactions: {total:,}")

    decision_counts = queue["decision"].value_counts()
    for decision, count in decision_counts.items():
        pct = count / total * 100
        print(f"  {decision}: {count:,} ({pct:.1f}%)")

    outcome_counts = queue["outcome"].value_counts()
    print("\nOutcomes (at block threshold):")
    for outcome, count in outcome_counts.items():
        pct = count / total * 100
        print(f"  {outcome}: {count:,} ({pct:.1f}%)")

    # Review queue workload
    review_queue = queue[queue["decision"] == "Manual Review"]
    print(f"\nManual review workload: {len(review_queue):,} transactions")

    if len(review_queue) > 0:
        fraud_in_review = (review_queue["actual_label"] == 1).sum()
        print(f"  Actual fraud in review: {fraud_in_review:,}")
        print(f"  Fraud rate in review: {fraud_in_review / len(review_queue) * 100:.2f}%")

    # Block queue
    block_queue = queue[queue["decision"] == "Block"]
    print(f"\nBlock queue: {len(block_queue):,} transactions")
    if len(block_queue) > 0:
        fraud_in_block = (block_queue["actual_label"] == 1).sum()
        print(f"  Actual fraud blocked: {fraud_in_block:,}")
        print(f"  False blocks (legitimate): {(block_queue['actual_label'] == 0).sum():,}")

    print("=" * 60)


def print_top_risk_transactions(
    queue: pd.DataFrame,
    n: int = 20,
    show_features: list[str] | None = None,
) -> None:
    """Print top N highest-risk transactions."""
    if show_features is None:
        show_features = ["Amount"]

    print(f"\n{'=' * 60}")
    print(f"TOP {n} HIGHEST RISK TRANSACTIONS")
    print(f"{'=' * 60}")

    top_n = queue.head(n)
    cols = ["transaction_index", "predicted_fraud_probability", "decision", "actual_label", "outcome"] + show_features

    for _, row in top_n.iterrows():
        prob = row["predicted_fraud_probability"]
        decision = row["decision"]
        actual = "Fraud" if row["actual_label"] == 1 else "Legit"
        outcome = row["outcome"]
        features = ", ".join(f"{c}={row[c]:.2f}" for c in show_features if c in row)
        print(f"  #{row['transaction_index']}: P(fraud)={prob:.4f} | {decision:14s} | Actual: {actual:6s} | {outcome} | {features}")

    print(f"{'=' * 60}")


def compare_decision_bands(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    approve_thresholds: list[float],
    block_thresholds: list[float],
) -> pd.DataFrame:
    """Compare multiple decision band configurations.

    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        approve_thresholds: List of approve thresholds to test
        block_thresholds: List of block thresholds to test

    Returns:
        DataFrame with metrics for each band configuration
    """
    rows = []
    for approve_thresh in approve_thresholds:
        for block_thresh in block_thresholds:
            if approve_thresh >= block_thresh:
                continue

            bands = DecisionBands(approve_threshold=approve_thresh, block_threshold=block_thresh)
            decisions = bands.decide_batch(y_prob)
            y_pred = (y_prob >= block_thresh).astype(int)

            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()

            review_mask = decisions == "Manual Review"
            review_count = review_mask.sum()
            fraud_in_review = y_true[review_mask].sum() if review_count > 0 else 0

            rows.append({
                "approve_threshold": approve_thresh,
                "block_threshold": block_thresh,
                "approve_count": (decisions == "Approve").sum(),
                "review_count": review_count,
                "block_count": (decisions == "Block").sum(),
                "true_positives": int(tp),
                "false_negatives": int(fn),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "fraud_in_review": int(fraud_in_review),
                "review_fraud_rate": fraud_in_review / review_count if review_count > 0 else 0,
            })

    return pd.DataFrame(rows)


def save_review_queue(queue: pd.DataFrame, path: str | Path) -> None:
    """Save review queue to CSV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    queue.to_csv(path, index=False)
    print(f"Review queue saved to {path}")