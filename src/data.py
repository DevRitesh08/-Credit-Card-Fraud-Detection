"""Data loading and validation utilities."""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def load_creditcard_data(path: str | Path) -> pd.DataFrame:
    """Load the credit card fraud dataset from CSV or RData.

    Args:
        path: Path to the data file (.csv or .Rdata/.RData)

    Returns:
        DataFrame with columns: Time, V1-V28, Amount, Class
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in (".rdata", ".rdata"):
        import pyreadr

        result = pyreadr.read_r(path)
        # Expect key 'creditcard' based on dataset convention
        key = "creditcard" if "creditcard" in result else list(result.keys())[0]
        df = result[key]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

    return df


def validate_dataframe(df: pd.DataFrame) -> dict:
    """Run validation checks on the dataset and return summary.

    Args:
        df: DataFrame to validate

    Returns:
        Dictionary with validation results
    """
    results = {}

    # Shape and columns
    results["shape"] = df.shape
    results["columns"] = list(df.columns)
    results["dtypes"] = df.dtypes.to_dict()

    # Missing values
    results["missing_values"] = df.isnull().sum().to_dict()
    results["total_missing"] = int(df.isnull().sum().sum())

    # Duplicates
    duplicate_count = int(df.duplicated().sum())
    results["duplicate_rows"] = duplicate_count

    # Class distribution
    class_counts = df["Class"].value_counts().sort_index()
    results["class_counts"] = class_counts.to_dict()
    results["class_percentages"] = (class_counts / len(df) * 100).round(4).to_dict()

    # Amount summary
    if "Amount" in df.columns:
        results["amount_summary"] = df["Amount"].describe().to_dict()

    # Verify Class contains only 0 and 1
    unique_classes = sorted(df["Class"].unique().tolist())
    results["unique_classes"] = unique_classes
    results["valid_classes"] = unique_classes == [0, 1]

    return results


def print_validation_report(validation: dict) -> None:
    """Print a formatted validation report."""
    print("=" * 60)
    print("DATA VALIDATION REPORT")
    print("=" * 60)

    print(f"\nShape: {validation['shape']}")
    print(f"Columns ({len(validation['columns'])}): {validation['columns']}")

    print(f"\nMissing values: {validation['total_missing']}")
    if validation["total_missing"] > 0:
        for col, count in validation["missing_values"].items():
            if count > 0:
                print(f"  {col}: {count}")

    print(f"\nDuplicate rows: {validation['duplicate_rows']}")

    print("\nClass distribution:")
    for cls, count in validation["class_counts"].items():
        pct = validation["class_percentages"][cls]
        label = "Fraud" if cls == 1 else "Legitimate"
        print(f"  Class {cls} ({label}): {count:,} ({pct:.4f}%)")

    if "amount_summary" in validation:
        print("\nAmount summary:")
        for stat, value in validation["amount_summary"].items():
            print(f"  {stat}: {value:.4f}")

    print(f"\nUnique classes: {validation['unique_classes']}")
    print(f"Valid classes (0, 1 only): {validation['valid_classes']}")

    print("=" * 60)


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create stratified train/test split.

    Args:
        df: Full dataset
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split

    X = df.drop(columns="Class")
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


def print_split_summary(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Print summary of train/test split."""
    print("\n" + "=" * 60)
    print("TRAIN/TEST SPLIT SUMMARY")
    print("=" * 60)

    for name, X, y in [("Train", X_train, y_train), ("Test", X_test, y_test)]:
        fraud_count = int(y.sum())
        total = len(y)
        pct = fraud_count / total * 100
        print(f"\n{name} set: {total:,} rows")
        print(f"  Fraud: {fraud_count:,} ({pct:.4f}%)")
        print(f"  Legitimate: {total - fraud_count:,} ({100 - pct:.4f}%)")

    print("=" * 60)