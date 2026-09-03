"""
Credit Card Fraud Detection using Deep Neural Network
Modular entry point using src/ directory components.
"""

import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Import from src modules
from src.data import load_creditcard_data, validate_dataframe, print_validation_report, split_data, print_split_summary
from src.model import FraudDetectionModel, train_model, predict_prob
from src.evaluate import compute_metrics, plot_confusion_matrix, print_classification_report

def main():
    # Set random seeds for reproducibility
    np.random.seed(2)
    torch.manual_seed(2)

    # 1. Load Data
    print("Loading data...")
    df = load_creditcard_data('data/raw/creditcard.csv')
    
    # Optional: print validation report
    # validation_report = validate_dataframe(df)
    # print_validation_report(validation_report)

    # 2. Preprocessing
    print("\nPreprocessing data...")
    # Drop Time column
    if 'Time' in df.columns:
        df = df.drop(['Time'], axis=1)

    # Normalize Amount column
    if 'Amount' in df.columns:
        scaler = StandardScaler()
        df['normalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        df = df.drop(['Amount'], axis=1)

    # 3. Train-Test Split
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.3, random_state=0)
    print_split_summary(X_train, X_test, y_train, y_test)

    # 4. Model Setup & Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    model = FraudDetectionModel(input_dim=X_train.shape[1])
    print(model)
    
    model = train_model(
        model=model, 
        X_train=X_train, 
        y_train=y_train, 
        X_test=X_test, 
        y_test=y_test, 
        epochs=20, 
        batch_size=15, 
        device=device
    )

    # Save the trained model and scaler for the Streamlit app
    import joblib
    torch.save(model.state_dict(), "fraud_model.pth")
    joblib.dump(scaler, "scaler.pkl")
    print("Model and scaler saved successfully!")

    # 5. Final Evaluation on Test Set
    print("\n" + "="*50)
    print("FINAL EVALUATION (TEST SET)")
    print("="*50)
    
    y_prob = predict_prob(model, X_test, device)
    y_pred = (y_prob >= 0.5).astype(int)
    y_test_np = y_test.values

    metrics = compute_metrics(y_test_np, y_pred, y_prob)
    print("\nTest Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    print_classification_report(y_test_np, y_pred)
    
    # Save Confusion Matrix plots
    plot_confusion_matrix(y_test_np, y_pred, title="Confusion Matrix (Test Set)", save_path="confusion_matrix.png")
    plot_confusion_matrix(y_test_np, y_pred, normalize=True, title="Normalized Confusion Matrix (Test Set)", save_path="confusion_matrix_normalized.png")

    # 6. Evaluation on Full Dataset
    print("\n" + "="*50)
    print("FULL DATASET EVALUATION")
    print("="*50)
    
    X_full = df.drop(columns="Class")
    y_full = df["Class"].values
    
    y_prob_full = predict_prob(model, X_full, device)
    y_pred_full = (y_prob_full >= 0.5).astype(int)
    
    metrics_full = compute_metrics(y_full, y_pred_full, y_prob_full)
    print("\nFull Dataset Metrics:")
    for k, v in metrics_full.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
            
    plot_confusion_matrix(y_full, y_pred_full, title="Confusion Matrix (Full Dataset)", save_path="confusion_matrix_full.png")
    plot_confusion_matrix(y_full, y_pred_full, normalize=True, title="Normalized Confusion Matrix (Full Dataset)", save_path="confusion_matrix_full_normalized.png")

    print("\n" + "="*50)
    print("PIPELINE COMPLETE!")
    print("="*50)

if __name__ == "__main__":
    main()
