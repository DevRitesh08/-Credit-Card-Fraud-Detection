import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Tuple

class FraudDetectionModel(nn.Module):
    def __init__(self, input_dim: int = 29):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 24),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(24, 20),
            nn.ReLU(),
            nn.Linear(20, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

def to_tensor(arr):
    arr = arr.values if hasattr(arr, "values") else arr
    return torch.tensor(np.asarray(arr), dtype=torch.float32)

def train_model(
    model: nn.Module,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    epochs: int = 20,
    batch_size: int = 15,
    device: torch.device = None
) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    X_train_t = to_tensor(X_train)
    y_train_t = to_tensor(y_train).view(-1, 1)
    X_test_t = to_tensor(X_test)
    y_test_t = to_tensor(y_test).view(-1, 1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print("\nTraining model...")
    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_X.size(0)
            correct += ((outputs > 0.5).float() == batch_y).sum().item()
            total += batch_y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation on test set
        model.eval()
        with torch.no_grad():
            X_test_dev = X_test_t.to(device)
            y_test_dev = y_test_t.to(device)
            test_outputs = model(X_test_dev)
            test_loss = criterion(test_outputs, y_test_dev).item()
            test_acc = ((test_outputs > 0.5).float() == y_test_dev).float().mean().item()

        print(f"Epoch {epoch + 1}/{epochs} - "
              f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f} - "
              f"test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}")
    
    return model

def predict_prob(model: nn.Module, X: pd.DataFrame, device: torch.device = None) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.eval()
    X_t = to_tensor(X).to(device)
    with torch.no_grad():
        outputs = model(X_t)
    
    return outputs.cpu().numpy().ravel()
