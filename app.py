import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from src.data import load_creditcard_data
from src.model import FraudDetectionModel, predict_prob
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("Credit Card Fraud Detection")
st.markdown("---")

@st.cache_resource
def load_model_and_data():
    df = load_creditcard_data('data/raw/creditcard.csv')
    if 'Time' in df.columns:
        df = df.drop(['Time'], axis=1)
    
    scaler = StandardScaler()
    if 'Amount' in df.columns:
        df['normalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        df = df.drop(['Amount'], axis=1)
        
    input_dim = df.shape[1] - 1
    model = FraudDetectionModel(input_dim=input_dim)
    
    try:
        # In a real scenario, you'd load model.state_dict() here
        pass
    except Exception:
        pass
        
    return df, model, scaler

df, model, scaler = load_model_and_data()

st.subheader("Transaction Predictor")
st.write("Input transaction features to predict fraud probability.")

col1, col2 = st.columns(2)
amount = col1.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)

features = []
with st.expander("Advanced Features (V1-V28)", expanded=False):
    cols = st.columns(4)
    for i in range(1, 29):
        val = cols[(i-1)%4].number_input(f"V{i}", value=0.0, format="%.2f")
        features.append(val)

if st.button("Predict Risk"):
    scaled_amount = scaler.transform(np.array([[amount]]))[0][0]
    
    input_data = pd.DataFrame([features + [scaled_amount]])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prob = predict_prob(model, input_data, device=device)[0]
    
    st.markdown("### Prediction Result")
    if prob > 0.1:
        st.error(f"High Risk Transaction (Probability: {prob:.4f})")
    elif prob > 0.05:
        st.warning(f"Manual Review Required (Probability: {prob:.4f})")
    else:
        st.success(f"Low Risk (Probability: {prob:.4f})")

st.markdown("---")
st.subheader("Dataset Overview")
st.dataframe(df.head(10), use_container_width=True)
