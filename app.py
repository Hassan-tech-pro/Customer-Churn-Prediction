import pandas as pd
import numpy as np
import xgboost as xgb
import gradio as gr
import pickle

# Load model, scaler, encoder
xgb_model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))

def predict_churn(Age, Gender, Tenure, Usage_Frequency, Support_Calls, Payment_Delay, Subscription_Type, Contract_Length, Total_Spend):
    input_data = pd.DataFrame({
        'Age': [Age],
        'Gender': [Gender],
        'Tenure': [Tenure],
        'Usage Frequency': [Usage_Frequency],
        'Support Calls': [Support_Calls],
        'Payment Delay': [Payment_Delay],
        'Subscription Type': [Subscription_Type],
        'Contract Length': [Contract_Length],
        'Total Spend': [Total_Spend]
    })
    
    # Encode categorical
    categorical_cols = ["Gender", "Subscription Type", "Contract Length"]
    input_encoded_cat = encoder.transform(input_data[categorical_cols])
    
    # Numerical
    input_numerical = input_data.drop(categorical_cols, axis=1).values
    
    # Combine
    input_encoded = np.concatenate([input_numerical, input_encoded_cat], axis=1)
    
    # Scale
    input_scaled = scaler.transform(input_encoded)
    
    # Predict
    pred = xgb_model.predict(input_scaled)[0]
    prob = xgb_model.predict_proba(input_scaled)[0][1]
    
    return f"Churn Prediction: {'Yes' if pred == 1 else 'No'} (Probability: {prob:.2f})"

# Define inputs
inputs = [
    gr.Number(label="Age"),
    gr.Dropdown(["Male", "Female"], label="Gender"),
    gr.Number(label="Tenure"),
    gr.Number(label="Usage Frequency"),
    gr.Number(label="Support Calls"),
    gr.Number(label="Payment Delay"),
    gr.Dropdown(["Basic", "Standard", "Premium"], label="Subscription Type"),
    gr.Dropdown(["Monthly", "Quarterly", "Annual"], label="Contract Length"),
    gr.Number(label="Total Spend")
]

# Define output
output = gr.Textbox(label="Prediction")

# Launch interface
gr.Interface(fn=predict_churn, inputs=inputs, outputs=output, title="Customer Churn Prediction").launch(share=True)