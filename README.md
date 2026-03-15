# Customer Churn Prediction

A machine learning project to predict customer churn using XGBoost and a Gradio web interface.

## Features

- Train an XGBoost model on customer data
- Predict churn probability with a user-friendly web interface
- Preprocessing with one-hot encoding and scaling

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Train the model:
   ```
   python train.py
   ```

3. Run the web interface:
   ```
   python app.py
   ```

## Files

- `train.py`: Script to train the model and save artifacts
- `app.py`: Gradio web interface for predictions
- `model.pkl`: Trained XGBoost model
- `scaler.pkl`: Fitted MinMaxScaler
- `encoder.pkl`: Fitted OneHotEncoder
- `data/`: Directory with training and testing CSV files
- `requirements.txt`: Python dependencies

## Usage

Enter customer details in the web interface to get churn prediction and probability.