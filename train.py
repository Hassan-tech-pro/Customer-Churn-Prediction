import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.metrics import classification_report
import xgboost as xgb
import pickle

# Load data
data = pd.read_csv("data/customer_churn_dataset-training-master.csv")
# test = pd.read_csv("data/customer_churn_dataset-testing-master.csv")  # Commented out if not available

def data_preprocessing(data):
    # Drop ID
    data = data.drop(["CustomerID", "Last Interaction"], axis=1)

    
    # Clean
    data = data.dropna()
    
    # Encode
    categorical_cols = ["Gender", "Subscription Type", "Contract Length"]
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(data[categorical_cols])
    
    data_encoded = encoder.transform(data[categorical_cols])
    data = data.drop(categorical_cols, axis=1)
    data = pd.concat([data, pd.DataFrame(data_encoded, columns=encoder.get_feature_names_out(), index=data.index)], axis=1)
    
    # Split train/val
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        data.drop("Churn", axis=1), 
        data["Churn"], 
        test_size=0.2, 
        random_state=42
    )
    
    # Scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    scaler.feature_names_in_ = None  # Disable feature name checking
    X_val = scaler.transform(X_val_raw)
    
    return X_train, X_val, y_train, y_val, scaler, encoder

# Preprocess
X_train, X_val, y_train, y_val, scaler, encoder = data_preprocessing(data)

print("Class distributions:")
print(f"Train: {np.bincount(y_train.astype(int))}")
print(f"Val: {np.bincount(y_val.astype(int))}")

# Train XGBoost with adjusted scale_pos_weight for training distribution
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    max_depth=3,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    eval_metric='logloss',
    random_state=42
)

xgb_model.fit(X_train, y_train)

# Default threshold evaluation
print("\n=== DEFAULT THRESHOLD (0.5) ===")
y_val_pred = xgb_model.predict(X_val)

print("Validation:")
print(classification_report(y_val, y_val_pred))
print(f"Accuracy: {xgb_model.score(X_val, y_val):.4f}")

# Save model, scaler, encoder
pickle.dump(xgb_model, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(encoder, open('encoder.pkl', 'wb'))

print("Model, scaler, and encoder saved.")