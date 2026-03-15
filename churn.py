import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
import xgboost as xgb


# Load data
data = pd.read_csv("customer_churn_dataset-training-master.csv")
test = pd.read_csv("customer_churn_dataset-testing-master.csv")

def data_preprocessing(data, test):
    # Drop ID
    data = data.drop(["CustomerID", "Last Interaction"], axis=1)
    test = test.drop(["CustomerID", "Last Interaction"], axis=1)

    
    # Clean
    data = data.dropna()
    test = test.dropna()
    
    # Encode
    data = pd.get_dummies(data, columns=["Gender", "Subscription Type", "Contract Length"])
    test = pd.get_dummies(test, columns=["Gender", "Subscription Type", "Contract Length"])
    
    # Align columns
    train_cols = data.columns.drop("Churn")
    test = test.reindex(columns=train_cols.tolist() + ["Churn"], fill_value=0)
    
    # Split train/val
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        data.drop("Churn", axis=1), 
        data["Churn"], 
        test_size=0.2, 
        random_state=42
    )
    
    # Separate test X and y
    X_test_raw = test.drop("Churn", axis=1)
    y_test = test["Churn"].values
    
    # Scale
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Preprocess
X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing(data, test)

print("Class distributions:")
print(f"Train: {np.bincount(y_train.astype(int))}")
print(f"Val: {np.bincount(y_val.astype(int))}")
print(f"Test: {np.bincount(y_test.astype(int))}")

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
y_test_pred = xgb_model.predict(X_test)

print("Validation:")
print(classification_report(y_val, y_val_pred))
print(f"Accuracy: {xgb_model.score(X_val, y_val):.4f}")

print("\nTest:")
print(classification_report(y_test, y_test_pred))
print(f"Accuracy: {xgb_model.score(X_test, y_test):.4f}")

