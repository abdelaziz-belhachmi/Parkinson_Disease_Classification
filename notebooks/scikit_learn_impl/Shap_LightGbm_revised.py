import pandas as pd
import lightgbm as lgb
import numpy as np
import shap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Step 1: Load train and test data
# train_path = "../../data/processed/train_data.csv"  # Replace with your train file path
train_path = "../../data/processed/afterDroplessThanMean_Shap_train_data.csv"  # Replace with your train file path

# test_path = "../../data/processed/test_data.csv"  # Replace with your test file path
test_path = "../../data/processed/afterDroplessThanMean_Shap_test_data.csv"  # Replace with your test file path

# Load data
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Split features and target
# X_train = train_data.drop(columns=["target", "id"])  # Drop target and id
X_train = train_data.drop(columns=["target"])
y_train = train_data["target"]

# X_test = test_data.drop(columns=["target", "id"])  # Drop target and id
X_test = test_data.drop(columns=["target"])
y_test = test_data["target"]

# Step 2: Train an initial LightGBM model
lgbm_model = lgb.LGBMClassifier(objective="binary", random_state=42)
lgbm_model.fit(X_train, y_train)

# Step 3: Compute SHAP values
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_train)
# pd.DataFrame(shap_values).to_csv("../../data/processed/shap/shapvalues.csv", index=False)

shap.summary_plot(shap_values, X_train, max_display=len(X_train.columns))
shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=len(X_train.columns))

# Step 4: Make predictions on the test set
y_pred = lgbm_model.predict(X_test)

# Step 5: Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy score
print(f"Accuracy Score: {accuracy:.4f}")
