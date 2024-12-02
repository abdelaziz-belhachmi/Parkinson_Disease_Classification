import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Step 1: Load train and test data
train_path = "../../data/processed/train_data.csv"  # Replace with your train file path
test_path = "../../data/processed/test_data.csv"  # Replace with your test file path

# Load data
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Split features and target
X_train = train_data.drop(columns=["target", "id"])  # Drop target and id
y_train = train_data["target"]

X_test = test_data.drop(columns=["target", "id"])  # Drop target and id
y_test = test_data["target"]

# Step 2: Train an initial Random Forest model
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Step 3: Compute SHAP values
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values, X_train,max_display=len(X_train.columns))
shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=len(X_train.columns))
