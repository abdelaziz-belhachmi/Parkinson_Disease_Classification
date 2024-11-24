import pandas as pd
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
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

# Step 2: Train an initial Logistic Regression model
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)

# Step 3: Compute SHAP values using KernelExplainer
# Reduce the number of background samples using shap.kmeans
background_data = shap.kmeans(X_train, 400)  # Use 100 centroids for background
explainer = shap.KernelExplainer(lr_model.predict_proba, background_data)
shap_values = explainer.shap_values(X_train)

# Plot SHAP summary plot
shap.summary_plot(shap_values[1], X_train, max_display=len(X_train.columns))
shap.summary_plot(shap_values[1], X_train, plot_type="bar", max_display=len(X_train.columns))

# Step 4: Final evaluation
y_pred = lr_model.predict(X_test)
final_accuracy = accuracy_score(y_test, y_pred)
print(f"Logistic Regression Model Accuracy: {final_accuracy:.4f}")

# Cross-validation with Logistic Regression model
cv_scores = cross_val_score(lr_model, X_train, y_train, cv=5, scoring="accuracy")
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f}")
