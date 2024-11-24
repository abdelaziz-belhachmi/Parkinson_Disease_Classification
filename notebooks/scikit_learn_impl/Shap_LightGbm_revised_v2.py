import pandas as pd
import lightgbm as lgb
import numpy as np
import shap
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

# Step 2: Train an initial LightGBM model
lgbm_model = lgb.LGBMClassifier(objective="binary", random_state=42, is_unbalance=True)
lgbm_model.fit(X_train, y_train)

# Step 3: Compute SHAP values
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_train)
# pd.DataFrame(shap_values).to_csv("../../data/processed/shap/shapvalues.csv", index=False)
#
# shap.summary_plot(shap_values, X_train,max_display=len(X_train.columns))
# shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=len(X_train.columns))
#

import seaborn as sns

# Compute the mean absolute SHAP values for each feature
mean_abs_shap_values = np.abs(shap_values).mean(axis=0)  # Use shap_values[1] for the positive class in binary classification

# Create a DataFrame for visualization
shap_importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Mean Absolute SHAP Value": mean_abs_shap_values
}).sort_values(by="Mean Absolute SHAP Value", ascending=False)

mnabsshapval = np.mean(mean_abs_shap_values)
# Plot the distribution of mean absolute SHAP values
plt.figure(figsize=(10, 6))
sns.histplot(mean_abs_shap_values, bins=20, kde=True, color="skyblue")
plt.axvline(x=mnabsshapval, color="red", linestyle="--", label="Mean SHAP Value")
plt.title("Distribution of Mean Absolute SHAP Values")
plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Optionally, show the sorted feature importance for reference
print(shap_importance_df)


# Identify features with null SHAP values (mean absolute SHAP value == 0)
null_shap_features = shap_importance_df[shap_importance_df["Mean Absolute SHAP Value"] <= mnabsshapval]["Feature"]

# Drop these features from the datasets
X_train_reduced = X_train.drop(columns=null_shap_features)
X_test_reduced = X_test.drop(columns=null_shap_features)

X_train_df = pd.DataFrame(X_train_reduced)
X_test_df = pd.DataFrame(X_test_reduced)

train_combined = pd.concat([X_train_df, y_train], axis=1)
test_combined = pd.concat([X_test_df, y_test], axis=1)

# Save combined datasets to CSV
train_combined.to_csv("../../data/processed/afterDroplessThanMean_Shap_train_data.csv", index=False)
test_combined.to_csv("../../data/processed/afterDroplessThanMean_Shap_test_data.csv", index=False)


print(f"Original number of features: {X_train.shape[1]}")
print(f"Number of features after dropping null SHAP features: {X_train_reduced.shape[1]}")
print(f"Features dropped: {list(null_shap_features)}")