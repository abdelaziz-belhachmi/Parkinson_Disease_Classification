import pandas as pd
import lightgbm as lgb
import numpy as np
import shap
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


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

# Compute the mean absolute SHAP values for each feature
mean_abs_shap_values = np.abs(shap_values).mean(axis=0)  # Use shap_values[1] for the positive class in binary classification
mnabsshapval = np.mean(mean_abs_shap_values)

# Create a DataFrame for visualization
shap_importance_df = pd.DataFrame({
    "Feature": X_train.columns,
    "Mean Absolute SHAP Value": mean_abs_shap_values
}).sort_values(by="Mean Absolute SHAP Value", ascending=False)


# Plot the distribution of mean absolute SHAP values
plt.figure(figsize=(10, 6))
sns.histplot(mean_abs_shap_values, bins=20, kde=True, color="skyblue")
plt.axvline(x=mnabsshapval, color="red", linestyle="--", label="Mean SHAP Value")
plt.title("Distribution of Mean Absolute SHAP Values")
plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Drop features with SHAP values less than the mean
features_to_keep = shap_importance_df[shap_importance_df["Mean Absolute SHAP Value"] > mnabsshapval]["Feature"]
X_train_reduced1 = X_train[features_to_keep]
X_test_reduced1 = X_test[features_to_keep]

# Recalculate SHAP importance after the first reduction
shap_importance_df_reduced = shap_importance_df[shap_importance_df["Feature"].isin(features_to_keep)].reset_index(drop=True)

# Compute cumulative contribution
shap_importance_df_reduced["Cumulative Contribution"] = shap_importance_df_reduced[
    "Mean Absolute SHAP Value"
].cumsum() / shap_importance_df_reduced["Mean Absolute SHAP Value"].sum()

# Plot cumulative contribution
plt.figure(figsize=(12, 6))
plt.plot(
    range(1, len(shap_importance_df_reduced) + 1),
    shap_importance_df_reduced["Cumulative Contribution"],
    marker="o",
    linestyle="-",
    color="blue",
    label="Cumulative Contribution"
)
plt.axhline(y=0.75, color="red", linestyle="--", label="75% Threshold")
plt.title("Cumulative Contribution of SHAP Values")
plt.xlabel("Number of Features")
plt.ylabel("Cumulative Contribution")
plt.legend()
plt.grid()
plt.show()

# Keep features contributing up to 75%
selected_features = shap_importance_df_reduced[
    shap_importance_df_reduced["Cumulative Contribution"] <= 0.75
]["Feature"]

# Final reduced dataset
X_train_reduced2 = X_train_reduced1[selected_features]
X_test_reduced2 = X_test_reduced1[selected_features]

# Save combined datasets to CSV
train_combined = pd.concat([X_train_reduced2, y_train], axis=1)
test_combined = pd.concat([X_test_reduced2, y_test], axis=1)

train_combined.to_csv("../../data/processed/afterDroplessThanMean_Shap_train_data2.csv", index=False)
test_combined.to_csv("../../data/processed/afterDroplessThanMean_Shap_test_data2.csv", index=False)

print(f"Original number of features: {X_train.shape[1]}")
print(f"Number of features after dropping SHAP < mean: {X_train_reduced1.shape[1]}")
print(f"Number of features after cumulative contribution: {X_train_reduced2.shape[1]}")
