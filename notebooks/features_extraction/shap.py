import pandas as pd
import lightgbm as lgb
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from itertools import combinations

def compute_shap_values(model, X):
    """
    Approximate SHAP values for a LightGBM model.
    """
    n_features = X.shape[1]
    n_samples = X.shape[0]
    shap_values = np.zeros_like(X.values, dtype=float)
    baseline = X.mean(axis=0).values  # Baseline values for missing features

    for i in range(n_features):
        contributions = []
        for sample_idx in range(n_samples):  # Loop through all rows
            baseline_sample = baseline.copy()
            for subset in combinations(range(n_features), n_features - 1):
                if i in subset:
                    continue

                subset_with = list(subset) + [i]
                subset_without = list(subset)

                # Fill missing features with baseline values
                X_with = baseline_sample.copy()
                X_with[subset_with] = X.iloc[sample_idx, subset_with].values

                X_without = baseline_sample.copy()
                X_without[subset_without] = X.iloc[sample_idx, subset_without].values

                # Predict with and without the feature
                pred_with = model.predict_proba(X_with.reshape(1, -1))[0, 1]
                pred_without = model.predict_proba(X_without.reshape(1, -1))[0, 1]

                contributions.append(pred_with - pred_without)

        # Average marginal contributions for feature `i`
        shap_values[:, i] = np.mean(contributions)

    return shap_values




# Step 1: Load train and test data
train_path = "../../data/processed/train_data.csv"  
test_path = "../../data/processed/test_data.csv"  

# Load data
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Split features and target
X_train = train_data.drop(columns=["target"])  # Drop target and id
y_train = train_data["target"]

X_test = test_data.drop(columns=["target"])  # Drop target and id
y_test = test_data["target"]

# Step 2: Train an initial LightGBM model
lgbm_model = lgb.LGBMClassifier(objective="binary", random_state=42, is_unbalance=True)
lgbm_model.fit(X_train, y_train)


# Calculate SHAP values
shap_values = compute_shap_values(lgbm_model, X_train)

# Visualize SHAP summary (horizontal bar plot)
feature_importance = np.abs(shap_values).mean(axis=0)
features = X_train.columns

plt.barh(features, feature_importance)
plt.xlabel("Mean Absolute SHAP Value")
plt.ylabel("Feature")
plt.title("Feature Importance (Approx. SHAP Values)")
plt.tight_layout()
plt.show()
