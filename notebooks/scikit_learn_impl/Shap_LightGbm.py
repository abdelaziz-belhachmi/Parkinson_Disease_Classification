import pandas as pd
import lightgbm as lgb
import numpy
import shap
from sklearn.metrics import accuracy_score


# Step 1: Load train and test data
train_path = "../../data/processed/train_data.csv"  # Replace with your train file path
test_path = "../../data/processed/test_data.csv"    # Replace with your test file path

# Assuming the target column is named 'target'
train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Split features and target
X_train = train_data.drop(columns=["target"])  # Replace "target" with the actual name
y_train = train_data["target"]

X_test = test_data.drop(columns=["target"])
y_test = test_data["target"]

X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])


# Step 2: Train an initial LightGBM model
lgbm_model = lgb.LGBMClassifier(objective="binary", random_state=42,is_unbalance=True)
lgbm_model.fit(X_train, y_train)



# Step 3: Compute SHAP values
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_train)[1]  # For binary classification, class 1



# Calculate mean absolute SHAP values
mean_shap_values = abs(shap_values).mean(axis=0)
shap_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Mean_SHAP_Value': mean_shap_values
}).sort_values(by='Mean_SHAP_Value', ascending=False)

print("Top Features by SHAP Importance:\n", shap_importance)



# Visualize combined SHAP values
# Compute SHAP values (directly returns a 2D array for binary classification)
shap_values = explainer.shap_values(X_train)  # Returns a single array for binary classification

# Plot SHAP summary
shap.summary_plot(shap_values, X_train, feature_names=X_train.columns)



# Step 4: Select top features based on SHAP
top_n = 100  # Adjust this number based on your analysis
selected_features = shap_importance['Feature'].iloc[:top_n].tolist()


# Filter datasets to only use selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]


# Step 5: Retrain LightGBM with selected features
lgbm_model_selected = lgb.LGBMClassifier(objective="binary", random_state=42,is_unbalance=True)
lgbm_model_selected.fit(X_train_selected, y_train)


# Step 6: Evaluate the final model
y_pred = lgbm_model_selected.predict(X_test_selected)
accuracy = accuracy_score(y_test, y_pred)
print("\nFinal Model Accuracy with Selected Features:", accuracy)


from sklearn.model_selection import cross_val_score

scores = cross_val_score(lgbm_model_selected, X_train_selected, y_train, cv=5, scoring="accuracy")
print("Cross-Validation Accuracy:", scores.mean())
