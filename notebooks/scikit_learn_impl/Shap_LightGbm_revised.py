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

shap.summary_plot(shap_values, X_train,max_display=len(X_train.columns))
shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=len(X_train.columns))


# # Calculate the mean absolute SHAP values for each feature
# mean_abs_shap_values = np.mean(np.abs(shap_values), axis=0)
#
# # Create a DataFrame with features and their corresponding mean absolute SHAP values
# shap_importance = pd.DataFrame({
#     'feature': X_train.columns,
#     'mean_abs_shap': mean_abs_shap_values
# })
#
# # Sort the features by mean absolute SHAP values in ascending order
# shap_importance_sorted = shap_importance.sort_values(by='mean_abs_shap', ascending=True)
#
#
# # Select the bottom N least important features
# least_important_features = shap_importance_sorted
#
# # Plot a bar chart for the least important features
# plt.barh(least_important_features['feature'], least_important_features['mean_abs_shap'])
# plt.xlabel('Mean Absolute SHAP Value')
# plt.title('Least Important Features (Based on SHAP Values)')
# plt.show()

#
# # Step 4: Iteratively remove least important features
# remaining_features = X_train.columns.tolist()
# accuracy_history = []
#
# while len(remaining_features) > 1:
#     # Calculate mean absolute SHAP values
#     mean_shap_values = abs(shap_values).mean(axis=0)
#     shap_importance = pd.DataFrame({
#         'Feature': remaining_features,
#         'Mean_SHAP_Value': mean_shap_values
#     }).sort_values(by='Mean_SHAP_Value', ascending=False)
#
#     # Evaluate current model
#     y_pred = lgbm_model.predict(X_test[remaining_features])
#     accuracy = accuracy_score(y_test, y_pred)
#     accuracy_history.append((len(remaining_features), accuracy))
#
#     print(f"Using {len(remaining_features)} features, accuracy: {accuracy:.4f}")
#
#     # Stop if removing features decreases accuracy significantly
#     if len(accuracy_history) > 1:
#         if accuracy < accuracy_history[-2][1] - 0.01:  # 1% drop in accuracy
#             print("Stopping feature removal as accuracy dropped.")
#             break
#
#     # Remove the least important feature
#     least_important_feature = shap_importance['Feature'].iloc[-1]
#     remaining_features.remove(least_important_feature)
#
#     # Retrain LightGBM with remaining features
#     X_train_filtered = X_train[remaining_features]
#     lgbm_model = lgb.LGBMClassifier(objective="binary", random_state=42, is_unbalance=True)
#     lgbm_model.fit(X_train_filtered, y_train)
#
#     # Recompute SHAP values with updated model
#     explainer = shap.TreeExplainer(lgbm_model)
#     shap_values = explainer.shap_values(X_train_filtered)[1]
#
# # Step 5: Final evaluation
# print("\nFinal Selected Features:", remaining_features)
# X_train_final = X_train[remaining_features]
# X_test_final = X_test[remaining_features]
#
# y_pred_final = lgbm_model.predict(X_test_final)
# final_accuracy = accuracy_score(y_test, y_pred_final)
# print(f"Final Model Accuracy with {len(remaining_features)} Features: {final_accuracy:.4f}")
#
# # Cross-validation with final feature set
# cv_scores = cross_val_score(lgbm_model, X_train_final, y_train, cv=5, scoring="accuracy")
# print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f}")
