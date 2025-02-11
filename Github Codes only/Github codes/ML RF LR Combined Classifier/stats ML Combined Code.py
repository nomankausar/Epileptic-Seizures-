import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE

# Load Dataset
file_path = r"C:\Documents\Highlighted_Merged all engel I mfdfa v4 data balanced.xlsx"
data = pd.read_excel(file_path)

# Preprocess Data
data.rename(columns={"Identity Code Value ": "SOZ_Label"}, inplace=True)
data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
data = data.dropna()

# Define Features and Target
features = [col for col in data.columns if "Segment_" in col]
X = data[features]
y = data["SOZ_Label"]

# Apply SMOTE for Balancing
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

# Standardize Features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

# Train Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_prob_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

# Compute Confusion Matrices
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)

# Function to Plot Confusion Matrix
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=["Non-SOZ", "SOZ"], yticklabels=["Non-SOZ", "SOZ"])

    tn, fp, fn, tp = cm.ravel()
    labels = [[f'TN={tn}', f'FP={fp}'], [f'FN={fn}', f'TP={tp}']]
    
    for i in range(2):
        for j in range(2):
            cell_text = labels[i][j]
            color = 'red' if cell_text.startswith('FP') or cell_text.startswith('FN') else 'white'  # Red for FP & FN, White for TN & TP
            plt.text(j + 0.5, i + 0.5, cell_text, ha='center', va='center', color=color, fontsize=14, fontweight='bold')

    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=16)
    plt.show()

# Plot Confusion Matrices
plot_confusion_matrix(conf_matrix_rf, "Random Forest")
plot_confusion_matrix(conf_matrix_lr, "Logistic Regression")

# Compute Performance Metrics Across K-Fold
metrics_rf = {"Accuracy": [], "Precision": [], "Recall": [], "F1-Score": [], "ROC AUC": []}
metrics_lr = {"Accuracy": [], "Precision": [], "Recall": [], "F1-Score": [], "ROC AUC": []}

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for train_idx, test_idx in kf.split(X_resampled, y_resampled):
    X_train_k, X_test_k = X_resampled.iloc[train_idx], X_resampled.iloc[test_idx]
    y_train_k, y_test_k = y_resampled.iloc[train_idx], y_resampled.iloc[test_idx]

    # Random Forest Model
    rf_model.fit(X_train_k, y_train_k)
    y_pred_k_rf = rf_model.predict(X_test_k)
    y_prob_k_rf = rf_model.predict_proba(X_test_k)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test_k, y_prob_k_rf)
    auc_rf = auc(fpr_rf, tpr_rf)

    metrics_rf["Accuracy"].append(accuracy_score(y_test_k, y_pred_k_rf))
    metrics_rf["Precision"].append(precision_score(y_test_k, y_pred_k_rf, zero_division=0))
    metrics_rf["Recall"].append(recall_score(y_test_k, y_pred_k_rf, zero_division=0))
    metrics_rf["F1-Score"].append(f1_score(y_test_k, y_pred_k_rf, zero_division=0))
    metrics_rf["ROC AUC"].append(auc_rf)

    # Logistic Regression Model
    log_reg.fit(X_train_k, y_train_k)
    y_pred_k_lr = log_reg.predict(X_test_k)
    y_prob_k_lr = log_reg.predict_proba(X_test_k)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test_k, y_prob_k_lr)
    auc_lr = auc(fpr_lr, tpr_lr)

    metrics_lr["Accuracy"].append(accuracy_score(y_test_k, y_pred_k_lr))
    metrics_lr["Precision"].append(precision_score(y_test_k, y_pred_k_lr, zero_division=0))
    metrics_lr["Recall"].append(recall_score(y_test_k, y_pred_k_lr, zero_division=0))
    metrics_lr["F1-Score"].append(f1_score(y_test_k, y_pred_k_lr, zero_division=0))
    metrics_lr["ROC AUC"].append(auc_lr)

# Print Performance Metrics
print("Random Forest Metrics:")
for metric, values in metrics_rf.items():
    print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

print("\nLogistic Regression Metrics:")
for metric, values in metrics_lr.items():
    print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")

# Plot Separate ROC Curves
plt.figure(figsize=(10, 6))
plt.plot(fpr_rf, tpr_rf, color='blue', label=f'Random Forest (AUC = {np.mean(metrics_rf["ROC AUC"]):.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=2, label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(fpr_lr, tpr_lr, color='green', label=f'Logistic Regression (AUC = {np.mean(metrics_lr["ROC AUC"]):.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=2, label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.grid()
plt.show()

# Plot Box Plots Separately
plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame(metrics_rf), palette="Set2")
plt.title("Performance Metrics - Random Forest", fontsize=16)
plt.ylabel("Metric Value", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame(metrics_lr), palette="Set2")
plt.title("Performance Metrics - Logistic Regression", fontsize=16)
plt.ylabel("Metric Value", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
