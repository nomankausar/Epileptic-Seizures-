import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from imblearn.over_sampling import SMOTE

# ================================================
# 🔹 Load and Preprocess Dataset
# ================================================
file_path = r"C:\Documents\all_subjects_q20.xlsx"  # ← Change path as needed
df = pd.read_excel(file_path)

X = df[["H(q=20)"]]
y = df["SOZ_Label"]

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Standardize for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================================
# *** 🔶 RANDOM FOREST MODEL ***
# ================================================
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
conf_rf = confusion_matrix(y_test, y_pred_rf)

# ================================================
# *** 🔷 LOGISTIC REGRESSION MODEL ***
# ================================================
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)
y_pred_lr = log_model.predict(X_test_scaled)
y_prob_lr = log_model.predict_proba(X_test_scaled)[:, 1]
conf_lr = confusion_matrix(y_test, y_pred_lr)

# ================================================
# 🔍 K-FOLD VALIDATION FOR BOTH MODELS
# ================================================
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
metrics_rf = {"Accuracy": [], "Precision": [], "Recall": [], "F1": [], "AUC": []}
metrics_lr = {"Accuracy": [], "Precision": [], "Recall": [], "F1": [], "AUC": []}

for train_idx, test_idx in kf.split(X_resampled, y_resampled):
    X_ktrain, X_ktest = X_resampled.iloc[train_idx], X_resampled.iloc[test_idx]
    y_ktrain, y_ktest = y_resampled.iloc[train_idx], y_resampled.iloc[test_idx]

    X_ktrain_scaled = scaler.fit_transform(X_ktrain)
    X_ktest_scaled = scaler.transform(X_ktest)

    # --- Random Forest ---
    rf_model.fit(X_ktrain, y_ktrain)
    pred_rf = rf_model.predict(X_ktest)
    prob_rf = rf_model.predict_proba(X_ktest)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_ktest, prob_rf)
    auc_rf = auc(fpr_rf, tpr_rf)

    metrics_rf["Accuracy"].append(accuracy_score(y_ktest, pred_rf))
    metrics_rf["Precision"].append(precision_score(y_ktest, pred_rf, zero_division=0))
    metrics_rf["Recall"].append(recall_score(y_ktest, pred_rf, zero_division=0))
    metrics_rf["F1"].append(f1_score(y_ktest, pred_rf, zero_division=0))
    metrics_rf["AUC"].append(auc_rf)

    # --- Logistic Regression ---
    log_model.fit(X_ktrain_scaled, y_ktrain)
    pred_lr = log_model.predict(X_ktest_scaled)
    prob_lr = log_model.predict_proba(X_ktest_scaled)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_ktest, prob_lr)
    auc_lr = auc(fpr_lr, tpr_lr)

    metrics_lr["Accuracy"].append(accuracy_score(y_ktest, pred_lr))
    metrics_lr["Precision"].append(precision_score(y_ktest, pred_lr, zero_division=0))
    metrics_lr["Recall"].append(recall_score(y_ktest, pred_lr, zero_division=0))
    metrics_lr["F1"].append(f1_score(y_ktest, pred_lr, zero_division=0))
    metrics_lr["AUC"].append(auc_lr)

# ================================================
# 📊 Print Metric Summary
# ================================================
print("\n🔶 RANDOM FOREST AVERAGE METRICS")
for m, v in metrics_rf.items():
    print(f"{m}: {np.mean(v):.4f} ± {np.std(v):.4f}")

print("\n🔷 LOGISTIC REGRESSION AVERAGE METRICS")
for m, v in metrics_lr.items():
    print(f"{m}: {np.mean(v):.4f} ± {np.std(v):.4f}")

# ================================================
# 🧩 Confusion Matrix Plotting
# ================================================
def plot_confusion(cm, model_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Non-SOZ", "SOZ"], yticklabels=["Non-SOZ", "SOZ"])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.show()

plot_confusion(conf_rf, "Random Forest")
plot_confusion(conf_lr, "Logistic Regression")

# ================================================
# 📈 ROC Curve Plot
# ================================================
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {np.mean(metrics_rf['AUC']):.2f})", color='blue')
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC = {np.mean(metrics_lr['AUC']):.2f})", color='green')
plt.plot([0, 1], [0, 1], 'k--', label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - RF vs LR")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# ================================================
# 📦 Boxplot for Metric Comparison
# ================================================
plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.DataFrame(metrics_rf))
plt.title("Random Forest Metrics (10-fold)")
plt.grid(True)
plt.tight_layout()
plt.savefig("boxplot_rf.png", dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.DataFrame(metrics_lr))
plt.title("Logistic Regression Metrics ")
plt.grid(True)
plt.tight_layout()
plt.savefig("boxplot_lr.png", dpi=300, bbox_inches='tight')
plt.show()
