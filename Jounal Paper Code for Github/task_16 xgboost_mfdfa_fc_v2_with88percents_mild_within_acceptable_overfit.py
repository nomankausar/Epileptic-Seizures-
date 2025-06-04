import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, kurtosis
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek

# ================================
#  Load and Prepare Dataset
# ================================
file_path = r"C:\Documents\all_subjects_H(q) on q_values -40 to +40 with increment 0.01_with_fractional.csv"
df = pd.read_csv(file_path)
df.drop(columns=["Subject_ID", "Channel"], inplace=True)

X = df.drop(columns=["is_soz"])
y = df["is_soz"]

# ================================
#  Imputation + Feature Engineering
# ================================
X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
X_df = X.copy()
X_df["Entropy"] = np.apply_along_axis(lambda x: entropy(np.abs(x) + 1e-10), 1, X)
X_df["Kurtosis"] = np.apply_along_axis(kurtosis, 1, X)

# ================================
# ⚙️ Scaling, Feature Selection, Resampling
# ================================
X_scaled = StandardScaler().fit_transform(X_df)
X_selected = SelectKBest(mutual_info_classif, k=50).fit_transform(X_scaled, y)
X_resampled, y_resampled = SMOTETomek(random_state=42).fit_resample(X_selected, y)

# ================================
#  XGBoost with GridSearchCV (GPU)
# ================================
param_grid = {
    'n_estimators': [200, 500],
    'max_depth': [4, 5],
    'learning_rate': [0.01, 0.05],
    'subsample': [0.7, 0.85],
    'colsample_bytree': [0.7, 0.85],
    'reg_alpha': [0.1, 0.3],
    'reg_lambda': [1.0, 2.0],
    'gamma': [0.1, 0.3]
}

xgb_base = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    verbosity=0,
    tree_method='gpu_hist',
    predictor='gpu_predictor'
)

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    scoring='f1',
    cv=kf,
    n_jobs=-1,
    verbose=0
)
grid.fit(X_resampled, y_resampled)
best_model = grid.best_estimator_

# ================================
# 10-Fold CV Evaluation + Overfit Plot
# ================================
metrics = {m: [] for m in ["Accuracy", "Precision", "Recall", "F1", "AUC"]}
conf_matrix_total = np.zeros((2, 2), dtype=int)
fpr_final, tpr_final = None, None

for fold, (train_idx, test_idx) in enumerate(kf.split(X_resampled, y_resampled)):
    X_train, X_test = X_resampled[train_idx], X_resampled[test_idx]
    y_train, y_test = y_resampled[train_idx], y_resampled[test_idx]

    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    if fold == 0:
        fpr_final, tpr_final = fpr, tpr
        y_train_prob = best_model.predict_proba(X_train)[:, 1]
        fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
        auc_train = auc(fpr_train, tpr_train)
        auc_test = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr_train, tpr_train, label=f"Train ROC (AUC = {auc_train:.2f})", color="blue")
        plt.plot(fpr, tpr, label=f"Test ROC (AUC = {auc_test:.2f})", color="orange")
        plt.plot([0, 1], [0, 1], 'k--', label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Train vs Test ROC - XGBoost (GPU) - Fold 0")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs("results_xgb_gpu_final", exist_ok=True)
        plt.savefig("results_xgb_gpu_final/train_vs_test_roc_xgb_gpu.png", dpi=300)
        plt.show()

    conf_matrix_total += confusion_matrix(y_test, y_pred)
    metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
    metrics["Precision"].append(precision_score(y_test, y_pred))
    metrics["Recall"].append(recall_score(y_test, y_pred))
    metrics["F1"].append(f1_score(y_test, y_pred))
    metrics["AUC"].append(auc(fpr, tpr))

# ================================
# Save Outputs
# ================================
output_dir = "results_xgb_gpu_final"
os.makedirs(output_dir, exist_ok=True)
pd.DataFrame(metrics).to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
pd.DataFrame([grid.best_params_]).to_csv(os.path.join(output_dir, "best_params.csv"), index=False)

# Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix_total, annot=True, fmt='d', cmap='coolwarm',
            xticklabels=["Non-SOZ", "SOZ"], yticklabels=["Non-SOZ", "SOZ"])
plt.title("Confusion Matrix - XGBoost (GPU)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_xgb_gpu.png"), dpi=300)
plt.show()

# ROC Curve
plt.figure(figsize=(8, 6))
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_final, tpr_final, label=f"AUC = {np.mean(metrics['AUC']):.2f}", color='green')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost (GPU)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "roc_curve_xgb_gpu.png"), dpi=300)
plt.show()

# Boxplot of Metrics
plt.figure(figsize=(10, 6))
sns.boxplot(data=pd.DataFrame(metrics))
plt.title("XGBoost (GPU) Metrics - 10-Fold")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "boxplot_xgb_gpu.png"), dpi=300)
plt.show()

# ================================
#  Learning Curve (Improved)
# ================================
def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=5, scoring='roc_auc',
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_mean, 'o-', label="Train AUC", color='blue')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
    plt.plot(train_sizes, val_mean, 'o-', label="Validation AUC", color='orange')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='orange')
    plt.title(title)
    plt.xlabel("Training Set Size")
    plt.ylabel("AUC Score")
    plt.xticks(train_sizes, [f"{int(s / len(X) * 100)}%" for s in train_sizes])
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{title.lower().replace(' ', '_')}.png"), dpi=300)
    plt.show()

plot_learning_curve(best_model, X_resampled, y_resampled, "XGBoost GPU Learning Curve")

# ================================
#  Print Summary
# ================================
print("\n Best Parameters:")
print(grid.best_params_)

print("\n 10-Fold Average Metrics:")
for m in metrics:
    print(f"{m}: {np.mean(metrics[m]):.4f} ± {np.std(metrics[m]):.4f}")
