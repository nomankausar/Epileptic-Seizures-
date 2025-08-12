import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patheffects as patheffects

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc
)
from imblearn.over_sampling import SMOTE
import matplotlib as mpl
mpl.rcParams['savefig.dpi'] = 300



plt.rcParams.update({
    'font.size': 18,       # Default text
    'axes.titlesize': 18,  # Plot title
    'axes.labelsize': 18,  # X and Y labels
    'xtick.labelsize': 18, # X tick labels
    'ytick.labelsize': 18, # Y tick labels
    'legend.fontsize': 18  # Legend
    
})
# -----------------------------
# Load Dataset
# -----------------------------
file_path =  r"D:\conference paper works replicate\ML RF LR Combined Classifier\Highlighted_Merged all engel I mfdfa v4 data balanced.xlsx"
data = pd.read_excel(file_path)

# Preprocess Data
data = data.loc[:, ~data.columns.astype(str).str.contains(r'^Unnamed', case=False, regex=True)]
if "SOZ_Label" not in data.columns and "Identity Code Value " in data.columns:
    data = data.rename(columns={"Identity Code Value ": "SOZ_Label"})
elif "SOZ_Label" not in data.columns and "Identity Code Value" in data.columns:
    data = data.rename(columns={"Identity Code Value": "SOZ_Label"})
data = data.dropna()

# Define Features and Target
features = [c for c in data.columns if "Segment_" in c]
X = data[features].copy()
y = data["SOZ_Label"].astype(int).copy()

# Apply SMOTE for Balancing (do it once here to mirror your original flow)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# -----------------------------
# Hold-out split for quick plots
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Models
rf_model = RandomForestClassifier(random_state=42)

# LR in a pipeline with scaling
lr_pipe = Pipeline(steps=[
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("lr", LogisticRegression(max_iter=1000, n_jobs=None))
])

# Train & predict (Hold-out)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

lr_pipe.fit(X_train, y_train)
y_pred_lr = lr_pipe.predict(X_test)
y_prob_lr = lr_pipe.predict_proba(X_test)[:, 1]

# -----------------------------
# Confusion matrix plotter
# -----------------------------
def plot_confusion_matrix(cm, model_name, save=False, fname=None, out_dir=".", dpi=300):
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        cm, annot=False, fmt='d', cmap='Blues', cbar=False,
        xticklabels=["Non-SOZ", "SOZ"], yticklabels=["Non-SOZ", "SOZ"],
        linewidths=0.5, linecolor='white'
    )
    tn, fp, fn, tp = cm.ravel()
    counts = [[tn, fp], [fn, tp]]
    tags   = [["TN", "FP"], ["FN", "TP"]]

    for i in range(2):
        for j in range(2):
            ax.text(
                j + 0.5, i + 0.45, str(counts[i][j]),
                ha='center', va='center', fontsize=16, fontweight='bold',
                color='white',
                path_effects=[patheffects.Stroke(linewidth=2, foreground='black'),
                              patheffects.Normal()]
            )
            tag_color = 'red' if tags[i][j] in ('FP', 'FN') else 'white'
            ax.text(
                j + 0.5, i + 0.78, tags[i][j],
                ha='center', va='center', fontsize=12, fontweight='bold',
                color=tag_color,
                path_effects=[patheffects.Stroke(linewidth=2, foreground='black'),
                              patheffects.Normal()]
            )

    plt.xlabel("Predicted Labels", fontsize=20)
    plt.ylabel("True Labels", fontsize=20)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=20)
    plt.tight_layout()
    plt.savefig("Confusion Matrix LG.png", dpi=300)
    if save and fname:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, fname), dpi=300)
        plt.savefig("Confusion Matrix.png", dpi=300)
    plt.show()

# Plot Confusion Matrices (Hold-out)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
plot_confusion_matrix(conf_matrix_rf, "Random Forest")
plot_confusion_matrix(conf_matrix_lr, "Logistic Regression")

# -----------------------------
# 10-fold CV metrics
# -----------------------------
metrics_rf = {"Accuracy": [], "Precision": [], "Recall": [], "Specificity": [], "F1-Score": []}
metrics_lr = {"Accuracy": [], "Precision": [], "Recall": [], "Specificity": [], "F1-Score": []}

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for train_idx, test_idx in kf.split(X_resampled, y_resampled):
    X_train_k, X_test_k = X_resampled.iloc[train_idx], X_resampled.iloc[test_idx]
    y_train_k, y_test_k = y_resampled.iloc[train_idx], y_resampled.iloc[test_idx]

    # Random Forest
    rf_model.fit(X_train_k, y_train_k)
    y_pred_k_rf = rf_model.predict(X_test_k)
    y_prob_k_rf = rf_model.predict_proba(X_test_k)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test_k, y_prob_k_rf)
    auc_rf = auc(fpr_rf, tpr_rf)

    tn, fp, fn, tp = confusion_matrix(y_test_k, y_pred_k_rf).ravel()
    specificity_rf = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics_rf["Accuracy"].append(accuracy_score(y_test_k, y_pred_k_rf))
    metrics_rf["Precision"].append(precision_score(y_test_k, y_pred_k_rf, zero_division=0))
    metrics_rf["Recall"].append(recall_score(y_test_k, y_pred_k_rf, zero_division=0))
    metrics_rf["Specificity"].append(specificity_rf)
    metrics_rf["F1-Score"].append(f1_score(y_test_k, y_pred_k_rf, zero_division=0))
  
    # Logistic Regression (Pipeline handles scaling)
    lr_pipe.fit(X_train_k, y_train_k)
    y_pred_k_lr = lr_pipe.predict(X_test_k)
    y_prob_k_lr = lr_pipe.predict_proba(X_test_k)[:, 1]
    fpr_lr, tpr_lr, _ = roc_curve(y_test_k, y_prob_k_lr)
    auc_lr = auc(fpr_lr, tpr_lr)

    tn, fp, fn, tp = confusion_matrix(y_test_k, y_pred_k_lr).ravel()
    specificity_lr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics_lr["Accuracy"].append(accuracy_score(y_test_k, y_pred_k_lr))
    metrics_lr["Precision"].append(precision_score(y_test_k, y_pred_k_lr, zero_division=0))
    metrics_lr["Recall"].append(recall_score(y_test_k, y_pred_k_lr, zero_division=0))
    metrics_lr["Specificity"].append(specificity_lr)
    metrics_lr["F1-Score"].append(f1_score(y_test_k, y_pred_k_lr, zero_division=0))


# -----------------------------
# Print CV performance
# -----------------------------
def print_summary(name, metrics_dict):
    print(f"\n{name} Metrics (10-fold CV):")
    for metric, vals in metrics_dict.items():
        vals = np.array(vals, dtype=float)
        print(f"{metric}: {vals.mean():.4f} ± {vals.std(ddof=1):.4f}")

print_summary("Random Forest", metrics_rf)
print_summary("Logistic Regression", metrics_lr)

# -----------------------------
# ROC curves from hold-out split
# -----------------------------
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
auc_rf = auc(fpr_rf, tpr_rf)
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
auc_lr = auc(fpr_lr, tpr_lr)

# ==== PLOT ROC CURVES ====
plt.figure(figsize=(10, 6))
plt.plot(fpr_rf, tpr_rf, color='blue', label=f'Random Forest (AUC = {auc_rf:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=2, label='Chance')
plt.xlabel('False Positive Rate', fontsize=20)
plt.ylabel('True Positive Rate', fontsize=20)
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.grid()
plt.show()
plt.savefig("RF ROC.png", dpi=300)

plt.figure(figsize=(10, 6))
plt.plot(fpr_lr, tpr_lr, color='green', label=f'Logistic Regression (AUC = {auc_lr:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='red', linewidth=2, label='Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve Logistic Regression")
plt.legend()
plt.grid()
plt.show()
plt.savefig("LG ROC.png", dpi=300)
# -----------------------------
# Box plots for CV metrics
# -----------------------------
plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame(metrics_rf))
plt.title("Performance Metrics - Random Forest ", fontsize=20)
plt.ylabel("Metric Value(%)", fontsize=20)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.ylim(0.90, 1) 
plt.tight_layout()
plt.show()
plt.savefig("RF Perfomance Metrics.png", dpi=300)

plt.figure(figsize=(12, 6))
sns.boxplot(data=pd.DataFrame(metrics_lr))
plt.title("Performance Metrics - Logistic Regression ", fontsize=20)
plt.ylabel("Metric Value(%)", fontsize=20)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.ylim(0.45, 0.80) 
plt.tight_layout()
plt.show()
plt.savefig("LG Perfomance Metrics.png", dpi=300)