import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix
)
from imblearn.combine import SMOTETomek

# ========== Load Dataset ==========
file_path = r"C:\Documents\all_subjects_H(q) on q_values -40 to +40 with increment 0.01_with_fractional.csv"
df = pd.read_csv(file_path)
df.drop(columns=["Subject_ID", "Channel"], inplace=True)
X = df.drop(columns=["is_soz"])
y = df["is_soz"]

# ========== Feature Engineering ==========
X = pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(X), columns=X.columns)
X_df = X.copy()
X_df["Entropy"] = np.apply_along_axis(lambda x: entropy(np.abs(x) + 1e-10), axis=1, arr=X)
X_df["Kurtosis"] = np.apply_along_axis(kurtosis, axis=1, arr=X)

# ========== Scale + Feature Selection + Resample ==========
X_scaled = StandardScaler().fit_transform(X_df)
X_selected = SelectKBest(score_func=mutual_info_classif, k=50).fit_transform(X_scaled, y)
X_res, y_res = SMOTETomek(random_state=42).fit_resample(X_selected, y)

# ========== Models ==========
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=4,
        max_leaf_nodes=100,
        max_features='sqrt',
        class_weight='balanced',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),
    "Logistic Regression": LogisticRegression(max_iter=10000, solver='lbfgs', random_state=42)
}

# ========== Output Folder ==========
output_dir = "results_rf_logistic"
os.makedirs(output_dir, exist_ok=True)

# ========== Overfit AUC Plot Function ==========
def plot_mean_overfit_roc(train_aucs, test_aucs, model_name, filename):
    plt.figure(figsize=(8, 6))
    plt.plot(train_aucs, label='Train AUC', marker='o', color='blue')
    plt.plot(test_aucs, label='Test AUC', marker='o', color='orange')
    plt.fill_between(range(len(train_aucs)), train_aucs, test_aucs, color='red', alpha=0.1)
    plt.title(f"Mean Overfit ROC Gap - {model_name}")
    plt.xlabel("Fold")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.show()

# ========== Evaluation Loop ==========
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n🧪 Evaluating: {name}")
    metrics = {m: [] for m in ["Accuracy", "Precision", "Recall", "F1", "AUC"]}
    conf_matrix_total = np.zeros((2, 2), dtype=int)
    fpr_final, tpr_final = None, None
    train_auc_all, test_auc_all = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_res, y_res)):
        X_train, X_test = X_res[train_idx], X_res[test_idx]
        y_train, y_test = y_res[train_idx], y_res[test_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            print(f"⚠️ Skipping fold {fold} for {name} due to class imbalance.")
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        y_train_prob = model.predict_proba(X_train)[:, 1]
        auc_train = auc(*roc_curve(y_train, y_train_prob)[:2])
        auc_test = auc(*roc_curve(y_test, y_prob)[:2])
        train_auc_all.append(auc_train)
        test_auc_all.append(auc_test)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        if fpr_final is None:
            fpr_final, tpr_final = fpr, tpr

        conf_matrix_total += confusion_matrix(y_test, y_pred)
        metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["Precision"].append(precision_score(y_test, y_pred))
        metrics["Recall"].append(recall_score(y_test, y_pred))
        metrics["F1"].append(f1_score(y_test, y_pred))
        metrics["AUC"].append(auc_test)

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(os.path.join(output_dir, f"metrics_{name.replace(' ', '_')}.csv"), index=False)

    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix_total, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=["Non-SOZ", "SOZ"], yticklabels=["Non-SOZ", "SOZ"])
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{name.replace(' ', '_')}.png"), dpi=300)
    plt.show()

    # ROC Curve
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_final, tpr_final, label=f"AUC = {np.mean(metrics['AUC']):.2f}", color='green')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"roc_curve_{name.replace(' ', '_')}.png"), dpi=300)
    plt.show()

    # Boxplot of Metrics
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_metrics)
    plt.title(f"{name} Metrics - 10-Fold")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"boxplot_{name.replace(' ', '_')}.png"), dpi=300)
    plt.show()

    # Train vs Test AUC
    plt.figure(figsize=(8, 6))
    plt.plot(train_auc_all, label="Train AUC", marker='o', color='blue')
    plt.plot(test_auc_all, label="Test AUC", marker='o', color='orange')
    plt.fill_between(range(len(train_auc_all)), train_auc_all, test_auc_all, color='red', alpha=0.1)
    plt.title(f"Train vs Test AUC per Fold - {name}")
    plt.xlabel("Fold")
    plt.ylabel("AUC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"overfit_auc_gap_{name.replace(' ', '_')}.png"), dpi=300)
    plt.show()

    # Mean Overfit ROC Plot
    plot_mean_overfit_roc(train_auc_all, test_auc_all, name, f"mean_overfit_roc_{name.replace(' ', '_')}.png")

    # Final Results Print
    auc_gap = np.mean(train_auc_all) - np.mean(test_auc_all)
    print(f"\n📌 10-Fold Average Metrics ({name}):")
    for m in metrics:
        print(f"{m}: {np.mean(metrics[m]):.4f} ± {np.std(metrics[m]):.4f}")
    print(f"\n🔍 Overfit check ({name}):")
    print(f"Mean Train AUC: {np.mean(train_auc_all):.4f}")
    print(f"Mean Test  AUC: {np.mean(test_auc_all):.4f}")
    print(f"AUC Gap       : {auc_gap:.4f}")
    if auc_gap > 0.05:
        print("⚠️ Likely Overfitting!")
    else:
        print("✅ No strong overfitting detected.")
