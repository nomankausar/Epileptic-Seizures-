import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from mne.io import read_raw_edf, read_raw_fif
import difflib

# ------------ Configuration ------------ #
INPUT_FOLDER = r"C:\Engel 1 Subjects\allfiles"
SOZ_CSV = r"C:\Engel 1 Subjects\Meta Data\SOZ_Channels_info.csv"
OUTPUT_FOLDER = os.path.join(INPUT_FOLDER, "EZ_NON EZ_CSD_Boxplot7")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

WINDOW_SEC = 10
STEP_SEC = 5
BANDPASS = (1, 60)

# ------------ Load SOZ CSV ------------ #
soz_data = pd.read_csv(SOZ_CSV)
soz_dict = {
    row['Subject'].strip(): row.dropna()[1:].astype(str).str.strip().tolist()
    for _, row in soz_data.iterrows()
}

def extract_subject_id(filename):
    base = os.path.basename(filename)
    parts = base.split("_")
    for part in parts:
        if part.startswith("sub-"):
            return part.replace("sub-", "")
    return parts[0].replace(".edf", "").replace(".fif", "")

def fuzzy_match(channel_name, channel_list):
    matches = difflib.get_close_matches(channel_name.lower(), [c.lower() for c in channel_list], n=1, cutoff=0.8)
    return bool(matches)

# ------------ Metric Computation ------------ #
def compute_sliding_metrics(data, sfreq, window_sec=10, step_sec=5):
    win = int(window_sec * sfreq)
    step = int(step_sec * sfreq)
    var_list, acf_list, times = [], [], []

    for start in range(0, len(data) - win, step):
        seg = data[start:start + win]
        var = np.var(seg)
        x = seg - np.mean(seg)
        acf = np.corrcoef(x[:-1], x[1:])[0, 1]
        var_list.append(var)
        acf_list.append(acf)
        times.append((start + win // 2) / sfreq)
    
    return np.array(times), np.array(var_list), np.array(acf_list)

# ------------ EEG Loader ------------ #
def load_eeg(file_path):
    if file_path.endswith(".edf"):
        return read_raw_edf(file_path, preload=True, verbose=False)
    elif file_path.endswith(".fif"):
        return read_raw_fif(file_path, preload=True, verbose=False)
    return None

# ------------ Main Analysis ------------ #
all_results = []  # Store all EZ/NON EZ values
ttest_results = []  # Store per-subject t-test p-values

def process_file(file_path):
    subj_id = extract_subject_id(file_path)
    raw = load_eeg(file_path)
    if raw is None or not raw.ch_names:
        print(f"Failed to load or no channels found: {file_path}")
        return

    print(f"Loaded {file_path} with {len(raw.ch_names)} channels.")
    raw.filter(*BANDPASS)
    sfreq = raw.info['sfreq']
    soz_channels = soz_dict.get(subj_id, [])

    print(f"Processing {subj_id} | SOZ Channels: {soz_channels}")
    results = {"EZ": {"var": [], "acf": []}, "NON EZ": {"var": [], "acf": []}}

    for ch in raw.ch_names:
        data, _ = raw.copy().pick([ch])[:]
        if data.shape[1] < WINDOW_SEC * sfreq:
            continue
        signal = data[0]
        times, var, acf = compute_sliding_metrics(signal, sfreq, WINDOW_SEC, STEP_SEC)
        group = "EZ" if fuzzy_match(ch, soz_channels) else "NON EZ"
        results[group]["var"].append(var)
        results[group]["acf"].append(acf)

    # Append values to all_results
    for metric in ["var", "acf"]:
        for group in ["EZ", "NON EZ"]:
            if results[group][metric]:
                vals = np.concatenate(results[group][metric])
                for val in vals:
                    all_results.append((subj_id, group, metric, val))

    # Individual Boxplot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for idx, metric in enumerate(["var", "acf"]):
        ez_vals = np.concatenate(results["EZ"][metric]) if results["EZ"][metric] else []
        nonez_vals = np.concatenate(results["NON EZ"][metric]) if results["NON EZ"][metric] else []

        bp = axes[idx].boxplot(
            [ez_vals, nonez_vals],
            patch_artist=True,
            boxprops=dict(color='black'),
            medianprops=dict(color='black'),
            flierprops=dict(markerfacecolor='black', marker='o', markersize=5),
        )
        colors = ['red', 'blue']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        axes[idx].set_title(f"{metric.upper()} Distribution")
        axes[idx].set_ylabel("Variance (µV²)" if metric == "var" else "Autocorrelation")

    fig.suptitle(f"{subj_id} | EZ vs NON EZ | Boxplot | Window: {WINDOW_SEC}s")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, f"{subj_id}_EZ_NOZ_Boxplot.png"))
    plt.close()

    # T-test now inside the function
    for metric in ["var", "acf"]:
        ez_vals = np.concatenate(results["EZ"][metric]) if results["EZ"][metric] else []
        nonez_vals = np.concatenate(results["NON EZ"][metric]) if results["NON EZ"][metric] else []
        pval = np.nan
        if len(ez_vals) > 0 and len(nonez_vals) > 0:
            _, pval = ttest_ind(ez_vals, nonez_vals, equal_var=False)
        ttest_results.append((subj_id, metric, len(ez_vals), len(nonez_vals), pval))


    # # Individual Boxplot
    # fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # for idx, metric in enumerate(["var", "acf"]):
    #     ez_vals = np.concatenate(results["EZ"][metric]) if results["EZ"][metric] else []
    #     nonez_vals = np.concatenate(results["NON EZ"][metric]) if results["NON EZ"][metric] else []

    #     axes[idx].boxplot(
    #         [ez_vals, nonez_vals],
    #         labels=["EZ", "NON EZ"],
    #         patch_artist=True,
    #         boxprops=dict(facecolor='red', color='black'),
    #         medianprops=dict(color='black'),
    #         flierprops=dict(markerfacecolor='red', marker='o', markersize=5),
    #     )
    #     if len(axes[idx].artists) > 1:
    #         axes[idx].artists[1].set_facecolor("blue")

    #     axes[idx].set_title(f"{metric.upper()} Distribution")
    #     axes[idx].set_ylabel("Variance (µV²)" if metric == "var" else "Autocorrelation")

    # fig.suptitle(f"{subj_id} | EZ vs NON EZ | Boxplot")
    # plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_FOLDER, f"{subj_id}_EZ_NOZ_Boxplot.png"))
    # plt.close()
    



# ------------ Run All ------------ #
if __name__ == "__main__":
    all_files = [os.path.join(INPUT_FOLDER, f) 
                 for f in os.listdir(INPUT_FOLDER) 
                 if f.endswith((".edf", ".fif"))]

    for file_path in all_files:
        try:
            process_file(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # ----------- Export CSVs ----------- #
    df_values = pd.DataFrame(all_results, columns=["Subject", "Group", "Metric", "Value"])
    df_values.to_csv(os.path.join(OUTPUT_FOLDER, "EZ_NON_EZ_Values.csv"), index=False)

    df_ttests = pd.DataFrame(ttest_results, columns=["Subject", "Metric", "EZ_Count", "NON_EZ_Count", "p_value"])
    df_ttests.to_csv(os.path.join(OUTPUT_FOLDER, "EZ_NON_EZ_TTest_Results.csv"), index=False)

    # ----------- Combined Boxplot ----------- #
    for metric in ["var", "acf"]:
        plt.figure(figsize=(20, 8))
        metric_df = df_values[df_values["Metric"] == metric]
        subjects = sorted(metric_df["Subject"].unique())
        positions = np.arange(len(subjects))
        width = 0.35

        data_ez = [metric_df[(metric_df["Subject"] == s) & (metric_df["Group"] == "EZ")]["Value"].values for s in subjects]
        data_nonez = [metric_df[(metric_df["Subject"] == s) & (metric_df["Group"] == "NON EZ")]["Value"].values for s in subjects]

        b1 = plt.boxplot(data_nonez, positions=positions - width/2, widths=width, patch_artist=True,
                         boxprops=dict(facecolor='blue'))
        b2 = plt.boxplot(data_ez, positions=positions + width/2, widths=width, patch_artist=True,
                         boxprops=dict(facecolor='red'))

        plt.xticks(positions, subjects, rotation=45, ha="right")
        plt.xlabel("Subject")
        plt.ylabel("Variance (µV²)" if metric == "var" else "Autocorrelation")
        plt.title(f"All Subjects - {metric.upper()} Boxplot")
        plt.legend([b1["boxes"][0], b2["boxes"][0]], ["NON EZ", "EZ"])
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"ALLSUBJECTS_{metric.upper()}_Boxplot.png"))
        plt.close()

    # ----------- Per-Site Combined Boxplots ----------- #
    print("Generating site-wise plots...")
    df_values["Site"] = df_values["Subject"].apply(lambda s: ''.join([c for c in s if not c.isdigit()]))
    for site in df_values["Site"].unique():
        site_df = df_values[df_values["Site"] == site]
        for metric in ["var", "acf"]:
            plt.figure(figsize=(18, 7))
            metric_df = site_df[site_df["Metric"] == metric]
            subjects = sorted(metric_df["Subject"].unique())
            positions = np.arange(len(subjects))
            width = 0.35

            data_ez = [metric_df[(metric_df["Subject"] == s) & (metric_df["Group"] == "EZ")]["Value"].values for s in subjects]
            data_nonez = [metric_df[(metric_df["Subject"] == s) & (metric_df["Group"] == "NON EZ")]["Value"].values for s in subjects]

            b1 = plt.boxplot(data_nonez, positions=positions - width/2, widths=width,
                             patch_artist=True, boxprops=dict(facecolor='blue'))
            b2 = plt.boxplot(data_ez, positions=positions + width/2, widths=width,
                             patch_artist=True, boxprops=dict(facecolor='red'))

            plt.xticks(positions, subjects, rotation=45, ha="right")
            plt.xlabel("Subject")
            plt.ylabel("Variance (µV²)" if metric == "var" else "Autocorrelation")
            plt.title(f"{site.upper()} - {metric.upper()} Boxplot")
            plt.legend([b1["boxes"][0], b2["boxes"][0]], ["NON EZ", "EZ"])
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, f"{site}_EZ_NON_EZ_{metric.upper()}_Boxplot.png"))
            plt.close()
