import os
import re
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from scipy.stats import ks_2samp

# === PARAMETERS ===
data_dir           = r"C:\Engel 1 Subjects\allfiles"
soz_csv            = os.path.join(data_dir, "SOZ_Channels_info.csv")
seconds_to_analyze = None
q_values           = np.linspace(-40.0, 40.0,801)
scale_min, scale_max = 16, 4096

# === LOAD SOZ INFO AND CLEAN ===
soz_df = pd.read_csv(soz_csv)
soz_long = (
    soz_df
    .melt(id_vars="Subject", var_name="Channel_Index", value_name="Channel")
    .dropna(subset=["Channel"])
)
soz_long["Subject_ID"] = soz_long["Subject"].astype(str)\
    .apply(lambda s: re.sub(r'[^A-Za-z0-9]', '', s).upper())
soz_long["Clean_Channel"] = soz_long["Channel"].astype(str)\
    .apply(lambda s: re.sub(r'[^A-Za-z0-9]', '', s).upper())

subject_soz_map = (
    soz_long
    .groupby("Subject_ID")["Clean_Channel"]
    .apply(list)
    .to_dict()
)

# === HELPERS ===
def extract_subject_id(fname):
    m = re.search(r"sub-([A-Za-z0-9]+)_", fname)
    return m.group(1).upper() if m else None

def clean_channel_name(name):
    return re.sub(r'[^A-Za-z0-9]', '', str(name)).upper()

def fuzzy_match(ch, soz_list):
    return any(ch in soz or soz in ch for soz in soz_list)
def MFDFA(signal, scale_min, scale_max, q_vals):
    x = np.cumsum(signal - np.mean(signal))
    scales = 2 ** np.arange(int(np.log2(scale_min)),
                           int(np.log2(scale_max))+1)
    # instantiation variables
    fluct = np.zeros((len(scales), len(q_vals)))
    Hq    = np.zeros(len(q_vals))

    # find fluctuations across all scales
    for si, s in enumerate(scales):
        segs = len(x)//s # length of the segment
        F_s  = [] # fluctuations
        for i in range(segs): # iterate over all segments
            seg = x[i*s:(i+1)*s] # extract each data segment
            t   = np.arange(s) # array from 0 to s-1
            cfs = np.polyfit(t, seg, 1)
            trend = np.polyval(cfs, t)
            F_s.append(np.sqrt(np.mean((seg - trend)**2))) # compute fluctuation and append
        F_s = np.array(F_s) # convert to a numpy array
        F_s = F_s[F_s>1e-8] # check if fluctuation is greater thn 1e-8 and replace with 0
        for qi, q in enumerate(q_vals):
            if q==0:
                fluct[si,qi] = np.exp(0.5*np.mean(np.log(F_s**2)))
            else:
                fluct[si,qi] = np.mean(F_s**q)**(1.0/q)

    log_sc = np.log2(scales)
    for qi in range(len(q_vals)):
        log_F = np.log2(fluct[:,qi])
        Hq[qi] = np.polyfit(log_sc, log_F,1)[0]

    return Hq

def process_channel(args):
    data, ch_name, soz_list = args
    label = 'EZ' if fuzzy_match(clean_channel_name(ch_name), soz_list) else 'Non-EZ'
    hq    = MFDFA(data, scale_min, scale_max,q_values)
    return label, hq

def process_file(filepath, soz_list, max_dur=None):
    ext = filepath.lower().split('.')[-1]
    reader = mne.io.read_raw_edf if ext=='edf' else mne.io.read_raw_fif
    raw = reader(filepath, preload=False, verbose=False)
    if max_dur:
        raw.crop(0, max_dur)
    raw.load_data()
    raw.notch_filter(60)
    eeg, _ = raw.get_data(return_times=True)
    chs    = raw.info['ch_names']
    args   = [(eeg[i], chs[i], soz_list) for i in range(len(chs))]
    with Pool(cpu_count()) as p:
        return p.map(process_channel, args)

def main():
    from openpyxl import Workbook
    from pandas import ExcelWriter

    all_subject_dfs = {}  # New: hold subject-wise dataframes for Excel sheets

    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(('.edf', '.fif')):
            continue

        sub = extract_subject_id(fname)
        if not sub:
            continue

        soz_list = subject_soz_map.get(sub, [])
        if not soz_list:
            continue

        fp      = os.path.join(data_dir, fname)
        results = process_file(fp, soz_list, seconds_to_analyze)
        base    = os.path.splitext(fname)[0]

        # split EZ vs Non‑EZ
        ez_hqs  = [hq for label, hq in results if label=='EZ']
        non_hqs = [hq for label, hq in results if label=='Non-EZ']
        EZ      = np.vstack(ez_hqs)
        NON_EZ  = np.vstack(non_hqs)

        # compute mean curves
        mean_ez   = EZ.mean(axis=0)
        mean_non  = NON_EZ.mean(axis=0)

        # save the mean-H(q) CSV
        df_mean = pd.DataFrame(
            [mean_non, mean_ez],
            index=['Non-EZ','EZ'],
            columns=[f"{q:.1f}" for q in q_values]
        )
        df_mean.index.name = 'Group'
        df_mean.to_csv(os.path.join(data_dir, f"{base}_mean_hurst.csv"))
        print(f"→ Saved mean-H(q) to {base}_mean_hurst.csv")

        # === PLOT ALL CHANNELS INDIVIDUALLY ===
        plt.figure(figsize=(10, 6))

        # Plot Non-EZ channels in blue
        for hq in non_hqs:
            plt.plot(q_values, hq, color='blue', alpha=0.3, linewidth=1.0)

        # Plot EZ channels in red
        for hq in ez_hqs:
            plt.plot(q_values, hq, color='red', alpha=0.6, linewidth=1.5)

        # Labels and Title
        plt.xlabel('q‑order')
        plt.ylabel('H(q)')
        plt.title(f'MFDFA H(q) Curves — {sub}')
        plt.grid('--', lw=0.5)
        plt.tight_layout()

        # Save Plot
        plt.savefig(os.path.join(data_dir, f"{base}_all_channels_hurst.png"), dpi=300)
        plt.close()

        # === NEW: Store results for this subject in a sheet ===
        chs = mne.io.read_raw_fif(fp, verbose=False).info['ch_names'] if fp.endswith('.fif') \
              else mne.io.read_raw_edf(fp, verbose=False).info['ch_names']
        subject_rows = []
        for (label, hq), ch in zip(results, chs):
            clean_ch = clean_channel_name(ch)
            row = {
                'Subject_ID': sub,
                'Channel': ch,
                'is_soz': 1 if fuzzy_match(clean_ch, soz_list) else 0,
                **{f"{q:.1f}": val for q, val in zip(q_values, hq)}
            }
            subject_rows.append(row)

        df_subject = pd.DataFrame(subject_rows)
        all_subject_dfs[sub] = df_subject

    # === NEW: Write all subject DataFrames into one Excel with multiple sheets ===
    if all_subject_dfs:
        excel_path = os.path.join(data_dir, "all_subjects_q_values_by_sheet.xlsx")
        with ExcelWriter(excel_path, engine='openpyxl') as writer:
            for sub_id, df in all_subject_dfs.items():
                # Truncate long sheet names to 31 chars to avoid Excel limits
                sheet_name = sub_id[:31]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"→ Saved all H(q) values to {excel_path} (split by subject)")




if __name__ == "__main__":
    main()
