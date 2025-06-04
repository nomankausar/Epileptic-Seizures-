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
q_values           = np.linspace(-40.0, 40.0, 801)
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
                            int(np.log2(scale_max)) + 1)
    # instantiation variables
    fluct = np.zeros((len(scales), len(q_vals)))
    Hq = np.zeros(len(q_vals))

    # find fluctuations across all scales
    for si, s in enumerate(scales):
        segs = len(x) // s  # length of the segment
        F_s = []  # fluctuations
        for i in range(segs):  # iterate over all segments
            seg = x[i*s:(i+1)*s]  # extract each data segment
            t = np.arange(s)  # array from 0 to s-1
            cfs = np.polyfit(t, seg, 1)
            trend = np.polyval(cfs, t)
            fluctuation = np.sqrt(np.mean((seg - trend)**2))
            if fluctuation > 1e-8:
                F_s.append(fluctuation)

        F_s = np.array(F_s)
        if F_s.size == 0:
            continue

        for qi, q in enumerate(q_vals):
            try:
                if q == 0:
                    fluct[si, qi] = np.exp(0.5 * np.mean(np.log(F_s**2)))
                else:
                    fluct[si, qi] = np.mean(F_s**q)**(1.0 / q)
            except:
                fluct[si, qi] = np.nan

    log_sc = np.log2(scales)
    for qi in range(len(q_vals)):
        log_F = np.log2(np.clip(fluct[:, qi], 1e-10, None))
        Hq[qi] = np.polyfit(log_sc, log_F, 1)[0]

    return Hq

def process_channel(args):
    data, ch_name, soz_list = args
    label = 'EZ' if fuzzy_match(clean_channel_name(ch_name), soz_list) else 'Non-EZ'
    hq = MFDFA(data, scale_min, scale_max, q_values)
    return label, hq

def process_file(filepath, soz_list, max_dur=None):
    ext = filepath.lower().split('.')[-1]
    reader = mne.io.read_raw_edf if ext == 'edf' else mne.io.read_raw_fif
    raw = reader(filepath, preload=False, verbose=False)
    if max_dur:
        raw.crop(0, max_dur)
    raw.load_data()
    raw.notch_filter(60)
    eeg, _ = raw.get_data(return_times=True)
    chs = raw.info['ch_names']
    args = [(eeg[i], chs[i], soz_list) for i in range(len(chs))]
    with Pool(cpu_count()) as p:
        return p.map(process_channel, args)

def main():
    from openpyxl import Workbook
    from pandas import ExcelWriter

    segment_duration = 30  # seconds
    segment_output_dir = os.path.join(data_dir, "segments")
    os.makedirs(segment_output_dir, exist_ok=True)
    all_subject_dfs = {}

    for fname in os.listdir(data_dir):
        if not fname.lower().endswith(('.edf', '.fif')):
            continue

        sub = extract_subject_id(fname)
        if not sub:
            continue

        soz_list = subject_soz_map.get(sub, [])
        if not soz_list:
            continue

        # === STEP 1: Segment original file into 30s FIF files ===
        fp = os.path.join(data_dir, fname)
        if not os.path.exists(fp):
            print(f"Skipping missing file: {fp}")
            continue

        ext = fp.lower().split('.')[-1]
        reader = mne.io.read_raw_edf if ext == 'edf' else mne.io.read_raw_fif
        raw = reader(fp, preload=True, verbose=False)

        raw.notch_filter(60)
        sfreq = raw.info['sfreq']
        total_duration = raw.times[-1]
        n_segments = int(total_duration // segment_duration)

        seg_files = []

        for seg_idx in range(n_segments):
            tmin = seg_idx * segment_duration
            tmax = (seg_idx + 1) * segment_duration
            raw_seg = raw.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)

            # Use correct MNE file naming convention
            seg_fname = f"{os.path.splitext(fname)[0]}_seg_{seg_idx+1:03d}_ieeg.fif"
            seg_path = os.path.join(segment_output_dir, seg_fname)
            raw_seg.save(seg_path, overwrite=True)
            seg_files.append((seg_path, seg_idx + 1))

        print(f"→ Segmented {fname} into {n_segments} files.")

        # === STEP 2: Run MFDFA on each 30s segment ===
        subject_rows = []

        for seg_path, seg_num in seg_files:
            raw_seg = mne.io.read_raw_fif(seg_path, preload=True, verbose=False)
            raw_seg.load_data()
            eeg_data, _ = raw_seg.get_data(return_times=True)
            ch_names = raw_seg.info['ch_names']
            args = [(eeg_data[i], ch_names[i], soz_list) for i in range(len(ch_names))]

            with Pool(cpu_count()) as p:
                results = p.map(process_channel, args)

            for (label, hq), ch in zip(results, ch_names):
                clean_ch = clean_channel_name(ch)
                row = {
                    'Subject_ID': sub,
                    'Channel': ch,
                    'Segment': seg_num,
                    'is_soz': 1 if fuzzy_match(clean_ch, soz_list) else 0,
                    **{f"{q:.1f}": val for q, val in zip(q_values, hq)}
                }
                subject_rows.append(row)

        df_subject = pd.DataFrame(subject_rows)
        all_subject_dfs[sub] = df_subject

    # === STEP 3: Write Excel with all subjects split in sheets ===
    if all_subject_dfs:
        excel_path = os.path.join(data_dir, "all_subjects_q_values_by_segment.xlsx")
        with ExcelWriter(excel_path, engine='openpyxl') as writer:
            for sub_id, df in all_subject_dfs.items():
                df.to_excel(writer, sheet_name=sub_id[:31], index=False)
        print(f"→ Saved all segmented MFDFA H(q) values to {excel_path}")

        # === STEP 4: Plot all channels (SOZ = red, Non-SOZ = blue) ===
        for sub_id, df in all_subject_dfs.items():
            df = df.dropna(axis=1, how='any')  # remove NaNs if any
            q_cols = [col for col in df.columns if re.match(r'^-?\d+\.?\d*$', col)]
            q_vals = np.array([float(q) for q in q_cols])

            # Sort q_cols by numeric value
            sorted_indices = np.argsort(q_vals)
            q_vals = q_vals[sorted_indices]
            q_cols = [q_cols[i] for i in sorted_indices]

            plt.figure(figsize=(12, 6))

            for _, row in df.iterrows():
                hq = row[q_cols].values[sorted_indices]
                color = 'red' if row['is_soz'] == 1 else 'blue'
                plt.plot(q_vals, hq, color=color, alpha=0.4)

            plt.title(f"MFDFA H(q) Curves — {sub_id}", fontsize=14)
            plt.xlabel("q-order", fontsize=12)
            plt.ylabel("H(q)", fontsize=12)
            plt.grid(True)
            plt.tight_layout()

            # Save figure
            fig_dir = os.path.join(data_dir, "mfdfa_all_channels_combined")
            os.makedirs(fig_dir, exist_ok=True)
            fig_path = os.path.join(fig_dir, f"{sub_id}_all_channels_Hq.png")
            plt.savefig(fig_path, dpi=300)
            plt.close()

        print("→ Combined H(q) plots for all channels saved in 'mfdfa_all_channels_combined'")

if __name__ == "__main__":
    main()
