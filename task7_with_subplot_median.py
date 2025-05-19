import os
import re
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from scipy.stats import ks_2samp  # though no KS used, kept as per original code

# === PARAMETERS ===
data_dir = r"C:\Engel 1 Subjects\allfiles"
soz_csv = os.path.join(data_dir, "SOZ_Channels_info.csv")
seconds_to_analyze = None  # or set a float
q_values = np.linspace(-20.0, 20.0, 401)
scale_min, scale_max = 16, 4096

# === LOAD SOZ INFO AND CLEAN ===
soz_df = pd.read_csv(soz_csv)
# pivot so that each row is one (Subject, Channel)
soz_long = (
    soz_df
    .melt(id_vars="Subject", var_name="Channel_Index", value_name="Channel")
    .dropna(subset=["Channel"])
)
# clean up both subject IDs and channel names to alphanumeric uppercase
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

        if len(ez_hqs) == 0 or len(non_hqs) == 0:
            print(f"Skipping {sub} — insufficient EZ/Non-EZ data.")
            continue

        EZ      = np.vstack(ez_hqs)
        NON_EZ  = np.vstack(non_hqs)

        # compute mean curves (still saving CSV)
        mean_ez   = EZ.mean(axis=0)
        mean_non  = NON_EZ.mean(axis=0)
        df_mean = pd.DataFrame(
            [mean_non, mean_ez],
            index=['Non-EZ','EZ'],
            columns=[f"{q:.1f}" for q in q_values]
        )
        df_mean.index.name = 'Group'
        df_mean.to_csv(os.path.join(data_dir, f"{base}_mean_hurst.csv"))
        print(f"→ Saved mean-H(q) to {base}_mean_hurst.csv")

        # === SUBPLOT VERSION WITH MEDIAN OVERLAY ===
        fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

        # Top: EZ Channels (red)
        for hq in ez_hqs:
            axs[0].plot(q_values, hq, color='red', alpha=0.3, linewidth=1.0)
        median_ez = np.median(EZ, axis=0)
        axs[0].plot(q_values, median_ez, color='red', linewidth=3.0, label='EZ Median')
        axs[0].set_ylabel('H(q)')
        axs[0].set_title(f'EZ Channels — {sub}')
        axs[0].grid('--', lw=0.5)
        axs[0].legend()

        # Bottom: Non-EZ Channels (blue)
        for hq in non_hqs:
            axs[1].plot(q_values, hq, color='blue', alpha=0.2, linewidth=1.0)
        median_non = np.median(NON_EZ, axis=0)
        axs[1].plot(q_values, median_non, color='blue', linewidth=3.0, label='Non-EZ Median')
        axs[1].set_xlabel('q‑order')
        axs[1].set_ylabel('H(q)')
        axs[1].set_title(f'Non-EZ Channels — {sub}')
        axs[1].grid('--', lw=0.5)
        axs[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f"{base}_all_channels_subplot_median.png"), dpi=1200)
        plt.close()
        print(f"→ Saved subplot H(q) plot to {base}_all_channels_subplot_median.png")

if __name__ == "__main__":
    main()
