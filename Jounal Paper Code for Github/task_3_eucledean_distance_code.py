import os
import re
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from scipy.stats import ks_2samp

# === PARAMETERS ===
data_dir = r"C:\Engel 1 Subjects\allfiles"
soz_csv = os.path.join(data_dir, "SOZ_Channels_info.csv")
seconds_to_analyze = None   # or set a float
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

# build a map: subject_id → list of clean SOZ channel names
subject_soz_map = (
    soz_long
    .groupby("Subject_ID")["Clean_Channel"]
    .apply(list)
    .to_dict()
)

# === HELPER FUNCTIONS ===
def extract_subject_id(filename):
    """Turn 'sub-PY18015_ses-01_task…edf' → 'PY18015'"""
    m = re.search(r"sub-([A-Za-z0-9]+)_", filename)
    return m.group(1).upper() if m else None

def clean_channel_name(name):
    return re.sub(r'[^A-Za-z0-9]', '', str(name)).upper()

def fuzzy_match(ch, soz_list):
    """
    Simple substring‐based fuzzy match:
    returns True if ch is contained in any SOZ entry or vice versa.
    """
    for soz in soz_list:
        if ch in soz or soz in ch:
            return True
    return False
# === MFDFA (fixed) ===
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

# === CHANNEL‐LEVEL WORKER ===
def process_channel(args):
    data, ch_name, soz_list = args
    clean_nm = clean_channel_name(ch_name)
    label = 'SOZ' if fuzzy_match(clean_nm, soz_list) else 'Normal'
    return (label, ch_name, *MFDFA(data, scale_min, scale_max,q_values))

# === FILE‐LEVEL PROCESSING ===
def process_file(filepath, soz_list, max_dur=None):
    print(f"→ Processing {os.path.basename(filepath)}")
    ext = filepath.lower().split('.')[-1]
    raw = (mne.io.read_raw_edf if ext=='edf' else mne.io.read_raw_fif)(
        filepath, preload=False, verbose=False
    )
    if max_dur:
        raw.crop(0, max_dur)
    raw.load_data()
    raw.notch_filter(60)
    eeg, _ = raw.get_data(return_times=True)
    chs = raw.info['ch_names']
    args = [(eeg[i], chs[i], soz_list) for i in range(len(chs))]
    with Pool(cpu_count()) as p:
        return p.map(process_channel, args)

# === MAIN LOOP ===
def main():
    for fname in os.listdir(data_dir):
        if not (fname.endswith('.edf') or fname.endswith('.fif')):
            continue
        sub = extract_subject_id(fname)
        if not sub:
            print(f"⚠ Could not parse subject from {fname}, skipping.")
            continue
        soz_list = subject_soz_map.get(sub, [])
        if not soz_list:
            print(f"⚠ No SOZ channels found for subject {sub}, skipping.")
            continue

        fp = os.path.join(data_dir, fname)
        results = process_file(fp, soz_list, seconds_to_analyze)
        # --- NEW FEATURE: if we’ve matched every SOZ in soz_list, print their names ---
        matched = [r[1] for r in results if r[0] == 'SOZ']                        # original ch names
        clean_matched = [clean_channel_name(ch) for ch in matched]               # cleaned names
        if set(clean_matched) >= set(soz_list):                                  # all GT SOZ channels found?
           print(f"→ All SOZ channels matched for subject {sub}: {matched}")


        # split SOZ vs Normal
        soz_hq  = np.array([r[2] for r in results if r[0]=='SOZ'])
        norm_hq = np.array([r[2] for r in results if r[0]=='Normal'])


         # split EZ vs Non‑EZ
        ez_hqs  = [hq for label, hq in results if label=='EZ']
        non_hqs = [hq for label, hq in results if label=='Non-EZ']
        EZ      = np.vstack(ez_hqs)
        NON_EZ  = np.vstack(non_hqs)

         # compute mean curves
        mean_ez   = EZ.mean(axis=0)
        mean_non  = NON_EZ.mean(axis=0)

        #  # save the mean-H(q) CSV
        # df_mean = pd.DataFrame(
        #     [mean_non, mean_ez],
        #     index=['Non-EZ','EZ'],
        #     columns=[f"{q:.1f}" for q in q_values]
        # )
        # df_mean.index.name = 'Group'
        # df_mean.to_csv(os.path.join(data_dir, f"{base}_mean_hurst.csv"))
        # print(f"→ Saved mean-H(q) to {base}_mean_hurst.csv")

         # Hurst plot 
        plt.figure(figsize=(7,5))
        plt.plot(q_values, mean_ez,  'r', label='EZ', linewidth=5.0)
        plt.plot(q_values, mean_non, 'b', label='Non‑EZ', linewidth=5.0)
        plt.xlabel('q‑order'); plt.ylabel('H(q)')
        plt.title(f'Hurst H(q) — {sub}')
        plt.legend(); plt.grid('--', lw=0.5)
        plt.tight_layout()
        # plt.savefig(os.path.join(data_dir, f"{base}_hurst.png"), dpi=300)
        plt.close()


        # === Task 3: Euclidean distance annotation at q = 20 ===
        q_target = 20.0
        idx_q20 = np.where(np.isclose(q_values, q_target))[0][0]
        hz_soz_q20 = mean_ez[idx_q20]
        hz_norm_q20 = mean_non[idx_q20]
        euclidean_dist = np.abs(hz_soz_q20 - hz_norm_q20)
        print(f"→ Euclidean distance at q={q_target}: {euclidean_dist:.5f}")

        # Annotate arrow and label
        x_pos = 20.0
        plt.annotate(
             '', xy=(x_pos, hz_soz_q20), xytext=(x_pos, hz_norm_q20),
        arrowprops=dict(arrowstyle='<->', color='black', lw=1.5)
        )
        mid_y = (hz_soz_q20 + hz_norm_q20) / 2
        plt.text(x_pos + 0.5, mid_y, f'D={euclidean_dist:.4f}', va='center', fontsize=11, weight='bold', color='black')

        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f"{base}_hurst.png"), dpi=300)
        plt.close()

        
        


if __name__ == "__main__":
    main()
