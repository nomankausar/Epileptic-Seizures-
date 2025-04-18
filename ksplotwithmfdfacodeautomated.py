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
q_values = np.linspace(-5.0, 5.0, 101)
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
def MFDFA(signal, scale_min, scale_max, _ignored, q_values):
    # 1. integrate the zero‑mean signal
    x = np.cumsum(signal - np.mean(signal))

    # 2. define scales
    scales = 2 ** np.arange(
        int(np.log2(scale_min)),
        int(np.log2(scale_max)) + 1
    )

    # 3. pre‑allocate storage
    fluctuation      = np.zeros((len(scales), len(q_values)))
    hurst_exponents  = np.zeros(len(q_values))
    regression_lines = []
    tq_values        = np.zeros(len(q_values))

    # 4. compute F(s) for each segment and each scale
    for s_idx, s in enumerate(scales):
        n_segments = len(x) // s
        F_s = []  # collect per‑segment RMS

        for i in range(n_segments):
            segment = x[i*s:(i+1)*s]
            time    = np.arange(s)
            coeffs  = np.polyfit(time, segment, 1)
            trend   = np.polyval(coeffs, time)
            F_s.append(np.sqrt(np.mean((segment - trend)**2)))

        F_s = np.array(F_s)
        F_s = F_s[F_s > 1e-8]  # drop degenerate segments

        # 5. fluctuation function for each q
        for qi, q in enumerate(q_values):
            if q < 0:
                fluctuation[s_idx, qi] = np.exp(np.mean(np.log(F_s)))
            elif q == 0:
                fluctuation[s_idx, qi] = np.exp(0.5 * np.mean(np.log(F_s**2)))
            else:
                fluctuation[s_idx, qi] = np.mean(F_s**q)**(1.0/q)

    # 6. fit log‑log to get H(q), regression lines, and t(q)
    log_scales = np.log2(scales)
    for qi, q in enumerate(q_values):
        log_F = np.log2(np.clip(fluctuation[:, qi], 1e-8, None))
        C     = np.polyfit(log_scales, log_F, 1)
        Hq    = C[0]
        hurst_exponents[qi] = Hq
        regression_lines.append(np.polyval(C, log_scales))
        tq_values[qi] = Hq * q - 1

    # 7. compute multifractal spectrum derivative d(q)
    dq_values = np.diff(tq_values) / np.diff(q_values)

    return scales, fluctuation, hurst_exponents, regression_lines, tq_values, dq_values


# === CHANNEL‐LEVEL WORKER ===
def process_channel(args):
    data, ch_name, soz_list = args
    clean_nm = clean_channel_name(ch_name)
    label = 'SOZ' if fuzzy_match(clean_nm, soz_list) else 'Normal'
    return (label, ch_name, *MFDFA(data, scale_min, scale_max, None, q_values))

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

        # split SOZ vs Normal
        soz_hq  = np.array([r[4] for r in results if r[0]=='SOZ'])
        norm_hq = np.array([r[4] for r in results if r[0]=='Normal'])
        soz_tq  = np.array([r[5] for r in results if r[0]=='SOZ'])
        norm_tq = np.array([r[5] for r in results if r[0]=='Normal'])

        mean_soz_h = soz_hq.mean(0)
        mean_n_h  = norm_hq.mean(0)
        mean_soz_t = soz_tq.mean(0)
        mean_n_t   = norm_tq.mean(0)

        base = os.path.splitext(fname)[0]
        # Hurst plot
        plt.figure(figsize=(7,5))
        plt.plot(q_values, mean_soz_h, 'o-', label='EZ')
        plt.plot(q_values, mean_n_h,  's-', label='Non‑EZ')
        plt.xlabel('q‑order'); plt.ylabel('H(q)')
        plt.title(f'Hurst H(q) — {sub}')
        plt.legend(); plt.grid('--', lw=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f"{base}_hurst.png"), dpi=300)
        plt.close()

        # # Mass exponent plot
        # plt.figure(figsize=(7,5))
        # plt.plot(q_values, mean_soz_t, 'o-', label='EZ')
        # plt.plot(q_values, mean_n_t,  's-', label='Non‑EZ')
        # plt.xlabel('q‑order'); plt.ylabel('t(q)')
        # plt.title(f'Mass Exponent t(q) — {sub}')
        # plt.legend(); plt.grid('--', lw=0.5)
        # plt.tight_layout()
        # plt.savefig(os.path.join(data_dir, f"{base}_tq.png"), dpi=300)
        # plt.close()

        # KS tests per q
        ks_s, ks_p = [], []
        for i in range(len(q_values)):
            s_vals = soz_hq[:,i]
            n_vals = norm_hq[:,i]
            st, pv = ks_2samp(s_vals, n_vals)
            ks_s.append(st); ks_p.append(pv)

        plt.figure(figsize=(12,5))
        plt.subplot(1,2,1)
        plt.plot(q_values, ks_s, label='KS stat'); plt.grid('--', lw=0.5)
        plt.title(f'KS Statistic — {sub}'); plt.xlabel('q'); plt.ylabel('D')

        plt.subplot(1,2,2)
        plt.plot(q_values, ks_p, label='p‑value'); 
        plt.axhline(0.05, ls='--', color='r', label='α=0.05')
        plt.title(f'p‑value — {sub}'); plt.xlabel('q'); plt.ylabel('p')
        plt.legend(); plt.grid('--', lw=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f"{base}_ks.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
