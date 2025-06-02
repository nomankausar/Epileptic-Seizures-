import os
import re
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

# === PARAMETERS ===
data_dir = r"C:\Engel 1 Subjects\allfiles"
soz_csv = os.path.join(data_dir, "SOZ_Channels_info.csv")
seconds_to_analyze = None
q_values = np.linspace(-20.0, 20.0, 401)
scale_min, scale_max = 16, 4096

# === LOAD SOZ INFO ===
soz_df = pd.read_csv(soz_csv)
soz_long = (
    soz_df
    .melt(id_vars="Subject", var_name="Channel_Index", value_name="Channel")
    .dropna(subset=["Channel"])
)
soz_long["Subject_ID"] = soz_long["Subject"].astype(str).apply(lambda s: re.sub(r'[^A-Za-z0-9]', '', s).upper())
soz_long["Clean_Channel"] = soz_long["Channel"].astype(str).apply(lambda s: re.sub(r'[^A-Za-z0-9]', '', s).upper())
subject_soz_map = soz_long.groupby("Subject_ID")["Clean_Channel"].apply(list).to_dict()

# === HELPERS ===
def extract_subject_id(filename):
    m = re.search(r"sub-([A-Za-z0-9]+)_", filename)
    return m.group(1).upper() if m else None

def clean_channel_name(name):
    return re.sub(r'[^A-Za-z0-9]', '', str(name)).upper()

def fuzzy_match(ch, soz_list):
    return any(ch in soz or soz in ch for soz in soz_list)

# === MFDFA CORE ===
def MFDFA(signal, scale_min, scale_max, q_vals):
    x = np.cumsum(signal - np.mean(signal))
    scales = 2 ** np.arange(int(np.log2(scale_min)), int(np.log2(scale_max)) + 1)
    fluct = np.zeros((len(scales), len(q_vals)))
    Hq = np.zeros(len(q_vals))

    for si, s in enumerate(scales):
        segs = len(x) // s
        F_s = []
        for i in range(segs):
            seg = x[i * s:(i + 1) * s]
            t = np.arange(s)
            trend = np.polyval(np.polyfit(t, seg, 1), t)
            F_s.append(np.sqrt(np.mean((seg - trend) ** 2)))
        F_s = np.array(F_s)
        F_s = F_s[F_s > 1e-8]
        for qi, q in enumerate(q_vals):
            if q == 0:
                fluct[si, qi] = np.exp(0.5 * np.mean(np.log(F_s ** 2)))
            else:
                fluct[si, qi] = np.mean(F_s ** q) ** (1.0 / q)

    log_sc = np.log2(scales)
    for qi in range(len(q_vals)):
        log_F = np.log2(fluct[:, qi])
        Hq[qi] = np.polyfit(log_sc, log_F, 1)[0]
    return Hq

# === WORKERS ===
def process_channel(args):
    data, ch_name, soz_list = args
    label = 'SOZ' if fuzzy_match(clean_channel_name(ch_name), soz_list) else 'Normal'
    return (label, ch_name, MFDFA(data, scale_min, scale_max, q_values))

def process_file(filepath, soz_list, max_dur=None):
    print(f"→ Processing {os.path.basename(filepath)}")
    ext = filepath.lower().split('.')[-1]
    raw = (mne.io.read_raw_edf if ext == 'edf' else mne.io.read_raw_fif)(filepath, preload=True, verbose=False)
    if max_dur:
        raw.crop(0, max_dur)
    raw.notch_filter(60)
    eeg = raw.get_data()
    chs = raw.info['ch_names']
    args = [(eeg[i], chs[i], soz_list) for i in range(len(chs))]
    with Pool(cpu_count()) as p:
        return p.map(process_channel, args)

# === MAIN ===
def main():
    for fname in os.listdir(data_dir):
        if not (fname.endswith('.edf') or fname.endswith('.fif')):
            continue
        sub = extract_subject_id(fname)
        if not sub or sub not in subject_soz_map:
            print(f"⚠ Skipping {fname}")
            continue
        soz_list = subject_soz_map[sub]
        fp = os.path.join(data_dir, fname)
        results = process_file(fp, soz_list, seconds_to_analyze)

        matched = [r[1] for r in results if r[0] == 'SOZ']
        clean_matched = [clean_channel_name(ch) for ch in matched]
        if set(clean_matched) >= set(soz_list):
            print(f"→ All SOZ channels matched for subject {sub}: {matched}")

        soz_hq = np.array([r[2] for r in results if r[0] == 'SOZ'])
        norm_hq = np.array([r[2] for r in results if r[0] == 'Normal'])
        if soz_hq.size == 0 or norm_hq.size == 0:
            print(f"⚠ Skipping {sub} due to missing SOZ or Normal data.")
            continue

        mean_soz = np.mean(soz_hq, axis=0).squeeze()
        mean_norm = np.mean(norm_hq, axis=0).squeeze()

        # === Save mean H(q)
        base = os.path.splitext(fname)[0]
        df_mean = pd.DataFrame(
            [mean_norm, mean_soz],
            index=['Non-EZ', 'EZ'],
            columns=[f"{q:.1f}" for q in q_values]
        )
        df_mean.index.name = 'Group'
        df_mean.to_csv(os.path.join(data_dir, f"{base}_mean_hurst.csv"))

        # === Plot
        plt.figure(figsize=(7, 5))
        plt.plot(q_values, mean_soz, 'o-', label='EZ', color='red')
        plt.plot(q_values, mean_norm, 's-', label='Non‑EZ', color='blue')
        plt.xlabel('q‑order'); plt.ylabel('H(q)')
        plt.title(f'Hurst H(q) — {sub}')
        plt.legend(); plt.grid('--', lw=0.5)

        # === Task 3: Euclidean Distance at q=20
        idx_q20 = np.where(np.isclose(q_values, 20.0))[0][0]
        hz_soz_q20 = mean_soz[idx_q20]
        hz_norm_q20 = mean_norm[idx_q20]
        euclidean_dist = np.abs(hz_soz_q20 - hz_norm_q20)
        print(f"→ Euclidean distance at q=20: {euclidean_dist:.5f}")

        # Annotate arrow and label
        x_pos = 19.5
        plt.annotate('', xy=(x_pos, hz_soz_q20), xytext=(x_pos, hz_norm_q20),
                     arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
        mid_y = (hz_soz_q20 + hz_norm_q20) / 2
        plt.text(x_pos + 0.5, mid_y, f'D={euclidean_dist:.4f}', va='center',
                 fontsize=11, weight='bold', color='black')

        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f"{base}_hurst.png"), dpi=300)
        plt.close()

if __name__ == "__main__":
    main()
