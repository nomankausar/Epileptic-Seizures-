import os
import re
import numpy as np
import pandas as pd
import mne
from multiprocessing import Pool, cpu_count

# === PARAMETERS ===
data_dir = r"C:\Engel 1 Subjects\allfiles"
soz_csv = os.path.join(data_dir, "SOZ_Channels_info.csv")
seconds_to_analyze = None  # set to a float if needed
q_values = np.linspace(-20.0, 20.0, 401)
scale_min, scale_max = 16, 4096

# === LOAD SOZ CHANNEL INFO ===
soz_df = pd.read_csv(soz_csv)
soz_long = (
    soz_df.melt(id_vars="Subject", var_name="Channel_Index", value_name="Channel")
          .dropna(subset=["Channel"])
)
soz_long["Subject_ID"] = soz_long["Subject"].astype(str)\
    .apply(lambda s: re.sub(r'[^A-Za-z0-9]', '', s).upper())
soz_long["Clean_Channel"] = soz_long["Channel"].astype(str)\
    .apply(lambda s: re.sub(r'[^A-Za-z0-9]', '', s).upper())
subject_soz_map = (
    soz_long.groupby("Subject_ID")["Clean_Channel"]
            .apply(list)
            .to_dict()
)

# === HELPER FUNCTIONS ===
def extract_subject_id(filename):
    m = re.search(r"sub-([A-Za-z0-9]+)_", filename)
    return m.group(1).upper() if m else None

def clean_channel_name(name):
    return re.sub(r'[^A-Za-z0-9]', '', str(name)).upper()

def fuzzy_match(ch, soz_list):
    for soz in soz_list:
        if ch in soz or soz in ch:
            return True
    return False

def MFDFA(signal, scale_min, scale_max, _ignored, q_values):
    x = np.cumsum(signal - np.mean(signal))
    scales = 2 ** np.arange(int(np.log2(scale_min)), int(np.log2(scale_max)) + 1)
    fluctuation = np.zeros((len(scales), len(q_values)))
    hurst_exponents = np.zeros(len(q_values))

    for s_idx, s in enumerate(scales):
        n_segments = len(x) // s
        F_s = []
        for i in range(n_segments):
            segment = x[i*s:(i+1)*s]
            time = np.arange(s)
            coeffs = np.polyfit(time, segment, 1)
            trend = np.polyval(coeffs, time)
            F_s.append(np.sqrt(np.mean((segment - trend)**2)))
        F_s = np.array(F_s)
        F_s = F_s[F_s > 1e-8]  # Avoid log(0) issues

        for qi, q in enumerate(q_values):
            if q < 0:
                fluctuation[s_idx, qi] = np.exp(np.mean(np.log(F_s)))
            elif q == 0:
                fluctuation[s_idx, qi] = np.exp(0.5 * np.mean(np.log(F_s**2)))
            else:
                fluctuation[s_idx, qi] = np.mean(F_s**q)**(1.0/q)

    log_scales = np.log2(scales)
    for qi, q in enumerate(q_values):
        log_F = np.log2(np.clip(fluctuation[:, qi], 1e-8, None))
        hurst_exponents[qi] = np.polyfit(log_scales, log_F, 1)[0]

    return hurst_exponents

def process_channel(args):
    data, ch_name, soz_list = args
    clean_nm = clean_channel_name(ch_name)
    label = 'SOZ' if fuzzy_match(clean_nm, soz_list) else 'Normal'
    hq = MFDFA(data, scale_min, scale_max, None, q_values)
    return label, ch_name, hq

def process_file(filepath, soz_list, max_dur=None):
    print(f"→ Processing {os.path.basename(filepath)}")
    ext = filepath.lower().split('.')[-1]
    raw = (mne.io.read_raw_edf if ext == 'edf' else mne.io.read_raw_fif)(
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

# === MAIN FUNCTION ===
def main():
    all_rows = []

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

        filepath = os.path.join(data_dir, fname)
        results = process_file(filepath, soz_list, seconds_to_analyze)

        idx_q20 = np.where(np.isclose(q_values, 20.0))[0][0]

        for label, ch_name, hq in results:
            hq_q20 = hq[idx_q20]
            all_rows.append({
                'Subject': sub,
                'Channel': ch_name,
                'H(q=20)': hq_q20,
                'SOZ_Label': 1 if label == 'SOZ' else 0
            })

    if all_rows:
        df_all = pd.DataFrame(all_rows)
        out_file = os.path.join(data_dir, "all_subjects_q20.xlsx")
        df_all.to_excel(out_file, index=False)
        print(f"\n✅ All subject data saved to: {out_file}")
    else:
        print("\n⚠ No EEG data was processed.")

if __name__ == "__main__":
    main()
