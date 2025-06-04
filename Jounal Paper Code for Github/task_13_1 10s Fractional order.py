import os
import re
import numpy as np
import pandas as pd
import mne
from scipy.io import loadmat
from fracModel import fracOrdUU

# === PARAMETERS ===
data_dir           = r"C:\Fractional Order Research\All 24 Subjects Data Files"
soz_csv            = os.path.join(data_dir, "SOZ_Channels_info.csv")
segment_duration   = 10  # in seconds
output_csv         = os.path.join(data_dir, "frac_order_features.csv")

# === LOAD SOZ MAPPING ===
soz_df = pd.read_csv(soz_csv)
soz_long = (
    soz_df.melt(id_vars="Subject", var_name="Channel_Index", value_name="Channel")
    .dropna(subset=["Channel"])
)
soz_long["Subject_ID"] = soz_long["Subject"].astype(str).apply(lambda s: re.sub(r'[^A-Za-z0-9]', '', s).upper())
soz_long["Clean_Channel"] = soz_long["Channel"].astype(str).apply(lambda s: re.sub(r'[^A-Za-z0-9]', '', s).upper())
subject_soz_map = soz_long.groupby("Subject_ID")["Clean_Channel"].apply(list).to_dict()

def extract_subject_id(fname):
    m = re.search(r"sub-([A-Za-z0-9]+)_", fname)
    return m.group(1).upper() if m else None

def clean_channel_name(name):
    return re.sub(r'[^A-Za-z0-9]', '', str(name)).upper()

# === MAIN PROCESS ===
all_features = []

for fname in os.listdir(data_dir):
    if not fname.lower().endswith(('.edf', '.fif')):
        continue

    sub = extract_subject_id(fname)
    if not sub or sub not in subject_soz_map:
        continue

    file_path = os.path.join(data_dir, fname)
    ext = fname.split('.')[-1].lower()
    reader = mne.io.read_raw_edf if ext == 'edf' else mne.io.read_raw_fif

    try:
        raw = reader(file_path, preload=True, verbose=False)
        raw.notch_filter(60)
    except Exception as e:
        print(f" Failed to read {fname}: {e}")
        continue

    eeg_data, _ = raw.get_data(return_times=True)
    ch_names = raw.info['ch_names']
    sfreq = raw.info['sfreq']
    soz_list = subject_soz_map[sub]

    total_duration = raw.times[-1]
    n_segments = int(total_duration // segment_duration)

    for seg_idx in range(n_segments):
        tmin = seg_idx * segment_duration
        tmax = (seg_idx + 1) * segment_duration
        segment = raw.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)
        segment.load_data()
        X = segment.get_data()

        if X.shape[1] < X.shape[0]:
            continue  # not enough data

        try:
            model = fracOrdUU(verbose=0)
            model.fit(X)

            for ch_idx, ch_name in enumerate(ch_names):
                clean_ch = clean_channel_name(ch_name)
                is_soz = int(any(clean_ch in soz or soz in clean_ch for soz in soz_list))
                all_features.append({
                    'Subject_ID': sub,
                    'Segment': seg_idx + 1,
                    'Channel': ch_name,
                    'is_soz': is_soz,
                    'frac_order': model._order[ch_idx]
                })

        except Exception as e:
            print(f"⚠️ Error processing segment {seg_idx} of {fname}: {e}")

# === SAVE TO CSV ===
df_out = pd.DataFrame(all_features)
df_out.to_csv(output_csv, index=False)
print(f"✅ Saved fractional order features to: {output_csv}")
