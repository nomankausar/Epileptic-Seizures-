import os
import re
import numpy as np
import pandas as pd
import mne
from fracModel import fracOrdUU

# === PARAMETERS ===
data_dir = r"C:\Reaserch Works\one subject"
soz_csv = os.path.join(data_dir, "SOZ_Channels_info.csv")
segment_duration = 100  # seconds
output_csv = os.path.join(data_dir, "eigen_with_soz_all_subjects_mag_complex_angle.csv")

# === Load SOZ Channel Mapping ===
soz_df = pd.read_csv(soz_csv)
soz_long = (
    soz_df.melt(id_vars="Subject", var_name="Channel_Index", value_name="Channel")
    .dropna(subset=["Channel"])
)
soz_long["Subject_ID"] = soz_long["Subject"].astype(str).apply(lambda s: re.sub(r'[^A-Za-z0-9]', '', s).upper())
soz_long["Clean_Channel"] = soz_long["Channel"].astype(str).apply(lambda s: re.sub(r'[^A-Za-z0-9]', '', s).upper())
subject_soz_map = soz_long.groupby("Subject_ID")["Clean_Channel"].apply(list).to_dict()

# === Helpers ===
def extract_subject_id(fname):
    m = re.search(r"sub-([A-Za-z0-9]+)_", fname)
    return m.group(1).upper() if m else None

def clean_channel_name(name):
    return re.sub(r'[^A-Za-z0-9]', '', str(name)).upper()

# === MAIN LOOP ===
all_data = []

for fname in os.listdir(data_dir):
    if not fname.lower().endswith(('.edf', '.fif')):
        continue

    sub = extract_subject_id(fname)
    if not sub or sub not in subject_soz_map:
        print(f"⚠️ Skipping {fname} (no SOZ match or invalid ID)")
        continue

    file_path = os.path.join(data_dir, fname)
    ext = fname.split('.')[-1].lower()
    reader = mne.io.read_raw_edf if ext == 'edf' else mne.io.read_raw_fif

    try:
        print(f"\n📥 Loading {fname}")
        raw = reader(file_path, preload=True, verbose=False)
        raw.notch_filter(60)
    except Exception as e:
        print(f"❌ Failed to read {fname}: {e}")
        continue

    ch_names = raw.info['ch_names']
    sfreq = raw.info['sfreq']
    total_duration = raw.times[-1]
    n_segments = int(total_duration // segment_duration)
    soz_list = subject_soz_map[sub]

    for segment_index in range(n_segments):
        tmin = segment_index * segment_duration
        tmax = tmin + segment_duration

        try:
            segment = raw.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)
            segment.load_data()
            X = segment.get_data()

            if X.shape[1] < X.shape[0]:
                print(f"⛔ Skipping segment {segment_index+1} of {sub} — too short")
                continue

            model = fracOrdUU(verbose=0)
            model.fit(X)

            A = model._AMat[-1]
            eigvals, eigvecs = np.linalg.eig(A)
            max_idx = np.argmax(np.abs(eigvals))
            max_eigval = eigvals[max_idx]
            max_eigvec = eigvecs[:, max_idx]

            for i, ch in enumerate(ch_names):
                clean_ch = clean_channel_name(ch)
                is_soz = int(any(clean_ch == soz for soz in soz_list))
                all_data.append({
                    'Subject_ID': sub,
                    'Segment': segment_index + 1,
                    'Channel': ch,
                    'is_soz': is_soz,
                    'Eigenvector_Magnitude': np.abs(max_eigvec[i]),
                    'Eigenvector_Real': np.real(max_eigvec[i]),
                    'Eigenvector_Imag': np.imag(max_eigvec[i]),
                    'Eigenvector_Phase': np.angle(max_eigvec[i]),
                    'Max_Eigenvalue': max_eigval
                })

            print(f"✅ {sub} Segment {segment_index+1}/{n_segments} done")

        except Exception as e:
            print(f"⚠️ Error in {sub} Segment {segment_index+1}: {e}")

# === Save to CSV ===
df_all = pd.DataFrame(all_data, columns=[
    'Subject_ID', 'Segment', 'Channel', 'is_soz',
    'Eigenvector_Magnitude', 'Eigenvector_Real', 'Eigenvector_Imag', 'Eigenvector_Phase',
    'Max_Eigenvalue'
])
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df_all.to_csv(output_csv, index=False)
print(f"\n✅ All subject results saved to:\n{output_csv}")
