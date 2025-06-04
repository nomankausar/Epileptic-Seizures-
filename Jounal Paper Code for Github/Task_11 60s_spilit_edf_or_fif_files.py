import mne
import os

# === Load the original FIF file ===
file_path = r"C:\Engel 1 Subjects\allfiles\sub-PY18N013_ses-extraoperative_task-interictal_run-01_ieeg"
raw = mne.io.read_raw_fif(file_path, preload=True)

# === Get basic info ===
sfreq = raw.info['sfreq']              # Sampling frequency
total_duration = raw.times[-1]        # Total duration in seconds
segment_duration = 60                 # Segment duration in seconds
n_segments = int(total_duration // segment_duration)

# === Create output directory ===
output_dir = r"C:\Engel 1 Subjects\segments_fif"
os.makedirs(output_dir, exist_ok=True)

# === Segment and save ===
segment_files = []
for i in range(n_segments):
    tmin = i * segment_duration
    tmax = (i + 1) * segment_duration
    raw_segment = raw.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)
    
    segment_file_path = os.path.join(output_dir, f"segment_{i+1:03d}.fif")
    raw_segment.save(segment_file_path, overwrite=True)
    segment_files.append(segment_file_path)

# === Print preview of saved files ===
print("First 5 segment files:")
for path in segment_files[:5]:
    print(path)
