import os
import mne
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from antropy import detrended_fluctuation

def process_channel(args):
    """Process a single channel using Antropy DFA."""
    channel_name, data, segment_duration = args
    dfa_values = []

    num_segments = len(data) // segment_duration
    for seg_idx in range(num_segments):
        segment_data = data[seg_idx * segment_duration:(seg_idx + 1) * segment_duration]
        if np.std(segment_data) == 0:
            print(f"Warning: Segment {seg_idx} in channel {channel_name} has zero standard deviation.")
            dfa_values.append(np.nan)
            continue
        dfa_values.append(detrended_fluctuation(segment_data))

    return channel_name, dfa_values

def process_subject(file_path, output_dir, segment_duration):
    """Process a subject's raw EEG file (.edf or .fif) using Antropy DFA."""
    if file_path.endswith(".fif"):
        raw = mne.io.read_raw_fif(file_path, preload=True)
    elif file_path.endswith(".edf"):
        raw = mne.io.read_raw_edf(file_path, preload=True)
    else:
        print(f"Skipping unsupported file format: {file_path}")
        return

    raw.notch_filter(freqs=60)  # Remove powerline noise
    data = raw.get_data()
    sampling_rate = int(raw.info['sfreq'])
    segment_duration *= sampling_rate  # Convert to samples

    args_list = [(raw.ch_names[i], data[i], segment_duration) for i in range(data.shape[0])]

    with Pool(cpu_count()) as pool:
        results = pool.map(process_channel, args_list)

    df = pd.DataFrame({channel: values for channel, values in results}).T
    df.reset_index(inplace=True)
    df.columns = ['Channel'] + [f'Segment_{i+1}' for i in range(df.shape[1] - 1)]

    output_path = os.path.join(output_dir, os.path.basename(file_path).replace('.fif', '_dfa_antropy.xlsx').replace('.edf', '_dfa_antropy.xlsx'))
    df.to_excel(output_path, index=False)
    print(f"Saved Antropy DFA results to {output_path}")

if __name__ == "__main__":
    data_folder = r"C:\\Engel 1 Subjects\\allfiles"
    output_folder = r"C:\\Engel 1 Subjects\\resul180sswithscalingw16_32_8192antropydfa"
    os.makedirs(output_folder, exist_ok=True)

    segment_duration = 180  # in seconds

    for file in os.listdir(data_folder):
        if file.endswith(".edf") or file.endswith(".fif"):
            process_subject(os.path.join(data_folder, file), output_folder, segment_duration)
