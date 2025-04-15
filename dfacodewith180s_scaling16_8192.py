import os
import mne
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

def DFA(signal, scales, q):
    """Perform DFA on the signal."""
    signal = np.cumsum(signal - np.mean(signal))
    fluctuation = []
    for s in scales:
        n_segments = len(signal) // s
        F_s = np.zeros(n_segments)
        for i in range(n_segments):
            segment = signal[i * s:(i + 1) * s]
            time = np.arange(s)
            poly_fit = np.polyfit(time, segment, 1)
            trend = np.polyval(poly_fit, time)
            F_s[i] = np.sqrt(np.mean((segment - trend) ** 2))
        fluctuation.append(np.mean(F_s ** q))
    fluctuation = np.array(fluctuation) ** (1.0 / q)
    log_scales = np.log(scales)
    log_fluctuation = np.log(fluctuation)
    poly_fit = np.polyfit(log_scales, log_fluctuation, 1)
    return poly_fit[0]

def process_channel(args):
    """Process a single channel."""
    channel_name, data, scales, segment_duration = args
    dfa_values = []
    
    num_segments = len(data) // segment_duration
    for seg_idx in range(num_segments):
        segment_data = data[seg_idx * segment_duration:(seg_idx + 1) * segment_duration]
        if np.std(segment_data) == 0:
            print(f"Warning: Segment {seg_idx} in channel {channel_name} has zero standard deviation.")
        dfa_values.append(DFA(segment_data, scales, 2))
    
    return channel_name, dfa_values

def process_subject(file_path, output_dir, scales, segment_duration):
    """Process a subject's raw data file (EDF or FIF)."""
    if file_path.endswith(".fif"):
        raw = mne.io.read_raw_fif(file_path, preload=True)
    elif file_path.endswith(".edf"):
        raw = mne.io.read_raw_edf(file_path, preload=True)
    else:
        print(f"Skipping unsupported file format: {file_path}")
        return
    # Apply a notch filter at 60 Hz to remove power line noise
    raw.notch_filter(freqs=60)
    data = raw.get_data()
    sampling_rate = int(raw.info['sfreq'])
    segment_duration *= sampling_rate

    args_list = [(raw.ch_names[i], data[i], scales, segment_duration) for i in range(data.shape[0])]
    
    with Pool(cpu_count()) as pool:
        results = pool.map(process_channel, args_list)
    
    df = pd.DataFrame({channel: values for channel, values in results}).T
    df.reset_index(inplace=True)
    df.columns = ['Channel'] + [f'Segment_{i+1}' for i in range(df.shape[1] - 1)]
    
    output_path = os.path.join(output_dir, os.path.basename(file_path).replace('.fif', '_dfa.xlsx').replace('.edf', '_dfa.xlsx'))
    df.to_excel(output_path, index=False)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    data_folder = r"C:\\Engel 1 Subjects\\allfiles"
    output_folder = r"C:\\Engel 1 Subjects\\resul1msswithscalingw16_32_8192"
    os.makedirs(output_folder, exist_ok=True)

    scales = np.array([16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192])  # Updated DFA scales
    segment_duration =180  # Segment duration in seconds

    for file in os.listdir(data_folder):
        if file.endswith(".fif") or file.endswith(".edf"):
            process_subject(os.path.join(data_folder, file), output_folder, scales, segment_duration)
