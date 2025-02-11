import os
import mne
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt

def preprocess_data(data):
    """Preprocess data by removing mean and normalizing."""
    data -= np.mean(data)
    std_dev = np.std(data)
    if std_dev == 0:
        # Handle constant segments by returning zeros
        return np.zeros_like(data)
    data /= std_dev
    return data

def MFDFA(signal, scale_min, scale_max, scale_res, q):
    """Perform MFDFA on the signal."""
    signal = np.cumsum(signal - np.mean(signal))
    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), num=scale_res).astype(int)
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
    channel_name, data, scale_params, segment_duration = args
    scale_min, scale_max, scale_res, q = scale_params
    mfdfa_values = []
    
    # Divide into segments
    num_segments = len(data) // segment_duration
    for seg_idx in range(num_segments):
        segment_data = data[seg_idx * segment_duration:(seg_idx + 1) * segment_duration]
        if np.std(segment_data) == 0:
            print(f"Warning: Segment {seg_idx} in channel {channel_name} has zero standard deviation.")
        preprocessed_data = preprocess_data(segment_data)
        mfdfa_values.append(MFDFA(preprocessed_data, scale_min, scale_max, scale_res, q))
    
    return channel_name, mfdfa_values

def process_subject(fif_file, output_dir, scale_params, segment_duration):
    """Process a subject's raw data file."""
    raw = mne.io.read_raw_fif(fif_file, preload=True)
    data = raw.get_data()
    sampling_rate = int(raw.info['sfreq'])
    segment_duration *= sampling_rate  # Convert duration from seconds to samples

    args_list = [
        (raw.ch_names[i], data[i], scale_params, segment_duration)
        for i in range(data.shape[0])
    ]

    with Pool(cpu_count()) as pool:
        results = pool.map(process_channel, args_list)

    # Save results to Excel
    df = pd.DataFrame({name: values for name, values in results})
    output_path = os.path.join(output_dir, os.path.basename(fif_file).replace('.fif', '_mfdfa.xlsx'))
    df.to_excel(output_path, index=False)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    fif_folder = r"C:\\Engel 1 Subjects\\100s cropped raw fif file"
    output_folder = r"C:\\Engel 1 Subjects\\results"
    os.makedirs(output_folder, exist_ok=True)

    scale_params = (16, 1024, 10, 2)  # MFDFA parameters
    segment_duration = 5  # Segment duration in seconds

    for fif_file in os.listdir(fif_folder):
        if fif_file.endswith(".fif"):
            process_subject(os.path.join(fif_folder, fif_file), output_folder, scale_params, segment_duration)
