import numpy as np
import mne
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from scipy.stats import sem

# Define the directory containing the FIF files
fif_dir = r'C:\Engel 1 Subjects\100s cropped raw fif file'  # Modify this path as needed

# Define the selected q-values for fast computation
q_values = np.linspace(-5.00, 5.00, num=int((5.00 - (-5.00)) / 0.1) + 1)

# Define SOZ channels
soz_channels = ['POL LF1', 'POL LA1', 'POL LA2', 'POL LA3', 'POL LH1', 'POL LH2', 'POL LH3']

def MFDFA(signal, scale_min, scale_max, scale_res, q_values):
    signal = np.cumsum(signal - np.mean(signal))  # Cumulative sum to obtain profile
    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), num=scale_res).astype(int)
    fluctuation = np.zeros((len(scales), len(q_values)))
    hurst_exponents = np.zeros(len(q_values))
    regression_lines = []
    tq_values = np.zeros(len(q_values))
    dq_values = np.zeros(len(q_values) - 1)

    for s_idx, s in enumerate(scales):
        n_segments = len(signal) // s
        F_s = np.zeros(n_segments)

        for i in range(n_segments):
            segment = signal[i * s:(i + 1) * s]
            time = np.arange(s)
            poly_fit = np.polyfit(time, segment, 1)  # 1st-order polynomial detrending
            trend = np.polyval(poly_fit, time)
            F_s[i] = np.sqrt(np.mean((segment - trend) ** 2))  # Fixed: Removed division by s

        F_s = F_s[F_s > 1e-8]  # Avoid numerical instability

        for nq, q in enumerate(q_values):
            if q < 0:
                fluctuation[s_idx, nq] = np.exp(np.mean(np.log(F_s[F_s > 0])))  # Logarithmic averaging
            elif q == 0:
                fluctuation[s_idx, nq] = np.exp(0.5 * np.mean(np.log(F_s**2)))
            else:
                fluctuation[s_idx, nq] = np.mean(F_s**q) ** (1.0 / q)
    
    # Compute q-order Hurst exponent and mass exponent
    for nq in range(len(q_values)):
        log_scales = np.log2(scales)
        log_fluctuation = np.log2(np.clip(fluctuation[:, nq], 1e-8, None))  # Clipping for stability
        C = np.polyfit(log_scales, log_fluctuation, 1)
        hurst_exponents[nq] = C[0]
        regression_lines.append(np.polyval(C, log_scales))
        tq_values[nq] = hurst_exponents[nq] * q_values[nq] - 1  # Compute mass exponent t_q
    
    # Compute singularity dimension D_q
    dq_values = np.diff(tq_values) / np.diff(q_values)
    
    return scales, fluctuation, hurst_exponents, regression_lines, tq_values, dq_values

def process_channel(args):
    channel_data, ch, soz_channels = args
    scales, fluctuation, hurst_exponents, regression_lines, tq_values, dq_values = MFDFA(channel_data, scale_min=16, scale_max=8192, scale_res=30, q_values=q_values)
    return ('SOZ' if ch in soz_channels else 'Normal', ch, scales, fluctuation, hurst_exponents, regression_lines, tq_values, dq_values)

def process_file_sequentially(file_path):
    print(f"Processing file: {file_path}")
    raw = mne.io.read_raw_fif(file_path, preload=True)
    eeg_data, _ = raw.get_data(return_times=True)
    print("EEG Data Min:", np.min(eeg_data))
    print("EEG Data Max:", np.max(eeg_data))
    print("EEG Data Mean:", np.mean(eeg_data))
    print("EEG Data Std Dev:", np.std(eeg_data))
    channels = raw.info['ch_names']
    args = [(eeg_data[channels.index(ch)], ch, soz_channels) for ch in channels if ch in channels]
    with Pool(cpu_count()) as pool:
        results = pool.map(process_channel, args)
    return results

def main():
    fif_files = [os.path.join(fif_dir, f) for f in os.listdir(fif_dir) if f.endswith('.fif')]
    results = []
    for file_path in fif_files:
        results.extend(process_file_sequentially(file_path))
    
    scales = np.logspace(np.log10(16), np.log10(8192), num=30).astype(int)
    soz_hurst = np.mean([hurst for category, _, _, _, hurst, _, _, _ in results if category == 'SOZ'], axis=0)
    normal_hurst = np.mean([hurst for category, _, _, _, hurst, _, _, _ in results if category == 'Normal'], axis=0)
    soz_tq = np.mean([tq for category, _, _, _, _, _, tq, _ in results if category == 'SOZ'], axis=0)
    normal_tq = np.mean([tq for category, _, _, _, _, _, tq, _ in results if category == 'Normal'], axis=0)
    soz_dq = np.mean([dq for category, _, _, _, _, _, _, dq in results if category == 'SOZ'], axis=0)
    normal_dq = np.mean([dq for category, _, _, _, _, _, _, dq in results if category == 'Normal'], axis=0)
    
    # Plot q-order Hurst exponent
    plt.figure(figsize=(7, 5))
    plt.plot(q_values, soz_hurst, '-o', label='EZ', color='red')
    plt.plot(q_values, normal_hurst, '-s', label='Non-EZ', color='blue')
    plt.xlabel('q-order')
    plt.ylabel('q-order Hurst exponent H(q)')
    plt.title('q-order Hurst Exponent H(q)')
    plt.legend()
    plt.savefig("new generalized hurst exponent.png", dpi=1200)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.show()

    # # Plot Mass Exponent t_q
    # plt.figure(figsize=(7, 5))
    # plt.plot(q_values, soz_tq, '-o', label='SOZ', color='red')
    # plt.plot(q_values, normal_tq, '-s', label='Normal', color='blue')
    # plt.xlabel('q-order')
    # plt.ylabel('Mass Exponent t_q')
    # plt.title('q-order Mass Exponent t_q')
    # plt.legend()
    # plt.grid(True, linestyle='--', linewidth=0.5)
    # plt.show()

if __name__ == "__main__":
    main()
