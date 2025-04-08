import os
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from fuzzywuzzy import process, fuzz

# === PARAMETERS ===
data_dir = r"C:\Engel 1 Subjects\allfiles"
soz_csv_path = r"C:\Engel 1 Subjects\Meta Data\SOZ_Channels_info.csv"
seconds_to_analyze = None  # Use full data
q_values = np.linspace(-5.0, 5.0, 101)

# === LOAD SOZ MAPPING ===
soz_mapping_df = pd.read_csv(soz_csv_path)
raw_names = soz_mapping_df.iloc[:, 0].tolist()
canonical_names = soz_mapping_df.iloc[:, 1].tolist()

fuzzy_soz_map = {}
for raw_name in raw_names:
    match, score = process.extractOne(raw_name, canonical_names, scorer=fuzz.ratio)
    if score > 80:
        fuzzy_soz_map[raw_name] = match

# === MFDFA ===
def MFDFA(signal, scale_min, scale_max, scale_res, q_values):
    signal = np.cumsum(signal - np.mean(signal))
    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), num=scale_res).astype(int)
    fluctuation = np.zeros((len(scales), len(q_values)))
    hurst_exponents = np.zeros(len(q_values))
    regression_lines = []
    tq_values = np.zeros(len(q_values))

    for s_idx, s in enumerate(scales):
        n_segments = len(signal) // s
        F_s = np.zeros(n_segments)
        for i in range(n_segments):
            segment = signal[i * s:(i + 1) * s]
            time = np.arange(s)
            poly_fit = np.polyfit(time, segment, 1)
            trend = np.polyval(poly_fit, time)
            F_s[i] = np.sqrt(np.mean((segment - trend) ** 2))
        F_s = F_s[F_s > 1e-8]
        for nq, q in enumerate(q_values):
            if q < 0:
                fluctuation[s_idx, nq] = np.exp(np.mean(np.log(F_s[F_s > 0])))
            elif q == 0:
                fluctuation[s_idx, nq] = np.exp(0.5 * np.mean(np.log(F_s**2)))
            else:
                fluctuation[s_idx, nq] = np.mean(F_s**q) ** (1.0 / q)

    for nq in range(len(q_values)):
        log_scales = np.log2(scales)
        log_fluctuation = np.log2(np.clip(fluctuation[:, nq], 1e-8, None))
        C = np.polyfit(log_scales, log_fluctuation, 1)
        hurst_exponents[nq] = C[0]
        regression_lines.append(np.polyval(C, log_scales))
        tq_values[nq] = hurst_exponents[nq] * q_values[nq] - 1

    dq_values = np.diff(tq_values) / np.diff(q_values)
    return scales, fluctuation, hurst_exponents, regression_lines, tq_values, dq_values

# === CHANNEL PROCESSING ===
def process_channel(args):
    channel_data, ch_name, soz_canonical = args
    is_soz = 'SOZ' if ch_name in soz_canonical else 'Normal'
    return (is_soz, ch_name, *MFDFA(channel_data, 16, 8192, 30, q_values))

# === FILE PROCESSING ===
def process_file(filepath, soz_canonical, max_duration=None):
    print(f"Processing: {os.path.basename(filepath)}")
    if filepath.endswith('.edf'):
        raw = mne.io.read_raw_edf(filepath, preload=False, verbose=False)
    else:
        raw = mne.io.read_raw_fif(filepath, preload=False, verbose=False)

    if max_duration is not None:
        raw.crop(tmin=0, tmax=max_duration)
    raw.load_data()
    raw.notch_filter(freqs=60)

    eeg_data, _ = raw.get_data(return_times=True)
    ch_names = raw.info['ch_names']
    args = [(eeg_data[i], ch, soz_canonical) for i, ch in enumerate(ch_names)]

    with Pool(cpu_count()) as pool:
        return pool.map(process_channel, args)

# === MAIN LOOP ===
def main():
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.edf', '.fif'))]

    for file in all_files:
        subject_id = os.path.splitext(os.path.basename(file))[0]
        results = process_file(file, canonical_names, seconds_to_analyze)

        # Extract values by index: 0=label, 4=hurst, 6=tq
        soz_hurst = np.mean([r[4] for r in results if r[0] == 'SOZ'], axis=0)
        norm_hurst = np.mean([r[4] for r in results if r[0] == 'Normal'], axis=0)
        soz_tq = np.mean([r[6] for r in results if r[0] == 'SOZ'], axis=0)
        norm_tq = np.mean([r[6] for r in results if r[0] == 'Normal'], axis=0)
        
        # Plot Hurst Exponent
        plt.figure(figsize=(7, 5))
        plt.plot(q_values, soz_hurst, '-o', label='EZ', color='red')
        plt.plot(q_values, norm_hurst, '-s', label='Non-EZ', color='blue')
        plt.xlabel('q-order')
        plt.ylabel('H(q)')
        plt.title(f'Hurst Exponent H(q): {subject_id}')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f"{subject_id}_hurst.png"), dpi=300)
        plt.close()

        # Plot Mass Exponent t(q)
        plt.figure(figsize=(7, 5))
        plt.plot(q_values, soz_tq, '-o', label='EZ', color='red')
        plt.plot(q_values, norm_tq, '-s', label='Non-EZ', color='blue')
        plt.xlabel('q-order')
        plt.ylabel('Mass Exponent t(q)')
        plt.title(f'Mass Exponent t(q): {subject_id}')
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(data_dir, f"{subject_id}_tq.png"), dpi=300)
        plt.close()

if __name__ == '__main__':
    main()
