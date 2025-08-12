import numpy as np
import mne
import os
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count

plt.rcParams.update({
    'font.size': 18,       # Default text
    'axes.titlesize':18,  # Plot title
    'axes.labelsize': 18,  # X and Y labels
    'xtick.labelsize': 18, # X tick labels
    'ytick.labelsize': 18, # Y tick labels
    'legend.fontsize': 18  # Legend
})


# Define the directory containing the FIF files
fif_dir = r'D:\conference paper works replicate\Fluctutations vs Scales\Data Files'  # Modify this path as needed

# Define the selected q-values
q_values = np.array([-3, -1, 1, 3])

# Define SOZ channels
soz_channels = ['POL LF1', 'POL LA1', 'POL LA2', 'POL LA3', 'POL LH1', 'POL LH2', 'POL LH3']

def MFDFA(signal, scale_min, scale_max, scale_res, q_values):
    signal = np.cumsum(signal - np.mean(signal))  # Cumulative sum to obtain profile
    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), num=scale_res).astype(int)
    fluctuation = np.zeros((len(scales), len(q_values)))

    for s_idx, s in enumerate(scales):
        n_segments = len(signal) // s
        F_s = np.zeros(n_segments)

        for i in range(n_segments):
            segment = signal[i * s:(i + 1) * s]
            time = np.arange(s)
            poly_fit = np.polyfit(time, segment, 1)
            trend = np.polyval(poly_fit, time)
            F_s[i] = np.sqrt(np.mean((segment - trend) ** 2))  # Root Mean Square deviation

        F_s = F_s[F_s > 1e-8]  # Avoid numerical instability

        for nq, q in enumerate(q_values):
            if len(F_s) > 0:  # Ensure we have valid F_s values
                if q < 0:
                    fluctuation[s_idx, nq] = (np.mean(np.abs(F_s)**q)) ** (1.0 / q)
                elif q == 0:
                    fluctuation[s_idx, nq] = np.exp(0.5 * np.mean(np.log(F_s**2)))
                else:
                    fluctuation[s_idx, nq] = (np.mean(F_s**q)) ** (1.0 / q)
    
    return scales, fluctuation

def process_channel(args):
    channel_data, ch, soz_channels = args
    scales, fluctuation = MFDFA(channel_data, scale_min=16, scale_max=1024, scale_res=10, q_values=q_values)
    a=np.array(scales)
    print(a)
    b=np.array(fluctuation)
    print(b)
    return ('SOZ' if ch in soz_channels else 'Normal', ch, scales, fluctuation)

def process_file_sequentially(file_path):
    print(f"Processing file: {file_path}")
    raw = mne.io.read_raw_fif(file_path, preload=True)
    eeg_data, _ = raw.get_data(return_times=True)
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

    scales = np.logspace(np.log10(16), np.log10(1024), num=10).astype(int)

    # Define line styles for better readability
    line_styles = { 
        -3: 'solid', -2: 'dashed', -1: 'dotted', 
         1: 'dashdot', 2: (0, (3, 5, 1, 5)), 3: (0, (5, 10)) 
    }

    # Define colors for consistency
    colors = {
        'SOZ': 'red',
        'Normal': 'blue'
    }

    plt.figure(figsize=(12, 6))

    for q_idx, q in enumerate(q_values):
        # Ensure data is available before processing
        soz_values = [fluctuation[:, q_idx] for category, _, _, fluctuation in results if category == 'SOZ']
        normal_values = [fluctuation[:, q_idx] for category, _, _, fluctuation in results if category == 'Normal']

        if len(soz_values) > 0 and len(normal_values) > 0:
            soz_fluctuation = np.mean(np.array(soz_values), axis=0)
            normal_fluctuation = np.mean(np.array(normal_values), axis=0)

            # Plot SOZ
            plt.loglog(scales, soz_fluctuation, linestyle=line_styles[q], color=colors['SOZ'], 
                       label=f'EZ (q={q})', linewidth=2.5)

            # Plot Normal
            plt.loglog(scales, normal_fluctuation, linestyle=line_styles[q], color=colors['Normal'], 
                       label=f'Non-EZ (q={q})', linewidth=1.5)

            # Add inline text annotation at a midpoint for clarity
            mid_index = len(scales) // 2
            plt.text(scales[mid_index], soz_fluctuation[mid_index], f'q={q}', 
                     color='black', fontsize=8)
            plt.text(scales[mid_index], normal_fluctuation[mid_index], f'q={q}', 
                     color='black', fontsize=8)

    plt.xlabel('Scale (Segment Sample Sizes)')
    plt.ylabel('Fluctuation Functions')
    plt.title('Fluctuation Function for Different q-values')

    plt.xticks([16, 32, 128, 512,1024], labels=['16', '32', '128', '512','1024'])
    plt.yticks([1e-5, 1e-4, 1e-3, 1e-2], labels=['0.00001', '0.0001', '0.001','0.01' ])

    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()
    plt.savefig("Fq .png", dpi=300)
if __name__ == "__main__":
    main()
