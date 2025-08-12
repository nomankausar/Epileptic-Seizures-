import os
import mne
import matplotlib.pyplot as plt

# ==== GLOBAL PLOT SETTINGS ====
plt.rcParams.update({
    'font.size': 17,         # Base font size
    'axes.titlesize': 17,    # Title font size
    'axes.labelsize': 17,    # X/Y label font size
    'xtick.labelsize': 17,   # X-tick font size
    'ytick.labelsize': 17,   # Y-tick font size
    'legend.fontsize': 17    # Legend font size
})

# List of channels to be red
red_channels = ["POL LF1", "POL LA1", "POL LA2", "POL LA3",
                "POL LH1", "POL LH2", "POL LH3"]

# Define the folder containing the .fif files
fif_folder = r"D:\conference paper works replicate\test"

# Function to apply a notch filter
def apply_notch_filter(raw, freq=60.0, notch_widths=2.0):
    raw.notch_filter(freqs=[freq], notch_widths=notch_widths, method='fir')
    return raw

# Process each .fif file
for root, dirs, files in os.walk(fif_folder):
    for file in files:
        if file.endswith('.fif'):
            file_path = os.path.join(root, file)
            subject_folder = os.path.join(root, os.path.splitext(file)[0])

            # Create subject-specific folder for outputs
            os.makedirs(subject_folder, exist_ok=True)

            # Load the .fif file
            raw = mne.io.read_raw_fif(file_path, preload=True)

            # Apply 60 Hz notch filter
            raw = apply_notch_filter(raw)

            # Get channel names
            channel_names = raw.info['ch_names']

            # Loop through each channel
            for channel in channel_names:
                # Choose color based on channel name
                color = 'red' if channel in red_channels else 'blue'

                # Select the specific channel
                raw_channel = raw.copy().pick_channels([channel])

                # === Time Series Plot ===
                fig, ax = plt.subplots(figsize=(10, 5))
                data, times = raw_channel[:]
                ax.plot(times, data[0] * 1e6, color=color)  # µV
                ax.set_title(f"{channel} - Time Series")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude (µV)")
                ax.grid(True)
                time_series_path = os.path.join(subject_folder, f"{channel}_time_series.png")
                plt.savefig(time_series_path, dpi=600, bbox_inches='tight')
                plt.close(fig)

                # === PSD Plot ===
                fig, ax = plt.subplots(figsize=(10, 5))
                raw_channel.plot_psd(fmax=100, ax=ax, show=False)
                for line in ax.get_lines():
                    line.set_color(color)  # Apply color to PSD line
                ax.set_title(f"{channel} - Power Spectral Density")
                ax.set_xlabel("Frequency (Hz)")
                ax.set_ylabel("Power (dB)")
                psd_path = os.path.join(subject_folder, f"{channel}_psd.png")
                plt.savefig(psd_path, dpi=600, bbox_inches='tight')
                plt.close(fig)

                print(f"Saved plots for channel {channel} in {file}")
