import os
import mne
import matplotlib.pyplot as plt

# Define the folder containing the .fif files
fif_folder = r"C:\Engel 1 Subjects\100s cropped fif file"

# Function to apply a notch filter
def apply_notch_filter(raw, freq=60.0, notch_widths=2.0):
    raw.notch_filter(freqs=[freq], notch_widths=notch_widths, method='iir')
    return raw

# Process each .fif file
for root, dirs, files in os.walk(fif_folder):
    for file in files:
        if file.endswith('.fif'):
            file_path = os.path.join(root, file)
            subject_folder = os.path.join(root, os.path.splitext(file)[0])

            # Create subject-specific folder for outputs
            if not os.path.exists(subject_folder):
                os.makedirs(subject_folder)

            # Load the .fif file
            raw = mne.io.read_raw_fif(file_path, preload=True)

            # Apply 60 Hz notch filter
            raw = apply_notch_filter(raw)

            # Get channel names
            channel_names = raw.info['ch_names']

            # Loop through each channel
            for i, channel in enumerate(channel_names):
                # Select the specific channel
                raw_channel = raw.copy().pick_channels([channel])

                # Plot time series
                fig, ax = plt.subplots(figsize=(10, 5))
                data, times = raw_channel[:]
                ax.plot(times, data[0] * 1e6)  # Convert to microvolts
                ax.set_title(f"{channel} - Time Series")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude (µV)")
                ax.grid(True)

                # Save time series plot
                time_series_path = os.path.join(subject_folder, f"{channel}_time_series.png")
                plt.savefig(time_series_path)
                plt.close(fig)

                # Plot PSD
                fig, ax = plt.subplots(figsize=(10, 5))
                raw_channel.plot_psd(fmax=100, ax=ax, show=False)
                ax.set_title(f"{channel} - Power Spectral Density")
                
                # Save PSD plot
                psd_path = os.path.join(subject_folder, f"{channel}_psd.png")
                plt.savefig(psd_path)
                plt.close(fig)

                print(f"Saved plots for channel {channel} in {file}")
