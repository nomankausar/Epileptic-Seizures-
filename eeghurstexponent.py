# -*- coding: utf-8 -*-
""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 11:50:06 2024

@author: Md.Abu Noman Kausar
"""
import mne
import numpy as np
import matplotlib.pyplot as plt
from hurst import compute_Hc

# Function to preprocess the data (e.g., remove mean, normalize)
def preprocess_data(data):
    if len(data) == 0:
        raise ValueError("Data array is empty.")
    
    # Remove mean
    data -= np.mean(data)
    
    # Check for zero standard deviation to avoid division by zero
    std_dev = np.std(data)
    if std_dev == 0:
        raise ValueError("Standard deviation of the data is zero, cannot normalize.")
    
    # Normalize the data
    data /= std_dev
    return data

# Load the EEG data from EDF file using MNE
edf_file = 'subPY19N026.edf'  # Replace with your EDF file name
try:
    raw = mne.io.read_raw_edf(edf_file, preload=True)
except FileNotFoundError:
    raise FileNotFoundError(f"EDF file '{edf_file}' not found.")
except Exception as e:
    raise RuntimeError(f"Error loading EDF file: {e}")

# Get data and channel information
try:
    eeg_data = raw.get_data()
    channel_names = raw.ch_names
except Exception as e:
    raise RuntimeError(f"Error extracting data or channel names: {e}")

# Print all channel names
for i, name in enumerate(channel_names):
    print(f"Channel {i+1}: {name}")

print("All channel names:", channel_names) 

# List of channels to highlight in red
highlight_channels = [
    'POL RAM1', 'POL RAM2', 'POL RAM3', 'POL RAM4', 'POL RAH1', 
    'POL RAH2', 'POL RAH3', 'POL RPH1', 'POL RPH2', 'POL RPH3', 
    'POL RPHG1', 'POL RPHG2', 'POL RPHG3'
]

# Ensure all highlight channels are valid
invalid_channels = [ch for ch in highlight_channels if ch not in channel_names]
if invalid_channels:
    print(f"Warning: These highlight channels were not found: {invalid_channels}")

# List to store Hurst exponent values
hurst_values = []

# Iterate over all channels
for i, name in enumerate(channel_names):
    try:
        # Select data for the current channel
        channel_data = preprocess_data(eeg_data[i])
        
        # Compute the Hurst exponent
        H, c, data = compute_Hc(channel_data, kind='random_walk', simplified=True)
        
    except FloatingPointError:
        H = np.nan
        print(f"FloatingPointError encountered for channel: {name}")
        
    except ValueError as ve:
        H = np.nan
        print(f"ValueError for channel {name}: {ve}")
        
    except Exception as e:
        H = np.nan
        print(f"Unexpected error for channel {name}: {e}")
    
    # Store the Hurst exponent value
    hurst_values.append(H)

# Plotting the Hurst exponent values as a bar chart
plt.figure(figsize=(14, 7))
colors = ['red' if name in highlight_channels else 'skyblue' for name in channel_names]
plt.bar(channel_names, hurst_values, color=colors)
plt.xlabel('Channel')
plt.ylabel('Hurst Exponent')
plt.title('Hurst Exponent for Each EEG Channel')
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
