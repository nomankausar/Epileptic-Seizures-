# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 22:22:57 2024

@author: abuno
"""

import os
import mne

# Define the folder containing the .fif files
fif_folder = r"C:\Engel 1 Subjects\edf"

# Manual file mapping for Engel-I subjects
file_mapping = {
    "sub-NIH1_ses-extraoperative_task-interictal_run-01_ieeg.fif": "sub-NIH1_ses-extraoperative_task-interictal_run-01_channels.tsv",
      "sub-NIH2_ses-extraoperative_task-interictal_run-01_ieeg.fif": "sub-NIH2_ses-extraoperative_task-interictal_run-01_channels.tsv",
      "sub-NIH4_ses-extraoperative_task-interictal_run-01_ieeg.fif": "sub-NIH4_ses-extraoperative_task-interictal_run-01_channels.tsv",
      "sub-NIH5_ses-extraoperative_task-interictal_run-01_ieeg.fif": "sub-NIH5_ses-extraoperative_task-interictal_run-01_channels.tsv",
      "sub-PY18N013_ses-extraoperative_task-interictal_run-01_ieeg.fif": "sub-PY18N013_ses-extraoperative_task-interictal_run-01_channels.tsv",
      "sub-PY18N015_ses-extraoperative_task-interictal_run-01_ieeg.fif": "sub-PY18N015_ses-extraoperative_task-interictal_run-01_channels.tsv",
      "sub-PY19N023_ses-extraoperative_task-interictal_run-01_ieeg.fif": "sub-PY19N023_ses-extraoperative_task-interictal_run-01_channels.tsv",
    "sub-PY19N026_ses-extraoperative_task-interictal_run-01_ieeg.fif": "sub-PY19N026_ses-extraoperative_task-interictal_run-01_channels.tsv",
    "sub-rns006_ses-extraoperative_task-interictal_run-01_ieeg.fif": "sub-rns006_ses-extraoperative_task-interictal_run-01_channels.tsv",
}


# Function to crop the first 100 seconds and retain the rest
def crop_first_100_seconds(file_path, save_path):
    # Load the .fif file
    raw = mne.io.read_raw_fif(file_path, preload=True)
    
    # Get the total duration of the recording in seconds
    total_duration = raw.times[-1]
    
    # Check if the recording is longer than 100 seconds
    if total_duration > 100:
        # Crop out the first 100 seconds
        raw.crop(tmin=100, tmax=None)
        # Save the modified file
        raw.save(save_path, overwrite=True)
        print(f"Processed and saved: {save_path}")
    else:
        print(f"File {file_path} is less than 100 seconds long and was skipped.")

# Process each .fif file in the directory
for root, dirs, files in os.walk(fif_folder):
    for file in files:
        if file.endswith('.fif'):
            file_path = os.path.join(root, file)
            save_path = os.path.join(root, f"cropped_{file}")
            
            # Crop and save the file
            crop_first_100_seconds(file_path, save_path)
