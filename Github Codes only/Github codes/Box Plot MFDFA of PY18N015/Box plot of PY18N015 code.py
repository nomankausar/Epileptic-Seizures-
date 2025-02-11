# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 00:42:06 2025

@author: Md.Abu Noman Kausar
"""

import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file
file_path = r"C:\Engel 1 Subjects\box plot generation\sub-PY18N015_ses-extraoperative_task-interictal_run-01_ieeg.xlsx"
data = pd.read_excel(file_path)

# Define the corrected SOZ channels
soz_channels = ['POL LF1', 'POL LA1', 'POL LA2', 'POL LA3', 'POL LH1', 'POL LH2', 'POL LH3']

# Filter SOZ and normal channels
data_soz = data[data["Channel"].isin(soz_channels)]
data_normal = data[~data["Channel"].isin(soz_channels)]

# Extract numerical data for plotting
segment_columns = [col for col in data.columns if "Segment" in col]
plot_data = [data_normal[segment_columns].values.flatten(), data_soz[segment_columns].values.flatten()]

# Define box plot colors
box_colors = ["blue", "red"]

# Create box plot
plt.figure(figsize=(8, 6))
box = plt.boxplot(plot_data, labels=["Non-EZ Channels", "EZ Channels"], patch_artist=True)

# Set colors
for patch, color in zip(box["boxes"], box_colors):
    patch.set_facecolor(color)

plt.title("Comparison of H(q)  Non-EZ vs EZ Channels")
plt.ylabel("Hurst Exponent H(q)")
plt.grid(axis="y")

# Save and show the plot
plt.savefig("Comparison_of_H(q)_Non-EZ_vs_EZ_Channels.png", dpi=600, bbox_inches="tight")
plt.show()
