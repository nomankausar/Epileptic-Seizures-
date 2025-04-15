import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import matplotlib.patches as mpatches

# Folder containing your Excel files
folder_path = r"C:\Engel 1 Subjects\resul180sswithscalingw16_32_8192antropydfa\updated_excels"

# Get all .xlsx files in the folder
file_paths = glob.glob(os.path.join(folder_path, "*.xlsx"))
print(file_paths)

# Prepare list to collect data
data_list = []

# Process each file
for file_path in file_paths:
    subject_name = os.path.basename(file_path).split("_")[0]  # Extract subject name
    df = pd.read_excel(file_path)

    # Clean up column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Check for SOZ and segment data
    if "identity_code_value" in df.columns:
        segment_cols = [col for col in df.columns if col.startswith("segment_")]
        if segment_cols:
            df_long = df.melt(
                id_vars=["identity_code_value"],
                value_vars=segment_cols,
                var_name="segment",
                value_name="value"
            )
            df_long.rename(columns={"identity_code_value": "soz"}, inplace=True)
            df_long["subject"] = subject_name
            data_list.append(df_long)

# Combine all data
combined_df = pd.concat(data_list, ignore_index=True)

# Plotting
plt.figure(figsize=(14, 7))
sns.boxplot(
    x="subject",
    y="value",
    hue="soz",
    data=combined_df,
    palette={0: "blue", 1: "red"}
)

plt.xlabel("Subject Name")
plt.ylabel("Hurst Exponent H(q)")
plt.title("EZ vs Non-EZ DFA Analysis")
plt.xticks(rotation=45)
# Create custom legend patches to match the color palette
non_soz_patch = mpatches.Patch(color='blue', label='Non-EZ')
soz_patch = mpatches.Patch(color='red', label='EZ')

plt.legend(handles=[non_soz_patch, soz_patch])
plt.grid(True)

# Save the figure
plt.savefig(os.path.join(folder_path, "hurst_exponent_corrected.png"), dpi=300)
plt.show()
