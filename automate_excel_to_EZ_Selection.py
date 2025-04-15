import pandas as pd
import os
import re

# === SETTINGS ===
input_folder = r"C:\Engel 1 Subjects\resul180sswithscalingw16_32_8192"  # Replace with your actual folder path
soz_csv = os.path.join(input_folder, "SOZ_Channels_info.csv")
output_folder = os.path.join(input_folder, "updated_excels")
os.makedirs(output_folder, exist_ok=True)

# === LOAD SOZ INFO ===
soz_info_df = pd.read_csv(soz_csv)
soz_long_df = soz_info_df.melt(id_vars=["Subject"], var_name="Channel_Index", value_name="Channel").dropna()
soz_long_df["Clean_Channel"] = soz_long_df["Channel"].apply(lambda x: re.sub(r'[^A-Za-z0-9]', '', str(x)).upper())
soz_long_df["Subject"] = soz_long_df["Subject"].str.upper()

# === HELPER FUNCTIONS ===
def extract_subject_id(filename):
    match = re.search(r"sub-([A-Za-z0-9]+)_", filename)
    return match.group(1).upper() if match else None

def clean_channel_name(name):
    return re.sub(r'[^A-Za-z0-9]', '', str(name)).upper()

def fuzzy_match(channel, soz_list):
    for soz in soz_list:
        if channel in soz or soz in channel:
            return 1
    return 0

# === PROCESS FILES ===
for file in os.listdir(input_folder):
    if not file.endswith(".xlsx") or file == "SOZ_Channels_info.xlsx":
        continue

    filepath = os.path.join(input_folder, file)
    try:
        df = pd.read_excel(filepath)
    except Exception as e:
        print(f"❌ Could not open {file}: {e}")
        continue

    if "Channel" not in df.columns:
        print(f"❌ Skipping {file}: 'Channel' column not found.")
        continue

    subject_id = extract_subject_id(file)
    if not subject_id:
        print(f"❌ Skipping {file}: No subject ID found in filename.")
        continue

    soz_channels = soz_long_df[soz_long_df["Subject"] == subject_id]["Clean_Channel"].tolist()
    df["Clean_Channel"] = df["Channel"].apply(clean_channel_name)
    df["Identity Code Value"] = df["Clean_Channel"].apply(lambda ch: fuzzy_match(ch, soz_channels))

    # Move "Identity Code Value" column after "Channel"
    cols = df.columns.tolist()
    if "Channel" in cols and "Identity Code Value" in cols:
        channel_index = cols.index("Channel")
        cols.insert(channel_index + 1, cols.pop(cols.index("Identity Code Value")))
        df = df[cols]

    df.drop(columns=["Clean_Channel"], inplace=True)
    output_path = os.path.join(output_folder, file)
    df.to_excel(output_path, index=False)
    print(f"✅ Processed: {file}")

print("\n🎉 All files processed and saved to:", output_folder)
