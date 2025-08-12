import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from multiprocessing import Pool, cpu_count

plt.rcParams.update({
    'font.size': 18,       # Default text
    'axes.titlesize': 18,  # Plot title
    'axes.labelsize': 18,  # X and Y labels
    'xtick.labelsize': 18, # X tick labels
    'ytick.labelsize': 18, # Y tick labels
    'legend.fontsize': 18  # Legend
})
# Channels to be plotted in red
red_channels = ["POL LF1", "POL LA1", "POL LA2", "POL LA3", "POL LH1", "POL LH2", "POL LH3"]

def process_channel(args):
    """Process a single channel and save the plot."""
    time_axis, channel_data, channel_name, output_folder = args
    color = "red" if channel_name.strip() in red_channels else "blue"

    # Create plot
    plt.figure(figsize=(6, 4))
    plt.plot(time_axis, channel_data, color=color, linewidth=2)

    plt.title(f"Channel: {channel_name}", fontsize=20, fontweight="bold")
    plt.xlabel("Segment(5s)", fontsize=20, fontweight="bold")
    plt.ylabel("H(q)", fontsize=20, fontweight="bold")
    plt.xticks(fontsize=20, fontweight="bold")
    plt.yticks(fontsize=20, fontweight="bold")
    plt.grid(True, linewidth=0.6, alpha=0.7)
    

    # Save as high-resolution PNG
    output_file = os.path.join(output_folder, f"{channel_name}.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    return f"Saved plot for {channel_name} in {output_file}"


def process_file(file_path, fluctuation_directory, output_directory):
    """Process a single fluctuation file."""
    file_name = os.path.basename(file_path)
    subject_name = os.path.splitext(file_name)[0]
    subject_output_dir = os.path.join(output_directory, subject_name)
    os.makedirs(subject_output_dir, exist_ok=True)

    if file_name.endswith(".csv"):
        data = pd.read_csv(file_path)
    elif file_name.endswith(".xlsx"):
        data = pd.read_excel(file_path)

    if "Channel" not in data.columns:
        print(f"Skipping {file_name} as 'Channel' column is missing.")
        return

    segments = data.columns[1:]
    time_axis = np.arange(len(segments))

    args_list = []
    for _, row in data.iterrows():
        channel_name = str(row["Channel"])
        channel_data = row[1:]
        args_list.append((time_axis, channel_data, channel_name, subject_output_dir))

    with Pool(cpu_count()) as pool:
        results = pool.map(process_channel, args_list)

    for result in results:
        print(result)


if __name__ == "__main__":
    fluctuation_directory = r"D:\conference paper works replicate\test"
    output_directory = os.path.join(fluctuation_directory, "Plots")
    os.makedirs(output_directory, exist_ok=True)

    for file_name in os.listdir(fluctuation_directory):
        if file_name.endswith(".xlsx") or file_name.endswith(".csv"):
            file_path = os.path.join(fluctuation_directory, file_name)
            process_file(file_path, fluctuation_directory, output_directory)

    print(f"All plots saved in: {output_directory}")
