
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from multiprocessing import Pool, cpu_count

def process_channel(args):
    """Process a single channel and save the plot."""
    time_axis, channel_data, channel_name, output_folder = args

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_axis, channel_data)
    plt.title(f"Channel: {channel_name}")
    plt.xlabel("Segment")
    plt.ylabel("MF-DFA Fluctuations")
    plt.grid()

    # Save the plot
    output_file = os.path.join(output_folder, f"{channel_name}.png")
    plt.savefig(output_file)
    plt.close()

    return f"Saved plot for {channel_name} in {output_file}"


def process_file(file_path, fluctuation_directory, output_directory):
    """Process a single fluctuation file."""
    # Extract subject name from file name
    file_name = os.path.basename(file_path)
    subject_name = os.path.splitext(file_name)[0]
    subject_output_dir = os.path.join(output_directory, subject_name)
    os.makedirs(subject_output_dir, exist_ok=True)

    # Load the data
    if file_name.endswith(".csv"):
        data = pd.read_csv(file_path)
    elif file_name.endswith(".xlsx"):
        data = pd.read_excel(file_path)

    # Validate data structure
    if "Channel" not in data.columns:
        print(f"Skipping {file_name} as 'Channel' column is missing.")
        return

    # Get the time axis and column names
    segments = data.columns[1:]  # Exclude 'Channel'
    time_axis = np.arange(len(segments))

    # Prepare arguments for multiprocessing
    args_list = []
    for idx, row in data.iterrows():
        channel_name = row["Channel"]
        channel_data = row[1:]  # Exclude the 'Channel' column
        args_list.append((time_axis, channel_data, channel_name, subject_output_dir))

    # Use multiprocessing to process channels
    with Pool(cpu_count()) as pool:
        results = pool.map(process_channel, args_list)

    # Print results
    for result in results:
        print(result)


if __name__ == "__main__":
    # Define directories
    fluctuation_directory = r"C:\Engel 1 Subjects\results\rowwise"  # Replace with your actual directory
    output_directory = os.path.join(fluctuation_directory, "Plots")
    os.makedirs(output_directory, exist_ok=True)

    # Process each file in the directory
    for file_name in os.listdir(fluctuation_directory):
        if file_name.endswith(".xlsx") or file_name.endswith(".csv"):
            file_path = os.path.join(fluctuation_directory, file_name)
            process_file(file_path, fluctuation_directory, output_directory)

    print(f"All plots saved in: {output_directory}")

