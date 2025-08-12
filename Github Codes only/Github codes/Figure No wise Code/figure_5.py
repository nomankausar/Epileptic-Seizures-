import pandas as pd  # Import pandas for handling dataframes
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import seaborn as sns  # Import seaborn for statistical plotting
import os  # Import os for handling file paths
import glob  # Import glob for retrieving file paths
# Set global font sizes
plt.rcParams.update({
    'font.size': 18,       # Default text
    'axes.titlesize': 18,  # Plot title
    'axes.labelsize': 18,  # X and Y labels
    'xtick.labelsize': 18, # X tick labels
    'ytick.labelsize': 18, # Y tick labels
    'legend.fontsize': 18  # Legend
})

# Define the file path directory containing the Excel files
folder_path = r"D:\conference paper works replicate\Stats Combined box plot on each subject\Excel Datasets"

# Get a list of all Excel files in the specified directory
file_paths = glob.glob(os.path.join(folder_path, "*.xlsx"))

# Initialize an empty list to store processed data from each file
data_list = []

# Loop through each file in the folder
for file_path in file_paths:
    subject_name = os.path.basename(file_path).split("_")[0]  # Extract subject ID from the filename
    
    # Read the Excel file into a Pandas dataframe
    df = pd.read_excel(file_path)
    
    # Remove any leading or trailing spaces from column names
    df.columns = df.columns.str.strip()
    
    # Ensure that the required columns exist in the dataframe
    if "Identity Code Value" in df.columns and any(col.startswith("Segment_") for col in df.columns):
        # Identify all columns that start with "Segment_", which contain the MFDFA values
        value_columns = [col for col in df.columns if col.startswith("Segment_")]
        
        # Reshape the dataframe from wide format to long format (melting)
        df_long = df.melt(
            id_vars=["Identity Code Value"],  # Keep "Idenity Code Value" as identifier (SOZ)
            value_vars=value_columns,  # Columns to convert from wide to long
            var_name="Segment",  # Name for the new column that identifies segment names
            value_name="Value"  # Name for the new column that holds MFDFA values
        )
        
        # Rename the "Idenity Code Value" column to "SOZ" for clarity
        df_long.rename(columns={"Identity Code Value": "SOZ"}, inplace=True)
        
        # Add a new column that contains the subject ID (extracted from the filename)
        df_long["Subject"] = subject_name  

        # Append the processed dataframe to the data list
        data_list.append(df_long)

# Combine all processed data into a single dataframe
combined_df = pd.concat(data_list, ignore_index=True)

# Create a figure for the box plots
plt.figure(figsize=(12, 6))  # Set figure size to 12 inches by 6 inches

# Generate box plots comparing SOZ vs Non-SOZ per subject
sns.boxplot(
    x="Subject",  # X-axis: Subject ID
    y="Value",  # Y-axis: MFDFA values
    hue="SOZ",  # Differentiate between SOZ (1) and Non-SOZ (0) using colors
    data=combined_df,  # Data source
    palette={0: "blue", 1: "red"}  # Define color scheme (gray for Non-SOZ, blue for SOZ)
)

# Label the x-axis
plt.xlabel("Subject Name")

# Label the y-axis
plt.ylabel("Hurst Exponent H(q)")

# Set a title for the plot
plt.title("EZ vs Non-EZ DFA Analysis")

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Add a legend with corrected color labels
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, ["Non-EZ ", "EZ "])

# Add gridlines to improve readability
plt.grid(True)

# Adjust layout to prevent overlapping labels
plt.tight_layout()
plt.ylim(0,2)
plt.savefig("Hurst_exponent.png", dpi=1200)

# Display the plot
plt.show()
