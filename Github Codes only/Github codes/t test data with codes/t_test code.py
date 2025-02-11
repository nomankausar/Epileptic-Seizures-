import os  # Library to interact with the operating system
import pandas as pd  # Library for handling data in tabular format
import scipy.stats as stats  # Library for statistical analysis (t-test)
import matplotlib.pyplot as plt  # Library for plotting graphs
import seaborn as sns  # Library for creating visually appealing plots

# Set your folder path where the Excel files are located
folder_path = r"C:\t test"  # Change this to your local directory

# List all Excel files in the folder
file_list = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]

# Store results for each patient and for global analysis
patient_results = {}  # Dictionary to store t-test results for each patient
all_soz = []  # List to store all SOZ (Seizure Onset Zone) values
all_non_soz = []  # List to store all Non-SOZ values

# Process each file
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)  # Create full file path

    try:
        df = pd.ExcelFile(file_path)  # Load the Excel file
        
        # Read the first sheet from the Excel file
        sheet_df = df.parse(df.sheet_names[0])

        # Strip spaces from column names and rename incorrect columns
        sheet_df.columns = sheet_df.columns.str.strip()
        sheet_df.rename(columns={"Idenity Code Value": "Identity Code"}, inplace=True)

        # Identify MFDFA columns (all columns starting with "Segment_")
        mfdfa_cols = [col for col in sheet_df.columns if col.startswith("Segment_")]

        # Ensure both "Identity Code" and MFDFA values are present in the dataset
        if "Identity Code" in sheet_df.columns and mfdfa_cols:
            # Compute the mean MFDFA value per row (averaging across segments)
            sheet_df["MFDFA Value"] = sheet_df[mfdfa_cols].mean(axis=1)

            # Separate SOZ (1) and non-SOZ (0) values
            soz_values = sheet_df[sheet_df["Identity Code"] == 1]["MFDFA Value"].dropna()
            non_soz_values = sheet_df[sheet_df["Identity Code"] == 0]["MFDFA Value"].dropna()

            # Store values for global analysis
            all_soz.extend(soz_values)
            all_non_soz.extend(non_soz_values)

            # Perform t-test for this patient if both groups have at least 2 values
            if len(soz_values) > 1 and len(non_soz_values) > 1:
                t_stat, p_value = stats.ttest_ind(soz_values, non_soz_values, equal_var=False)
                patient_results[file_name] = {"t-statistic": t_stat, "p-value": p_value}

                # Generate a box plot for visualization
                plt.figure(figsize=(6, 4))
                sns.boxplot(data=sheet_df, x="Identity Code", y="MFDFA Value")
                plt.xticks(ticks=[0, 1], labels=["Non-SOZ", "SOZ"])  # Set x-axis labels
                plt.title(f"Boxplot of MFDFA - {file_name}")  # Title of the plot
                plt.show()

        else:
            print(f"Skipping {file_name}: Required columns not found.")  # Print if missing columns

    except Exception as e:
        print(f"Error processing {file_name}: {e}")  # Print error message if any issue occurs

# Perform global t-test across all patients
if len(all_soz) > 1 and len(all_non_soz) > 1:
    global_t_stat, global_p_value = stats.ttest_ind(all_soz, all_non_soz, equal_var=False)
    patient_results["Global Analysis (All Patients)"] = {"t-statistic": global_t_stat, "p-value": global_p_value}

# Convert results to DataFrame and save to CSV
results_df = pd.DataFrame.from_dict(patient_results, orient="index")
results_df.to_csv(os.path.join(folder_path, "t_test_results.csv"))  # Save results as CSV file

print("T-test results saved to 't_test_results.csv'.")  # Notify user of successful execution
