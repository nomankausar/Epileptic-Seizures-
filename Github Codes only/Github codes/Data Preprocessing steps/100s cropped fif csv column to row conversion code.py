import os
import pandas as pd

# Function to convert all Excel files in a folder, retaining channel names
def convert_folder_columns_to_rows(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")

    # Iterate through all Excel files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.xlsx'):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, f"converted_{filename}")

            try:
                # Read the Excel file into a DataFrame
                df = pd.read_excel(input_file)
                print(f"Processing file: {input_file}")

                # Transpose the DataFrame (convert columns to rows)
                transposed_df = df.T
                print(f"Transposed DataFrame shape: {transposed_df.shape}")

                # Align columns and headers dynamically
                transposed_df.columns = [f"Column_{i+1}" for i in range(len(transposed_df.columns))]
                transposed_df.rename(columns={transposed_df.columns[0]: "Channel Names"}, inplace=True)

                # Save the transposed DataFrame to a new Excel file
                transposed_df.to_excel(output_file, index=False)
                print(f"Converted and saved: {output_file}")

            except Exception as e:
                print(f"Error processing file {input_file}: {e}")

# Specify the input folder and the output folder
input_folder = r"C:\\Engel 1 Subjects\\results\\columnwise"  # Replace with your input folder path
output_folder = r"C:\\Engel 1 Subjects\\results\\converted"  # Replace with your output folder path

# Call the function to process all files in the folder
convert_folder_columns_to_rows(input_folder, output_folder)
