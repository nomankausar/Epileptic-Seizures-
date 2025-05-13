import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Define your data folder
data_dir = r"C:\p_value" # Change this to your folder containing 30+ CSVs
output_dir = os.path.join(data_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

# Define output PDF path
pdf_output_file = os.path.join(output_dir, "All_SOZ_Pvalue_Boxplots.pdf")

# Create a PDF file to save all plots
with PdfPages(pdf_output_file) as pdf:
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            file_path = os.path.join(data_dir, filename)
            df = pd.read_csv(file_path)

            plt.figure(figsize=(10, 6))
            sns.boxplot(x='SOZ_Channel', y='p_value', data=df)
            plt.title(f'P-Value Distribution per SOZ Channel\n{filename}')
            plt.xlabel('SOZ Channel')
            plt.ylabel('P-Value (KS Test)')
            plt.yscale('log')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save PNG
            output_file_png = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_pvalue_boxplot.png")
            plt.savefig(output_file_png, dpi=300)

            # Save in PDF as a page
            pdf.savefig(dpi=300)
            plt.close()

print(f"All plots saved in {output_dir}")
print(f"Combined PDF saved at {pdf_output_file}")
