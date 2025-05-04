import os
import pandas as pd

# Path to the main dataset folder
base_dir = "dataset"  # Replace with your actual path

# List to collect all combined dataframes
combined_data = []

# Loop over folder names from 1 to 160
for i in range(1, 161):
    folder_name = str(i)
    folder_path = os.path.join(base_dir, folder_name)
    
    output_csv_path = os.path.join(folder_path, "output.csv")
    data_out_csv_path = os.path.join(folder_path, "data_out.csv")
    
    # Read both CSVs
    try:
        output_df = pd.read_csv(output_csv_path)
        data_out_df = pd.read_csv(data_out_csv_path)
    except FileNotFoundError:
        print(f"Skipping folder {folder_name}: Missing files")
        continue
    
    # Modify 'image' column to include folder name
    output_df["Image"] = output_df["Image"].apply(lambda x: f"{folder_name}_{x}")
    
    # Combine output and data_out CSVs side by side
    merged_df = pd.concat([output_df, data_out_df], axis=1)
    
    combined_data.append(merged_df)

# Concatenate all combined dataframes into one
final_df = pd.concat(combined_data, ignore_index=True)

# Drop rows where all columns except the last 4 are NaN, and the last 4 are exactly 0.0
num_cols = final_df.shape[1]
last_four_cols = final_df.columns[-4:]

mask = (
    final_df.iloc[:, :-4].isnull().all(axis=1) & 
    (final_df[last_four_cols] == 0.0).all(axis=1)
)

# Invert mask to keep only rows that do not match the pattern
final_df = final_df[~mask]

# Save to CSV
final_df.to_csv("merged_dset.csv", index=False)

print("Filtered and combined CSV saved as merged_dset.csv")
