import os
import pandas as pd

def merge_csv_files(input_folder_path, output_folder_path):
    # Ensure the output folder exists
    os.makedirs(output_folder_path, exist_ok=True)
    
    # Lists to store DataFrames
    improv_dfs = []
    non_improv_dfs = []

    # Iterate over all files in the folder
    for filename in os.listdir(input_folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder_path, filename)
            df = pd.read_csv(file_path)
            
            if 'improv' in filename:
                improv_dfs.append(df)
            else:
                non_improv_dfs.append(df)

    # Merge all DataFrames
    if improv_dfs:
        merged_improv_df = pd.concat(improv_dfs, ignore_index=True)
        merged_improv_df.to_csv(os.path.join(output_folder_path, 'reb_improv.csv'), index=False)
    
    if non_improv_dfs:
        merged_non_improv_df = pd.concat(non_improv_dfs, ignore_index=True)
        merged_non_improv_df.to_csv(os.path.join(output_folder_path, 'reb_non_improv.csv'), index=False)

# Example usage
input_folder_path = '/home/wiss/zhang/Jinhe/singularity/paper_results/rebbutal'  # Replace with the path to your input folder
output_folder_path = '/home/wiss/zhang/Jinhe/singularity/paper_results/rebbutal/rebuttal_final'  # Replace with the path to your output folder
merge_csv_files(input_folder_path, output_folder_path)
