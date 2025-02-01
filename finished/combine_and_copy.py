import os
import pandas as pd

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory


main_df = pd.DataFrame()  # Create an empty DataFrame to store the combined data

for file in files:
    if file.endswith('.csv'):  # Check if the file is a CSV file
        df = pd.read_csv(file)  # Read the CSV file into a DataFrame
        if main_df.empty:  # If the main DataFrame is empty, set it to the first DataFrame read
            main_df = df
        else:
            main_df = pd.concat([main_df, df], ignore_index=True)  # Concatenate the DataFrame to the main DataFrame

main_df.to_csv('combined.csv', index=False)  # Write the combined DataFrame to a new CSV file