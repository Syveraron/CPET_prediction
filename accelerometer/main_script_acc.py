
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy 
import pandas as pd
import polars as pl
import pytz
from datetime import timedelta
import subprocess
from altering_format import reformat_acc

#set path to REMOTES folder
path = '../../data'

#open dataframe with file names & start times
df = pd.read_excel(path + '/data_labels.xlsx')

#loop through the dataframe, extract the Patient ID values, reformat the data save to new csv file, and run the accProcess program
for index, row in df.iloc[5:6].iterrows():
    
    patient_id = row['Patient ID']
    file_name = df[df['Patient ID'] == patient_id]['file_name'].values[0]
    start_time = df[df['Patient ID'] == patient_id]['Start'].values[0]
    

    #reformat the data into the csv file needed
    print(f"Reformatting data for patient {patient_id}...")
    reformat_acc(patient_id, file_name, start_time)

    # Identify the path of the new csv file
    reformat_path = os.path.join(path, f"bdf_files/{file_name}/{patient_id}/{patient_id}_combined.csv")

    # Ensure the destination directory exists
    output_dir = os.path.join(path + "/activity_class")
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the accProcess without redirecting output because it generates two files automatically
    subprocess.run(["accProcess", reformat_path, "--sampleRate",  "25"])

    # Move the generated files to the output directory
    subprocess.run(["mv", path + f"/bdf_files/{file_name}/{patient_id}/{patient_id}_combined-summary.json", f"{output_dir}"])
    subprocess.run(["mv", path + f"/bdf_files/{file_name}/{patient_id}/{patient_id}_combined-timeSeries.csv.gz", f"{output_dir}"])

    

