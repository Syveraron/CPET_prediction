import numpy as np
import os

from datetime import datetime, timedelta
from forest.oak.base import run
from forest.constants import Frequency

import scipy 
import pandas as pd
import polars as pl
import pytz
from datetime import timedelta
import subprocess

from altering_format import save_to_csv

print('READING DATA LABELS FILE',flush=True)

#set path to REMOTES folder
path = '../../data'



#open dataframe with file names & start times
df = pd.read_excel(path + '/data_labels.xlsx')




print('starting_loop')
for index, row in df.iterrows():

    patient_id = row['Patient ID']
    file_name = df[df['Patient ID'] == patient_id]['file_name'].values[0]
    start_time = df[df['Patient ID'] == patient_id]['Start'].values[0]

    #reformat the data into the csv file needed
    print(f"Processing Patient ID: {patient_id}, File Name: {file_name}",flush=True)
    numeric_patient_id = patient_id.lstrip('R')
    numeric_patient_id = int(numeric_patient_id)
    numeric_patient_id = str(numeric_patient_id)
    print(f"Numeric Patient ID: {numeric_patient_id}",flush=True)

    #open the accelerometer data
    df_accx = pl.read_parquet(f"{path}/bdf_files/{file_name}/{patient_id}/ACC_X.parquet")
    df_accy = pl.read_parquet(f"{path}/bdf_files/{file_name}/{patient_id}/ACC_Y.parquet")
    df_accz = pl.read_parquet(f"{path}/bdf_files/{file_name}/{patient_id}/ACC_Z.parquet")

    acc_x = df_accx.to_numpy().reshape(-1)
    acc_y = df_accy.to_numpy().reshape(-1)
    acc_z = df_accz.to_numpy().reshape(-1)

    start_time = datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')
    print(f"Start Time: {start_time}",flush=True)

    output_dir = f"{path}/steps/input"
    save_to_csv(acc_x, acc_y, acc_z, start_time, output_dir, patient_id)

    print(f"CSV file for {patient_id} created",flush=True)

    tz_str = 'Europe/London'

    # Get the start and end time of the signal in this format "2023-01-09 10_32_00"
    time_start = start_time.strftime('%Y-%m-%d %H_%M_%S')
    time_end = (start_time + timedelta(seconds=acc_x.shape[0] / 25)).strftime('%Y-%m-%d %H_%M_%S')

    study_folder = f"{path}/steps/input"
    output_folder = f"{path}/steps/results"

    frequency = Frequency.MINUTE
    beiwe_id = [numeric_patient_id]

    source_folder = os.path.join(study_folder, numeric_patient_id, "accelerometer")
    print(source_folder, flush=True)
    if not os.path.exists(source_folder):
        print(f"Source folder not found for Patient ID: {patient_id}. Skipping.")
        continue


    print(f"Starting run function with beiwe_id: {beiwe_id} and source folder: {source_folder}")
    run(study_folder, output_folder, tz_str, frequency, time_start, time_end, beiwe_id)

    
    results_file = os.path.join(output_folder, 'minute', f'{numeric_patient_id}_gait_hourly.csv')
    if not os.path.exists(results_file):
        print(f"Results file not found for Patient ID: {patient_id}. Skipping.")
        continue
    
