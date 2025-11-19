import numpy as np
import os
import matplotlib.pyplot as plt
import scipy 
import neurokit2 as nk

import pandas as pd
import polars as pl

from orphanidou_nk import assess_qual_hr

#set path directory for data files
path = '../'

#read in signal freq -> dictionary
file = open(f"{path}/data/label_freq.txt", 'r')
label_freq = file.read()
label_freq = eval(label_freq)
file.close()

new_ecg_fs = 250
ecg_fs = label_freq['ECG_A']
fs = 250

#open the data file to loop through 
df = pd.read_csv(f"{path}/data/data_files.xlsx")

#loop through the files in the datarame to access each ECG file

for index, row in df.iterrows():
    #loop through the df get the patient id and the file name 
    patient_id = row['Patient ID']
    file_name = row['file_name']
    print(patient_id)
    
    #set path to the ECG file
    path = f"{path}/data/bdf_files/{file_name}/{patient_id}/" 
    
    #open the ECG A file 
    df_ecg = pl.read_parquet(path + 'ECG_A.parquet')

    #convert t0 1D numpy array
    ecg = df_ecg.to_numpy()
    ecg = ecg.reshape(-1)

    #resample the signal to 250Hz as extracting 10s at current Hz would leave discrepancies in the window size over time
    ecg = nk.signal_resample(ecg, desired_sampling_rate=new_ecg_fs, sampling_rate=ecg_fs)
    length = len(ecg)
    print(len(ecg))

    #calculate the window size for 10s of an ECG
    window = 10 * new_ecg_fs

    #caluclate what would be left over if dividing ecg in 10s and remove anything leftover
    leftover = length%window
    if leftover >0:
        ecg = ecg[:-leftover]

    n_windows = int(length/window)
    print(n_windows)

    #reshape the original 1D array into a 2D matrix with 10s windows
    rows = n_windows
    cols = window
    ecg = ecg.reshape(rows, cols)
    ecg.shape

    #initate lists to store the quality values
    quality_nk = np.zeros(n_windows)
    hr_nk = np.zeros(n_windows)

    for i in range(n_windows):
        #try:
        qual, hr_full, beats = assess_qual_hr(ecg[i], new_ecg_fs, thresh=0.66)
        quality_nk[i] = qual
        hr_nk[i] = hr_full
        #except:
            #print(f"error at index {i}")
            #quality_nk[i] = 0
            #hr_nk[i] = 0

    hr_nk = np.round(hr_nk).astype(int)

    #save the quality values array as a .npy file with the same name as the patient id
    np.save(f'{path}/data/hr_values/{patient_id}.npy', hr_nk)