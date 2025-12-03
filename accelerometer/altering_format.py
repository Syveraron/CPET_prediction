
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy 
import pandas as pd
import polars as pl
import pytz
from datetime import timedelta

path = '../../data/'


def reformat_acc(patient_id, file_name, start_time, fs=25):
    """
    Function that reads in the raw acceleration data from the parquet files and reformats it into a dataframe with the 
    following columns: 'timestamp', 'x', 'y', 'z, temp'. The timestamp is in seconds and the acceleration values are in m/s^2.
    
    Args:
    - patient_id: the patient ID of the patient whose data is to be reformatted.
    - fs: the sampling frequency of the data. Default is 25 Hz.
    
    Returns:
    - df: a dataframe containing the reformatted acceleration data.
    """
    
    
    #load in the parquet file for x, y and z
    x = pl.read_parquet(path + f"bdf_files/{file_name}/{patient_id}/ACC_X.parquet")
    y = pl.read_parquet(path + f"bdf_files/{file_name}/{patient_id}/ACC_Y.parquet")
    z = pl.read_parquet(path + f"bdf_files/{file_name}/{patient_id}/ACC_Z.parquet")

    # Convert to pandas DataFrame
    df_x = x.to_pandas()
    df_y = y.to_pandas()
    df_z = z.to_pandas()

    # Combine df_x, df_y, df_z into a single dataframe
    df_combined = pd.DataFrame({
        'x' : df_x.values.flatten(),
        'y' : df_y.values.flatten(),
        'z' : df_z.values.flatten()
    })

    #convert time into a timezone aware datetime object
    start_time = pd.to_datetime(start_time).tz_localize('Europe/London')

    # Generate the time column with 25 Hz frequency (1 sample every 40 milliseconds)
    time_stamps = [start_time + timedelta(seconds=i / 25) for i in range(len(df_combined))]

    # Convert to the required time format "YYYY-MM-DD HH:MM:SS.sss+zzzz [Europe/London]"
    df_combined['time'] = [ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3] + ts.strftime('%z') + ' [Europe/London]' for ts in time_stamps]

    # Move the 'time' column to the first position
    df_combined = df_combined[['time', 'x', 'y', 'z']]

    # Assuming df_combined is your pandas dataframe with time, x, y, z
    df_combined['temp'] = 0.0  # Add a temperature column with placeholder values (zeros)

    # Reorder columns to match the expected order: time, x, y, z, temp
    df_combined = df_combined[['time', 'x', 'y', 'z', 'temp']]

    #save to csv file to nobackup
    output_file = path + f"bdf_files/{file_name}/{patient_id}/{patient_id}_combined.csv"

    # Save to CSV
    df_combined.to_csv(output_file, index=False)

    print(f"CSV file for {patient_id} created")

def save_to_csv2(acc_x, acc_y, acc_z, start_time, output_dir, patient_id):
    sampling_frequency = 25
    samples_per_day = 24 * 60 * 60 * sampling_frequency
    min_samples_per_day = 1 * sampling_frequency

    total_samples = len(acc_x)
    total_days = total_samples // samples_per_day
    remaining_samples = total_samples % samples_per_day

    #extract the numeric part of the patient ID
    numeric_patient_id = patient_id.lstrip('R')
    numeric_patient_id = int(numeric_patient_id)
    numeric_patient_id = str(numeric_patient_id)    

    def format_data(arr1, arr2, arr3, start_time, sampling_rate=25):
        timestamps = []
        utc_times = []
        accuracy = np.full(len(arr1), 'unknown')
        
        current_time = start_time
        for i in range(len(arr1)):
            timestamps.append(int(current_time.timestamp() * 1000))
            utc_times.append(current_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3])
            current_time += timedelta(seconds=1/sampling_rate)
        
        formatted_data = np.column_stack((timestamps, utc_times, accuracy, arr1, arr2, arr3))
        formatted_data = [f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}" for row in formatted_data]
        
        # Add column titles
        column_titles = "timestamp,UTC time,accuracy,x,y,z"
        formatted_data.insert(0, column_titles)
        
        return "\n".join(formatted_data)

    for day in range(total_days):
        start_idx = day * samples_per_day
        end_idx = start_idx + samples_per_day

        start_time_str = (start_time + timedelta(days=day)).strftime('%Y-%m-%d %H_%M_%S')
        csv_data = format_data(acc_x[start_idx:end_idx], acc_y[start_idx:end_idx], acc_z[start_idx:end_idx], start_time + timedelta(days=day))

        file_path = os.path.join(output_dir, numeric_patient_id, 'accelerometer', f"{start_time_str}.csv")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as f:
            f.write(csv_data)

    if remaining_samples >= min_samples_per_day:
        start_idx = total_days * samples_per_day
        start_time_str = (start_time + timedelta(days=total_days)).strftime('%Y-%m-%d %H_%M_%S')
        csv_data = format_data(acc_x[start_idx:], acc_y[start_idx:], acc_z[start_idx:], start_time + timedelta(days=total_days))

        file_path = os.path.join(output_dir, numeric_patient_id, 'accelerometer', f"{start_time_str}.csv")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'w') as f:
            f.write(csv_data)

def save_to_csv(acc_x, acc_y, acc_z, start_time, output_dir, patient_id):
    """
    Processes accelerometer data and saves it in daily CSV files.

    Parameters:
        acc_x, acc_y, acc_z (list): X, Y, Z axis accelerometer data.
        start_time (datetime): Start time of data collection.
        output_dir (str): Directory to save the processed CSV files.
        patient_id (str): Unique patient identifier.
    
    Notes:
        - Data is saved in daily chunks, with a new file for each day.
        - A partial day file is saved if the remaining samples meet the minimum threshold.
    """
    
    sampling_frequency = 25  # Sampling rate in Hz
    samples_per_day = 24 * 60 * 60 * sampling_frequency  # Total samples in one day
    min_samples_per_day = 1 * sampling_frequency  # Minimum samples for partial day

    total_samples = len(acc_x)
    total_days = total_samples // samples_per_day  # Full days of data
    remaining_samples = total_samples % samples_per_day  # Samples left over

    # Extract numeric part of the patient ID for filename consistency
    numeric_patient_id = patient_id.lstrip('R')
    numeric_patient_id = int(numeric_patient_id)
    numeric_patient_id = str(numeric_patient_id)    

    def format_data(arr1, arr2, arr3, start_time, sampling_rate=25):
        """
        Formats data into CSV-ready rows with timestamps and UTC times.

        Parameters:
            arr1, arr2, arr3 (list): Data for each axis (x, y, z).
            start_time (datetime): Timestamp for start of data.
            sampling_rate (int): Frequency at which data was sampled.
        
        Returns:
            str: Formatted CSV string with timestamps, UTC times, and accelerometer data.
        """
        
        # Prepare timestamps and UTC times
        timestamps = []
        utc_times = []
        accuracy = np.full(len(arr1), 'unknown')  # Placeholder for accuracy data
        
        current_time = start_time
        for i in range(len(arr1)):
            # Convert timestamp to milliseconds
            timestamps.append(int(current_time.timestamp() * 1000))
            # Format UTC time string to millisecond precision
            utc_times.append(current_time.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3])
            # Increment time by sampling interval
            current_time += timedelta(seconds=1 / sampling_rate)
        
        # Stack data into columns and format as strings
        formatted_data = np.column_stack((timestamps, utc_times, accuracy, arr1, arr2, arr3))
        formatted_data = [f"{row[0]},{row[1]},{row[2]},{row[3]},{row[4]},{row[5]}" for row in formatted_data]
        
        # Add column headers to the CSV output
        column_titles = "timestamp,UTC time,accuracy,x,y,z"
        formatted_data.insert(0, column_titles)
        
        return "\n".join(formatted_data)

    # Process each full day of data and save to individual CSV files
    for day in range(total_days):
        start_idx = day * samples_per_day
        end_idx = start_idx + samples_per_day

        # Generate filename with start time for each day's data
        start_time_str = (start_time + timedelta(days=day)).strftime('%Y-%m-%d %H_%M_%S')
        csv_data = format_data(
            acc_x[start_idx:end_idx],
            acc_y[start_idx:end_idx],
            acc_z[start_idx:end_idx],
            start_time + timedelta(days=day)
        )

        # Define file path and ensure directory exists
        file_path = os.path.join(output_dir, numeric_patient_id, 'accelerometer', f"{start_time_str}.csv")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write formatted data to CSV file
        with open(file_path, 'w') as f:
            f.write(csv_data)

    # Save any remaining data if it meets the minimum sample threshold
    if remaining_samples >= min_samples_per_day:
        start_idx = total_days * samples_per_day
        start_time_str = (start_time + timedelta(days=total_days)).strftime('%Y-%m-%d %H_%M_%S')
        csv_data = format_data(
            acc_x[start_idx:],
            acc_y[start_idx:],
            acc_z[start_idx:],
            start_time + timedelta(days=total_days)
        )

        # Define file path and ensure directory exists
        file_path = os.path.join(output_dir, numeric_patient_id, 'accelerometer', f"{start_time_str}.csv")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Write formatted data to CSV file
        with open(file_path, 'w') as f:
            f.write(csv_data)