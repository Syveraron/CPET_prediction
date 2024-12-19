
import pandas as pd
import numpy as np
import polars as pl
import time
import matplotlib.pyplot as plt


def average_hr_30s(hr_values):
    """
    Convert 10-second HR values to 30-second averages.
    
    Args:
        hr_values (list): List of HR values, each representing a 10-second period.
    
    Returns:
        list: Averaged HR values for each 30-second period.
    """
    hr_30s_values = []

    # Process HR values in groups of 3 (each representing 10 seconds)
    for i in range(0, len(hr_values), 3):
        group = hr_values[i:i+3]

        # Filter out zero values unless all values are zero
        non_zero_values = [val for val in group if val != 0]

        if non_zero_values:
            hr_30s_values.append(np.median(non_zero_values))
        else:
            hr_30s_values.append(0)  # Append 0 if all are zero

    return hr_30s_values

def calculate_sdann_hr24(hr_values, segment_duration):
    """
    Calculate SDANN_HR24 from heart rate (HR) values in bpm.

    Parameters:
    - hr_values: A list or numpy array of HR values (in bpm) for each minute of a 24-hour period.
    - segment_duration: The duration of each segment in minutes.

    Returns:
    - sdann_hr24: The SDANN_HR24 value in ms.
    """
    # Number of HR values per segment
    values_per_segment = segment_duration

    # Convert HR values to ANN (Average NN intervals in seconds)
    ann_values = 60 / hr_values

    # Calculate mean ANN for each segment
    mean_ann_segments = [
        np.mean(ann_values[i:i + values_per_segment])
        for i in range(0, len(ann_values), values_per_segment)
    ]

    # Calculate SDANN_HR24 as the standard deviation of the mean ANN values across all segments
    sdann_hr24 = np.std(mean_ann_segments) * 1000  # Convert to milliseconds

    return sdann_hr24

def impute_missing_hr(hr_values, max_gap_duration=60):
    """
    Impute missing HR values (0s) with linear interpolation if the gap is less than max_gap_duration.

    Parameters:
    - hr_values: A numpy array of heart rate values where each element represents a 10-second period.
    - max_gap_duration: Maximum gap duration (in 10s periods) for linear interpolation (default is 60 periods = 10 minutes).

    Returns:
    - hr_values: The numpy array with interpolated values where appropriate.
    """

    # Convert to pandas Series for easy interpolation
    hr_series = pd.Series(hr_values)

    # Detect where values are zero (assumed to be missing)
    missing = hr_series == 0

    # Create groups of consecutive zeros (missing values)
    missing_groups = (missing != missing.shift()).cumsum()
    
    # Interpolate missing values in groups of zeros less than max_gap_duration
    for group in missing_groups[missing].unique():
        group_size = missing_groups[missing_groups == group].size
        if group_size <= max_gap_duration:  # Only interpolate small gaps
            hr_series[missing_groups == group] = np.nan

    # Use linear interpolation to fill the gaps
    hr_series.interpolate(method='linear', inplace=True)

    # Fill remaining NaNs (which were not interpolated due to large gaps) with zeros or leave them as NaNs
    hr_series.fillna(0, inplace=True)

    return hr_series.values



def upsample_acc_df(acc_df):
    new_rows = []
    
    for index, row in acc_df.iterrows():
        # Get the original time and add two new times at 10 and 20 seconds intervals
        original_time = pd.to_datetime(row['time'].split(" [")[0])  # Clean the time string
        
        for i in range(3):  # For 0s, 10s, and 20s
            new_row = row.copy()
            new_time = original_time + pd.Timedelta(seconds=i * 10)
            new_row['time'] = new_time.strftime('%Y-%m-%d %H:%M:%S.%f%z')
            new_rows.append(new_row)

    # Create the upsampled dataframe
    upsampled_df = pd.DataFrame(new_rows)
    return upsampled_df

def find_sleep_period(acc_df):
    # Ensure the time column is in datetime format
    #acc df is a value every 10 s
    acc_df['time'] = pd.to_datetime(acc_df['time'], errors='coerce', utc=True)

    def find_period(search_for_sleep=True):
        best_period = None
        min_hr_zeros = float('inf')  # Initialize with a very high number of HR zeros
        
        # Iterate over all potential 90-row periods (15 minutes)
        for i in range(len(acc_df) - 90 + 1):
            sleep_period = acc_df['sleep'].iloc[i:i + 90]
            sedentary_period = acc_df['sedentary'].iloc[i:i + 90]
            hr_period = acc_df['HR'].iloc[i:i + 90]
            time_period = acc_df['time'].iloc[i:i + 90]
            
            # Get the start time of the period
            start_time = time_period.iloc[0]
            if pd.isnull(start_time):  # Handle any missing/invalid time values
                continue

            # Check if the time is between 11 PM and 8 AM
            if not (start_time.hour >= 23 or start_time.hour < 8):
                continue  # Skip if outside of the time range
            
            # If searching for sleep, require at least 81 rows of sleep
            if search_for_sleep:
                if (sleep_period == 1).sum() < 81:
                    continue  # Skip if insufficient sleep
            else:
                # If searching for a mix of sleep and sedentary, check for at least 81 rows
                if (sleep_period == 1).sum() + (sedentary_period == 1).sum() < 81:
                    continue  # Skip if there are not enough sleep/sedentary rows combined

            # Count the number of HR zeros in the period
            hr_zeros = (hr_period == 0).sum()

            # If this period has fewer HR zeros than the current best, update best_period
            if hr_zeros < min_hr_zeros:
                min_hr_zeros = hr_zeros
                best_period = (i, i + 90)
        
        return best_period
    
    # First try to find a sleep period
    best_sleep_period = find_period(search_for_sleep=True)
    
    if best_sleep_period is not None:
        return best_sleep_period  # Return the best sleep period if found

    # If no sleep period is found, search for a mix of sleep or sedentary period
    best_sedentary_period = find_period(search_for_sleep=False)
    return best_sedentary_period  # Return the best sleep/sedentary period (if found)



def resample_hr_data(hr_data, chunk_size=6):
    """
    Resample HR data to a specific interval by taking the mean of each chunk, excluding zeros.

    Parameters:
        hr_data (numpy array): The input HR data.
        chunk_size (int): The size of each chunk for resampling. Default is 6.

    Returns:
        list: Resampled HR data.
    """
    hr_resampled = []
    for i in range(0, len(hr_data), chunk_size):
        # Take each chunk
        chunk = hr_data[i:i+chunk_size]
        # Filter out zeros
        non_zero_chunk = chunk[chunk != 0]
        # Calculate mean
        median = np.mean(non_zero_chunk) if len(non_zero_chunk) > 0 else 0
        hr_resampled.append(median)
    return hr_resampled



# Ensure hr_30s_values and acc_df have the same length
def align_hr_and_acc(hr_30s_values, acc_df):
    """
    Aligns HR values with accelerometer data by trimming or padding the HR values.

    Parameters:
        hr_30s_values (list): List of heart rate values.
        acc_df (DataFrame): Accelerometer data as a pandas DataFrame.

    Returns:
        DataFrame: Accelerometer data with an added HR column.
    """
    if len(hr_30s_values) > len(acc_df):
        # Trim HR values if they are longer than accelerometer data
        trim_amount = len(hr_30s_values) - len(acc_df)
        hr_30s_values = hr_30s_values[:len(acc_df)]
        print(f'Trimmed {trim_amount} values')
    elif len(hr_30s_values) < len(acc_df):
        # Pad HR values with NaN if they are shorter
        padded_amount = len(acc_df) - len(hr_30s_values)
        hr_30s_values += [np.nan] * padded_amount
        print(f'Padded {padded_amount} values')

    acc_df['HR'] = hr_30s_values
    return acc_df

# Extract sleep-related data between 03:00 and 07:00
def extract_sleep_data(acc_df):
    """
    Extract non-zero HR values during sleep or sedentary periods between 03:00 and 07:00.

    Parameters:
        acc_df (DataFrame): Accelerometer data with a time column and HR values.

    Returns:
        DataFrame: Filtered DataFrame with sleep or sedentary rows and non-zero HR values.
    """
    # Ensure 'time_cleaned' column is in datetime format
    acc_df['time_cleaned'] = pd.to_datetime(acc_df['time'].str.split("[").str[0])

    # Extract the time of day as a datetime.time object
    acc_df['time_of_day'] = acc_df['time_cleaned'].dt.time

    # Filter rows where the time is between 03:00 and 07:00
    sleep_df = acc_df[(acc_df['time_of_day'] >= pd.to_datetime('03:00').time()) & 
                      (acc_df['time_of_day'] <= pd.to_datetime('07:00').time())]

    # Further filter rows where 'sleep' or 'sedentary' is true
    sleep_df = sleep_df[(sleep_df['sleep'] == 1) | (sleep_df['sedentary'] == 1)]

    return sleep_df