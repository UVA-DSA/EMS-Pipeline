import pandas as pd
import numpy as np
import sys
from scipy.signal import find_peaks
import os

def detect_positive_peaks_with_timestamps(file_path, data_column, timestamp_column):
    
    df = pd.read_csv(file_path)

    data = df[data_column].values
    timestamps = df[timestamp_column]
    
    peaks, _ = find_peaks(data, distance=6, prominence=5)
    
    # Filter out only positive peaks (where the value of the peak is positive)
    positive_peaks = [peak for peak in peaks if data[peak] > 0]
    
    peak_timestamps = timestamps[positive_peaks]
    
    if len(peak_timestamps) < 2:
        print("ERROR: Not enough peaks to find differences, change peak detection parameters")
        sys.exit()
    
    time_differences = np.diff(peak_timestamps)
    
    time_differences_float = time_differences / 1_000_000_000
    
    # Drop 2 outliers on either end - too far or too short
    if len(time_differences_float) > 10:
        sorted_differences = np.sort(time_differences_float)
        filtered_differences = sorted_differences[2:-2]
    else:
        print(f"Error, not enough peaks detected:  {file_path}")
        sys.exit()
    
    avg_time_difference = np.mean(filtered_differences) if filtered_differences.size > 0 else 0
    std_dev_time_difference = np.std(filtered_differences) if filtered_differences.size > 0 else 0
    
    avg_rate = 60 / avg_time_difference if avg_time_difference > 0 else 0
    std_dev_rate = (60 / (avg_time_difference ** 2)) * std_dev_time_difference if avg_time_difference > 0 else 0
    
    return len(positive_peaks), avg_rate, std_dev_rate

def process_csv_files(directory, data_column, timestamp_column):
    results = []
    
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            try:
                num_peaks, avg_rate, std_dev_rate = detect_positive_peaks_with_timestamps(file_path, data_column, timestamp_column)
                results.append({
                    'File': filename,
                    'Average Rate (peaks/min)': avg_rate,
                    'Standard Deviation Rate (peaks/min)': std_dev_rate
                })
                print(f"File: {filename}, Number of Peaks: {num_peaks}, Average Rate (peaks/min): {avg_rate:.4f}, Std. deviation (peaks/min): {std_dev_rate:.4f}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    return results

directory_path = 'ToPlot'
data_column = 'value_X_Axis'
timestamp_column = 'sw_epoch_ms'
results = process_csv_files(directory_path, data_column, timestamp_column)

df_results = pd.DataFrame(results)

output_csv_path = 'peak_analysis_results.csv'
df_results.to_csv(output_csv_path, index=False)

print(f"\nResults have been saved to {output_csv_path}")
