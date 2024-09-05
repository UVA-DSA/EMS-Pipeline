import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def plot_peaks(file_path, data_column):
    
    df = pd.read_csv(file_path)
    data = df[data_column].values
    
    
    
    x_values = np.arange(len(data))

    peaks, _ = find_peaks(data, distance=6, prominence=5)
    
    positive_peaks = [peak for peak in peaks if data[peak] > 0]
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_values, data, label='Data', color='blue')
    plt.scatter(x_values[positive_peaks], data[positive_peaks], color='red', marker='o', label='Peaks')
    plt.xlabel('Frame')
    plt.ylabel('Values')
    plt.title('Smartwatch Data with Identified Peaks')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage
file_path = 'ng1_2_cpr_2_clipped.csv'  # Replace with your file path
data_column = 'value_X_Axis'
plot_peaks(file_path, data_column)
