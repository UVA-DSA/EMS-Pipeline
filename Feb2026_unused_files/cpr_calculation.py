import matplotlib.pyplot as plt
from scipy.signal import find_peaks,find_peaks_cwt,peak_widths
from scipy import signal
import datetime as dt
import collections
import pylab
import numpy as np

# functions for smartwatch activity

def thresholding_algo(y, lag, threshold, influence):
    signals = np.zeros(len(y))
    filteredY = np.array(y)
    avgFilter = [0]*len(y)
    stdFilter = [0]*len(y)
    avgFilter[lag - 1] = np.mean(y[0:lag])
    stdFilter[lag - 1] = np.std(y[0:lag])
    for i in range(lag, len(y)):
        if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
            if y[i] > avgFilter[i-1]:
                signals[i] = 1
            else:
                signals[i] = -1

            filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
        else:
            signals[i] = 0
            filteredY[i] = y[i]
            avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
            stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

    return dict(signals = np.asarray(signals),
                avgFilter = np.asarray(avgFilter),
                stdFilter = np.asarray(stdFilter))

def robust_peaks_detection_zscore(df,lag,threshold,influence):
    rate = thresholding_algo(df['value_Magnitude_XYZ'],lag,threshold,influence)
    indices = np.where(rate['signals'] == 1)[0]
    robust_peaks_time = df.iloc[indices]['sw_epoch_ms']
    robust_peaks_value = df.iloc[indices]['value_Magnitude_XYZ']
    indices = np.where(rate['signals'] == -1)[0]
    robust_valleys_time = df.iloc[indices]['sw_epoch_ms']
    robust_valleys_value = df.iloc[indices]['value_Magnitude_XYZ']
    # # #Plotting
    # fig = plt.figure()
    # ax = fig.subplots()
    # ax.plot(acc_df[‘EPOCH_Time_ms’].tolist(),acc_df[‘Value_Magnitude_XYZ’].tolist())
    # ax.scatter(robust_valleys_time, robust_valleys_value, color = ‘gold’, s = 15, marker = ‘v’, label = ‘Minima’)
    # ax.scatter(robust_peaks_time, robust_peaks_value, color = ‘b’, s = 15, marker = ‘X’, label = ‘Robust Peaks’)
    # ax.legend()
    # ax.grid()
    # plt.show()
    return [robust_peaks_time,robust_peaks_value,robust_valleys_time,robust_valleys_value]

"""
Function to find the peaks and valleys
1. uses scipy find_peaks implementation.
2. paramter tuning is highly important !!!
3. todo
"""
# def find_peaks_valleys_cwt(df):
#     peaks = find_peaks_cwt(df['value_Magnitude_XYZ'],np.arange(100,2000))
#     print(peaks)
#     height = peaks[1][‘peak_heights’] #list of the heights of the peaks
#     peak_pos = data_frame.iloc[peaks[0]] #list of the peaks positions
#     # #Finding the minima
#     y2 = df[‘Value_Magnitude_XYZ’]*-1
#     minima = find_peaks(y2,height = -5, distance = 1)
#     min_pos = data_frame.iloc[minima[0]] #list of the minima positions
#     min_height = y2.iloc[minima[0]] #list of the mirrored minima heights
    # return peaks
"""
Function to find the peaks and valleys
1. uses scipy find_peaks implementation.
2. paramter tuning is highly important !!!
3. todo
"""
def find_peaks_valleys(df,height,distance,prominence):
    # print("find_peaks",df)
    peaks = find_peaks(df['value_Magnitude_XYZ'], height = height,  distance = distance,prominence=prominence)
    height = peaks[1]['peak_heights'] #list of the heights of the peaks
    peak_pos = df.iloc[peaks[0]] #list of the peaks positions
    # #Finding the minima
    y2 = df['value_Magnitude_XYZ']*-1
    minima = find_peaks(y2,height = -5, distance = 1)
    min_pos = df.iloc[minima[0]] #list of the minima positions
    min_height = y2.iloc[minima[0]] #list of the mirrored minima heights
    # print("Peaks",peaks)
    return [peak_pos,min_pos,height,min_height]
"""
Function to calculate the CPR rate
1. finds time differences between peaks and return the average rate per minute
"""
def find_cpr_rate(peaks):
    time_diff_between_peaks=np.diff(peaks['sw_epoch_ms'])
    is_not_empty=len(time_diff_between_peaks) > 0
    if is_not_empty:
        avg_time_btwn_peaks_in_seconds_scipy = np.average(time_diff_between_peaks)/1000
        # print ("Average time between peaks in seconds (scipy): ", str(avg_time_btwn_peaks_in_seconds_scipy))
        # print("CPR Rate Per Minute (scipy): ", (1/avg_time_btwn_peaks_in_seconds_scipy)*60)
        return [(avg_time_btwn_peaks_in_seconds_scipy), ((1/avg_time_btwn_peaks_in_seconds_scipy)*60)]
    return [0,0]

""" 
Function to preprocess data
1. calculate magnitude of x,y,z values combined  sqrt(x^2 + y^2 + z^2) * eliminates effects from orientations
2. todo
"""
def preprocess_data(df):
    magnitude_xyz_df = np.sqrt(np.square(df[['value_X_Axis','value_Y_Axis','value_Z_Axis']]).sum(axis=1))
    df['value_Magnitude_XYZ'] = magnitude_xyz_df
    return df


def vid_streaming_Cpr(y_vals, image_times):
    mean=np.convolve(y_vals, np.ones(50)/50, mode='valid')
    mean=np.pad(mean,(len(y_vals)-len(mean),0),'edge')
    #normalize by removing mean 
    wrist_data_norm=y_vals-mean
    #detect peaks for hand detection
    peaks, _ = find_peaks(wrist_data_norm, height=0.002)
    peak_times = np.take(image_times, peaks)

    #find time difference between peaks and calculate cpr rate
    time_diff_between_peaks=np.diff(peak_times)
    avg_time_btwn_peaks_in_seconds = np.average(time_diff_between_peaks)/1000
    
    # print ("Average time between peaks in seconds video): ", str(avg_time))
    # print("CPR Rate Per Minute (video): ", cpr_rate)

    return avg_time_btwn_peaks_in_seconds