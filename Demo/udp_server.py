

import socket
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks,find_peaks_cwt,peak_widths
from scipy import signal
import datetime as dt
import collections
"""
Function to find the peaks and valleys using the robust method using z-score and dispersion
1. uses Brakel, J.P.G. van (2014). “Robust peak detection algorithm using z-scores”. Stack Overflow. Available at: https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data/22640362#22640362 (version: 2020-11-08).
2. paramter tuning is highly important !!!
"""
def robust_peaks_detection_zscore(df,lag,threshold,influence):
    rate = thresholding_algo(df['Value_Magnitude_XYZ'],lag,threshold,influence)
    indices = np.where(rate['signals'] == 1)[0]
    robust_peaks_time = df.iloc[indices]['EPOCH_Time_ms']
    robust_peaks_value = df.iloc[indices]['Value_Magnitude_XYZ']
    indices = np.where(rate['signals'] == -1)[0]
    robust_valleys_time = df.iloc[indices]['EPOCH_Time_ms']
    robust_valleys_value = df.iloc[indices]['Value_Magnitude_XYZ']
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
def find_peaks_valleys_cwt(df):
    peaks = find_peaks_cwt(df['Value_Magnitude_XYZ'],np.arange(100,2000))
    print(peaks)
#     height = peaks[1][‘peak_heights’] #list of the heights of the peaks
#     peak_pos = data_frame.iloc[peaks[0]] #list of the peaks positions
#     # #Finding the minima
#     y2 = df[‘Value_Magnitude_XYZ’]*-1
#     minima = find_peaks(y2,height = -5, distance = 1)
#     min_pos = data_frame.iloc[minima[0]] #list of the minima positions
#     min_height = y2.iloc[minima[0]] #list of the mirrored minima heights
    return peaks
"""
Function to find the peaks and valleys
1. uses scipy find_peaks implementation.
2. paramter tuning is highly important !!!
3. todo
"""
def find_peaks_valleys(df,height,distance,prominence):
    peaks = find_peaks(df['Value_Magnitude_XYZ'], height = height,  distance = distance,prominence=prominence)
    height = peaks[1]['peak_heights'] #list of the heights of the peaks
    peak_pos = data_frame.iloc[peaks[0]] #list of the peaks positions
    # #Finding the minima
    y2 = df['Value_Magnitude_XYZ']*-1
    minima = find_peaks(y2,height = -5, distance = 1)
    min_pos = data_frame.iloc[minima[0]] #list of the minima positions
    min_height = y2.iloc[minima[0]] #list of the mirrored minima heights
    return [peak_pos,min_pos,height,min_height]
"""
Function to calculate the CPR rate
1. finds time differences between peaks and return the average rate per minute
"""
def find_cpr_rate(peaks):
    time_diff_between_peaks=np.diff(peaks['EPOCH_Time_ms'])
    is_not_empty=len(time_diff_between_peaks) > 0
    if is_not_empty:
        avg_time_btwn_peaks_in_seconds_scipy = np.average(time_diff_between_peaks)/1000
        # # printing difference list
        print ("Average time between peaks in seconds (scipy): ", str(avg_time_btwn_peaks_in_seconds_scipy))
        print("CPR Rate Per Minute (scipy): ", (1/avg_time_btwn_peaks_in_seconds_scipy)*60)

localIP     = "127.0.0.1"
localPort   = 20002
bufferSize  = 1024
msgFromServer       = "Hello UDP Client"
bytesToSend         = str.encode(msgFromServer)
# Create a datagram socket
UDPServerSocket = socket.socket(family=socket.AF_INET, type=socket.SOCK_DGRAM)
# Bind to address and ip
UDPServerSocket.bind((localIP, localPort))
print("UDP server up and listening")
buffer = collections.deque([])
# Listen for incoming datagrams
while(True):
    bytesAddressPair = UDPServerSocket.recvfrom(bufferSize)
    message = bytesAddressPair[0]
    address = bytesAddressPair[1]
    # clientMsg = “Message from Client:{}“.format(message.decode())
    clientIP  = "Client IP Address:{}".format(address)
    buffer.append(message.decode())
    current_buffer_size = len(buffer)
    print(current_buffer_size)
    print(clientIP)
    if current_buffer_size%1000 == 0:
        data_frame = pd.DataFrame(list(buffer))
        print(data_frame)
#         peaks,valleys,height,min_height = find_peaks_valleys(data_frame,height=32.5,distance=1,prominence=1)
#         find_cpr_rate(peaks)