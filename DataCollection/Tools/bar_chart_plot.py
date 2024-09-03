import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv("average_rates.csv")


fig, ax = plt.subplots(figsize=(14, 8))

bar_width = 0.35
index = np.arange(len(data["File"]))


depth_bars = ax.bar(index - bar_width/2, data["Depth Sensor Average Rate (compressions/min)"], 
                    bar_width, yerr=data["Depth Sensor Standard Deviation Rate (peaks/min)"], 
                    label='Depth Sensor', color='blue', capsize=5)


smartwatch_bars = ax.bar(index + bar_width/2, data["Smartwatch Average Rate (compressions/min)"], 
                        bar_width, yerr=data["Smartwatch Standard Deviation Rate (peaks/min)"], 
                        label='Smartwatch', color='orange', capsize=5)


ax.set_xlabel('Files')
ax.set_ylabel('Average Rate (compressions/min)')
ax.set_title('Average Compression Rates with Standard Deviations')
ax.set_xticks(index)
ax.set_xticklabels(data["File"], rotation=90)
ax.legend()


plt.tight_layout()
plt.show()

