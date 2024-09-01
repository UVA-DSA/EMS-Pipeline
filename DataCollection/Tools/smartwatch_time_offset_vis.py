import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
file_path = '../Server/test/smartwatch_data/sw_right/sw_data.csv'
# file_path = '~/Downloads/sw_data.csv'
df = pd.read_csv(file_path)

# check if server_epoch_ms is in nanoseconds
if df['server_epoch_ms'].max() > 1e13:

    df['server_epoch_ms'] = df['server_epoch_ms'] // 1e6


# Calculate the offset between smartwatch epoch time and server epoch time
df['epoch_offset_ms'] = df['sw_epoch_ms'] - df['server_epoch_ms']

# check if seq_num column is not present and add it if not
if 'seq_num' not in df.columns:
    # add column for sequence number
    df['seq_num'] = range(0, len(df))

# Calculate mean, standard deviation, min, and max of the offset
mean_offset = df['epoch_offset_ms'].mean()
std_offset = df['epoch_offset_ms'].std()
min_offset = df['epoch_offset_ms'].max()
max_offset = df['epoch_offset_ms'].min()

# Set seaborn style
sns.set(style="whitegrid")

# Plot the epoch offset
plt.figure(figsize=(10, 6))
sns.lineplot(x=df['seq_num'], y=df['epoch_offset_ms'], label='Epoch Offset (ms)', color='blue')

# Add the stats as text on the plot
plt.text(0.05, 0.60, f"Mean Offset (ms): {mean_offset:.2f}\n"
                     f"Std Dev (ms): {std_offset:.2f}\n"
                     f"Min Offset (ms): {min_offset}\n"
                     f"Max Offset (ms): {max_offset}",
         horizontalalignment='left', verticalalignment='top', transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', alpha=0.8))

plt.savefig('./Plots/epoch_offset.png')

plt.xlabel('Sequence Number')
plt.ylabel('Offset (ms)')
plt.title('Epoch Offset Between Smartwatch and Server')
plt.legend()
plt.grid(True)
plt.show()

