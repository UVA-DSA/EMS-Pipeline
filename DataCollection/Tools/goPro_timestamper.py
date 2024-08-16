import pandas as pd
import os


# Read the CSV file
input_file = 'file.csv'  # Change this to your input CSV file
df = pd.read_csv(input_file)

# Process each row
for index, row in df.iterrows():
    # Generate frame timestamps
    frame_timestamps = [df['st'][index]]
    time = df['st'][index]
    for i in range(1, df['goProFrames'][index]):
        time += 33333333
        frame_timestamps.append(time)

    # Create a DataFrame from the timestamps
    output_df = pd.DataFrame()
    output_df['timestamps'] = frame_timestamps

    # Get the filename from the 'name' column
    filename = row['name'] + '.csv'

    # Save the output DataFrame to a new CSV file
    output_df.to_csv(filename, index=False)

    print(f"Generated timestamps for {row['name']} and saved to {filename}")
