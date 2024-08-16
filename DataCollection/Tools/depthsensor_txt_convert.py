from datetime import datetime
import pandas as pd

def convert_txt_to_csv(input_txt_file, output_csv_file):
    timestamps = []
    values = []
    
    with open(input_txt_file, 'r') as file:
        for line in file:
            # Assuming each line is in the format "YYYY-MM-DD HH:MM:SS.ffffff value"
            parts = line.strip().split()
            if len(parts) == 3:  
                date_str, time_str, value_str = parts
                timestamps.append(date_str + " " + time_str)
                values.append(value_str)

    print("Timestamps:", timestamps)

    epoch_times = []
    for timestamp in timestamps:
            date_part, fractional_part = timestamp.split('.')
            epoch_time = int(datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S.%f").timestamp() * 1_000_000_000) + int(fractional_part)
            epoch_times.append(epoch_time)
      

    print("Epoch times:", epoch_times)

    # Create a DataFrame
    df = pd.DataFrame({
        'epoch': epoch_times,
        'value': values
    })

    # Save to CSV
    df.to_csv(output_csv_file, index=False)

# Example usage
convert_txt_to_csv('ng5_5.txt', 'ng5_5.csv')
