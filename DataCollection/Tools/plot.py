import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_data(csv_file, output_folder):
    df = pd.read_csv(csv_file)

    # Get the frame number
    frame = range(1, len(df) + 1)  

    plt.figure(figsize=(10, 6))
    plt.plot(frame, df['value_X_Axis'], label = 'X values')
    plt.plot(frame, df['value_Y_Axis'], label = 'Y values')
    plt.plot(frame, df['value_Z_Axis'], label = 'Z values')
    plt.title(f'Plot for {os.path.basename(csv_file)}')
    plt.xlabel('Frame')  
    plt.ylabel('Values')  
    
    plot_filename = os.path.join(output_folder, f"{os.path.basename(csv_file).replace('.csv', '.png')}")
    plt.savefig(plot_filename)
    plt.close()  



def process_csv_files(data_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

   
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            if file.endswith('.csv'):
                csv_file_path = os.path.join(root, file)
                plot_data(csv_file_path, output_directory)


data_directory = 'ToPlot' 
output_directory = 'Plots' 
process_csv_files(data_directory, output_directory)
