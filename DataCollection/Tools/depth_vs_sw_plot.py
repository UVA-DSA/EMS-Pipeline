import os
import pandas as pd
import matplotlib.pyplot as plt


main_folder = 'CSVs' 


for subdir, _, files in os.walk(main_folder):

    
    for file in files:
        if file.endswith('.txt'):
            text_file_path = os.path.join(subdir, file)
        elif file.endswith('.csv'):
            csv_file_path = os.path.join(subdir, file)

    if text_file_path and csv_file_path:
        # Load data from the text file
        with open(text_file_path, 'r') as f:
            lines = f.readlines()

        # Extract the values from the text file 
        values = [float(line.split()[2]) for line in lines]  

        # Load data from the CSV file
        csv_data = pd.read_csv(csv_file_path) 
        csv_X_values = csv_data['value_X_Axis'].tolist()
        csv_Y_values = csv_data['value_Y_Axis'].tolist()
        csv_Z_values = csv_data['value_Z_Axis'].tolist()

       
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot values from the text file
        ax1.plot(range(len(values)), values, linestyle='-', color='b')
        ax1.set_title(f'Values from Depth Camera ({os.path.basename(text_file_path)})')
        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Value')
        ax1.grid()

        # Plot values from the CSV file
        ax2.plot(range(len(csv_X_values)), csv_X_values, linestyle='-', color='r', label='X Axis')
        ax2.plot(range(len(csv_Y_values)), csv_Y_values, linestyle='-', color='b', label='Y Axis')
        ax2.plot(range(len(csv_Z_values)), csv_Z_values, linestyle='-', color='g', label='Z Axis')
        ax2.set_title(f'Values from Smart Watch ({os.path.basename(csv_file_path)})')
        ax2.set_xlabel('Frame')
        ax2.set_ylabel('Value')
        ax2.grid()
        ax2.legend()

        
        plt.tight_layout()
        output_file_path = os.path.join(subdir, 'plot.png')  
        plt.savefig(output_file_path)
        plt.close()  

        
    