import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_data(csv_file, output_folder):
    df = pd.read_csv(csv_file, index_col=False)

    print(df.head(5))

    # Plot x, y, z axis values with seq_num on the x-axis
    plt.figure(figsize=(10, 6))
    plt.plot(df['seq_num'], df['value_X_Axis'], label='X Axis')
    plt.plot(df['seq_num'], df['value_Y_Axis'], label='Y Axis')
    plt.plot(df['seq_num'], df['value_Z_Axis'], label='Z Axis')

    # Adding labels and title
    plt.xlabel('Sequence Number')
    plt.ylabel('Accelerometer Values')
    plt.title('Smartwatch Accelerometer Data')
    plt.legend()

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


data_directory = '../Server/test/smartwatch_data/sw_right/' 
output_directory = 'Plots' 
process_csv_files(data_directory, output_directory)
