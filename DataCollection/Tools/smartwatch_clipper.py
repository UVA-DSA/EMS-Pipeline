import pandas
import os

directory = 'CSVs'

#original data from rivanna for this trial
data = pandas.read_csv('sw_data.csv')

#Change the files in the CSV folder with each run, for each trial
#for example, put all NG1_*_t1_*_.csv files in CSVs and make the 
#smartwatch data file the corresponding trial csv for NG1
for filename in os.listdir(directory):
    
    file_path = os.path.join(directory, filename)

    df = pandas.read_csv(file_path)

    final_df = pandas.DataFrame(columns = data.columns)

    gopro_stamplist = df['sg_timestamps']

    print(gopro_stamplist)
    print(data['server_epoch_ms'])

    iterator = 0
    for i in gopro_stamplist:
        while True:
            try:
                if i >= data['server_epoch_ms'][iterator] and i < data['server_epoch_ms'][iterator+1]:
                    final_df = pandas.concat([final_df, data.iloc[[iterator]]], ignore_index=True)
                    break
                else:
                    iterator+=1
            except Exception as e:
                print(e)
                break
            print(i)

    clipped_name = f"{os.path.splitext(filename)[0]}_clipped.csv"

    final_df.to_csv(clipped_name, index = False)


    
            




