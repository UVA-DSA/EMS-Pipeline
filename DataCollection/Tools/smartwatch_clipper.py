import pandas
import os



for clip in os.listdir('Clips'):

#original data from rivanna for this trial
    clip_path = os.path.join('Clips', clip)
    clip_data = pandas.read_csv(clip_path)
    clip_prefix = '_'.join(clip.split('_')[:2])  # Get 'ng5_1'
    target_folder = os.path.join('CSVs', clip_prefix)
    target_csv = os.path.join(target_folder, 'sw_data.csv')
    
    if os.path.exists(target_csv):
        data = pandas.read_csv(target_csv)
    else:
        print("Error! Wrong path: " +  target_csv)

#Change the files in the CSV folder with each run, for each trial
#for example, put all NG1_*_t1_*_.csv files in CSVs and make the 
#smartwatch data file the corresponding trial csv for NG1


    final_df = pandas.DataFrame(columns = data.columns)

    gopro_stamplist = clip_data['epoch']

    print(gopro_stamplist)
    # print(data['sw_epoch_ms'])

    iterator = 0
    for i in gopro_stamplist:
        while True:
            try:
                if i >= data['server_epoch_ms'][iterator] and i < data['server_epoch_ms'][iterator+1]:
                    print(i, data['server_epoch_ms'][iterator])
                    final_df = pandas.concat([final_df, data.iloc[[iterator]]], ignore_index=True)
                    break
                else:
                    iterator+=1
            except Exception as e:
                print(e)
                break
            print(i)

    clipped_name = f"{os.path.splitext(clip)[0]}_clipped.csv"

    final_df.to_csv(clipped_name, index = False)


    
            




