import pandas
import os


csv = pandas.read_csv('smartGlass_synch.csv')


for index, row in csv.iterrows():
    
    part1 = str(row['participant']) 
    part2 = str(row['intervention'])  
    part3 = str(row['trial'])
    part4 = str(row['clip'])
    
    
    filename = f"{part1}_{part2}_t{part3}_c{part4}.csv"
    output_file = os.path.join('CSVs', filename)
    
    gopro_range = list(range(csv['gopro_sf'][index], csv['gopro_ef'][index] + 1))

    time = csv['start_ts'][index]
    timestamps = [csv['start_ts'][index]]
    for i in range(1, len(gopro_range)):
        time += 33333333
        timestamps.append(time)

    df = pandas.DataFrame(columns=['gopro_frames', 'sg_timestamps'])
    df['gopro_frames'] = gopro_range
    df['sg_timestamps'] = timestamps


    df.to_csv(output_file, index=False)





