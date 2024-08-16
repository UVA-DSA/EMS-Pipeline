import pandas
import os




output_folder = 'extracted_clips' 



clips = pandas.read_csv('goProFrames.csv')

for index, row in clips.iterrows():
    start_index = row['gp Start Frame']
    end_index = row['gp End Frame']
    clip_name = row['Participant_Trial_intervention']


    source_file_prefix = clip_name.split('_')[0:2]  # Get the prefix (e.g., 'ng1_2')
    source_file_name = '_'.join(source_file_prefix) + '.csv'

    source = pandas.read_csv(source_file_name)

    clip = pandas.DataFrame()
    clip['frame'] = range(start_index, end_index + 1)
    clip['epoch'] = source['epoch'].iloc[start_index:end_index + 1].reset_index(drop=True)

    clip_file_path = f"{clip_name}.csv"
    clip_file_path = os.path.join('Clipped', f"{clip_name}.csv")
    clip.to_csv(clip_file_path, index=False)

    print(f"Extracted clip '{clip_name}' from source file '{source_file_name}' from index {start_index} to {end_index}.")







