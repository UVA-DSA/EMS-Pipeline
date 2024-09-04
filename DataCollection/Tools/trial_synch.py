import os
import pandas as pd

base_folder_path = "CSVs"

# CSV file with offset values for each modality for each trial
df_input = pd.read_csv("synchronization.csv")

for idx, row in df_input.iterrows():
    result_rows = []
    trial_value = row['trial']
    print("working on: " + trial_value)
    sw_offset = row['sw_offset_seconds']
    depthsensor_offset = row['depthsensor_offset_seconds']
    depthcam_offset = row['depthcam_offset_frames'] 
    
    depthcam_frame = -depthcam_offset
    
    trial_folder_path = os.path.join(base_folder_path, str(trial_value))

    for file in os.listdir(trial_folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(trial_folder_path, file)
            if file == "sw_data.csv":
                sw_data_csv = pd.read_csv(file_path)
            elif file.startswith("depth"):
                depth_csv = pd.read_csv(file_path)
            elif file.startswith("GX"):
                gopro_csv = pd.read_csv(file_path)
    
    gopro_stamplist = gopro_csv['epoch'].astype('int64').tolist()
    sw_stamplist = sw_data_csv['server_epoch_ms'].astype('int64')
    depth_stamplist = depth_csv['epoch'].astype('int64')

    # Apply offsets to sw_data and depthsensor
    sw_offset_stamplist = [m + (sw_offset * 1000000000) for m in sw_stamplist]  
    depth_offset_stamplist = [n + (depthsensor_offset * 1000000000) for n in depth_stamplist]

    sw_index = 0  
    depth_index = 0  

    for i, gopro_epoch in enumerate(gopro_stamplist):  
        while sw_index < len(sw_offset_stamplist) - 1:
            if gopro_epoch >= sw_offset_stamplist[sw_index] and gopro_epoch < sw_offset_stamplist[sw_index+1]:
                sw_data_selected = {
                    'sw_value_X_Axis': sw_data_csv['value_X_Axis'].iloc[sw_index],
                    'sw_value_Y_Axis': sw_data_csv['value_Y_Axis'].iloc[sw_index],
                    'sw_value_Z_Axis': sw_data_csv['value_Z_Axis'].iloc[sw_index],
                    # 'sw_server_epoch_ms': sw_data_csv['server_epoch_ms'].iloc[sw_index],
                    # 'offset_sw_server_epoch_ms': sw_offset_stamplist[sw_index]
                }
                break
            else:
                if sw_index < len(sw_offset_stamplist) - 2:
                    sw_index += 1
                else:
                    sw_data_selected = {
                        'sw_value_X_Axis': 0,
                        'sw_value_Y_Axis': 0,
                        'sw_value_Z_Axis': 0,
                    #     'sw_server_epoch_ms': 0,
                    #     'offset_sw_server_epoch_ms': 0
                    }
                    sw_index = 0
                    break

        while depth_index < len(depth_offset_stamplist) - 1:
            if gopro_epoch >= depth_offset_stamplist[depth_index] and gopro_epoch < depth_offset_stamplist[depth_index+1]:
                depth_data_selected = {
                    'depth_value': depth_csv['value'].iloc[depth_index],
                    # 'original_depthsensor_epoch': depth_csv['epoch'].iloc[depth_index],
                    # 'offset_depthsensor_epoch': depth_offset_stamplist[depth_index]
                }
                break
            else:
                if depth_index < len(depth_offset_stamplist) - 2:
                    depth_index += 1
                else:
                    depth_data_selected = {
                        'depth_value': 0,
                        # 'original_depthsensor_epoch': 0,
                        # 'offset_depthsensor_epoch': 0
                    }
                    depth_index = 0
                    break

        if depthcam_frame < 1:
            depthcam_print = -1
        else:
            depthcam_print = depthcam_frame
        result_rows.append({
            'gopro_epoch': gopro_epoch,  
            'offset_depthcam_frame': depthcam_print,
            **sw_data_selected,
            **depth_data_selected,
        })
        
        depthcam_frame += 1

    # Save each trial's results to a CSV
    df_results = pd.DataFrame(result_rows)
    df_results.to_csv(f"{trial_value}.csv", index=False)  

print("Process complete.")
