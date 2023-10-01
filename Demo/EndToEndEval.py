from Pipeline import Pipeline
import pipeline_config
import csv

# -------------------------

if __name__ == '__main__':
    
    for recording in pipeline_config.recordings_to_test:
        # field names 
        notes = [f'average over {pipeline_config.num_trials_per_recording} runs']
        fields = ['time audio->transcript', 'transcript', 'whisper confidence', 'time protocol input->output', 'prediction', 'confidence'] 
        final_rows = None

        for i in range(1,pipeline_config.num_trials_per_recording+1):
            # data rows of csv file for this run
            pipeline_config.curr_segment = []
            pipeline_config.rows_trial = []
            Pipeline(recording=recording)
    
            # Write data to csv
            with open (f'evaluation_results/{recording}-eval-{i}.csv', 'w') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(fields)
                writer.writerows(pipeline_config.rows_trial)
        

    
