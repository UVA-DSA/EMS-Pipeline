import pandas as pd
import matplotlib.pyplot as plt

class InteractivePlot:
    def __init__(self, csv_file, x_column, y_columns, gopro_ranges, gopro_frames):
        self.data = pd.read_csv(csv_file)
        self.x_column = x_column
        self.y_columns = y_columns

        self.highlight_indicies = []
        for start, end in gopro_ranges:
            start_index = int((start / gopro_frames) * len(self.data))
            end_index = int((end / gopro_frames) * len(self.data))
            self.highlight_indicies.append((start_index, end_index))

        self.current_offset = 0
        
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        
        for y_col in self.y_columns:
            self.ax.plot(self.data[self.x_column], self.data[y_col], label=f'{y_col} vs {x_column}')
        
        
        self.draw_highlights()

        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.ax.set_xlabel(x_column)
        self.ax.set_ylabel('Values')
        self.ax.set_title(f'Plot of {", ".join(self.y_columns)} vs {x_column} with Highlights')
        self.ax.legend()
        self.ax.grid()

        plt.show()

    def draw_highlights(self):
        self.ax.clear()  
        for y_col in self.y_columns:
            self.ax.plot(self.data[self.x_column], self.data[y_col], label=f'{y_col} vs {self.x_column}')
        
        
        for start, end in self.highlight_indicies:
            
            adjusted_start = start + self.current_offset
            adjusted_end = end + self.current_offset
            
            
            if 0 <= adjusted_start < len(self.data) and 0 <= adjusted_end < len(self.data):
                x_start = self.data[self.x_column].iloc[adjusted_start]
                x_end = self.data[self.x_column].iloc[adjusted_end]
                self.ax.axvspan(x_start, x_end, color='red', alpha=0.3)

        
        self.ax.set_xlabel(self.x_column)
        self.ax.set_ylabel('Values')
        self.ax.set_title(f'Plot of {", ".join(self.y_columns)} vs {self.x_column} with Highlights')
        self.ax.legend()
        self.ax.grid()
        self.fig.canvas.draw_idle()

    def on_key_press(self, event):
        # Move all highlights left or right
        if event.key == 'right':
            self.current_offset += 100  # Move right
            self.draw_highlights()
        elif event.key == 'left':
            self.current_offset -= 100  # Move left
            self.draw_highlights()
        
        elif event.key == 's':
            self.save_highlight_indices()

    def save_highlight_indices(self):
        current_indicies = []
        for start, end in self.highlight_indicies:
            start_index = start + self.current_offset
            end_index = end + self.current_offset
            current_indicies.append((start_index, end_index))  

       
        save_data = {
            'participant_trial': ['ng5_5'], 
            'Current Highlight Offsets': [self.current_offset],
            'Highlight Ranges': [current_indicies]
        }
        
        
        df = pd.DataFrame(save_data)

        df.to_csv('highlight_indices.csv', mode='a', header=False, index=False)
        
        print(f'Saved current highlight offset: {self.current_offset} to highlight_indices.csv')


if __name__ == "__main__":
    csv_file_path = 'CSVs/ng1_4/sw_data.csv'  
    x_col = 'server_epoch_ms'                  
    y_cols = ['value_X_Axis', 'value_Y_Axis', 'value_Z_Axis']  
    goPro_ranges = [ (390, 1190), (1500, 1900), (2200, 2400)]  # expected CPR frames
    goPro_frames = 2621

    InteractivePlot(csv_file_path, x_col, y_cols, goPro_ranges, goPro_frames)
