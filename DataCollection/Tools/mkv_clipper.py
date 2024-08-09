import cv2
import os
from moviepy.editor import VideoFileClip

# Open the video file
video_path = '2024-05-23-20-36-20.mkv'
txt_path = '2024-05-23-20-36-20.txt'
name = "NG5-cpr-t1-c1"
csv = 'depth_camera.csv'
cap = cv2.VideoCapture(video_path)

skip_frames = 10

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0
start_frame = 0
end_frame = frame_count - 1

# Get the base name of the video file without extension
base_name = os.path.splitext(os.path.basename(video_path))[0]
print(f"Processing Video: {base_name} ({frame_count} frames)")

# Define the codec for video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for MKV (XVID is commonly used)

def display_frame(frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number}.")
        return
    cv2.putText(frame, f'Frame: {frame_number}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('Video', frame)

def save_clip_with_audio(start_frame, end_frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"Saving clip .", end='', flush=True)

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {start_frame}.")
        return
    
    output_file = f"{base_name}_clipped_with_audio_start_{start_frame}_end_{end_frame}.mkv"
    
    # Use moviepy to handle video and audio
    with VideoFileClip(video_path) as video:
        fps = video.fps
        start_time = (start_frame / fps)
        end_time = (end_frame / fps)
        
        clip = video.subclip(start_time, end_time)
        clip.write_videofile(output_file, codec='libx264', audio_codec='aac')
    
    print(f"\nClip saved as '{output_file}'")


def save_frame(current_frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {current_frame}.")
        return
    
    cv2.imwrite(f"{base_name}_frame_{current_frame}.jpg", frame)

def get_timestamps_set_csv(start_frame, end_frame):
    global name
    with open(csv, 'r') as file:
        lines = file.readlines()
        start_timestamp = lines[start_frame - 1].strip() 
        end_timestamp = lines[end_frame - 1].strip()

    new_row = [name, start_frame, start_timestamp, end_frame, end_timestamp]

    with open(csv, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(new_row) 
        
        

# Initial display
display_frame(current_frame)

while True:
    key = cv2.waitKey(0)
    if key == 120:  # X key to exit
        break
    elif key == 97:  # A key
        current_frame = max(0, current_frame - skip_frames)
        display_frame(current_frame)
    elif key == ('c'):  # C key
        current_frame = min(frame_count - 1, current_frame + 2)
        display_frame(current_frame)
    elif key == 100:  # D key
        current_frame = min(frame_count - 1, current_frame + skip_frames)
        display_frame(current_frame)
    elif key == ord('z'):  # Z key, super boost button
        current_frame = min(frame_count - 1, current_frame + 50)
        display_frame(current_frame)
    elif key == ord('s'):  # 's' key to set start frame
        start_frame = current_frame
        print(f"Start frame set to {start_frame}")
    elif key == ord('e'):  # 'e' key to set end frame
        end_frame = current_frame
        print(f"End frame set to {end_frame}")
    # elif key == ord('c'):  # 'c' key to save the clip
        #save_clip_with_audio(start_frame, end_frame)
        # get_timestamps_set_csv(start_frame, end_frame)
    elif key == ord('f'):  # 'f' key to save the frame
        save_frame(current_frame)

cap.release()
cv2.destroyAllWindows()
