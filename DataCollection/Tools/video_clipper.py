import cv2
import os

# Open the video file
video_path = '/home/kesharaw/Desktop/datasets/EMS Action Dataset/Data Collection/Interventions/North Garden/05-23-2024/GoPro/GX010314.MP4'
cap = cv2.VideoCapture(video_path)

skip_frames = 10

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
current_frame = 0

# Get the base name of the video file without extension
base_name = os.path.splitext(os.path.basename(video_path))[0]
print(f"Processing Video: {base_name} ({frame_count} frames)")

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4

def display_frame(frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {frame_number}.")
        return
    cv2.putText(frame, f'Frame: {frame_number}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('Video', frame)

def save_clip(start_frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"Saving clip .", end='', flush=True)

    ret, frame = cap.read()
    if not ret:
        print(f"Error: Could not read frame {start_frame}.")
        return
    
        
    # Get the frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_file = f"{base_name}_clipped.mp4"
    out = cv2.VideoWriter(output_file, fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))
    
    count = 0
    while ret:
        out.write(frame)
        ret, frame = cap.read()
        count += 1
        
        if(count % 30 == 0):
            print('.', end='', flush=True)
    
    out.release()
    print(f"\nClip saved as '{output_file}'")

# Initial display
display_frame(current_frame)

while True:
    key = cv2.waitKey(0)
    if key == 27:  # ESC key to exit
        break
    elif key == 81:  # Left arrow key
        current_frame = max(0, current_frame - skip_frames)
        display_frame(current_frame)
    elif key == 83:  # Right arrow key
        current_frame = min(frame_count - 1, current_frame + skip_frames)
        display_frame(current_frame)
    elif key == ord('s'):  # 's' key to save the clip
        save_clip(current_frame)

cap.release()
cv2.destroyAllWindows()
