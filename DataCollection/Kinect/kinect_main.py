import open3d as o3d
import datetime
from kinect_recorder import RecorderWithCallback  # Assuming the class is saved in recorder.py

def main():
    config = o3d.io.AzureKinectSensorConfig()
    filename = '{date:%Y-%m-%d-%H-%M-%S}.mkv'.format(date=datetime.datetime.now())
    device = 0
    align_depth_to_color = True

    recorder = RecorderWithCallback(config, device, filename, align_depth_to_color)
    
    # Start the visualization and recording in a background thread
    import threading
    recording_thread = threading.Thread(target=recorder.run)
    recording_thread.start()

    rootdir = 'D:/repos/EMS-Pipeline/DataCollection/DataCollected/26-08-2024/Keshara/BVM/0'
    # Control the recorder
    command = ""
    while command != "exit":
        command = input("Enter command (start, pause, stop, exit): ").strip().lower()
        if command == "start":
            recorder.start_recording(rootdir)
        elif command == "pause":
            recorder.pause_recording()
        elif command == "stop":
            recorder.stop_recording()

    recording_thread.join()

if __name__ == "__main__":
    main()
