import serial
import time
import csv
from datetime import datetime
import threading
import os
from multiprocessing import Process, Event

class ArduinoVL6180Recorder(Process):
    def __init__(self, port, recording_dir, thread_stop, baudrate=115200):
        super().__init__()
        self.serial_port = port
        self.baudrate = baudrate
        self.recording_dir = os.path.join(recording_dir, "VL6180/")
        os.makedirs(self.recording_dir, exist_ok=True)
        self.thread_stop = thread_stop
        self.running = False
        self.ser = None
        self.csv_filename = os.path.join(self.recording_dir, f"VL6180_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    def start_recording(self):
        if not self.running:
            self.running = True
            print("Arduino: Recording started.")

    def stop_recording(self):
        if self.running:
            self.running = False
            print("Arduino: Recording stopped.")

    def run(self):
        # Set up the serial connection in the process
        try:
            self.ser = serial.Serial(self.serial_port, self.baudrate)
            time.sleep(2)  # Wait for the connection to establish
            print(f"Arduino: Connected to Arduino on port {self.serial_port} at {self.baudrate} baud.")

            record = False
            # Open the CSV file for writing
            with open(self.csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp (ns)", "Range (mm)", "Sequence"])  # Write header
                print(f"Arduino: Recording started to {self.csv_filename}")
                print(self.thread_stop.is_set(), self.running)
                while not self.thread_stop.is_set() and self.running:
                    if self.ser.in_waiting > 0:
                        try:
                            line = self.ser.readline().decode('utf-8').strip()
                            # print("Arduino: ", line)    
                            if not line.startswith("Start") and not record:
                                continue

                            record = True

                            if record:
                                if line.startswith("Range:"):
                                    try:
                                        range_value = int(line.split(":")[1])
                                        seq_num = int(line.split(":")[-1])
                                        timestamp_ns = int(datetime.now().timestamp() * 1e9)  # Get current time in ns
                                        writer.writerow([timestamp_ns, range_value, seq_num])
                                        # print(f"Recorded: {timestamp_ns}, {range_value} mm, seq:{seq_num}")
                                    except ValueError:
                                        print("Error parsing range value.")
                                else:
                                    #if line has string Seq
                                    if "Seq" in line:
                                        # print(f"Error message received: {line}")
                                        seq_num = int(line.split(":")[-1])
                                        # Write the message to the file with a placeholder value
                                        timestamp_ns = int(datetime.now().timestamp() * 1e9)  # Get current time in ns
                                        writer.writerow([timestamp_ns, -1, seq_num])

                        except Exception as e:
                            print(f"Error: {e}")    

                    time.sleep(0.001)  # Slight delay to avoid excessive CPU usage

            print("Arduino: recording stopped...")

        except Exception as e:
            print(f"Arduino: Error: {e}")

        finally:
            self.close()

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Arduino: Serial connection closed.")

    def stop(self):
        self.join()
        print("Arduino: Recording process stopped.")

# Example usage:

if __name__ == "__main__":
    thread_stop = Event()
    thread_stop.clear()
    recorder = ArduinoVL6180Recorder(port='COM5', recording_dir='./recordings', thread_stop=thread_stop)  # Replace with your actual port

    try:
        recorder.start_recording()
        recorder.start()
        time.sleep(10)  # Run for 10 seconds as an example
        recorder.stop_recording()
        thread_stop.set()  # Signal the thread to stop
        recorder.stop()  # Ensure the process is properly stopped
    finally:
        recorder.close()
