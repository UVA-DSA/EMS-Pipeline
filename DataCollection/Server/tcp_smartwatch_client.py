import socket
import os
import csv
import time
from datetime import datetime
from multiprocessing import Queue
from typing import List
from threading import Event

def receive_smartwatch_data(server_ip: str, server_port: int, fifo_queue: Queue, recording_dir: str, smartwatch_id: str, thread_stop) -> None:
    try:
        # Create the necessary directories and CSV file for recording data
        columns: List[str] = ['sw_epoch_ms', 'wrist_position', 'sensor_type', 'value_X_Axis', 'value_Y_Axis', 'value_Z_Axis', 'seq_num', 'server_epoch_ms']
        curr_date = datetime.now()
        dt_string = curr_date.strftime("%d-%m-%Y-%H-%M-%S")
        newpath = os.path.join(recording_dir, f"smartwatch_data/sw_{smartwatch_id}/")
        
        os.makedirs(newpath, exist_ok=True)
        
        csv_file_path = os.path.join(newpath, 'sw_data.csv')
        
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            
            while True:
                client_socket = None
                connected = False
 
                while not connected:
                    if thread_stop.is_set():
                        break
                    try:
                        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        client_socket.settimeout(5)
                        print(f"[Smartwatch: Attempting to connect to server {server_ip}:{server_port}]")
                        client_socket.connect((server_ip, server_port))
                        connected = True
                        print(f"[Smartwatch: Successfully connected to server {server_ip}:{server_port}]")
                    except Exception as e:
                        print(f"[Smartwatch: Connection failed: {e}. Retrying in 5 seconds...]")
                        time.sleep(5)
                
                message = "Hello, Smart Watch!"
                
                if thread_stop.is_set():
                    print("[Smartwatch: Thread stop set, exiting..]")
                    break
                
                try:
                    while True:
                        try:
                            if thread_stop.is_set():
                                client_socket.close()
                                client_socket = None

                                file.flush()
                                file.close()
                        
                                print("[Smartwatch Receiver: Thread stop set, exiting..]")
                                break
                
                            # Send the message
                            client_socket.sendall(message.encode('utf-8'))

                            # Initialize a buffer to accumulate received data
                            buffer = b""
                            while True:
                                response = client_socket.recv(8192)
                                if not response:
                                    raise ConnectionError("[Smartwatch: Server closed the connection.]")
                                
                                buffer += response
                                
                                # Check if we received the "eof" marker
                                if b"eof" in buffer:
                                    break

                            # Decode the full buffer
                            received_message = buffer.decode('utf-8').replace("eof", "")
                            print(f"[Smartwatch Receiver: Received raw data: {received_message}]")

                            # Split the received message by semicolons to get individual data points
                            data_points = received_message.split(';')

                            curr_epoch_time = int(time.time_ns())

                            for data_point in data_points:
                                if data_point.strip():  # Check if the data_point is not empty
                                    sw_data = data_point.split(',')
                                    # print(sw_data, len(sw_data))
                                    # Check if the sw_data has the expected number of elements (8 columns)
                                    if len(sw_data) == 7:
                                        try:
                                            # Convert to proper types and append the server timestamp
                                            sw_data[0] = int(sw_data[0])  # sw_epoch_ms
                                            sw_data[3] = float(sw_data[3])  # value_X_Axis
                                            sw_data[4] = float(sw_data[4])  # value_Y_Axis
                                            sw_data[5] = float(sw_data[5])  # value_Z_Axis
                                            sw_data[6] = int(sw_data[6])  # seq_num
                                            sw_data.append(curr_epoch_time)

                                            # Write the data point to the CSV
                                            writer.writerow(sw_data)

                                            try:
                                                # Add it to the queue to be processed by the main process
                                                fifo_queue.put(sw_data, block=False)
                                            except Exception as e:
                                                print(f"Error when writing to the FIFO queue: {e}")
                                                while not fifo_queue.empty():
                                                    fifo_queue.get()
                                        except (ValueError, IndexError) as e:
                                            print(f"[Smartwatch Receiver: Error processing data point: {data_point}, {e}]")
                                            
                        except Exception as e:
                            print(f"[Error occurred while communicating with server: {e}]")
                            break  # Exit the inner loop and attempt to reconnect

                        except KeyboardInterrupt:
                            print("[Smartwatch receival interrupted by user. Exiting...]")
                            return

                except Exception as e:
                    print(f"[Smartwatch: Error: {e}]")
                
                except KeyboardInterrupt:
                    print("[Smartwatch: Smartwatch receival interrupted by user. Exiting...]")
                    return

                if client_socket:
                    client_socket.close()
                    print("[Smartwatch: Smartwatch connection closed!]")
                    time.sleep(5)

    except KeyboardInterrupt:
        print("[Smartwatch: Smartwatch receival interrupted by user. Exiting...]")
        return

# Example usage:
smartwatch_1_ip = '192.168.0.17'
smartwatch_port = 7889
smartwatch_1_id = 'right'
smartwatch_1_q = Queue()
thread_stop = Event()

receive_smartwatch_data(smartwatch_1_ip, smartwatch_port, smartwatch_1_q, './test/', smartwatch_1_id, thread_stop)
