import socket
import os
import csv
import time
from datetime import datetime



def receive_data(server_ip, server_port, recording_dir, commandqueue):
    # Create a socket object
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


    try:
        # Connect to the server
        client_socket.connect((server_ip, server_port))
        
        columns = ['sw_epoch_ms','wrist_position','sensor_type','value_X_Axis','value_Y_Axis','value_Z_Axis','server_epoch_ms']

        curr_date = datetime.now()
        dt_string = curr_date.strftime("%d-%m-%Y-%H-%M-%S")

        newpath = f"{recording_dir}/smartwatch_data/"
        
        message = "Hello, Smart Watch!"
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        
        with open(newpath+'sw_data.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)

            while True:
                try:
                    # Send the message
                    client_socket.sendall(message.encode('utf-8'))

                    # Receive response from the server (optional)
                    response = client_socket.recv(1024)
                    
                    sw_data = response.decode('utf-8').split(',')

                    curr_epoch_time = int(time.time_ns())
                    
                    # convert to proper types
                    sw_data[0] = int (sw_data[0])
                    sw_data[3] = float (sw_data[3])
                    sw_data[4] = float (sw_data[4])
                    sw_data[5] = float (sw_data[5])
                    
                    sw_data.append(curr_epoch_time) #check

                    writer.writerow(sw_data)
                    
                    print(f"Received from server: {sw_data}")
                 
                    if(not commandqueue.empty() ):
                        command = commandqueue.get()
                        print("Command received: ", command)
                        if(command == "stop"):
                            break   
            
                except Exception:
                    print("Error occured...")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Close the socket
        client_socket.close()


# # Call the function to send the message
# send_message(server_ip, server_port, message_to_send)
