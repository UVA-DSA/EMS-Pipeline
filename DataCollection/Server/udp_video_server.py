import socket
import base64

def start_udp_server(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((ip, port))

    print(f"UDP server up and listening at {ip}:{port}")

    while True:
        data, addr = sock.recvfrom(65536)  # Adjust buffer size if necessary
        message = data.decode('utf-8')

        # Strip the <start> and <end> tags
        if message.startswith("<start>") and message.endswith("<end>"):
            base64_image = message[len("<start>"):-len("<end>")]

            # Decode the base64 string
            image_data = base64.b64decode(base64_image)

            # Save the image
            with open("received_image.png", "wb") as image_file:
                image_file.write(image_data)

            print(f"Image received and saved from {addr}")
        else:
            print(f"Received message from {addr} does not have the correct format.")


if __name__ == "__main__":
    # Set the IP address and port
    SERVER_IP = "0.0.0.0"  # Listen on all available interfaces
    SERVER_PORT = 8899     # Port number

    start_udp_server(SERVER_IP, SERVER_PORT)
