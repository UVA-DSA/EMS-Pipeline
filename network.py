from Network.server_socket import server_network, listener

from multiprocessing import Queue

receiving_video_socket_queue = Queue()
receiving_audio_socket_queue = Queue()

server = server_network(IP="0.0.0.0",PORT=12345,WEB_PORT=8080,video_queue=receiving_video_socket_queue,audio_queue=receiving_audio_socket_queue)
audio_video_listener = listener(IP="0.0.0.0",SOCKET_PORT=12345,SERVER=server.return_server())
server.set_listener(audio_video_listener)
server.setup_server()