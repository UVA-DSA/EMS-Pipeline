import time
import socketio
import threading
from pipeline_config import socketio_ipaddr, feedback_topic

class FeedbackClient(threading.Thread):
    """Simple Socket.IO client that keeps retrying on connect error
    and exits cleanly when stop() is called.
    """
    def __init__(self):
        super().__init__()
        self.sio = socketio.Client()
        
        print("[FeedbackClient]: Initialized")

        # Stop signal
        self._sigstop = threading.Event()

        # Indicate connection status
        self.is_connected = threading.Event()

        # Register event handlers for socketio
        self.sio.on('connect', self.on_connect)
        self.sio.on('disconnect', self.on_disconnect)
        self.sio.on('connect_error', self.on_connect_error)

    def on_connect(self):
        print("[FeedbackClient] SocketIO connected.")
        self.is_connected.set()

    def on_disconnect(self):
        print("[FeedbackClient] SocketIO disconnected.")
        self.is_connected.clear()

    def on_connect_error(self, data):
        print("[FeedbackClient] SocketIO connection error:", data)
        self.is_connected.clear()

    def stop(self):
        """Signal the thread to stop on the next loop iteration."""
        print("[FeedbackClient] Received stop signal.")
        self._sigstop.set()

    def run(self):
        """Thread entry point: keep trying to connect if disconnected,
        and once connected, do small-sleep loops until stop is signaled.
        """
        while not self._sigstop.is_set():
            if not self.is_connected.is_set():
                # Attempt to connect if we're not currently connected
                try:
                    print("[FeedbackClient] Attempting to connect:", socketio_ipaddr)
                    self.sio.connect(socketio_ipaddr)
                except Exception as e:
                    print("[FeedbackClient] Connection failed, retrying in 5 seconds...", e)
                    time.sleep(5)
                    # Then continue the loop, possibly we try again
                    continue

            # If connected, or still attempting, let socketio process events.
            # We check stop event periodically so we can exit cleanly.
            self.sio.sleep(0.2)

        # Once stop is signaled, disconnect if still connected
        if self.sio.connected:
            self.sio.disconnect()

        print("[FeedbackClient] Exited feedback thread cleanly.")
        

    def send_message(self, message_obj, topic=feedback_topic):
        """Send a message if connected. Otherwise, do nothing."""
        if not self.is_connected.is_set():
            # Not connected yet
            return
        self.sio.emit(topic, message_obj)
        # Uncomment if you'd like verbose output:
        # print(f"[FeedbackClient] Sent message to '{topic}': {message_obj}")
