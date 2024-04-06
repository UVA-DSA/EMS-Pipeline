import socket
import netifaces as ni
import time
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot


def get_local_ipv4():
    for interface in ni.interfaces():
        addresses = ni.ifaddresses(interface)
        if ni.AF_INET in addresses:
            for link in addresses[ni.AF_INET]:
                if 'addr' in link and not link['addr'].startswith('127.'):
                    return link['addr']
    return 'No local IPv4 found.'


def is_internet_available(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False
    
class InternetCheckThread(QThread):
    internet_status_signal = pyqtSignal(bool)

    def __init__(self, check_interval=1, parent=None):
        super(InternetCheckThread, self).__init__(parent)
        self.check_interval = check_interval
        self.is_running = True
        self.prev_state = None

    def run(self):
        while self.is_running:
            # Replace 'is_internet_available' with the actual function you use to check internet availability
            internet_avail = is_internet_available()
            if(self.prev_state != internet_avail):
                self.internet_status_signal.emit(internet_avail)
            self.prev_state = internet_avail
            time.sleep(self.check_interval)

    def stop(self):
        self.is_running = False
        self.wait()
