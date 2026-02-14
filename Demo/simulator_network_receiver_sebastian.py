#!/usr/bin/env python3
"""
SRT Receiver Process - Receives multimodal SRT stream and writes to named pipes
This should run as a separate process to avoid blocking the GUI with network I/O.
"""
import os
import sys
import ctypes
import struct
import time
from multiprocessing import Process, Value
import numpy as np

# Load libsrt
if sys.platform == 'darwin':
    try:
        libsrt = ctypes.CDLL('/opt/homebrew/Cellar/srt/1.5.4/lib/libsrt.dylib')
    except OSError:
        try:
            libsrt = ctypes.CDLL('/opt/homebrew/lib/libsrt.dylib')
        except OSError:
            try:
                libsrt = ctypes.CDLL('libsrt.dylib')
            except OSError:
                print("Error: libsrt not found on macOS")
                print("Install with: brew install srt")
                sys.exit(1)
else:
    try:
        libsrt = ctypes.CDLL("libsrt.so.1")
    except OSError:
        try:
            libsrt = ctypes.CDLL("libsrt.so")
        except OSError:
            print("Error: libsrt not found")
            sys.exit(1)

# SRT function declarations
libsrt.srt_startup.argtypes = []
libsrt.srt_startup.restype = ctypes.c_int
libsrt.srt_create_socket.argtypes = []
libsrt.srt_create_socket.restype = ctypes.c_int
libsrt.srt_setsockopt.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
libsrt.srt_setsockopt.restype = ctypes.c_int
libsrt.srt_connect.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
libsrt.srt_connect.restype = ctypes.c_int
libsrt.srt_recv.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
libsrt.srt_recv.restype = ctypes.c_int
libsrt.srt_close.argtypes = [ctypes.c_int]
libsrt.srt_close.restype = ctypes.c_int
libsrt.srt_cleanup.argtypes = []
libsrt.srt_cleanup.restype = ctypes.c_int

# Platform-specific socket address structure
if sys.platform == 'darwin':
    class sockaddr_in(ctypes.Structure):
        _fields_ = [
            ("sin_len", ctypes.c_uint8),
            ("sin_family", ctypes.c_uint8),
            ("sin_port", ctypes.c_uint16),
            ("sin_addr", ctypes.c_byte * 4),
            ("sin_zero", ctypes.c_byte * 8),
        ]
else:
    class sockaddr_in(ctypes.Structure):
        _fields_ = [
            ("sin_family", ctypes.c_short),
            ("sin_port", ctypes.c_ushort),
            ("sin_addr", ctypes.c_byte * 4),
            ("sin_zero", ctypes.c_byte * 8),
        ]

# Pipe paths
PIPE_VIDEO = "/tmp/emsvid"
PIPE_AUDIO = "/tmp/emsaud"
PIPE_CSV = "/tmp/emscsv"


def setup_pipes():
    """Create all three named pipes."""
    for pipe_path in [PIPE_VIDEO, PIPE_AUDIO, PIPE_CSV]:
        try:
            if os.path.exists(pipe_path):
                os.unlink(pipe_path)
            os.mkfifo(pipe_path)
            print(f"[Pipe] Created {pipe_path}")
        except Exception as e:
            print(f"[Pipe] Error creating {pipe_path}: {e}")
            # Don't exit - pipes might already exist


def cleanup_pipes():
    """Remove all three named pipes."""
    for pipe_path in [PIPE_VIDEO, PIPE_AUDIO, PIPE_CSV]:
        try:
            if os.path.exists(pipe_path):
                os.unlink(pipe_path)
        except:
            pass


def srt_receiver_process(host, port, width, height, enabled_types, running_flag):
    """
    SRT receiver process that reads from network and writes to pipes.

    Args:
        host: SRT server hostname/IP
        port: SRT server port
        width: Video frame width
        height: Video frame height
        enabled_types: Set of enabled data types (1=video, 2=audio, 3=csv)
        running_flag: Shared Value to control process lifecycle
    """
    print(f"[SRTProcess] Starting, PID: {os.getpid()}")
    print(f"[SRTProcess] Connecting to {host}:{port}")
    print(f"[SRTProcess] Video: {width}x{height}")
    print(f"[SRTProcess] Enabled types: {enabled_types}")

    sock = None
    opened_pipes = {}

    try:
        # Initialize SRT
        if libsrt.srt_startup() < 0:
            print("[SRTProcess] Failed to initialize SRT")
            return

        sock = libsrt.srt_create_socket()
        if sock < 0:
            print("[SRTProcess] Failed to create socket")
            libsrt.srt_cleanup()
            return

        # Configure SRT socket
        rcvbuf = ctypes.c_int(48000000)
        libsrt.srt_setsockopt(sock, 0, 8, ctypes.byref(rcvbuf), ctypes.sizeof(rcvbuf))

        tlpktdrop = ctypes.c_int(0)
        libsrt.srt_setsockopt(sock, 0, 6, ctypes.byref(tlpktdrop), ctypes.sizeof(tlpktdrop))

        # Setup address
        addr = sockaddr_in()
        import socket as socket_module

        if sys.platform == 'darwin':
            addr.sin_len = ctypes.sizeof(sockaddr_in)
            addr.sin_family = 2
        else:
            addr.sin_family = 2

        addr.sin_port = socket_module.htons(port)
        ip_parts = [int(x) for x in host.split('.')]
        for i in range(4):
            addr.sin_addr[i] = ip_parts[i]

        # Connect
        print("[SRTProcess] Connecting to SRT server...")
        if libsrt.srt_connect(sock, ctypes.byref(addr), ctypes.sizeof(addr)) < 0:
            print("[SRTProcess] Failed to connect")
            libsrt.srt_close(sock)
            libsrt.srt_cleanup()
            return

        print("[SRTProcess] Connected to SRT server!")

        # Open pipes
        print("[SRTProcess] Opening pipes for writing...")
        pipe_map = {1: PIPE_VIDEO, 2: PIPE_AUDIO, 3: PIPE_CSV}

        for data_type in enabled_types:
            pipe_path = pipe_map[data_type]
            try:
                if data_type == 3:  # CSV is text mode
                    opened_pipes[data_type] = open(pipe_path, 'w', buffering=1)
                else:
                    opened_pipes[data_type] = open(pipe_path, 'wb', buffering=0)
                print(f"[SRTProcess] Opened {pipe_path}")
            except Exception as e:
                print(f"[SRTProcess] Failed to open {pipe_path}: {e}")
                return

        print(f"[SRTProcess] All {len(opened_pipes)} pipes opened! Starting to receive...")

        # Main receive loop
        frames_received = 0
        last_debug = time.time()

        frame_buffers = {}
        expected_chunks = {}
        current_frame = None

        while running_flag.value:
            try:
                # Receive one chunk
                buf = ctypes.create_string_buffer(2048)
                received = libsrt.srt_recv(sock, buf, 2048)

                if received <= 0:
                    if running_flag.value:
                        print("[SRTProcess] Connection closed")
                    break

                if received < 11:
                    continue

                data_type, frame_idx, chunk_num, total_chunks, data_size = struct.unpack('!BIHHH', buf.raw[:11])
                data = buf.raw[11:11+data_size]

                # Ignore disabled types
                if data_type not in enabled_types:
                    continue

                # Initialize frame buffer
                if frame_idx not in frame_buffers:
                    frame_buffers[frame_idx] = {1: {}, 2: {}, 3: {}}
                    expected_chunks[frame_idx] = {}
                    if current_frame is None:
                        current_frame = frame_idx

                # Store chunk
                frame_buffers[frame_idx][data_type][chunk_num] = data
                expected_chunks[frame_idx][data_type] = total_chunks

                # Check if current frame is complete
                if current_frame in frame_buffers:
                    fb = frame_buffers[current_frame]
                    ec = expected_chunks[current_frame]

                    all_complete = True
                    for dt in enabled_types:
                        if dt not in ec or len(fb[dt]) != ec[dt]:
                            all_complete = False
                            break

                    if all_complete:
                        # Assemble and write complete frame
                        for dt in enabled_types:
                            complete_data = b''.join(fb[dt][i] for i in range(ec[dt]))

                            if dt == 1:  # Video
                                video_bytes = len(complete_data).to_bytes(4, 'big') + complete_data
                                opened_pipes[1].write(video_bytes)
                                opened_pipes[1].flush()
                            elif dt == 2:  # Audio
                                audio_bytes = len(complete_data).to_bytes(4, 'big') + complete_data
                                opened_pipes[2].write(audio_bytes)
                                opened_pipes[2].flush()
                            elif dt == 3:  # CSV
                                csv_text = complete_data.decode('utf-8').strip()
                                opened_pipes[3].write(csv_text + '\n')
                                opened_pipes[3].flush()

                        frames_received += 1

                        # Debug
                        if frames_received % 30 == 0:
                            elapsed = time.time() - last_debug
                            fps = 30 / elapsed
                            print(f"[SRTProcess] Frames: {frames_received:5d} | FPS: {fps:.1f}")
                            last_debug = time.time()

                        # Cleanup
                        del frame_buffers[current_frame]
                        del expected_chunks[current_frame]
                        current_frame += 1

                        # Remove old frames
                        old_frames = [f for f in frame_buffers.keys() if f < current_frame - 5]
                        for old_frame in old_frames:
                            del frame_buffers[old_frame]
                            if old_frame in expected_chunks:
                                del expected_chunks[old_frame]

            except Exception as e:
                if running_flag.value:
                    print(f"[SRTProcess] Error in receive loop: {e}")
                    import traceback
                    traceback.print_exc()
                break

    except Exception as e:
        print(f"[SRTProcess] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("[SRTProcess] Cleaning up...")

        for pipe in opened_pipes.values():
            try:
                pipe.close()
            except:
                pass

        if sock is not None:
            libsrt.srt_close(sock)

        libsrt.srt_cleanup()

    print(f"[SRTProcess] Exiting, received {frames_received} frames")


class SRTReceiverProcess:
    """
    Manager class for the SRT receiver process.
    Use this instead of the old SRTReceiver class.
    """
    def __init__(self, host, port, width, height, enabled_types=None):
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.enabled_types = enabled_types if enabled_types is not None else {1, 2, 3}

        # Multiprocessing components
        self.running_flag = Value(ctypes.c_bool, True)

        # Receiver process
        self.receiver_process = Process(
            target=srt_receiver_process,
            args=(host, port, width, height, self.enabled_types, self.running_flag),
            daemon=True
        )

        print('[SRTReceiverProcess] Initialized')

    def start(self):
        """Start the SRT receiver process."""
        print("[SRTReceiverProcess] Starting...")
        self.running_flag.value = True
        self.receiver_process.start()
        print(f"[SRTReceiverProcess] Started (Process PID: {self.receiver_process.pid})")
        return True

    def stop(self):
        """Stop the SRT receiver process."""
        print("[SRTReceiverProcess] Stopping...")

        self.running_flag.value = False
        self.receiver_process.join(timeout=5)

        if self.receiver_process.is_alive():
            print("[SRTReceiverProcess] Process didn't stop, terminating...")
            self.receiver_process.terminate()
            self.receiver_process.join(timeout=2)

        print("[SRTReceiverProcess] Stopped")


# For backward compatibility
class SRTReceiver:
    """
    Wrapper to maintain backward compatibility with existing GUI.py code.
    Now runs as a separate process instead of a thread.
    """
    def __init__(self, host, port, width, height, enabled_types=None):
        self.manager = SRTReceiverProcess(host, port, width, height, enabled_types)

    def start(self):
        return self.manager.start()

    def stop(self):
        self.manager.stop()