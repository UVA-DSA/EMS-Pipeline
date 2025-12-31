#!/usr/bin/env python3
"""
SRT Client - Headless Pipe Writer
Writes synchronized data to three pipes:
- /tmp/emsvid - Video frames (raw BGR numpy array bytes)
- /tmp/emsaud - Audio data (raw PCM int16 bytes)
- /tmp/emscsv - CSV text data

Each frame writes in order: video -> audio -> csv
Maintains frame synchronization across all three streams.
"""
import os
import sys
import ctypes
import struct
import threading
import numpy as np
import time

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
            sys.exit(1)

def cleanup_pipes():
    """Remove all three named pipes."""
    for pipe_path in [PIPE_VIDEO, PIPE_AUDIO, PIPE_CSV]:
        try:
            if os.path.exists(pipe_path):
                os.unlink(pipe_path)
        except:
            pass

class SRTReceiver:
    """Handles SRT connection and receiving."""
    def __init__(self, host, port, width, height, enabled_types=None):
        self.host = host
        self.port = port
        self.width = width
        self.height = height
        self.sock = None
        self.running = False
        self.enabled_types = enabled_types if enabled_types is not None else {1, 2, 3}

    def recv_chunk(self):
        """Receive one SRT chunk."""
        buf = ctypes.create_string_buffer(2048)
        # This call blocks until data arrives
        received = libsrt.srt_recv(self.sock, buf, 2048)

        if received <= 0:
            return None

        if received < 11:
            return None

        data_type, frame_idx, chunk_num, total_chunks, data_size = struct.unpack('!BIHHH', buf.raw[:11])
        data = buf.raw[11:11+data_size]

        return (data_type, frame_idx, chunk_num, total_chunks, data)

    def receive_complete_data(self, data_type):
        """Receive all chunks for a data type - returns (frame_idx, data) or (None, None)."""
        chunks = {}
        total_chunks = None
        frame_idx = None
        max_attempts = 1000

        print(f"[receive_complete_data] Starting to receive data type {data_type}")
        for attempt in range(max_attempts):
            if attempt == 0:
                print(f"[receive_complete_data] Calling recv_chunk (may block)...")
            chunk_info = self.recv_chunk()
            if chunk_info is None:
                print(f"[receive_complete_data] recv_chunk returned None on attempt {attempt}")
                return (None, None)

            recv_type, recv_frame, chunk_num, num_chunks, data = chunk_info

            if attempt == 0:
                print(f"[receive_complete_data] First chunk: type={recv_type}, frame={recv_frame}, chunk={chunk_num}/{num_chunks}")

            if frame_idx is None:
                frame_idx = recv_frame

            if recv_frame != frame_idx:
                print(f"[receive_complete_data] Frame changed: {frame_idx} -> {recv_frame}")
                return (None, None)

            if recv_type != data_type:
                print(f"[receive_complete_data] Wrong type: expected {data_type}, got {recv_type}")
                return (None, None)

            chunks[chunk_num] = data
            total_chunks = num_chunks

            if len(chunks) == total_chunks:
                print(f"[receive_complete_data] Complete! Got all {total_chunks} chunks for type {data_type}")
                return (frame_idx, b''.join(chunks[i] for i in range(total_chunks)))

        print(f"[receive_complete_data] Max attempts reached for type {data_type}")
        return (None, None)

    def connect(self):
        """Connect to SRT server."""
        if libsrt.srt_startup() < 0:
            return False

        self.sock = libsrt.srt_create_socket()
        if self.sock < 0:
            libsrt.srt_cleanup()
            return False

        # Configure SRT socket
        rcvbuf = ctypes.c_int(48000000)
        libsrt.srt_setsockopt(self.sock, 0, 8, ctypes.byref(rcvbuf), ctypes.sizeof(rcvbuf))

        tlpktdrop = ctypes.c_int(0)
        libsrt.srt_setsockopt(self.sock, 0, 6, ctypes.byref(tlpktdrop), ctypes.sizeof(tlpktdrop))

        addr = sockaddr_in()

        import socket

        if sys.platform == 'darwin':
            addr.sin_len = ctypes.sizeof(sockaddr_in)
            addr.sin_family = 2
        else:
            addr.sin_family = 2

        addr.sin_port = socket.htons(self.port)

        ip_parts = [int(x) for x in self.host.split('.')]
        for i in range(4):
            addr.sin_addr[i] = ip_parts[i]

        if libsrt.srt_connect(self.sock, ctypes.byref(addr), ctypes.sizeof(addr)) < 0:
            libsrt.srt_close(self.sock)
            libsrt.srt_cleanup()
            return False

        return True

    def start(self):
        """Start receiving and writing to pipes."""
        if not self.connect():
            return False

        self.running = True
        self.thread = threading.Thread(
            target=self.receive_loop,
            daemon=True
        )
        self.thread.start()
        return True

    def stop(self):
        """Stop receiving."""
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=2)
        if self.sock is not None:
            libsrt.srt_close(self.sock)
        libsrt.srt_cleanup()

    def receive_loop(self):
        """Main receive loop - writes synchronized data to pipes."""
        print("[Receiver] Started, connecting to SRT...")

        # Wait for pipes to exist (created by display client)
        print("[Receiver] Waiting for pipes to be created by display client...")
        timeout = 30  # 30 second timeout
        start_time = time.time()

        # Map of data types to pipe paths
        pipe_paths = {
            1: PIPE_VIDEO,
            2: PIPE_AUDIO,
            3: PIPE_CSV
        }

        # Only wait for enabled pipes
        required_pipes = [pipe_paths[dt] for dt in self.enabled_types]

        while self.running:
            all_exist = all(os.path.exists(p) for p in required_pipes)
            if all_exist:
                break
            if time.time() - start_time > timeout:
                print("[Receiver] Timeout waiting for pipes. Is display client running?")
                return
            time.sleep(0.5)

        if not self.running:
            return

        print("[Receiver] Pipes found! Opening for writing...")

        # Map of data types to pipe paths
        pipe_paths = {
            1: PIPE_VIDEO,
            2: PIPE_AUDIO,
            3: PIPE_CSV
        }

        # Only open pipes for enabled types
        pipes_to_open = {dt: pipe_paths[dt] for dt in self.enabled_types}

        try:
            # Open pipes for writing in parallel (avoids deadlock with reader)
            pipe_holders = {dt: [None] for dt in self.enabled_types}
            errors = []

            def make_opener(data_type, path):
                def open_writer():
                    try:
                        print(f"[Receiver] Opening {path} for writing (type {data_type})...")
                        if data_type == 3:  # CSV is text mode
                            pipe_holders[data_type][0] = open(path, 'w', buffering=1)
                        else:  # Video and audio are binary
                            pipe_holders[data_type][0] = open(path, 'wb', buffering=0)
                        print(f"[Receiver] {path} opened for writing!")
                    except Exception as e:
                        errors.append(f"Type {data_type}: {e}")
                return open_writer

            # Create and start threads for each enabled pipe
            threads = []
            for data_type, path in pipes_to_open.items():
                t = threading.Thread(target=make_opener(data_type, path))
                t.start()
                threads.append(t)

            # Wait for all opens to complete
            for t in threads:
                t.join(timeout=30)

            if errors:
                print(f"[Receiver] Errors opening pipes: {errors}")
                return

            # Store opened pipes
            opened_pipes = {}
            for data_type in self.enabled_types:
                if pipe_holders[data_type][0] is None:
                    print(f"[Receiver] Failed to open pipe for type {data_type}")
                    return
                opened_pipes[data_type] = pipe_holders[data_type][0]

            print(f"[Receiver] Opened {len(opened_pipes)} pipe(s)! Starting to stream...")

            frames_received = 0
            last_debug = time.time()

            # Buffer to hold chunks by type and frame
            # Structure: {frame_idx: {data_type: {chunk_num: data}}}
            frame_buffers = {}

            # Track expected chunks for each type in current frame
            # Structure: {frame_idx: {data_type: total_chunks}}
            expected_chunks = {}

            print("[Receiver] Entering main receive loop...")
            print("[Receiver] Note: Chunks may arrive in any order, we'll buffer them...")

            current_frame = None

            while self.running:
                try:
                    # Receive one chunk (any type, any frame)
                    chunk_info = self.recv_chunk()
                    if chunk_info is None:
                        continue

                    recv_type, recv_frame, chunk_num, num_chunks, data = chunk_info

                    # Server sends: 1=video, 2=audio, 3=CSV
                    # Ignore any other types
                    if recv_type not in [1, 2, 3]:
                        continue

                    # Initialize frame buffer if needed
                    if recv_frame not in frame_buffers:
                        frame_buffers[recv_frame] = {1: {}, 2: {}, 3: {}}  # video, audio, csv
                        expected_chunks[recv_frame] = {}
                        if current_frame is None:
                            current_frame = recv_frame
                            print(f"[Receiver] Starting with frame {current_frame}")

                    # Store the chunk
                    frame_buffers[recv_frame][recv_type][chunk_num] = data
                    expected_chunks[recv_frame][recv_type] = num_chunks

                    # Check if we have complete data for the CURRENT frame
                    if current_frame in frame_buffers:
                        fb = frame_buffers[current_frame]
                        ec = expected_chunks[current_frame]

                        # Check if all ENABLED types are complete for current frame
                        all_complete = True
                        for data_type in self.enabled_types:
                            if data_type not in ec or len(fb[data_type]) != ec[data_type]:
                                all_complete = False
                                break

                        if all_complete:
                            # Assemble complete frame data for enabled types
                            frame_data = {}
                            for data_type in self.enabled_types:
                                frame_data[data_type] = b''.join(fb[data_type][i] for i in range(ec[data_type]))

                            # Write to pipes (only enabled types)
                            if 1 in self.enabled_types:  # Video
                                video_bytes = len(frame_data[1]).to_bytes(4, 'big') + frame_data[1]
                                opened_pipes[1].write(video_bytes)
                                opened_pipes[1].flush()

                            if 2 in self.enabled_types:  # Audio
                                audio_bytes = len(frame_data[2]).to_bytes(4, 'big') + frame_data[2]
                                opened_pipes[2].write(audio_bytes)
                                opened_pipes[2].flush()

                            if 3 in self.enabled_types:  # CSV
                                csv_text = frame_data[3].decode('utf-8').strip()
                                opened_pipes[3].write(csv_text + '\n')
                                opened_pipes[3].flush()

                            frames_received += 1

                            if frames_received == 1:
                                print(f"[Receiver] === Frame {current_frame} complete and written! ===")
                                type_info = []
                                if 1 in self.enabled_types:
                                    type_info.append(f"Video: {len(frame_data[1])} bytes")
                                if 2 in self.enabled_types:
                                    type_info.append(f"Audio: {len(frame_data[2])} bytes")
                                if 3 in self.enabled_types:
                                    type_info.append(f"CSV: {frame_data[3].decode('utf-8').strip()}")
                                print(f"[Receiver] {', '.join(type_info)}")
                                print("[Receiver] Continuing with reduced logging...")

                            # Clean up old frame
                            del frame_buffers[current_frame]
                            del expected_chunks[current_frame]

                            # Move to next frame
                            current_frame += 1

                            # Debug every 30 frames
                            if frames_received % 30 == 0:
                                elapsed = time.time() - last_debug
                                fps = 30 / elapsed
                                print(f"[Receiver] Frames: {frames_received:5d} | Frame idx: {current_frame} | FPS: {fps:.1f}")
                                last_debug = time.time()

                            # Clean up very old frames (in case we missed some)
                            old_frames = [f for f in frame_buffers.keys() if f < current_frame - 5]
                            for old_frame in old_frames:
                                del frame_buffers[old_frame]
                                if old_frame in expected_chunks:
                                    del expected_chunks[old_frame]

                except Exception as e:
                    print(f"[Receiver] Error in loop: {e}")
                    import traceback
                    traceback.print_exc()
                    break
        except Exception as e:
            print(f"[Receiver] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            try:
                for pipe in opened_pipes.values():
                    pipe.close()
            except:
                pass

        print("[Receiver] Stopped")

def main():
    if len(sys.argv) < 5:
        print("Usage: python srt_client_headless.py <host> <port> <width> <height> [data_types...]")
        print("Example: python srt_client_headless.py 127.0.0.1 9000 480 270")
        print("Example: python srt_client_headless.py 127.0.0.1 9000 480 270 1 3  # Only video and CSV")
        print("\nData types:")
        print("  1 = video (/tmp/emsvid)")
        print("  2 = audio (/tmp/emsaud)")
        print("  3 = CSV (/tmp/emscsv)")
        print("\nDefault: All three types (1 2 3) if not specified")
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2])
    width = int(sys.argv[3])
    height = int(sys.argv[4])

    # Parse optional data types to write
    if len(sys.argv) > 5:
        enabled_types = set(int(x) for x in sys.argv[5:])
        # Validate
        if not enabled_types.issubset({1, 2, 3}):
            print("Error: Data types must be 1, 2, or 3")
            sys.exit(1)
    else:
        enabled_types = {1, 2, 3}  # Default: all three

    type_names = {1: "video", 2: "audio", 3: "CSV"}
    enabled_names = [type_names[t] for t in sorted(enabled_types)]

    print(f"[Main] Connecting to {host}:{port} ({width}x{height})")
    print(f"[Main] Enabled data types: {enabled_names}")
    print("[Main] NOTE: Start pipe_display_client.py first (or cat the pipes)!")

    try:
        # Create and start receiver (it will wait for pipes and open them)
        receiver = SRTReceiver(host, port, width, height, enabled_types)
        if not receiver.start():
            print("[Main] Failed to connect to SRT server!")
            sys.exit(1)

        print("[Main] Streaming to pipes. Press Ctrl+C to stop.")

        # Keep running until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Main] Interrupted by user")

    except Exception as e:
        print(f"[Main] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("[Main] Shutting down...")
        receiver.stop()
        print("[Main] Done")

if __name__ == "__main__":
    main()