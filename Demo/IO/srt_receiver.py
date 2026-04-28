#!/usr/bin/env python3
"""
SRT Receiver Process - Receives multimodal SRT stream and writes to named pipes
Supports both display/playback pipes and ML pipeline pipes.

Pipe mapping:
  1 = /tmp/emsvid (video display)
  2 = /tmp/emsaud (audio playback)
  3 = /tmp/emscsv (CSV display)
  4 = /tmp/emsvidml (video ML)
  5 = /tmp/emsaudml (audio ML)
  6 = /tmp/emscsvml (CSV ML)
"""
import os
import sys
import ctypes
import ctypes.util
import struct
import time
import zlib
from multiprocessing import Process, Value

libsrt = None
libsrt_load_error = None


def _iter_libsrt_candidates():
    env_path = os.environ.get("SRT_LIBRARY_PATH")
    if env_path:
        yield env_path

    if sys.platform == 'darwin':
        candidates = [
            ctypes.util.find_library("srt"),
            "/opt/homebrew/opt/srt/lib/libsrt.dylib",
            "/opt/homebrew/lib/libsrt.dylib",
            "/usr/local/opt/srt/lib/libsrt.dylib",
            "/usr/local/lib/libsrt.dylib",
            "libsrt.dylib",
        ]
    else:
        candidates = [
            ctypes.util.find_library("srt"),
            ctypes.util.find_library("srt-gnutls"),
            ctypes.util.find_library("srt-openssl"),
            "libsrt.so.1",
            "libsrt.so",
            "libsrt-gnutls.so.1.5",
            "libsrt-gnutls.so",
            "libsrt-openssl.so.1.5",
            "libsrt-openssl.so",
        ]

    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        yield candidate


def _configure_libsrt(lib):
    lib.srt_startup.argtypes = []
    lib.srt_startup.restype = ctypes.c_int
    lib.srt_create_socket.argtypes = []
    lib.srt_create_socket.restype = ctypes.c_int
    lib.srt_setsockopt.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    lib.srt_setsockopt.restype = ctypes.c_int
    lib.srt_connect.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    lib.srt_connect.restype = ctypes.c_int
    lib.srt_recv.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.srt_recv.restype = ctypes.c_int
    lib.srt_close.argtypes = [ctypes.c_int]
    lib.srt_close.restype = ctypes.c_int
    lib.srt_cleanup.argtypes = []
    lib.srt_cleanup.restype = ctypes.c_int
    return lib


def _load_libsrt():
    errors = []
    for candidate in _iter_libsrt_candidates():
        try:
            return _configure_libsrt(ctypes.CDLL(candidate))
        except OSError as exc:
            errors.append(f"{candidate}: {exc}")

    if sys.platform == 'darwin':
        install_hint = "Install with: brew install srt"
    else:
        install_hint = (
            "Install with: sudo apt install libsrt1.5-gnutls libsrt-gnutls-dev "
            "(Ubuntu) or set SRT_LIBRARY_PATH to the full library path."
        )

    detail = "; ".join(errors) if errors else "no library candidates found"
    raise OSError(f"libsrt not found. {install_hint} Tried: {detail}")


def ensure_libsrt():
    global libsrt, libsrt_load_error

    if libsrt is not None:
        return libsrt

    try:
        libsrt = _load_libsrt()
        libsrt_load_error = None
        return libsrt
    except OSError as exc:
        libsrt_load_error = str(exc)
        raise RuntimeError(libsrt_load_error) from exc

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
PIPE_VIDEO_ML = "/tmp/emsvidml"
PIPE_AUDIO_ML = "/tmp/emsaudml"
PIPE_CSV_ML = "/tmp/emscsvml"
PIPE_EGOSIM_TRANSCRIPT = "/tmp/egosim_transcript"


def setup_pipes():
    """Create all six named pipes (display + ML)."""
    all_pipes = [PIPE_VIDEO, PIPE_AUDIO, PIPE_CSV,
                 PIPE_VIDEO_ML, PIPE_AUDIO_ML, PIPE_CSV_ML]#, PIPE_EGOSIM_TRANSCRIPT]

    for pipe_path in all_pipes:
        try:
            if os.path.exists(pipe_path):
                os.unlink(pipe_path)
            os.mkfifo(pipe_path)
            print(f"[Pipe] Created {pipe_path}")
        except Exception as e:
            print(f"[Pipe] Error creating {pipe_path}: {e}")
            # Don't exit - pipes might already exist


def cleanup_pipes():
    """Remove all six named pipes."""
    all_pipes = [PIPE_VIDEO, PIPE_AUDIO, PIPE_CSV,
                 PIPE_VIDEO_ML, PIPE_AUDIO_ML, PIPE_CSV_ML, PIPE_EGOSIM_TRANSCRIPT]

    for pipe_path in all_pipes:
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
        enabled_types: Set of enabled data types
                       1=video, 2=audio, 3=csv, 4=video_ml, 5=audio_ml, 6=csv_ml
        running_flag: Shared Value to control process lifecycle
    """
    print(f"[SRTProcess] Starting, PID: {os.getpid()}")
    print(f"[SRTProcess] Connecting to {host}:{port}")
    print(f"[SRTProcess] Video: {width}x{height}")
    print(f"[SRTProcess] Enabled types: {enabled_types}")

    # ---- Session logging setup -------------------------------------------
    _session_ts = time.strftime("%Y%m%d_%H%M%S")
    _log_dir    = "/tmp/ems_session_logs"
    os.makedirs(_log_dir, exist_ok=True)
    _stats_path    = f"{_log_dir}/srt_stats_{_session_ts}.tsv"
    _checksum_path = f"{_log_dir}/srt_checksums_{_session_ts}.tsv"

    _stats_f    = open(_stats_path,    "w", buffering=1)
    _checksum_f = open(_checksum_path, "w", buffering=1)

    # Stats log: human-readable events + timing
    _stats_f.write("# SRT receiver session log\n")
    _stats_f.write(f"# session_start\t{_session_ts}\n")
    _stats_f.write(f"# host\t{host}:{port}\n")
    _stats_f.write(f"# resolution\t{width}x{height}\n")
    _stats_f.write(f"# enabled_types\t{sorted(enabled_types)}\n")
    _stats_f.write("event\tframe_idx\ttimestamp\tdetail\n")

    # Checksum log: one row per complete frame, for comparison with local sampler
    _checksum_f.write("# SRT receiver frame checksums (adler32)\n")
    _checksum_f.write(f"# session_start\t{_session_ts}\n")
    _checksum_f.write("frame_idx\tvideo_adler32\taudio_adler32\tcsv_adler32\tframe_wall_time\n")

    def _log_event(event, frame_idx, detail=""):
        _stats_f.write(f"{event}\t{frame_idx}\t{time.time():.6f}\t{detail}\n")

    print(f"[SRTProcess] Session logs: {_log_dir}/srt_*_{_session_ts}.tsv")

    sock = None
    opened_pipes = {}
    frames_received = 0
    _last_complete_frame_idx = -1   # for detecting gaps / dropped frames
    _frame_arrival_times = []       # wall-clock time each frame was completed
    _reassembly_start = {}          # frame_idx -> wall time first chunk arrived

    try:
        libsrt = ensure_libsrt()

        # Initialize SRT
        if libsrt.srt_startup() < 0:
            print("[SRTProcess] Failed to initialize SRT")
            return

        sock = libsrt.srt_create_socket()
        if sock < 0:
            print("[SRTProcess] Failed to create socket")
            libsrt.srt_cleanup()
            return

        # PRE-CONNECT socket options (must be set before srt_connect)
        # SRTO_RCVBUF — large receive buffer for burst tolerance
        rcvbuf = ctypes.c_int(96000000)
        libsrt.srt_setsockopt(sock, 0, 8, ctypes.byref(rcvbuf), ctypes.sizeof(rcvbuf))

        # SRTO_RCVLATENCY (option 43) — how long SRT holds packets before
        # delivering them, in milliseconds.  Higher value = more buffer against
        # a slow receiver.  120ms is the default; we use 500ms so the server's
        # initial burst is absorbed inside the SRT layer rather than overflowing
        # into our application buffer before srt_recv has been called once.
        rcvlatency = ctypes.c_int(500)
        libsrt.srt_setsockopt(sock, 0, 43, ctypes.byref(rcvlatency), ctypes.sizeof(rcvlatency))

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

        # POST-CONNECT socket options
        # SRTO_TLPKTDROP (option 6) — must be set after connect so it
        # survives the SRT handshake negotiation.  Enables graceful dropping
        # of packets that can't be delivered in time instead of hard errors.
        tlpktdrop = ctypes.c_int(1)
        libsrt.srt_setsockopt(sock, 0, 6, ctypes.byref(tlpktdrop), ctypes.sizeof(tlpktdrop))

        # SRTO_SNDDROPDELAY (option 27, sender-side) — how long the sender
        # waits before dropping a packet it can't send.  Setting to 0 on the
        # receiver side is a no-op but harmless; the real effect is that we're
        # telling SRT to shed packets early rather than retrying until the
        # buffer is completely full.
        snddropdelay = ctypes.c_int(0)
        libsrt.srt_setsockopt(sock, 0, 27, ctypes.byref(snddropdelay), ctypes.sizeof(snddropdelay))

        # ------------------------------------------------------------------
        # Key insight: srt_recv MUST start immediately after connect.
        # Any blocking work done before the first srt_recv call lets the
        # server fill the receive buffer → "No room to store incoming packet".
        #
        # Architecture:
        #   • Main thread: tight srt_recv loop, puts assembled frames onto
        #     _write_queue (non-blocking put_nowait — drops if full rather
        #     than stalling the receive loop).
        #   • PipeOpener threads: open() each named pipe in background.
        #     Each open() blocks until a reader connects; that is now
        #     completely off the critical path.
        #   • PipeWriter thread: drains _write_queue and writes to whichever
        #     pipes are open at that moment.
        # ------------------------------------------------------------------

        import threading
        import queue as _queue_mod

        pipe_map = {
            1: PIPE_VIDEO,
            2: PIPE_AUDIO,
            3: PIPE_CSV,
            4: PIPE_VIDEO_ML,
            5: PIPE_AUDIO_ML,
            6: PIPE_CSV_ML,
        }

        # ---- Pipe opener threads (fire and forget) -------------------------
        def _open_pipe(data_type):
            path = pipe_map.get(data_type)
            if not path:
                return
            try:
                if data_type in (3, 6):
                    opened_pipes[data_type] = open(path, 'w', buffering=1)
                else:
                    opened_pipes[data_type] = open(path, 'wb', buffering=0)
                print(f"[SRTProcess] Opened pipe {path}")
            except Exception as exc:
                print(f"[SRTProcess] Failed to open {path}: {exc}")

        for dt in enabled_types:
            threading.Thread(target=_open_pipe, args=(dt,), daemon=True).start()

        # ---- Pipe writer thread -------------------------------------------
        # maxsize=8: small cushion so a momentarily slow consumer doesn't
        # instantly start dropping, but not large enough to grow unbounded.
        _write_queue = _queue_mod.Queue(maxsize=8)
        _writer_done = threading.Event()
        _queue_drops = [0]

        def _pipe_writer():
            while not _writer_done.is_set() or not _write_queue.empty():
                try:
                    fidx, payloads, btypes = _write_queue.get(timeout=0.05)
                except _queue_mod.Empty:
                    continue
                for bt in btypes:
                    data = payloads[bt]
                    try:
                        if bt == 1:
                            b = len(data).to_bytes(4, 'big') + data
                            if 1 in opened_pipes: opened_pipes[1].write(b); opened_pipes[1].flush()
                            if 4 in opened_pipes: opened_pipes[4].write(b); opened_pipes[4].flush()
                        elif bt == 2:
                            b = len(data).to_bytes(4, 'big') + data
                            if 2 in opened_pipes: opened_pipes[2].write(b); opened_pipes[2].flush()
                            if 5 in opened_pipes: opened_pipes[5].write(data); opened_pipes[5].flush()
                        elif bt == 3:
                            line = data.decode('utf-8').strip() + '\n'
                            if 3 in opened_pipes: opened_pipes[3].write(line); opened_pipes[3].flush()
                            if 6 in opened_pipes: opened_pipes[6].write(line); opened_pipes[6].flush()
                    except Exception as exc:
                        print(f"[SRTProcess] Pipe write error bt={bt} frame={fidx}: {exc}")

        _writer_thread = threading.Thread(target=_pipe_writer, daemon=True, name="PipeWriter")
        _writer_thread.start()

        # ---- Base-type mapping -------------------------------------------
        base_types_needed = set()
        if 1 in enabled_types or 4 in enabled_types:
            base_types_needed.add(1)
        if 2 in enabled_types or 5 in enabled_types:
            base_types_needed.add(2)
        if 3 in enabled_types or 6 in enabled_types:
            base_types_needed.add(3)

        # ---- Receive loop — nothing blocking except srt_recv itself -------
        frames_received = 0
        last_debug      = time.time()
        frame_buffers   = {}
        expected_chunks = {}
        current_frame   = None

        print("[SRTProcess] Receive loop running...")

        while running_flag.value:
            try:
                buf      = ctypes.create_string_buffer(2048)
                received = libsrt.srt_recv(sock, buf, 2048)

                if received <= 0:
                    if running_flag.value:
                        print("[SRTProcess] Connection closed")
                    break

                if received < 11:
                    continue

                data_type, frame_idx, chunk_num, total_chunks, data_size = struct.unpack(
                    '!BIHHH', buf.raw[:11]
                )
                data = buf.raw[11:11 + data_size]

                if data_type not in base_types_needed:
                    continue

                if frame_idx not in frame_buffers:
                    frame_buffers[frame_idx]   = {1: {}, 2: {}, 3: {}}
                    expected_chunks[frame_idx] = {}
                    _reassembly_start[frame_idx] = time.time()
                    if current_frame is None:
                        current_frame = frame_idx

                frame_buffers[frame_idx][data_type][chunk_num] = data
                expected_chunks[frame_idx][data_type] = total_chunks

                if current_frame not in frame_buffers:
                    continue

                fb = frame_buffers[current_frame]
                ec = expected_chunks[current_frame]

                if not all(dt in ec and len(fb[dt]) == ec[dt] for dt in base_types_needed):
                    continue

                # Frame complete
                _now = time.time()

                if _last_complete_frame_idx >= 0:
                    _gap = current_frame - _last_complete_frame_idx - 1
                    if _gap > 0:
                        _log_event("DROPPED", current_frame,
                                   f"gap={_gap} prev={_last_complete_frame_idx}")
                        print(f"[SRTProcess] WARNING: {_gap} dropped frame(s) before idx {current_frame}")

                if _frame_arrival_times:
                    _ift = _now - _frame_arrival_times[-1]
                    if _ift > 0.050:
                        _log_event("JITTER", current_frame, f"inter_frame_ms={_ift*1000:.1f}")
                _frame_arrival_times.append(_now)

                _rtime = _now - _reassembly_start.pop(current_frame, _now)

                _payloads = {
                    bt: b''.join(fb[bt][i] for i in range(ec[bt]))
                    for bt in base_types_needed
                }

                _vid_crc = zlib.adler32(_payloads.get(1, b'')) & 0xFFFFFFFF
                _aud_crc = zlib.adler32(_payloads.get(2, b'')) & 0xFFFFFFFF
                _csv_crc = zlib.adler32(_payloads.get(3, b'').rstrip(b'\n')) & 0xFFFFFFFF
                _checksum_f.write(
                    f"{current_frame}\t{_vid_crc}\t{_aud_crc}\t{_csv_crc}\t{_now:.6f}\n"
                )

                # Non-blocking hand-off to writer; drop frame if writer is behind
                try:
                    _write_queue.put_nowait((current_frame, _payloads, base_types_needed))
                except _queue_mod.Full:
                    _queue_drops[0] += 1
                    _log_event("QUEUE_FULL_DROP", current_frame,
                               f"total={_queue_drops[0]}")
                    if _queue_drops[0] % 30 == 1:
                        print(f"[SRTProcess] WARNING: writer behind, "
                              f"dropped frame {current_frame} "
                              f"(total drops: {_queue_drops[0]})")

                frames_received += 1
                _last_complete_frame_idx = current_frame

                if frames_received % 30 == 0:
                    elapsed = time.time() - last_debug
                    print(
                        f"[SRTProcess] Frames: {frames_received:5d} | "
                        f"FPS: {30/elapsed:.1f} | "
                        f"reassembly: {_rtime*1000:.1f}ms | "
                        f"queue_drops: {_queue_drops[0]}"
                    )
                    last_debug = time.time()

                del frame_buffers[current_frame]
                del expected_chunks[current_frame]
                current_frame += 1

                # Evict stale incomplete frames
                stale = [f for f in frame_buffers if f < current_frame - 5]
                for f in stale:
                    del frame_buffers[f]
                    expected_chunks.pop(f, None)
                    _reassembly_start.pop(f, None)
                    _log_event("ABANDONED", f, "chunks_never_completed")

            except Exception as exc:
                if running_flag.value:
                    print(f"[SRTProcess] Error in receive loop: {exc}")
                    import traceback
                    traceback.print_exc()
                break

        # Signal writer to drain and stop
        _writer_done.set()
        _writer_thread.join(timeout=3)
        if _writer_thread.is_alive():
            print("[SRTProcess] Warning: pipe writer thread did not exit cleanly")

    except Exception as e:
        print(f"[SRTProcess] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ---- Session summary ------------------------------------------------
        _end_ts = time.time()
        try:
            _duration = _end_ts - (
                _frame_arrival_times[0] if _frame_arrival_times else _end_ts
            )
            _actual_fps = (
                len(_frame_arrival_times) / _duration if _duration > 0 else 0.0
            )
            # Inter-frame time stats
            if len(_frame_arrival_times) > 1:
                _ifts = [
                    (_frame_arrival_times[i] - _frame_arrival_times[i-1]) * 1000
                    for i in range(1, len(_frame_arrival_times))
                ]
                _ift_mean = sum(_ifts) / len(_ifts)
                _ift_max  = max(_ifts)
                _ift_min  = min(_ifts)
                _jitter_frames = sum(1 for x in _ifts if x > 50)
            else:
                _ift_mean = _ift_max = _ift_min = 0.0
                _jitter_frames = 0

            _summary = (
                f"# SESSION SUMMARY\n"
                f"# frames_received\t{frames_received}\n"
                f"# duration_s\t{_duration:.2f}\n"
                f"# actual_fps\t{_actual_fps:.2f}\n"
                f"# ift_mean_ms\t{_ift_mean:.2f}\n"
                f"# ift_min_ms\t{_ift_min:.2f}\n"
                f"# ift_max_ms\t{_ift_max:.2f}\n"
                f"# jitter_frames_(>50ms)\t{_jitter_frames}\n"
            )
            _stats_f.write(_summary)
            print(f"[SRTProcess] {_summary.replace('#', '').strip()}")
        except Exception as _e:
            print(f"[SRTProcess] Warning: could not write session summary: {_e}")

        try:
            _stats_f.close()
        except Exception:
            pass
        try:
            _checksum_f.close()
        except Exception:
            pass

        # ---- Pipe + socket cleanup ------------------------------------------
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
        self.last_error = None

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
        try:
            ensure_libsrt()
        except RuntimeError as exc:
            self.last_error = str(exc)
            print(f"[SRTReceiverProcess] {self.last_error}")
            return False

        self.running_flag.value = True
        self.receiver_process.start()
        self.last_error = None
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