"""
LocalFileSampler - reads a local .mp4 + .csv and writes to the same named pipes
that the SRT receiver uses, with the identical wire format:

  video / audio pipes : 4-byte big-endian length  +  raw bytes
  CSV pipes           : newline-terminated UTF-8 text lines

Drop this file in IO/ alongside srt_receiver.py.
"""

import csv
import ctypes
import os
import struct
import subprocess
import sys
print(f"[LocalSampler] sys.prefix = {sys.prefix}")
print(f"[LocalSampler] sys.executable = {sys.executable}")
import threading
import time

from IO.srt_receiver import (
    PIPE_VIDEO,
    PIPE_AUDIO,
    PIPE_CSV,
    PIPE_VIDEO_ML,
    PIPE_AUDIO_ML,
    PIPE_CSV_ML,
)


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def local_file_sampler_process(
        video_file: str,
        csv_file: str,
        width: int,
        height: int,
        fps: int,
        enabled_types: set,
        running_flag,
        loop: bool = False,
):
    """
    Multiprocessing target.  Reads *video_file* via two FFmpeg subprocesses
    (one for video frames, one for audio PCM) and *csv_file* line-by-line,
    then writes each frame to the enabled named pipes in exactly the same
    format as SRTReceiverProcess.

    enabled_types mirrors srt_receiver convention:
        1 = video display   4 = video ML
        2 = audio playback  5 = audio ML
        3 = csv display     6 = csv ML
    """
    pid = os.getpid()
    print(f"[LocalSampler] Starting, PID={pid}")
    print(f"[LocalSampler] video={video_file}  csv={csv_file}")
    print(f"[LocalSampler] {width}x{height} @ {fps} fps  loop={loop}")
    print(f"[LocalSampler] enabled_types={enabled_types}")

    # ---- Resolve ffmpeg (spawn processes don't inherit conda PATH) ----------
    # Priority: conda-env ffmpeg first (self-consistent libs), then PATH, then system.
    # The system /usr/bin/ffmpeg breaks under conda's LD_LIBRARY_PATH due to
    # libffi/libp11-kit version conflicts.
    import shutil
    _conda_ffmpeg = os.path.join(sys.prefix, "bin", "ffmpeg")
    if os.path.isfile(_conda_ffmpeg):
        ffmpeg_bin = _conda_ffmpeg
    else:
        ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin in (None, "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
        _ems = "/home/cogems_nist/anaconda3/envs/EMS-Pipeline/bin/ffmpeg"
        if os.path.isfile(_ems):
            ffmpeg_bin = _ems
    if ffmpeg_bin is None:
        # Last-resort: check other known conda envs on this machine
        _home = os.path.expanduser("~")
        for candidate in (
                f"{_home}/anaconda3/envs/EMS-Pipeline/bin/ffmpeg",
                f"{_home}/anaconda3/envs/cogems/bin/ffmpeg",
                "/usr/bin/ffmpeg",
                "/usr/local/bin/ffmpeg",
        ):
            if os.path.isfile(candidate):
                ffmpeg_bin = candidate
                break
    if ffmpeg_bin is None:
        print("[LocalSampler] ERROR: ffmpeg not found — aborting")
        return
    print(f"[LocalSampler] Using ffmpeg: {ffmpeg_bin}")

    # ---- Quick probe: verify ffmpeg can read the file at all ----------------
    probe_log = "/tmp/ffmpeg_probe.log"
    try:
        probe = subprocess.run(
            [ffmpeg_bin, "-v", "error", "-i", video_file,
             "-f", "null", "-vframes", "1", "-"],
            stdout=subprocess.DEVNULL,
            stderr=open(probe_log, "w"),
            timeout=10,
        )
        with open(probe_log) as f:
            probe_output = f.read().strip()
        if probe_output:
            print(f"[LocalSampler] ffmpeg probe warnings/errors:\n{probe_output}")
        if probe.returncode != 0:
            print(f"[LocalSampler] ERROR: ffmpeg probe failed (rc={probe.returncode}) — check {probe_log}")
            return
        print("[LocalSampler] ffmpeg probe OK")
    except subprocess.TimeoutExpired:
        print("[LocalSampler] WARNING: ffmpeg probe timed out — proceeding anyway")
    except Exception as exc:
        print(f"[LocalSampler] WARNING: ffmpeg probe error: {exc} — proceeding anyway")

    pipe_map = {
        1: PIPE_VIDEO,
        2: PIPE_AUDIO,
        3: PIPE_CSV,
        4: PIPE_VIDEO_ML,
        5: PIPE_AUDIO_ML,
        6: PIPE_CSV_ML,
    }

    # Which base types do we actually need?
    need_video = (1 in enabled_types) or (4 in enabled_types)
    need_audio = (2 in enabled_types) or (5 in enabled_types)
    need_csv   = (3 in enabled_types) or (6 in enabled_types)

    audio_sample_rate = 16000
    audio_channels    = 1
    frame_size        = width * height * 3          # BGR24
    audio_samples_per_frame = int(audio_sample_rate / fps)
    audio_bytes_per_frame   = audio_samples_per_frame * 2 * audio_channels  # int16

    opened_pipes: dict = {}

    def _open_pipe(data_type):
        path = pipe_map[data_type]
        try:
            if data_type in (3, 6):
                opened_pipes[data_type] = open(path, "w", buffering=1)
            else:
                opened_pipes[data_type] = open(path, "wb", buffering=0)
            print(f"[LocalSampler] Opened pipe {path}")
        except Exception as exc:
            print(f"[LocalSampler] ERROR opening {path}: {exc}")

    # Open all required pipes in parallel (each open() blocks until a reader
    # connects, so we must not open them sequentially).
    open_threads = [
        threading.Thread(target=_open_pipe, args=(dt,), daemon=True)
        for dt in enabled_types
        if dt in pipe_map
    ]
    for t in open_threads:
        t.start()
    for t in open_threads:
        t.join()

    if not opened_pipes:
        print("[LocalSampler] No pipes opened — aborting")
        return

    iteration = 0
    frames_sent = 0

    try:
        while running_flag.value:
            iteration += 1
            print(f"[LocalSampler] Starting playback iteration {iteration}")

            # ---- Load CSV lines ------------------------------------------------
            csv_lines = []
            if need_csv:
                try:
                    with open(csv_file, "r", newline="") as f:
                        reader = csv.reader(f)
                        next(reader, None)   # skip header
                        for row in reader:
                            csv_lines.append(",".join(row))
                    print(f"[LocalSampler] Loaded {len(csv_lines)} CSV lines")
                except Exception as exc:
                    print(f"[LocalSampler] ERROR reading CSV: {exc}")

            # ---- Launch FFmpeg decoders ----------------------------------------
            video_proc = None
            audio_proc = None

            if need_video:
                _vlog = open("/tmp/ffmpeg_video_stderr.log", "w")
                try:
                    video_proc = subprocess.Popen(
                        [
                            ffmpeg_bin, "-i", video_file,
                            "-vf", f"scale={width}:{height}",
                            "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "pipe:1",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=_vlog,
                        bufsize=10 ** 8,
                    )
                except Exception as exc:
                    _vlog.flush(); _vlog.close()
                    print(f"[LocalSampler] ERROR launching video ffmpeg: {exc}")
                    break

            if need_audio:
                _alog = open("/tmp/ffmpeg_audio_stderr.log", "w")
                try:
                    audio_proc = subprocess.Popen(
                        [
                            ffmpeg_bin, "-i", video_file,
                            "-f", "s16le",
                            "-ar", str(audio_sample_rate),
                            "-ac", str(audio_channels),
                            "pipe:1",
                        ],
                        stdout=subprocess.PIPE,
                        stderr=_alog,
                        bufsize=10 ** 8,
                    )
                except Exception as exc:
                    _alog.flush(); _alog.close()
                    print(f"[LocalSampler] ERROR launching audio ffmpeg: {exc}")
                    break

            # ---- Streaming loop ------------------------------------------------
            frame_idx   = 0
            start_time  = time.time()
            frame_time  = 1.0 / fps
            last_debug  = time.time()

            csv_exhausted = (not need_csv) or (len(csv_lines) == 0)

            try:
                while running_flag.value:
                    # Pace output to the requested fps
                    target = start_time + frame_idx * frame_time
                    wait   = target - time.time()
                    if wait > 0:
                        time.sleep(wait)

                    # --- Video ---
                    video_data = b""
                    if video_proc is not None:
                        video_data = video_proc.stdout.read(frame_size)
                        if len(video_data) != frame_size:
                            rc = video_proc.poll()
                            try:
                                _vlog.flush()
                            except Exception:
                                pass
                            print(
                                f"[LocalSampler] Video stream ended after {frame_idx} frames "
                                f"(got {len(video_data)}B expected {frame_size}B, ffmpeg rc={rc})"
                            )
                            if frame_idx == 0:
                                with open("/tmp/ffmpeg_video_stderr.log") as _f:
                                    _tail = _f.read()[-2000:]
                                print(f"[LocalSampler] ffmpeg video stderr tail:\n{_tail}")
                            break
                        video_bytes = len(video_data).to_bytes(4, "big") + video_data
                        if 1 in opened_pipes:
                            opened_pipes[1].write(video_bytes)
                            opened_pipes[1].flush()
                        if 4 in opened_pipes:
                            opened_pipes[4].write(video_bytes)
                            opened_pipes[4].flush()

                    # --- Audio ---
                    if audio_proc is not None:
                        audio_data = audio_proc.stdout.read(audio_bytes_per_frame)
                        if len(audio_data) != audio_bytes_per_frame:
                            print("[LocalSampler] Audio stream ended")
                            break
                        audio_with_len = len(audio_data).to_bytes(4, "big") + audio_data
                        if 2 in opened_pipes:
                            opened_pipes[2].write(audio_with_len)
                            opened_pipes[2].flush()
                        # ML pipe gets raw PCM (no length prefix) — same as SRT receiver
                        if 5 in opened_pipes:
                            opened_pipes[5].write(audio_data)
                            opened_pipes[5].flush()

                    # --- CSV ---
                    if need_csv and not csv_exhausted:
                        if frame_idx < len(csv_lines):
                            csv_line = csv_lines[frame_idx] + "\n"
                            if 3 in opened_pipes:
                                opened_pipes[3].write(csv_line)
                                opened_pipes[3].flush()
                            if 6 in opened_pipes:
                                opened_pipes[6].write(csv_line)
                                opened_pipes[6].flush()
                        else:
                            csv_exhausted = True
                            print("[LocalSampler] CSV lines exhausted")

                    frames_sent += 1
                    frame_idx   += 1

                    if frames_sent % 30 == 0:
                        elapsed = time.time() - last_debug
                        print(
                            f"[LocalSampler] Frames: {frames_sent:5d} | "
                            f"FPS: {30/elapsed:.1f}"
                        )
                        last_debug = time.time()

            finally:
                for proc in (video_proc, audio_proc):
                    if proc is not None:
                        try:
                            proc.terminate()
                            proc.wait(timeout=2)
                        except Exception:
                            try:
                                proc.kill()
                            except Exception:
                                pass

            # Loop or stop
            if not loop or not running_flag.value:
                print("[LocalSampler] Playback finished")
                break
            else:
                print("[LocalSampler] Looping...")

    except Exception as exc:
        print(f"[LocalSampler] Unexpected error: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        for pipe in opened_pipes.values():
            try:
                pipe.close()
            except Exception:
                pass

    print(f"[LocalSampler] Exiting — sent {frames_sent} frames total")


# ---------------------------------------------------------------------------
# Manager  (same interface as SRTReceiverProcess in srt_receiver.py)
# ---------------------------------------------------------------------------

class LocalFileSamplerProcess:
    """
    Drop-in replacement for SRTReceiverProcess when the data source is
    'local files'.  Accepts the same start()/stop() interface used by the GUI.
    """

    def __init__(
            self,
            video_file: str,
            csv_file: str,
            width: int = 480,
            height: int = 270,
            fps: int = 30,
            enabled_types: set = None,
            loop: bool = False,
    ):
        self.video_file    = video_file
        self.csv_file      = csv_file
        self.width         = width
        self.height        = height
        self.fps           = fps
        self.enabled_types = enabled_types or {1, 2, 3, 4, 5}
        self.loop          = loop
        self.last_error    = None

        import multiprocessing as mp
        self._ctx          = mp.get_context("spawn")
        self._running_flag = self._ctx.Value(ctypes.c_bool, False)
        self._process      = None

    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Validate files, then spawn the sampler process. Returns False on error."""
        if not os.path.isfile(self.video_file):
            self.last_error = f"Video file not found: {self.video_file}"
            print(f"[LocalFileSamplerProcess] {self.last_error}")
            return False
        if not os.path.isfile(self.csv_file):
            self.last_error = f"CSV file not found: {self.csv_file}"
            print(f"[LocalFileSamplerProcess] {self.last_error}")
            return False

        self._running_flag.value = True
        self._process = self._ctx.Process(
            target=local_file_sampler_process,
            args=(
                self.video_file,
                self.csv_file,
                self.width,
                self.height,
                self.fps,
                self.enabled_types,
                self._running_flag,
                self.loop,
            ),
            daemon=True,
        )
        self._process.start()
        print(f"[LocalFileSamplerProcess] Spawned PID={self._process.pid}")
        return True

    # ------------------------------------------------------------------
    def stop(self):
        print("[LocalFileSamplerProcess] Stopping...")
        self._running_flag.value = False

        if self._process is not None:
            self._process.join(timeout=4)
            if self._process.is_alive():
                print("[LocalFileSamplerProcess] Terminating...")
                self._process.terminate()
                self._process.join(timeout=2)
            if self._process.is_alive():
                self._process.kill()
            self._process = None

        print("[LocalFileSamplerProcess] Stopped")