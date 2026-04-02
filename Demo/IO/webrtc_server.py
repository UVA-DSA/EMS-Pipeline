import asyncio
import json
import logging
import os
import random
import socket
import time
import wave

import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCRtpReceiver, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
import qrcode

try:
    from IO.bbox_engine import BBoxBroker
except ImportError:
    from bbox_engine import BBoxBroker

try:
    from IO.feedback_engine import FeedbackBroker, build_feedback_payload
except ImportError:
    from feedback_engine import FeedbackBroker, build_feedback_payload

logging.basicConfig(level=logging.INFO)

pcs = set()
video_tasks = set()
audio_tasks = set()
data_tasks = set()
bbox_broker = BBoxBroker()
feedback_broker = FeedbackBroker()

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8080"))
DISPLAY_VIDEO = os.environ.get("DISPLAY_VIDEO", "1") == "1"
PRINT_QR = os.environ.get("PRINT_QR", "1") == "1"
SEND_MOCK_BBOX = os.environ.get("SEND_MOCK_BBOX", "0") == "1"
_send_mock_feedback_env = os.environ.get("SEND_MOCK_FEEDBACK")
if _send_mock_feedback_env is None:
    SEND_MOCK_FEEDBACK = SEND_MOCK_BBOX
else:
    SEND_MOCK_FEEDBACK = _send_mock_feedback_env == "1"
DETECTION_CHANNEL_LABEL = os.environ.get("DETECTION_CHANNEL_LABEL", "detections")
PLAY_AUDIO = os.environ.get("PLAY_AUDIO", "0") == "1"
FORCE_AUDIO_CHANNELS = int(os.environ.get("FORCE_AUDIO_CHANNELS", "0") or "0")
AUDIO_DEVICE = os.environ.get("AUDIO_DEVICE", "").strip()
AUDIO_GAIN = float(os.environ.get("AUDIO_GAIN", "1.0"))
# AUDIO_OUTPUT_RATE = int(os.environ.get("AUDIO_OUTPUT_RATE", "0") or "0")
AUDIO_OUTPUT_RATE = 48000
RECORD_AUDIO = os.environ.get("RECORD_AUDIO", "1") == "1"
PIPE_VIDEO_PATH = os.environ.get("PIPE_VIDEO_PATH", "").strip()
PIPE_AUDIO_PATH = os.environ.get("PIPE_AUDIO_PATH", "").strip()
PIPE_AUDIO_ML_PATH = os.environ.get("PIPE_AUDIO_ML_PATH", "").strip()
PIPE_VIDEO_WIDTH = int(os.environ.get("PIPE_VIDEO_WIDTH", "480") or "480")
PIPE_VIDEO_HEIGHT = int(os.environ.get("PIPE_VIDEO_HEIGHT", "270") or "270")


def get_lan_ip():
    ip = "127.0.0.1"
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
    except OSError:
        pass
    finally:
        sock.close()
    return ip


def print_connection_info():
    base_url = os.environ.get("SERVER_BASE_URL")
    if not base_url:
        base_url = f"http://{get_lan_ip()}:{PORT}"

    offer_url = f"{base_url}/offer"
    ws_url = offer_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = ws_url.replace("/offer", "/ws")
    logging.info(f"Server listening on http://{HOST}:{PORT}")
    logging.info(f"Offer URL: {offer_url}")
    logging.info(f"WebSocket URL: {ws_url}")

    if PRINT_QR:
        logging.info("QR code (scan to get the WebSocket URL):")
        qr = qrcode.QRCode(border=1)
        qr.add_data(ws_url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)


class LengthPrefixedPipeWriter:
    def __init__(self, *paths):
        self.paths = [path for path in paths if path]
        self.handles = []

    @property
    def enabled(self):
        return bool(self.paths)

    def open(self):
        if self.handles:
            return
        for path in self.paths:
            self.handles.append(open(path, "wb", buffering=0))
            logging.info("Opened pipe writer: %s", path)

    def write(self, payload):
        if not self.enabled:
            return
        if not self.handles:
            self.open()

        packet = len(payload).to_bytes(4, "big") + payload
        for handle in self.handles:
            handle.write(packet)
            handle.flush()

    def close(self):
        while self.handles:
            handle = self.handles.pop()
            try:
                handle.close()
            except Exception:
                pass


class RawPipeWriter:
    def __init__(self, *paths):
        self.paths = [path for path in paths if path]
        self.handles = []

    @property
    def enabled(self):
        return bool(self.paths)

    def open(self):
        if self.handles:
            return
        for path in self.paths:
            self.handles.append(open(path, "wb", buffering=0))
            logging.info("Opened raw pipe writer: %s", path)

    def write(self, payload):
        if not self.enabled:
            return
        if not self.handles:
            self.open()

        for handle in self.handles:
            handle.write(payload)
            handle.flush()

    def close(self):
        while self.handles:
            handle = self.handles.pop()
            try:
                handle.close()
            except Exception:
                pass


def _has_video_pipe_output():
    return bool(PIPE_VIDEO_PATH)


def _has_audio_pipe_output():
    return bool(PIPE_AUDIO_PATH or PIPE_AUDIO_ML_PATH)


def configure_video_receiver(pc):
    """
    Work around aiortc 1.10.x crashing on negotiated RTX packets by excluding
    RTX from the server's preferred receive codecs.
    """
    if not (_has_video_pipe_output() or DISPLAY_VIDEO):
        return

    transceiver = pc.addTransceiver("video", direction="recvonly")
    capabilities = RTCRtpReceiver.getCapabilities("video")
    preferred_codecs = [
        codec
        for codec in capabilities.codecs
        if codec.mimeType.lower() != "video/rtx"
    ]
    transceiver.setCodecPreferences(preferred_codecs)
    logging.info(
        "Configured video receive codecs without RTX: %s",
        ", ".join(codec.mimeType for codec in preferred_codecs),
    )


async def mirror_video_to_pipes(track):
    logging.info("Video pipe output enabled")
    writer = LengthPrefixedPipeWriter(PIPE_VIDEO_PATH)
    frames_mirrored = 0
    video_idle = False
    try:
        while True:
            try:
                frame = await asyncio.wait_for(track.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                if not video_idle:
                    logging.warning("No video frames received from WebRTC track for 5.0s")
                    video_idle = True
                continue
            if video_idle:
                logging.info("WebRTC video track resumed")
                video_idle = False
            image = frame.to_ndarray(format="bgr24")
            frames_mirrored += 1

            if DISPLAY_VIDEO:
                cv2.imshow("Server View", image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            resized = cv2.resize(
                image,
                (PIPE_VIDEO_WIDTH, PIPE_VIDEO_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )
            writer.write(resized.tobytes())
            if frames_mirrored % 120 == 0:
                logging.info("Mirrored %s video frames to pipe", frames_mirrored)
    except Exception as exc:
        logging.info("Video pipe output stopped: %s", exc)
    finally:
        writer.close()
        if DISPLAY_VIDEO:
            cv2.destroyAllWindows()


async def mirror_audio_to_pipes(track):
    logging.info("Audio pipe output enabled")
    try:
        import av
    except Exception as exc:
        logging.info("PyAV not available for audio pipe output: %s", exc)
        await consume_audio(track)
        return

    playback_writer = LengthPrefixedPipeWriter(PIPE_AUDIO_PATH)
    ml_writer = RawPipeWriter(PIPE_AUDIO_ML_PATH)
    resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)

    try:
        while True:
            frame = await track.recv()
            frames = resampler.resample(frame)
            if frames is None:
                continue
            if not isinstance(frames, (list, tuple)):
                frames = [frames]

            for resampled in frames:
                pcm = np.ascontiguousarray(resampled.to_ndarray().reshape(-1), dtype=np.int16)
                if pcm.size:
                    payload = pcm.tobytes()
                    playback_writer.write(payload)
                    ml_writer.write(payload)
    except Exception as exc:
        logging.info("Audio pipe output stopped: %s", exc)
    finally:
        playback_writer.close()
        ml_writer.close()


async def handle_video_track(track):
    if _has_video_pipe_output():
        await mirror_video_to_pipes(track)
    else:
        await display_video(track)


async def handle_audio_track(track):
    if _has_audio_pipe_output():
        await mirror_audio_to_pipes(track)
    elif PLAY_AUDIO:
        await play_audio(track)
    else:
        await consume_audio(track)


async def display_video(track):
    logging.info("Video track started")
    try:
        while True:
            frame = await track.recv()
            if not DISPLAY_VIDEO:
                continue
            img = frame.to_ndarray(format="bgr24")
            cv2.imshow("Server View", img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except Exception as exc:
        logging.info(f"Video track ended: {exc}")
    finally:
        if DISPLAY_VIDEO:
            cv2.destroyAllWindows()


async def consume_audio(track):
    logging.info("Audio track started")
    try:
        while True:
            await track.recv()
    except Exception as exc:
        logging.info(f"Audio track ended: {exc}")


def _audio_process_main(queue_handle, stop_event, sample_rate, channels, device):
    try:
        import sounddevice as sd
    except Exception:
        return

    stream = None
    try:
        stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=channels,
            dtype="float32",
            blocksize=0,
            latency="low",
            device=device,
        )
        stream.start()
        while not stop_event.is_set():
            data = queue_handle.get()
            if data is None:
                break
            try:
                stream.write(data)
            except Exception:
                break
    finally:
        if stream is not None:
            try:
                stream.stop()
            finally:
                stream.close()


class AudioPlayer:
    def __init__(self, sample_rate, channels, max_queue=50, device=None):
        self.sample_rate = sample_rate
        self.channels = channels
        self.max_queue = max_queue
        self.queue = None
        self.process = None
        self.stop_event = None
        self.device = device

    def start(self):
        try:
            import sounddevice as _sd  # noqa: F401
        except Exception as exc:
            raise RuntimeError(f"sounddevice not available: {exc}") from exc

        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.queue = ctx.Queue(maxsize=self.max_queue)
        self.stop_event = ctx.Event()
        self.process = ctx.Process(
            target=_audio_process_main,
            args=(self.queue, self.stop_event, self.sample_rate, self.channels, self.device),
            daemon=True,
        )
        self.process.start()

    def put(self, data):
        if self.queue is None:
            return
        if self.queue.full():
            try:
                _ = self.queue.get_nowait()
            except Exception:
                pass
        try:
            self.queue.put_nowait(data)
        except Exception:
            pass

    def close(self):
        if self.stop_event is not None:
            self.stop_event.set()
        if self.queue is not None:
            try:
                self.queue.put_nowait(None)
            except Exception:
                pass
        if self.process is not None:
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.terminate()


class AudioRecorder:
    def __init__(self, directory="recordings", prefix="audio"):
        self.directory = directory
        self.prefix = prefix
        self.wave = None
        self.path = None
        self.sample_rate = None
        self.channels = None

    def start(self, sample_rate, channels):
        if self.wave is not None:
            self.close()
        os.makedirs(self.directory, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(
            self.directory, f"{self.prefix}_{ts}_{sample_rate}hz_{channels}ch.wav"
        )
        self.wave = wave.open(self.path, "wb")
        self.wave.setnchannels(channels)
        self.wave.setsampwidth(2)
        self.wave.setframerate(sample_rate)
        self.sample_rate = sample_rate
        self.channels = channels
        logging.info("Recording audio to %s", self.path)

    def ensure(self, sample_rate, channels):
        if self.wave is None:
            self.start(sample_rate, channels)
        elif self.sample_rate != sample_rate or self.channels != channels:
            self.start(sample_rate, channels)

    def write(self, data):
        if self.wave is None:
            return
        pcm = np.clip(data, -1.0, 1.0)
        pcm = (pcm * 32767.0).astype(np.int16, copy=False)
        self.wave.writeframes(pcm.tobytes())

    def close(self):
        if self.wave is not None:
            try:
                self.wave.close()
            finally:
                self.wave = None
                if self.path:
                    logging.info("Saved audio recording: %s", self.path)


async def play_audio(track):
    logging.info("Audio playback enabled")
    try:
        import sounddevice as sd
    except Exception as exc:
        logging.info(f"sounddevice not available, falling back to consume: {exc}")
        await consume_audio(track)
        return
    try:
        import av
    except Exception as exc:
        av = None
        logging.info(f"PyAV resampler not available: {exc}")

    player = None
    current_rate = None
    current_channels = None
    output_device = None
    output_rate = None
    resampler = None
    resampler_config = None
    recorder = AudioRecorder() if RECORD_AUDIO else None

    def resolve_output_device():
        nonlocal output_device
        if not AUDIO_DEVICE:
            return
        try:
            if AUDIO_DEVICE.isdigit():
                output_device = int(AUDIO_DEVICE)
                return
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                name = dev.get("name", "")
                if AUDIO_DEVICE.lower() in name.lower():
                    output_device = idx
                    return
        except Exception:
            output_device = None

    def query_max_output_channels():
        try:
            info = sd.query_devices(output_device, kind="output")
            val = info.get("max_output_channels", 2)
            if isinstance(val, (list, tuple)):
                val = val[0] if val else None
            if isinstance(val, np.ndarray):
                val = val.item() if val.size == 1 else (val.flat[0] if val.size else None)
            if isinstance(val, np.generic):
                val = val.item()
            if isinstance(val, (int, float)) and val > 0:
                return int(val)
            return 2
        except Exception:
            return 2
    def query_output_sample_rate():
        if AUDIO_OUTPUT_RATE > 0:
            return AUDIO_OUTPUT_RATE
        try:
            info = sd.query_devices(output_device, kind="output")
            rate = info.get("default_samplerate")
            if isinstance(rate, (list, tuple)):
                rate = rate[0] if rate else None
            if isinstance(rate, np.ndarray):
                rate = rate.item() if rate.size == 1 else (rate.flat[0] if rate.size else None)
            if isinstance(rate, np.generic):
                rate = rate.item()
            if isinstance(rate, (int, float)) and rate > 0:
                return int(rate)
            return None
        except Exception:
            return None

    def coerce_int(value, default=None):
        if isinstance(value, (list, tuple)):
            value = value[0] if value else None
        if isinstance(value, np.ndarray):
            value = value.item() if value.size == 1 else (value.flat[0] if value.size else None)
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float)):
            return int(value)
        return default

    resolve_output_device()
    max_output_channels = query_max_output_channels()
    print(f"Resolved audio output device: {output_device}, max channels: {max_output_channels}")
    output_rate = query_output_sample_rate()
    try:
        if output_device is not None:
            info = sd.query_devices(output_device, kind="output")
        else:
            info = sd.query_devices(kind="output")
        logging.info(
            "Audio output device: %s | default_samplerate=%s | max_output_channels=%s",
            info.get("name"),
            info.get("default_samplerate"),
            info.get("max_output_channels"),
        )
    except Exception as exc:
        logging.info("Audio output device info unavailable: %s", exc)
    if max_output_channels < 1:
        max_output_channels = 1
    if FORCE_AUDIO_CHANNELS in (1, 2):
        max_output_channels = FORCE_AUDIO_CHANNELS
    else:
        max_output_channels = 2 if max_output_channels >= 2 else 1

    def start_player(sample_rate, channels):
        nonlocal player, current_rate, current_channels
        if player is not None:
            player.close()
            player = None
        last_exc = None
        for ch in [channels, 1, 2]:
            try:
                player = AudioPlayer(sample_rate, ch, device=output_device)
                player.start()
                current_rate = sample_rate
                current_channels = ch
                logging.info(f"Audio output started: {sample_rate} Hz, {ch} channel(s)")
                return True
            except Exception as exc:
                last_exc = exc
        logging.info(f"Audio playback stopped: {last_exc}")
        return False

    def ensure_resampler(in_rate, out_rate, out_channels):
        nonlocal resampler, resampler_config
        if av is None:
            return None
        layout = "mono" if out_channels == 1 else "stereo"
        key = (in_rate, out_rate, out_channels)
        if resampler is None or resampler_config != key:
            resampler = av.AudioResampler(format="flt", layout=layout, rate=out_rate)
            resampler_config = key
        return resampler
    try:
        while True:
            frame = await track.recv()
            sample_rate = frame.sample_rate
            print(f"Received audio frame: {frame.format.name}, {sample_rate} Hz, layout={getattr(frame, 'layout', None)}")

            # Trust frame metadata for channel count.
            channels = 2


            target_rate = output_rate or sample_rate
            target_channels = channels

            print(f"Processing audio frame: {sample_rate} Hz, {channels} channel(s) -> {target_rate} Hz, {target_channels} channel(s)")
            frames = [frame]
            if av is not None and (sample_rate != target_rate or channels != target_channels):
                resampler = ensure_resampler(sample_rate, target_rate, target_channels)
                try:
                    frames = resampler.resample(frame)
                except Exception:
                    frames = [frame]

            for rframe in frames:
                samples = rframe.to_ndarray()
                print(f"Resampled audio frame: {samples.shape}, {rframe.format.name}, {rframe.sample_rate} Hz, layout={getattr(rframe, 'layout', None)}")
                sample_rate = rframe.sample_rate

                # print(
                #     f"Received audio frame: {samples.shape}, {rframe.format.name}, {sample_rate} Hz"
                # )

                # Normalize to interleaved float32: shape (frames, channels)
                rchannels = getattr(getattr(rframe, "layout", None), "channels", None)
                if rchannels is None:
                    rchannels = getattr(rframe, "channels", None)
                if rchannels is None:
                    rchannels = target_channels
                rchannels = coerce_int(rchannels, target_channels)

                # rchannels inferred from layout (stereo -> 2)
                # Handle packed/interleaved returned as (1, frames*channels)
                if samples.ndim == 2 and samples.shape[0] == 1 and rchannels > 1:
                    n = samples.shape[1]
                    if n % rchannels == 0:
                        data = samples.reshape(-1, rchannels)   # (frames, channels)
                    else:
                        # fallback
                        data = samples.T
                elif samples.ndim == 1:
                    # packed/interleaved returned as (frames*channels,)
                    if rchannels > 1 and (samples.size % rchannels == 0):
                        data = samples.reshape(-1, rchannels)
                    else:
                        data = samples.reshape(-1, 1)
                        rchannels = 1
                else:
                    # existing logic for planar etc.
                    if samples.shape[0] == rchannels:
                        data = samples.T
                    else:
                        data = samples

                print(f"Audio frame data shape: {data.shape}, dtype={data.dtype}")
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                elif data.dtype != np.float32:
                    data = data.astype(np.float32)
                data = np.ascontiguousarray(data, dtype=np.float32)
                if AUDIO_GAIN != 1.0:
                    data *= AUDIO_GAIN
                data = np.clip(data, -1.0, 1.0)

                if target_channels == 1 and rchannels > 1:
                    data = np.mean(data, axis=1, keepdims=True)
                elif target_channels == 2:
                    if rchannels == 1:
                        data = np.repeat(data, 2, axis=1)
                    elif rchannels > 2:
                        mono = np.mean(data, axis=1, keepdims=True)
                        data = np.repeat(mono, 2, axis=1)

                if player is None or sample_rate != current_rate or target_channels != current_channels:
                    if not start_player(sample_rate, target_channels):
                        await consume_audio(track)
                        return

                if recorder is not None:
                    recorder.ensure(sample_rate, data.shape[1])
                    recorder.write(data)

                player.put(data)
    except Exception as exc:
        logging.info(f"Audio playback stopped: {exc}")
    finally:
        if player is not None:
            player.close()
        if recorder is not None:
            recorder.close()

async def send_mock_feedback(channel):
    protocols = [
        "Maintain safe distance",
        "Watch cross-traffic",
        "Stop and scan",
        "Stay centered",
    ]
    actions = [
        "Turn left",
        "Turn right",
        "Slow down",
        "Proceed forward",
    ]
    try:
        while channel.readyState == "open":
            if random.random() < 0.5:
                payload = build_feedback_payload(
                    action=random.choice(actions),
                    protocol="",
                )
            else:
                payload = build_feedback_payload(
                    action="",
                    protocol=random.choice(protocols),
                )
            channel.send(json.dumps(payload))
            await asyncio.sleep(1.0)
    except Exception as exc:
        logging.info(f"DataChannel sender stopped: {exc}")


async def send_feedback_updates(channel):
    last_version = feedback_broker.current_version
    try:
        logging.info(
            "Feedback sender task started for channel=%s version=%s current=%s",
            channel.label,
            last_version,
            feedback_broker.current_feedback,
        )
        current_feedback = feedback_broker.current_feedback
        if current_feedback.get("action") or current_feedback.get("protocol"):
            payload = build_feedback_payload(
                action=current_feedback.get("action", ""),
                protocol=current_feedback.get("protocol", ""),
            )
            logging.info("Sending initial feedback snapshot: %s", payload["feedback"])
            channel.send(json.dumps(payload))
        else:
            logging.info("No initial feedback snapshot available for channel=%s", channel.label)

        while channel.readyState == "open":
            await asyncio.sleep(0.1)

            current_version = feedback_broker.current_version
            if current_version <= last_version:
                continue

            last_version = current_version
            feedback = feedback_broker.current_feedback
            logging.info(
                "Feedback broker produced update for channel=%s version=%s feedback=%s",
                channel.label,
                last_version,
                feedback,
            )
            payload = build_feedback_payload(
                action=feedback.get("action", ""),
                protocol=feedback.get("protocol", ""),
            )
            logging.info("Sending feedback update: %s", payload["feedback"])
            channel.send(json.dumps(payload))
    except Exception as exc:
        logging.info(f"DataChannel sender stopped: {exc}")


async def send_bbox_updates(channel):
    last_version = bbox_broker.current_version
    try:
        logging.info(
            "BBox sender task started for channel=%s version=%s current=%s",
            channel.label,
            last_version,
            bbox_broker.current_payload,
        )
        current_payload = bbox_broker.current_payload
        if current_payload is not None:
            logging.info(
                "Sending initial bbox snapshot: frame_id=%s boxes=%s",
                current_payload.get("frameId"),
                len(current_payload.get("boxes", [])),
            )
            channel.send(json.dumps(current_payload))
        else:
            logging.info("No initial bbox snapshot available for channel=%s", channel.label)

        while channel.readyState == "open":
            await asyncio.sleep(0.1)

            current_version = bbox_broker.current_version
            if current_version <= last_version:
                continue

            last_version = current_version
            payload = bbox_broker.current_payload
            if payload is None:
                continue

            logging.info(
                "BBox broker produced update for channel=%s version=%s "
                "frame_id=%s boxes=%s",
                channel.label,
                last_version,
                payload.get("frameId"),
                len(payload.get("boxes", [])),
            )
            channel.send(json.dumps(payload))
    except Exception as exc:
        logging.info(f"DataChannel sender stopped: {exc}")


async def send_mock_bboxes(channel):
    frame_id = 0
    try:
        while channel.readyState == "open":
            frame_id += 1
            payload = {
                "type": "bbox",
                "frameId": frame_id,
                "timestampMs": int(time.time() * 1000),
                "boxes": [
                    {
                        "x": round(random.uniform(0.1, 0.6), 3),
                        "y": round(random.uniform(0.1, 0.6), 3),
                        "w": 0.25,
                        "h": 0.35,
                        "label": "object",
                        "score": 0.9,
                    }
                ],
            }
            channel.send(json.dumps(payload))
            await asyncio.sleep(0.2)
    except Exception as exc:
        logging.info(f"DataChannel sender stopped: {exc}")


def attach_datachannel_handlers(channel):
    logging.info(f"DataChannel opened: {channel.label}")

    @channel.on("close")
    def on_close():
        logging.info(f"DataChannel closed: {channel.label}")

    if SEND_MOCK_BBOX and channel.label == DETECTION_CHANNEL_LABEL:
        task = asyncio.create_task(send_mock_bboxes(channel))
        data_tasks.add(task)
        task.add_done_callback(data_tasks.discard)
    elif channel.label == DETECTION_CHANNEL_LABEL:
        task = asyncio.create_task(send_bbox_updates(channel))
        data_tasks.add(task)
        task.add_done_callback(data_tasks.discard)

    if channel.label == DETECTION_CHANNEL_LABEL:
        if SEND_MOCK_FEEDBACK:
            task = asyncio.create_task(send_mock_feedback(channel))
        else:
            task = asyncio.create_task(send_feedback_updates(channel))
        data_tasks.add(task)
        task.add_done_callback(data_tasks.discard)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    configure_video_receiver(pc)
    pcs.add(pc)
    logging.info("Peer connection created")

    @pc.on("track")
    def on_track(track):
        logging.info(f"Track received: {track.kind}")
        if track.kind == "video":
            task = asyncio.create_task(handle_video_track(track))
            video_tasks.add(task)
            task.add_done_callback(video_tasks.discard)
        elif track.kind == "audio":
            task = asyncio.create_task(handle_audio_track(track))
            audio_tasks.add(task)
            task.add_done_callback(audio_tasks.discard)

    @pc.on("datachannel")
    def on_datachannel(channel):
        attach_datachannel_handlers(channel)

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return web.Response(
        content_type="application/json",
        text=json.dumps(
            {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}
        ),
    )


async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    pc = RTCPeerConnection()
    configure_video_receiver(pc)
    pcs.add(pc)
    logging.info("WebSocket peer connection created")

    @pc.on("track")
    def on_track(track):
        logging.info(f"Track received: {track.kind}")
        if track.kind == "video":
            task = asyncio.create_task(handle_video_track(track))
            video_tasks.add(task)
            task.add_done_callback(video_tasks.discard)
        elif track.kind == "audio":
            task = asyncio.create_task(handle_audio_track(track))
            audio_tasks.add(task)
            task.add_done_callback(audio_tasks.discard)

    @pc.on("datachannel")
    def on_datachannel(channel):
        attach_datachannel_handlers(channel)

    @pc.on("icecandidate")
    async def on_icecandidate(event):
        if event is None:
            return
        await ws.send_json(
            {
                "type": "candidate",
                "candidate": {
                    "candidate": candidate_to_sdp(event),
                    "sdpMid": event.sdpMid,
                    "sdpMLineIndex": event.sdpMLineIndex,
                },
            }
        )

    try:
        async for msg in ws:
            if msg.type != web.WSMsgType.TEXT:
                continue
            data = json.loads(msg.data)
            msg_type = data.get("type")
            if msg_type == "offer":
                offer = RTCSessionDescription(sdp=data["sdp"], type="offer")
                await pc.setRemoteDescription(offer)
                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)
                await ws.send_json(
                    {"type": "answer", "sdp": pc.localDescription.sdp}
                )
            elif msg_type == "candidate":
                candidate = data.get("candidate")
                if not candidate:
                    continue
                candidate_sdp = candidate.get("candidate")
                if not candidate_sdp:
                    continue
                ice = candidate_from_sdp(candidate_sdp)
                ice.sdpMid = candidate.get("sdpMid")
                ice.sdpMLineIndex = candidate.get("sdpMLineIndex")
                await pc.addIceCandidate(ice)
            elif msg_type == "bye":
                break
    finally:
        await pc.close()
        pcs.discard(pc)

    return ws


async def on_startup(app):
    if SEND_MOCK_BBOX:
        logging.info("Mock bbox enabled; skipping bbox broker startup")
    else:
        await bbox_broker.start()
        logging.info(
            "BBox broker startup complete: version=%s current=%s",
            bbox_broker.current_version,
            bbox_broker.current_payload,
        )

    if SEND_MOCK_FEEDBACK:
        logging.info("Mock feedback enabled; skipping feedback broker startup")
    else:
        await feedback_broker.start()
        logging.info(
            "Feedback broker startup complete: version=%s current=%s",
            feedback_broker.current_version,
            feedback_broker.current_feedback,
        )


async def on_shutdown(app):
    logging.info("Shutting down")
    await asyncio.gather(*[pc.close() for pc in pcs], return_exceptions=True)
    pcs.clear()
    if not SEND_MOCK_BBOX:
        await bbox_broker.stop()
    if not SEND_MOCK_FEEDBACK:
        await feedback_broker.stop()


def main():
    app = web.Application()
    app.router.add_post("/offer", offer)
    app.router.add_get("/ws", websocket_handler)
    app.on_startup.append(on_startup)
    app.on_shutdown.append(on_shutdown)
    print_connection_info()
    web.run_app(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
