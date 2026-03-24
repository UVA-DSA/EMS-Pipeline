import asyncio
import json
import logging
import os
import random
import socket
import queue
import threading
import time
import wave

import cv2
import numpy as np
from aiohttp import web
from aiortc import RTCPeerConnection, RTCIceCandidate, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
import qrcode

logging.basicConfig(level=logging.INFO)

pcs = set()
video_tasks = set()
audio_tasks = set()
data_tasks = set()

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
    assistance = [
        "Need extra light",
        "Hold camera steady",
        "Move closer",
        "Check blind spot",
    ]
    try:
        while channel.readyState == "open":
            send_all = random.random() < 0.5
            feedback = {}
            if send_all:
                feedback = {
                    "protocol": random.choice(protocols),
                    "action": random.choice(actions),
                    "assistance": random.choice(assistance),
                }
            else:
                kind = random.choice(["protocol", "action", "assistance"])
                if kind == "protocol":
                    feedback["protocol"] = random.choice(protocols)
                elif kind == "action":
                    feedback["action"] = random.choice(actions)
                else:
                    feedback["assistance"] = random.choice(assistance)
            payload = {
                "type": "feedback",
                "feedback": feedback,
            }
            channel.send(json.dumps(payload))
            await asyncio.sleep(1.0)
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

    if SEND_MOCK_FEEDBACK and channel.label == DETECTION_CHANNEL_LABEL:
        task = asyncio.create_task(send_mock_feedback(channel))
        data_tasks.add(task)
        task.add_done_callback(data_tasks.discard)


async def offer(request):
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)
    logging.info("Peer connection created")

    @pc.on("track")
    def on_track(track):
        logging.info(f"Track received: {track.kind}")
        if track.kind == "video":
            task = asyncio.create_task(display_video(track))
            video_tasks.add(task)
            task.add_done_callback(video_tasks.discard)
        elif track.kind == "audio":
            task = asyncio.create_task(play_audio(track) if PLAY_AUDIO else consume_audio(track))
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
    pcs.add(pc)
    logging.info("WebSocket peer connection created")

    @pc.on("track")
    def on_track(track):
        logging.info(f"Track received: {track.kind}")
        if track.kind == "video":
            task = asyncio.create_task(display_video(track))
            video_tasks.add(task)
            task.add_done_callback(video_tasks.discard)
        elif track.kind == "audio":
            task = asyncio.create_task(play_audio(track) if PLAY_AUDIO else consume_audio(track))
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


async def on_shutdown(app):
    logging.info("Shutting down")
    await asyncio.gather(*[pc.close() for pc in pcs], return_exceptions=True)
    pcs.clear()


def main():
    app = web.Application()
    app.router.add_post("/offer", offer)
    app.router.add_get("/ws", websocket_handler)
    app.on_shutdown.append(on_shutdown)
    print_connection_info()
    web.run_app(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
