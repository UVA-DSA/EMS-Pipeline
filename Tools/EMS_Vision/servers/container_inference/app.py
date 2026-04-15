import argparse
import asyncio
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from aiohttp import web

from container_inference.backends import DETRTensorRTBackend, MTRSAPTensorRTBackend


def build_detr_backend(args: argparse.Namespace) -> DETRTensorRTBackend:
    return DETRTensorRTBackend(
        engine_path=args.engine,
        detr_version=args.detr_version,
        threshold=args.threshold,
        engine_height=args.engine_height,
        engine_width=args.engine_width,
        device=args.device,
    )


def build_activity_backend(args: argparse.Namespace) -> Optional[MTRSAPTensorRTBackend]:
    if args.activity_engine is None:
        return None
    return MTRSAPTensorRTBackend(
        engine_path=args.activity_engine,
        class_map_path=args.activity_class_map,
        window_size=args.activity_window_size,
        stride=args.activity_stride,
        resize_short_side=args.activity_resize_short_side,
        center_crop_size=args.activity_center_crop_size,
        model_seq_len=args.activity_model_seq_len,
        feature_engine_path=args.activity_feature_engine,
        feature_extractor_weights=args.activity_feature_weights,
        device=args.device,
    )


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    if not image_bytes:
        raise ValueError("No image bytes were provided.")
    buffer = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image bytes.")
    return image


async def parse_request_payload(request: web.Request) -> Tuple[bytes, Optional[str], Optional[str]]:
    content_type = request.content_type.lower()
    if content_type.startswith("multipart/"):
        reader = await request.multipart()
        image_bytes = b""
        frame_id = None
        timestamp = None

        async for part in reader:
            if part.name == "image":
                image_bytes = await part.read(decode=False)
            elif part.name == "frame_id":
                frame_id = (await part.text()).strip() or None
            elif part.name == "timestamp":
                timestamp = (await part.text()).strip() or None

        return image_bytes, frame_id, timestamp

    image_bytes = await request.read()
    frame_id = request.headers.get("x-frame-id")
    timestamp = request.headers.get("x-timestamp")
    return image_bytes, frame_id, timestamp


async def health(request: web.Request) -> web.Response:
    detr_backend: DETRTensorRTBackend = request.app["detr_backend"]
    activity_backend: Optional[MTRSAPTensorRTBackend] = request.app["activity_backend"]
    activity_enabled = activity_backend is not None
    return web.json_response(
        {
            "status": "ok",
            "backends": {
                "detr": detr_backend.metadata(),
                "activity": activity_backend.metadata() if activity_enabled else {"enabled": False},
            },
        }
    )


@web.middleware
async def json_error_middleware(request: web.Request, handler):
    try:
        return await handler(request)
    except web.HTTPException:
        raise
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    except FileNotFoundError as exc:
        return web.json_response({"error": str(exc)}, status=404)
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=500)


async def infer_detr(request: web.Request) -> web.Response:
    backend: DETRTensorRTBackend = request.app["detr_backend"]
    lock: asyncio.Lock = request.app["detr_lock"]

    image_bytes, frame_id, timestamp = await parse_request_payload(request)
    image_bgr = decode_image_bytes(image_bytes)

    async with lock:
        result = backend.infer(image_bgr=image_bgr, frame_id=frame_id, timestamp=timestamp)

    return web.json_response(result.to_dict())


async def infer_activity(request: web.Request) -> web.Response:
    backend: Optional[MTRSAPTensorRTBackend] = request.app["activity_backend"]
    if backend is None:
        return web.json_response({"error": "Activity backend is disabled."}, status=503)

    stream_id = request.match_info["stream_id"].strip()
    lock: asyncio.Lock = request.app["activity_lock"]

    image_bytes, frame_id, timestamp = await parse_request_payload(request)
    image_bgr = decode_image_bytes(image_bytes)

    async with lock:
        result = backend.infer_stream_frame(
            stream_id=stream_id,
            image_bgr=image_bgr,
            frame_id=frame_id,
            timestamp=timestamp,
        )

    return web.json_response(result.to_dict())


async def reset_activity_stream(request: web.Request) -> web.Response:
    backend: Optional[MTRSAPTensorRTBackend] = request.app["activity_backend"]
    if backend is None:
        return web.json_response({"error": "Activity backend is disabled."}, status=503)

    stream_id = request.match_info["stream_id"].strip()
    cleared = backend.reset_stream(stream_id)
    return web.json_response(
        {
            "stream_id": stream_id,
            "cleared": cleared,
        }
    )


def build_parser(default_detr_engine: Path, default_activity_class_map: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Container-friendly realtime inference server.")
    parser.add_argument("--backend", type=str, default="detr-trt", choices=["detr-trt"], help="Kept for backward compatibility.")
    parser.add_argument("--engine", type=Path, default=default_detr_engine, help="Path to the DETR TensorRT engine (.ts/.ep).")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="HTTP bind host.")
    parser.add_argument("--port", type=int, default=8000, help="HTTP bind port.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device string.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations before serving requests.")
    parser.add_argument("--detr-version", type=str, default="ems", choices=["ems", "base"], help="DETR class set.")
    parser.add_argument("--threshold", type=float, default=0.7, help="Detection confidence threshold.")
    parser.add_argument("--engine-height", type=int, default=480, help="DETR compile-time engine input height.")
    parser.add_argument("--engine-width", type=int, default=640, help="DETR compile-time engine input width.")
    parser.add_argument("--activity-engine", type=Path, default=None, help="Optional path to the MTRSAP TensorRT engine (.ts/.ep).")
    parser.add_argument(
        "--activity-feature-engine",
        type=Path,
        default=None,
        help="Optional path to the ResNet50 TensorRT feature extractor (.ts/.ep). If omitted, PyTorch ResNet50 is used.",
    )
    parser.add_argument(
        "--activity-class-map",
        type=Path,
        default=default_activity_class_map,
        help="Path to activity class mapping JSON for human-readable labels.",
    )
    parser.add_argument("--activity-window-size", type=int, default=30, help="Number of buffered feature steps per activity window.")
    parser.add_argument("--activity-stride", type=int, default=1, help="How many new frames to wait between activity inferences.")
    parser.add_argument(
        "--activity-model-seq-len",
        type=int,
        default=None,
        help="Optional compile-time sequence length expected by the MTRSAP TensorRT engine.",
    )
    parser.add_argument(
        "--activity-resize-short-side",
        type=int,
        default=256,
        help="Resize the shorter side to this value before center crop for feature extraction.",
    )
    parser.add_argument(
        "--activity-center-crop-size",
        type=int,
        default=224,
        help="Center crop size for activity feature extraction.",
    )
    parser.add_argument(
        "--activity-feature-weights",
        type=str,
        default="imagenet1k_v1",
        choices=["imagenet1k_v1", "none"],
        help="Weights used by the current ResNet50 feature extractor implementation.",
    )
    return parser


def create_app(args: argparse.Namespace) -> web.Application:
    detr_backend = build_detr_backend(args)
    detr_backend.warmup(args.warmup)

    activity_backend = build_activity_backend(args)
    if activity_backend is not None:
        activity_backend.warmup(args.warmup)

    app = web.Application(client_max_size=32 * 1024 * 1024, middlewares=[json_error_middleware])
    app["detr_backend"] = detr_backend
    app["activity_backend"] = activity_backend
    app["detr_lock"] = asyncio.Lock()
    app["activity_lock"] = asyncio.Lock()
    app.add_routes(
        [
            web.get("/health", health),
            web.post("/infer", infer_detr),
            web.post("/infer/detr", infer_detr),
            web.post("/infer/activity/{stream_id}", infer_activity),
            web.post("/infer/activity/{stream_id}/reset", reset_activity_stream),
        ]
    )
    return app


def main() -> None:
    inference_root = Path(__file__).resolve().parents[2]
    default_detr_engine = inference_root / "checkpoints" / "ems_finetuned_detr_trt.ts"
    default_activity_class_map = inference_root / "conversion" / "action_class_mapping.json"
    parser = build_parser(default_detr_engine=default_detr_engine, default_activity_class_map=default_activity_class_map)
    args = parser.parse_args()

    app = create_app(args)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
