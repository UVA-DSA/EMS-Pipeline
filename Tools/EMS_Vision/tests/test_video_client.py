import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import requests


# Input source configuration.
USE_WEBCAM = False
WEBCAM_INDEX = 1

# Set this to the video you want to test when USE_WEBCAM is False.
VIDEO_PATH = Path("/mnt/f/repos/EgoEMS/Tools/inference/videos/GX010335_encoded_trimmed.mp4")

# Server settings.
SERVER_BASE_URL = "http://localhost:8000"
DETR_ENDPOINT = "/infer/detr"
ACTIVITY_ENDPOINT_TEMPLATE = "/infer/activity/{stream_id}"
ACTIVITY_STREAM_ID = "test_stream"
REQUEST_TIMEOUT_SECONDS = 30

# Optional throttles for easier debugging.
MAX_FRAMES = None
SEND_EVERY_NTH_FRAME = 1
JPEG_QUALITY = 90
SHOW_PREVIEW = False
SAVE_OUTPUT_VIDEO = True
PREVIEW_WINDOW_NAME = "Server Test Client"


def format_detections(detections: List[Dict[str, Any]]) -> str:
    if not detections:
        return "no detections"

    parts = []
    for det in detections:
        label = det.get("label", "unknown")
        score = det.get("score", 0.0)
        box = det.get("box_xyxy", [])
        parts.append(f"{label}({score:.2f}) {box}")
    return "; ".join(parts)


def format_activity(activity_result: Dict[str, Any]) -> str:
    status = activity_result.get("status", "unknown")
    if status == "ready" and isinstance(activity_result.get("activity"), dict):
        activity = activity_result["activity"]
        label = activity.get("label", "unknown")
        score = activity.get("score", 0.0)
        return f"ready {label} ({score:.2f})"

    buffer_size = activity_result.get("buffer_size")
    window_size = activity_result.get("window_size")
    if buffer_size is not None and window_size is not None:
        return f"{status} {buffer_size}/{window_size}"
    return status


def encode_frame(frame_bgr, frame_id: int) -> Tuple[bytes, Dict[str, str]]:
    ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError(f"Failed to encode frame {frame_id} as JPEG.")

    headers = {
        "Content-Type": "application/octet-stream",
        "x-frame-id": str(frame_id),
        "x-timestamp": f"{time.time():.6f}",
    }
    return encoded.tobytes(), headers


def send_request(session: requests.Session, url: str, image_bytes: bytes, headers: Dict[str, str]) -> Dict[str, Any]:
    response = session.post(
        url,
        data=image_bytes,
        headers=headers,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def open_capture() -> cv2.VideoCapture:
    if USE_WEBCAM:
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open webcam index {WEBCAM_INDEX}.")
        return cap

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")
    return cap


def determine_output_path() -> Path:
    if USE_WEBCAM:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return Path.cwd() / f"inference_webcam_{timestamp}.mp4"
    return VIDEO_PATH.parent / f"inference_{VIDEO_PATH.name}"


def build_video_writer(cap: cv2.VideoCapture, first_frame) -> Optional[cv2.VideoWriter]:
    if not SAVE_OUTPUT_VIDEO:
        return None

    output_path = determine_output_path()
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video writer: {output_path}")

    print(f"[client] output video: {output_path}")
    return writer


def draw_text_block(
    image,
    lines: List[str],
    origin: Tuple[int, int],
    text_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    line_height = 24
    padding = 6

    widths = []
    for line in lines:
        (w, h), _ = cv2.getTextSize(line, font, font_scale, thickness)
        widths.append((w, h))

    block_width = max((w for w, _ in widths), default=0) + padding * 2
    block_height = len(lines) * line_height + padding
    cv2.rectangle(image, (x, y - 20), (x + block_width, y - 20 + block_height), bg_color, thickness=-1)

    current_y = y
    for line in lines:
        cv2.putText(image, line, (x + padding, current_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
        current_y += line_height


def draw_detections(image, detections: List[Dict[str, Any]]) -> None:
    for det in detections:
        box = det.get("box_xyxy", [])
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        label = det.get("label", "unknown")
        score = det.get("score", 0.0)
        color = (0, 220, 0)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {score:.2f}"
        cv2.putText(image, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def annotate_frame(
    frame_bgr,
    frame_idx: int,
    sent_count: int,
    detr_result: Optional[Dict[str, Any]],
    activity_result: Optional[Dict[str, Any]],
) -> Any:
    annotated = frame_bgr.copy()

    detections = detr_result.get("detections", []) if isinstance(detr_result, dict) else []
    draw_detections(annotated, detections)

    detr_lines = [
        f"frame={frame_idx} sent={sent_count}",
        "DETR: no result" if detr_result is None else (
            f"DETR infer={detr_result.get('inference_ms', -1.0):.2f} ms "
            f"det={len(detections)}"
        ),
    ]
    if detections:
        top_det = detections[0]
        detr_lines.append(f"top={top_det.get('label', 'unknown')} {top_det.get('score', 0.0):.2f}")

    if activity_result is None:
        activity_lines = ["ACT: no result"]
    else:
        activity_lines = [
            f"ACT status={activity_result.get('status', 'unknown')}",
            (
                f"ACT feat={activity_result.get('feature_extraction_ms', -1.0):.2f} ms "
                f"infer={activity_result.get('inference_ms', -1.0) if activity_result.get('inference_ms') is not None else -1.0:.2f} ms"
            ),
        ]
        activity = activity_result.get("activity")
        if isinstance(activity, dict):
            activity_lines.append(f"pred={activity.get('label', 'unknown')} {activity.get('score', 0.0):.2f}")
        else:
            activity_lines.append(
                f"buffer={activity_result.get('buffer_size', '?')}/{activity_result.get('window_size', '?')}"
            )

    draw_text_block(annotated, detr_lines + activity_lines, origin=(10, 28))
    return annotated


def main() -> None:
    if SEND_EVERY_NTH_FRAME <= 0:
        raise ValueError("SEND_EVERY_NTH_FRAME must be > 0")
    if MAX_FRAMES is not None and MAX_FRAMES <= 0:
        raise ValueError("MAX_FRAMES must be > 0 when provided")

    detr_url = f"{SERVER_BASE_URL}{DETR_ENDPOINT}"
    activity_url = f"{SERVER_BASE_URL}{ACTIVITY_ENDPOINT_TEMPLATE.format(stream_id=ACTIVITY_STREAM_ID)}"

    cap = open_capture()

    if USE_WEBCAM:
        print(f"[client] webcam index: {WEBCAM_INDEX}")
        print("[client] press 'q' in the preview window to stop")
    else:
        print(f"[client] video: {VIDEO_PATH}")
    print(f"[client] DETR url    : {detr_url}")
    print(f"[client] activity url: {activity_url}")

    frame_idx = 0
    sent_count = 0
    start_time = time.perf_counter()
    video_writer: Optional[cv2.VideoWriter] = None

    session = requests.Session()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1

            detr_result: Optional[Dict[str, Any]] = None
            activity_result: Optional[Dict[str, Any]] = None

            if frame_idx % SEND_EVERY_NTH_FRAME == 0:
                image_bytes, headers = encode_frame(frame_bgr=frame, frame_id=frame_idx)
                detr_result = send_request(session=session, url=detr_url, image_bytes=image_bytes, headers=headers)
                activity_result = send_request(session=session, url=activity_url, image_bytes=image_bytes, headers=headers)
                sent_count += 1

                detections = detr_result.get("detections", [])
                print(
                    f"[frame {frame_idx}] "
                    f"DETR pre={detr_result.get('preprocess_ms', -1.0):.2f} ms "
                    f"infer={detr_result.get('inference_ms', -1.0):.2f} ms "
                    f"post={detr_result.get('postprocess_ms', -1.0):.2f} ms | "
                    f"{format_detections(detections)}"
                )
                print(
                    f"[frame {frame_idx}] "
                    f"ACT feat={activity_result.get('feature_extraction_ms', -1.0):.2f} ms "
                    f"infer={activity_result.get('inference_ms', -1.0) if activity_result.get('inference_ms') is not None else -1.0:.2f} ms | "
                    f"{format_activity(activity_result)}"
                )

            annotated = annotate_frame(
                frame_bgr=frame,
                frame_idx=frame_idx,
                sent_count=sent_count,
                detr_result=detr_result,
                activity_result=activity_result,
            )

            if video_writer is None:
                video_writer = build_video_writer(cap=cap, first_frame=annotated)

            if video_writer is not None:
                video_writer.write(annotated)

            if SHOW_PREVIEW:
                cv2.imshow(PREVIEW_WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if MAX_FRAMES is not None and sent_count >= MAX_FRAMES:
                break
    finally:
        cap.release()
        session.close()
        if video_writer is not None:
            video_writer.release()
        if SHOW_PREVIEW:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    fps = sent_count / elapsed if elapsed > 0 else 0.0
    print(f"[client] sent {sent_count} frames in {elapsed:.2f}s ({fps:.2f} request-pairs/s)")


if __name__ == "__main__":
    main()
