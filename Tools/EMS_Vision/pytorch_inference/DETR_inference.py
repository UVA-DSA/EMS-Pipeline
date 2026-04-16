import argparse
import ctypes
import os
import site
import time
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import cv2
import torch
import torchvision.transforms as T
from PIL import Image


def _torch_tensorrt_runtime_dirs() -> List[Path]:
    dirs: List[Path] = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        dirs.append(Path(conda_prefix) / "lib")

    dirs.append(Path(torch.__file__).resolve().parent / "lib")

    for site_dir in site.getsitepackages():
        root = Path(site_dir)
        dirs.extend(
            [
                root / "tensorrt_libs",
                root / "nvidia" / "cuda_runtime" / "lib",
                root / "nvidia" / "cu13" / "lib",
                root / "nvidia" / "cu12" / "lib",
            ]
        )

    unique_dirs: List[Path] = []
    seen = set()
    for path in dirs:
        if path.exists() and path not in seen:
            unique_dirs.append(path)
            seen.add(path)
    return unique_dirs


def _preload_torch_tensorrt_runtime_libs() -> List[Path]:
    loaded: List[Path] = []
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for lib_name in ("libcudart.so.13", "libnvinfer.so.10", "libnvinfer_plugin.so.10"):
        for directory in _torch_tensorrt_runtime_dirs():
            candidate = directory / lib_name
            if not candidate.exists():
                continue
            try:
                ctypes.CDLL(str(candidate), mode=mode)
                loaded.append(candidate)
                break
            except OSError:
                continue
    return loaded


def ensure_torch_tensorrt_runtime() -> None:
    try:
        import torch_tensorrt  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch_tensorrt is required to load this TensorRT TorchScript engine."
        ) from exc
    except OSError as exc:
        loaded_libs = _preload_torch_tensorrt_runtime_libs()
        if loaded_libs:
            try:
                import torch_tensorrt  # noqa: F401
                return
            except OSError as retry_exc:
                exc = retry_exc

        runtime_dirs = _torch_tensorrt_runtime_dirs()
        runtime_path_hint = ":".join(str(path) for path in runtime_dirs)
        raise RuntimeError(
            "torch_tensorrt failed to load its shared libraries.\n"
            f"Original error: {exc}\n"
            f"Detected torch CUDA build: {torch.version.cuda or 'unknown'}\n"
            "If you are running from WSL/conda, export these runtime directories before launching Python:\n"
            f"export LD_LIBRARY_PATH=\"{runtime_path_hint}:$LD_LIBRARY_PATH\"\n"
            "If that still fails, reinstall a torch_tensorrt build that matches your current torch/CUDA versions."
        ) from exc


def load_trt_engine(engine_path: Path, device: torch.device):
    if engine_path.suffix.lower() == ".ep":
        exported_program = torch.export.load(str(engine_path))
        module = exported_program.module().eval()
        if hasattr(module, "to"):
            module = module.to(device)
        return module
    return torch.jit.load(str(engine_path), map_location=device).eval()


def run_model_once(module, model_input: torch.Tensor):
    if callable(module):
        return module(model_input)
    if hasattr(module, "forward"):
        return module.forward(model_input)
    if hasattr(module, "module"):
        inner = module.module()
        return inner(model_input)
    raise RuntimeError(f"Unsupported TensorRT module type: {type(module)}")


def get_class_names(detr_version: str) -> List[str]:
    if detr_version == "ems":
        return ["IV needle", "bp monitor", "bvm", "defib pads", "dummy", "hands"]
    return [
        "N/A",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "N/A",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "N/A",
        "backpack",
        "umbrella",
        "N/A",
        "N/A",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "N/A",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "N/A",
        "dining table",
        "N/A",
        "N/A",
        "toilet",
        "N/A",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "N/A",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]


def cv2_to_pil(cv2_image):
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(1)
    return torch.stack([(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)], dim=1)


def rescale_bboxes(out_bbox: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    return b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=b.device)


def postprocess(logits: torch.Tensor, boxes: torch.Tensor, img_size: Tuple[int, int], threshold: float):
    probas = logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    out_prob = probas[keep].detach().cpu()
    out_boxes = rescale_bboxes(boxes[0, keep], img_size).detach().cpu()
    return out_prob, out_boxes


def scale_boxes_xyxy(boxes: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
    out = boxes.clone()
    out[:, 0] = out[:, 0] * sx
    out[:, 2] = out[:, 2] * sx
    out[:, 1] = out[:, 1] * sy
    out[:, 3] = out[:, 3] * sy
    return out


def draw_results(
    frame,
    prob: torch.Tensor,
    boxes: torch.Tensor,
    class_names: List[str],
):
    colors = [
        [0.000, 0.447, 0.741],
        [0.850, 0.325, 0.098],
        [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556],
        [0.466, 0.674, 0.188],
        [0.301, 0.745, 0.933],
    ]
    highest_confidence_objects = {}

    for p, box in zip(prob, boxes):
        cl = p.argmax().item()
        if cl == 1:
            continue

        confidence = p[cl].item()
        if cl not in highest_confidence_objects or confidence > highest_confidence_objects[cl][0]:
            xmin, ymin, xmax, ymax = box
            box_coordinates = [(int(xmin), int(ymin)), (int(xmax), int(ymax))]
            highest_confidence_objects[cl] = (confidence, box_coordinates)

    for cl, (confidence, box_coordinates) in highest_confidence_objects.items():
        name = class_names[cl] if cl < len(class_names) else f"class_{cl}"
        label = f"{name}: {confidence:.2f}"
        color = [int(x * 255) for x in colors[cl % len(colors)]]
        cv2.rectangle(
            frame,
            (box_coordinates[0][0], box_coordinates[0][1]),
            (box_coordinates[1][0], box_coordinates[1][1]),
            color,
            2,
        )
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(
            frame,
            (box_coordinates[0][0], box_coordinates[0][1] - label_size[1] - 10),
            (box_coordinates[0][0] + label_size[0], box_coordinates[0][1]),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            label,
            (box_coordinates[0][0], box_coordinates[0][1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return frame


def parse_args(default_engine: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DETR TensorRT inference on video with visualization and timing.")
    parser.add_argument("--engine", type=Path, default=default_engine, help="Path to TensorRT engine (.ts/.ep)")
    parser.add_argument("--video", type=Path, default=None, help="Video path. If omitted, first video in ./videos is used.")
    parser.add_argument("--detr-version", type=str, default="ems", choices=["ems", "base"], help="Class set")
    parser.add_argument("--threshold", type=float, default=0.7, help="Detection confidence threshold")
    parser.add_argument("--engine-height", type=int, default=480, help="Engine input height used at conversion time.")
    parser.add_argument("--engine-width", type=int, default=640, help="Engine input width used at conversion time.")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI display and save annotated video to ./outputs.")
    parser.add_argument("--display-width", type=int, default=960, help="Display width (0 to disable resizing)")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup frames before reporting avg timing")
    parser.add_argument("--print-every", type=int, default=1, help="Print latency every N frames")
    return parser.parse_args()


def find_default_video(script_dir: Path):
    videos_dir = script_dir / "videos"
    if not videos_dir.exists():
        return None
    valid_exts = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
    for name in sorted(videos_dir.iterdir()):
        if name.suffix.lower() in valid_exts:
            return name
    return None


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_engine = script_dir / "checkpoints" / "ems_finetuned_detr_trt.ts"
    args = parse_args(default_engine=default_engine)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT inference.")
    if not args.engine.exists():
        raise FileNotFoundError(f"Engine not found: {args.engine}")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")
    if args.print_every <= 0:
        raise ValueError("--print-every must be > 0")
    if args.engine_height <= 0 or args.engine_width <= 0:
        raise ValueError("--engine-height and --engine-width must be > 0")

    video_path = args.video if args.video is not None else find_default_video(script_dir)
    if video_path is None or not video_path.exists():
        raise FileNotFoundError("Video not found. Pass --video or place a video in ./videos.")

    ensure_torch_tensorrt_runtime()
    device = torch.device("cuda")
    module = load_trt_engine(engine_path=args.engine, device=device)

    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    class_names = get_class_names(args.detr_version)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    print(f"[DETR TRT] Engine: {args.engine}")
    print(f"[DETR TRT] Video : {video_path}")

    writer = None
    output_path = None
    if args.no_display:
        outputs_dir = script_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        output_path = outputs_dir / f"{video_path.stem}_detr_trt.mp4"
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        fps = float(input_fps) if input_fps and input_fps > 0 else 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_w, frame_h))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create output video writer: {output_path}")
        print(f"[DETR TRT] Output: {output_path}")

    frame_idx = 0
    inference_times_ms = []
    total_times_ms = []

    with torch.inference_mode():
        while True:
            t_total_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break

            orig_h, orig_w = frame.shape[:2]
            infer_frame = cv2.resize(frame, (args.engine_width, args.engine_height), interpolation=cv2.INTER_LINEAR)
            img = cv2_to_pil(infer_frame)
            model_input = transform(img).unsqueeze(0).to(device)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            pred_logits, pred_boxes = run_model_once(module, model_input)
            end_event.record()
            torch.cuda.synchronize()
            infer_ms = float(start_event.elapsed_time(end_event))

            probs, scaled_boxes = postprocess(
                logits=pred_logits,
                boxes=pred_boxes,
                img_size=(args.engine_width, args.engine_height),
                threshold=args.threshold,
            )
            sx = float(orig_w) / float(args.engine_width)
            sy = float(orig_h) / float(args.engine_height)
            scaled_boxes = scale_boxes_xyxy(scaled_boxes, sx=sx, sy=sy)
            annotated = draw_results(
                frame=frame,
                prob=probs,
                boxes=scaled_boxes,
                class_names=class_names,
            )

            total_ms = (time.perf_counter() - t_total_start) * 1000.0
            frame_idx += 1
            if frame_idx > args.warmup:
                inference_times_ms.append(infer_ms)
                total_times_ms.append(total_ms)

            inst_fps = 1000.0 / max(total_ms, 1e-6)
            cv2.putText(
                annotated,
                f"Infer: {infer_ms:.2f} ms | Total: {total_ms:.2f} ms | FPS: {inst_fps:.1f}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 0),
                2,
            )

            if writer is not None:
                writer.write(annotated)

            if not args.no_display:
                if args.display_width > 0:
                    h, w = annotated.shape[:2]
                    out_w = args.display_width
                    out_h = int(h * (out_w / w))
                    display = cv2.resize(annotated, (out_w, out_h))
                else:
                    display = annotated
                cv2.imshow("DETR TensorRT Inference", display)
            if frame_idx % args.print_every == 0:
                print(f"[frame {frame_idx}] infer={infer_ms:.2f} ms | total={total_ms:.2f} ms | fps={inst_fps:.2f}")

            if not args.no_display:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    if inference_times_ms:
        avg_infer = mean(inference_times_ms)
        avg_total = mean(total_times_ms)
        avg_fps = 1000.0 / max(avg_total, 1e-6)
        print("\n=== Timing Summary (after warmup) ===")
        print(f"Frames measured : {len(inference_times_ms)}")
        print(f"Avg infer time  : {avg_infer:.2f} ms")
        print(f"Avg total time  : {avg_total:.2f} ms")
        print(f"Avg FPS         : {avg_fps:.2f}")
    else:
        print("\nNo frames measured after warmup.")
    if output_path is not None:
        print(f"Saved annotated video to: {output_path}")


if __name__ == "__main__":
    main()
