import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import time
import os
import argparse
import pickle
import json
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import cv2
from PIL import Image
from dataclasses import dataclass
from pathlib import Path

try:
    from classes import DetectionObj
except ImportError:
    @dataclass
    class DetectionObj:
        box_coords: list
        name: str
        confidence: str

try:
    from pipeline_config import detr_threshold
except ImportError:
    detr_threshold = 0.7


class DETREngine:
    def __init__(self, detr_version="base", checkpoint_path=None, threshold=None, device=None):
        print(torch.__version__, torch.cuda.is_available())
        torch.set_grad_enabled(False)
        
        print("[DETR Engine] Initializing DETR Engine")

        self.threshold = detr_threshold if threshold is None else threshold
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.detr_version = detr_version
        inference_dir = os.path.dirname(os.path.abspath(__file__))
        default_ems_ckpt = os.path.join(inference_dir, "checkpoints", "ems_finetuned_detr_checkpoint.pth")

        if (self.detr_version == "ems"):
            self.finetuned_classes = [
                'IV needle', 'bp monitor', 'bvm', 'defib pads', 'dummy', 'hands'
            ]
            if checkpoint_path is None and os.path.exists(default_ems_ckpt):
                checkpoint_path = default_ems_ckpt

        else:
            self.finetuned_classes = [
                'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
                'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
                'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
                'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
                'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
                'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
                'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
                'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
                'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush'
            ]
            if checkpoint_path is None:
                old_default = './EMS_Vision/weights/detr-r50-e632da11.pth'
                if os.path.exists(old_default):
                    checkpoint_path = old_default

        self.num_classes = len(self.finetuned_classes)

        self.COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                       [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

        self.transform = T.Compose([
            # T.Resize((512, 512)),  # Resize the image to 224x224 pixels
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.model = torch.hub.load(
            'facebookresearch/detr', 'detr_resnet50', pretrained=False, num_classes=self.num_classes)
        if checkpoint_path is None:
            raise FileNotFoundError("No checkpoint path provided/found. Pass --checkpoint to run standalone inference.")
        print(f"[DETR Engine] Loading model from checkpoint: {checkpoint_path}")
        checkpoint = self._load_checkpoint(checkpoint_path)
        model_state = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
        self.model.load_state_dict(model_state, strict=True)
        print("[DETR_Engine] DETR Model loaded")

        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_checkpoint(checkpoint_path):
        """Load checkpoint compatibly across torch versions (incl. 2.6 weights_only default change)."""
        try:
            return torch.load(checkpoint_path, map_location='cpu')
        except (pickle.UnpicklingError, RuntimeError) as exc:
            msg = str(exc)
            if "Weights only load failed" in msg or "Unsupported global" in msg:
                print(
                    "[DETR Engine] Retrying checkpoint load with weights_only=False. "
                    "Use only with trusted checkpoints."
                )
                return torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            raise

    @staticmethod
    def cv2_to_pil(cv2_image):
        """Convert a cv2 image (numpy array in BGR) to a PIL Image in RGB."""
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        return pil_image

    @staticmethod
    def box_cxcywh_to_xyxy(x):
        """Convert bbox coordinates from (center_x, center_y, w, h) to [(x1, y1), (x2, y2)]"""
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    @staticmethod
    def rescale_bboxes(out_bbox, size):
        """Rescale bounding boxes from ratio values to pixel values"""
        img_w, img_h = size
        b = DETREngine.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32).to(b.device)
        return b

    def filter_bboxes_from_outputs(self, outputs, img_size, threshold=0.7):
        """Recover bounding-boxes with prediction confidence above threshold (default=0.7)"""
        probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold
        probas_to_keep = probas[keep]
        bboxes_scaled = self.rescale_bboxes(
            outputs['pred_boxes'][0, keep], img_size)
        return probas_to_keep, bboxes_scaled



    def detections_from_outputs(self, prob=None, boxes=None):
        if prob is None or boxes is None:
            return []

        detections = []
        for p, box in zip(prob, boxes):
            cl = p.argmax().item()
            confidence = p[cl].item()
            xmin, ymin, xmax, ymax = box.tolist()
            x1, y1, x2, y2 = int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))
            name = self.finetuned_classes[cl]
            detections.append({
                "class_id": int(cl),
                "name": name,
                "confidence": float(confidence),
                "box_coords": [(x1, y1), (x2, y2)],
                "bbox_xyxy": [x1, y1, x2, y2],
            })

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        return detections

    def plot_finetuned_results(self, cv2_img, detections=None):
        """Draw every kept detection on the image and return serialized detection data."""
        if detections is None:
            detections = []

        detection_objects = []
        for detection in detections:
            cl = detection["class_id"]
            confidence = detection["confidence"]
            box_coordinates = detection["box_coords"]
            label = f'{detection["name"]}: {confidence:.2f}'
            color = [int(x * 255) for x in self.COLORS[cl % len(self.COLORS)]]

            cv2.rectangle(
                cv2_img,
                (box_coordinates[0][0], box_coordinates[0][1]),
                (box_coordinates[1][0], box_coordinates[1][1]),
                color,
                2,
            )

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(
                cv2_img,
                (box_coordinates[0][0], box_coordinates[0][1] - label_size[1] - 10),
                (box_coordinates[0][0] + label_size[0], box_coordinates[0][1]),
                color,
                cv2.FILLED,
            )

            cv2.putText(
                cv2_img,
                label,
                (box_coordinates[0][0], box_coordinates[0][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            detection_objects.append(
                DetectionObj(
                    box_coords=box_coordinates,
                    name=detection["name"],
                    confidence=f'{confidence:.2f}',
                )
            )

        return cv2_img, detection_objects


    def run_workflow(self, my_image):
        start_t = time.time()
        img = self.cv2_to_pil(my_image)
        # print(f"[DETR Engine] Image conversion time: {time.time() - start_t}")
        img = self.transform(img).unsqueeze(0).to(self.device)
        outputs = self.model(img)
        # print(f"[DETR Engine] Inference time: {time.time() - start_t}")
        img_h, img_w = my_image.shape[:2]
        probas_to_keep, bboxes_scaled = self.filter_bboxes_from_outputs(
            outputs, img_size=(img_w, img_h), threshold=self.threshold)

        detections = self.detections_from_outputs(probas_to_keep, bboxes_scaled)
        result_image, detection_objects = self.plot_finetuned_results(my_image, detections)

        detection_results_serialized = []
        for detection, detection_object in zip(detections, detection_objects):
            serialized = vars(detection_object)
            serialized["class_id"] = detection["class_id"]
            serialized["bbox_xyxy"] = detection["bbox_xyxy"]
            serialized["confidence"] = round(detection["confidence"], 6)
            detection_results_serialized.append(serialized)
        

        return result_image, detection_results_serialized


def _find_default_video(video_arg):
    if video_arg:
        return video_arg

    inference_dir = os.path.dirname(os.path.abspath(__file__))
    videos_dir = os.path.join(inference_dir, "videos")
    if not os.path.isdir(videos_dir):
        return None

    valid_exts = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
    for name in sorted(os.listdir(videos_dir)):
        if name.lower().endswith(valid_exts):
            return os.path.join(videos_dir, name)
    return None


def main():
    parser = argparse.ArgumentParser(description="Standalone DETR video inference with realtime visualization.")
    parser.add_argument("--video", type=str, default=None, help="Path to input video. If omitted, first video in ./videos is used.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pth file.")
    parser.add_argument("--detr-version", type=str, default="ems", choices=["ems", "base"], help="Class set to use.")
    parser.add_argument("--threshold", type=float, default=None, help="Detection confidence threshold.")
    parser.add_argument("--device", type=str, default=None, help="Torch device override (e.g., cpu, cuda:0).")
    parser.add_argument("--display-width", type=int, default=960, help="Display width while preserving aspect ratio.")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI display.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Run inference on every Nth frame.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many processed frames.")
    parser.add_argument("--print-every", type=int, default=1, help="Print progress every N processed frames.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON output path for per-frame detections.")
    parser.add_argument("--save-annotated-video", type=str, default=None, help="Optional path for annotated output video.")
    args = parser.parse_args()

    video_path = _find_default_video(args.video)
    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError("Video not found. Pass a valid --video path.")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be > 0")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be > 0")

    engine = DETREngine(
        detr_version=args.detr_version,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        device=args.device,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps_value = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    outputs_dir = Path(__file__).resolve().parents[1] / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = Path(args.output_json) if args.output_json else outputs_dir / f"{Path(video_path).stem}_detr_pytorch.json"

    writer = None
    if args.save_annotated_video:
        save_annotated_video = Path(args.save_annotated_video)
        save_annotated_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(save_annotated_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps_value if fps_value > 0 else 30.0,
            (frame_width, frame_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create output video: {save_annotated_video}")

    print(f"[DETR Engine] Running realtime inference on: {video_path}")
    prev_t = time.time()
    frame_index = 0
    processed_frames = 0
    frame_results = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame_index = frame_index
        frame_index += 1
        if current_frame_index % args.frame_stride != 0:
            continue

        annotated, detections = engine.run_workflow(frame)
        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now
        processed_frames += 1

        frame_results.append({
            "frame_index": current_frame_index,
            "timestamp_ms": round((current_frame_index / fps_value) * 1000.0, 3) if fps_value and fps_value > 0 else None,
            "detections": detections,
        })

        if args.print_every > 0 and processed_frames % args.print_every == 0:
            print(f"[frame {current_frame_index}] detections={len(detections)}")

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        if writer is not None:
            writer.write(annotated)

        if not args.no_display and args.display_width > 0:
            h, w = annotated.shape[:2]
            new_w = args.display_width
            new_h = int(h * (new_w / w))
            display_frame = cv2.resize(annotated, (new_w, new_h))
        elif not args.no_display:
            display_frame = annotated

        if not args.no_display:
            cv2.imshow("DETR Realtime Inference", display_frame)
            if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                break

        if args.max_frames is not None and processed_frames >= args.max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    output_payload = {
        "model": {
            "name": "detr_pytorch",
            "detr_version": args.detr_version,
            "checkpoint_path": args.checkpoint,
            "device": args.device or str(engine.device),
            "threshold": engine.threshold,
            "class_names": engine.finetuned_classes,
        },
        "video": {
            "path": video_path,
            "fps": fps_value,
            "width": frame_width,
            "height": frame_height,
            "frame_count": frame_count,
            "frame_stride": args.frame_stride,
        },
        "summary": {
            "processed_frames": processed_frames,
        },
        "frames": frame_results,
    }
    output_json_path.write_text(json.dumps(output_payload, indent=2))
    print(f"[DETR Engine] Wrote frame detections to: {output_json_path}")
    if args.save_annotated_video:
        print(f"[DETR Engine] Wrote annotated video to: {args.save_annotated_video}")


if __name__ == "__main__":
    main()
