import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from container_inference.backends.base import InferenceBackend
from container_inference.trt_runtime import ensure_torch_tensorrt_runtime, load_trt_engine, run_model_once
from container_inference.types import Detection, InferenceResult


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


def box_cxcywh_to_xyxy(x: torch.Tensor) -> torch.Tensor:
    x_c, y_c, w, h = x.unbind(1)
    return torch.stack([(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)], dim=1)


def rescale_bboxes(out_bbox: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    scale = torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32, device=b.device)
    return b * scale


class DETRTensorRTBackend(InferenceBackend):
    def __init__(
        self,
        engine_path: Path,
        detr_version: str = "ems",
        threshold: float = 0.7,
        engine_height: int = 480,
        engine_width: int = 640,
        device: str = "cuda",
    ) -> None:
        if engine_height <= 0 or engine_width <= 0:
            raise ValueError("engine_height and engine_width must be > 0")
        if threshold <= 0.0 or threshold >= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        ensure_torch_tensorrt_runtime()

        self._device = torch.device(device)
        self._engine_path = Path(engine_path)
        if not self._engine_path.exists():
            raise FileNotFoundError(f"Engine not found: {self._engine_path}")
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for DETR TensorRT inference.")

        self._threshold = threshold
        self._engine_height = engine_height
        self._engine_width = engine_width
        self._class_names = get_class_names(detr_version)
        self._detr_version = detr_version
        self._module = load_trt_engine(engine_path=self._engine_path, device=self._device)

        self._mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=self._device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=self._device).view(1, 3, 1, 1)

    @property
    def model_name(self) -> str:
        return "detr_tensorrt"

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "engine_path": str(self._engine_path),
            "detr_version": self._detr_version,
            "threshold": self._threshold,
            "engine_height": self._engine_height,
            "engine_width": self._engine_width,
            "device": str(self._device),
        }

    def warmup(self, num_iters: int) -> None:
        if num_iters <= 0:
            return
        dummy = torch.zeros(
            (1, 3, self._engine_height, self._engine_width),
            dtype=torch.float32,
            device=self._device,
        )
        with torch.inference_mode():
            for _ in range(num_iters):
                run_model_once(self._module, dummy)
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)

    def infer(
        self,
        image_bgr: np.ndarray,
        frame_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> InferenceResult:
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected image_bgr to have shape [H, W, 3].")

        orig_h, orig_w = image_bgr.shape[:2]

        t0 = time.perf_counter()
        resized = cv2.resize(
            image_bgr,
            (self._engine_width, self._engine_height),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        model_input = torch.from_numpy(rgb).to(self._device, dtype=torch.float32)
        model_input = model_input.permute(2, 0, 1).unsqueeze(0) / 255.0
        model_input = (model_input - self._mean) / self._std
        preprocess_ms = (time.perf_counter() - t0) * 1000.0

        with torch.inference_mode():
            if self._device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                pred_logits, pred_boxes = run_model_once(self._module, model_input)
                end_event.record()
                torch.cuda.synchronize(self._device)
                inference_ms = float(start_event.elapsed_time(end_event))
            else:
                infer_start = time.perf_counter()
                pred_logits, pred_boxes = run_model_once(self._module, model_input)
                inference_ms = (time.perf_counter() - infer_start) * 1000.0

        t2 = time.perf_counter()
        detections = self._postprocess(
            logits=pred_logits,
            boxes=pred_boxes,
            original_width=orig_w,
            original_height=orig_h,
        )
        postprocess_ms = (time.perf_counter() - t2) * 1000.0

        return InferenceResult(
            model_name=self.model_name,
            frame_id=frame_id,
            timestamp=timestamp,
            image_width=orig_w,
            image_height=orig_h,
            inference_ms=inference_ms,
            preprocess_ms=preprocess_ms,
            postprocess_ms=postprocess_ms,
            detections=detections,
        )

    def _postprocess(
        self,
        logits: torch.Tensor,
        boxes: torch.Tensor,
        original_width: int,
        original_height: int,
    ) -> List[Detection]:
        probas = logits.softmax(-1)[0, :, :-1]
        scores, class_ids = probas.max(-1)
        keep = scores > self._threshold

        if not keep.any():
            return []

        kept_scores = scores[keep].detach().cpu()
        kept_class_ids = class_ids[keep].detach().cpu()
        kept_boxes = rescale_bboxes(
            boxes[0, keep],
            size=(self._engine_width, self._engine_height),
        ).detach().cpu()

        sx = float(original_width) / float(self._engine_width)
        sy = float(original_height) / float(self._engine_height)

        detections: List[Detection] = []
        for score, class_id, box in zip(kept_scores.tolist(), kept_class_ids.tolist(), kept_boxes.tolist()):
            x1, y1, x2, y2 = box
            scaled_box = [
                max(0.0, x1 * sx),
                max(0.0, y1 * sy),
                min(float(original_width), x2 * sx),
                min(float(original_height), y2 * sy),
            ]
            label = self._class_names[class_id] if class_id < len(self._class_names) else f"class_{class_id}"
            detections.append(
                Detection(
                    label=label,
                    class_id=int(class_id),
                    score=float(score),
                    box_xyxy=scaled_box,
                )
            )
        return detections
