import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from container_inference.activity_state import ActivityStreamStore
from container_inference.trt_runtime import ensure_torch_tensorrt_runtime, load_trt_engine, run_model_once
from container_inference.types import ActivityInferenceResult, ActivityPrediction


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def load_class_names(class_map_path: Path, num_classes: int) -> List[str]:
    labels = [f"class_{idx}" for idx in range(num_classes)]
    if not class_map_path.exists():
        return labels

    try:
        with open(class_map_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        new_id_to_class = data.get("new_id_to_class", {})
        if isinstance(new_id_to_class, dict) and new_id_to_class:
            for class_id, label in new_id_to_class.items():
                try:
                    class_idx = int(class_id)
                except (TypeError, ValueError):
                    continue
                if 0 <= class_idx < num_classes and isinstance(label, str):
                    labels[class_idx] = label
            return labels

        class_to_new_id = data.get("class_to_new_id", {})
        if isinstance(class_to_new_id, dict) and class_to_new_id:
            for label, class_id in class_to_new_id.items():
                if isinstance(class_id, int) and 0 <= class_id < num_classes and isinstance(label, str):
                    labels[class_id] = label
            return labels

        keysteps = data.get("keysteps", {})
        if isinstance(keysteps, dict) and keysteps:
            for label, class_id in keysteps.items():
                if isinstance(class_id, int) and 0 <= class_id < num_classes and isinstance(label, str):
                    labels[class_id] = label
            return labels

        return labels
    except Exception:
        return labels

    return labels


def adjust_seq_len(features: torch.Tensor, target_seq_len: int) -> torch.Tensor:
    current_seq_len = int(features.shape[1])
    if current_seq_len == target_seq_len:
        return features
    if current_seq_len > target_seq_len:
        return features[:, :target_seq_len, :].contiguous()
    pad_len = target_seq_len - current_seq_len
    pad = features[:, -1:, :].repeat(1, pad_len, 1)
    return torch.cat([features, pad], dim=1).contiguous()


class FrameFeatureExtractor(ABC):
    @property
    @abstractmethod
    def output_dim(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def extract(self, image_bgr: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def warmup(self, num_iters: int) -> None:
        raise NotImplementedError


class ResNet50FeatureExtractor(FrameFeatureExtractor):
    def __init__(
        self,
        device: torch.device,
        resize_short_side: int,
        center_crop_size: int,
        weights_name: str,
    ) -> None:
        try:
            import torchvision.models as models
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("torchvision is required for activity feature extraction.") from exc

        self._device = device
        self._resize_short_side = resize_short_side
        self._center_crop_size = center_crop_size

        if weights_name == "none":
            weights = None
        elif weights_name == "imagenet1k_v1":
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        else:
            raise ValueError(f"Unsupported feature extractor weights: {weights_name}")

        backbone = models.resnet50(weights=weights)
        self._backbone = nn.Sequential(*list(backbone.children())[:-1]).to(device).eval()
        for param in self._backbone.parameters():
            param.requires_grad = False

        self._mean = IMAGENET_MEAN.to(device)
        self._std = IMAGENET_STD.to(device)

    @property
    def output_dim(self) -> int:
        return 2048

    @property
    def name(self) -> str:
        return "resnet50_pytorch"

    def extract(self, image_bgr: np.ndarray) -> torch.Tensor:
        image_tensor = self._preprocess(image_bgr)
        with torch.inference_mode():
            feature = self._backbone(image_tensor).flatten(1)[0]
        return feature.detach().cpu()

    def warmup(self, num_iters: int) -> None:
        if num_iters <= 0:
            return
        dummy = torch.zeros((1, 3, self._center_crop_size, self._center_crop_size), dtype=torch.float32, device=self._device)
        with torch.inference_mode():
            for _ in range(num_iters):
                _ = self._backbone(dummy)
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = self._resize_shorter_side(rgb, self._resize_short_side)
        cropped = self._center_crop(resized, self._center_crop_size)
        image = torch.from_numpy(cropped).permute(2, 0, 1).unsqueeze(0).to(self._device, dtype=torch.float32) / 255.0
        return (image - self._mean) / self._std

    @staticmethod
    def _resize_shorter_side(image_rgb: np.ndarray, shorter_side: int) -> np.ndarray:
        h, w, _ = image_rgb.shape
        if min(h, w) == shorter_side:
            return image_rgb
        if h <= w:
            new_h = shorter_side
            new_w = int(round((w / h) * shorter_side))
        else:
            new_w = shorter_side
            new_h = int(round((h / w) * shorter_side))
        return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _center_crop(image_rgb: np.ndarray, crop_size: int) -> np.ndarray:
        h, w, _ = image_rgb.shape
        if crop_size > h or crop_size > w:
            raise ValueError(f"Center crop size ({crop_size}) exceeds resized frame size ({h}, {w}).")
        top = (h - crop_size) // 2
        left = (w - crop_size) // 2
        return image_rgb[top : top + crop_size, left : left + crop_size]


class ResNet50TensorRTFeatureExtractor(FrameFeatureExtractor):
    def __init__(
        self,
        engine_path: Path,
        device: torch.device,
        resize_short_side: int,
        center_crop_size: int,
    ) -> None:
        self._device = device
        self._engine_path = Path(engine_path)
        if not self._engine_path.exists():
            raise FileNotFoundError(f"Feature extractor engine not found: {self._engine_path}")
        self._resize_short_side = resize_short_side
        self._center_crop_size = center_crop_size
        self._module = load_trt_engine(engine_path=self._engine_path, device=self._device)
        self._mean = IMAGENET_MEAN.to(device)
        self._std = IMAGENET_STD.to(device)
        self._output_dim = self._infer_output_dim()

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def name(self) -> str:
        return "resnet50_tensorrt"

    def extract(self, image_bgr: np.ndarray) -> torch.Tensor:
        image_tensor = self._preprocess(image_bgr)
        with torch.inference_mode():
            feature = run_model_once(self._module, image_tensor)
        if isinstance(feature, (tuple, list)):
            feature = feature[0]
        if feature.ndim != 2 or feature.shape[0] != 1:
            raise RuntimeError(f"Expected TRT feature output shape [1, F], got {tuple(feature.shape)}")
        return feature[0].detach().cpu()

    def warmup(self, num_iters: int) -> None:
        if num_iters <= 0:
            return
        dummy = torch.zeros((1, 3, self._center_crop_size, self._center_crop_size), dtype=torch.float32, device=self._device)
        with torch.inference_mode():
            for _ in range(num_iters):
                _ = run_model_once(self._module, dummy)
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)

    def _infer_output_dim(self) -> int:
        dummy = torch.zeros((1, 3, self._center_crop_size, self._center_crop_size), dtype=torch.float32, device=self._device)
        with torch.inference_mode():
            feature = run_model_once(self._module, dummy)
        if isinstance(feature, (tuple, list)):
            feature = feature[0]
        if feature.ndim != 2 or feature.shape[0] != 1:
            raise RuntimeError(f"Expected TRT feature output shape [1, F], got {tuple(feature.shape)}")
        return int(feature.shape[1])

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = ResNet50FeatureExtractor._resize_shorter_side(rgb, self._resize_short_side)
        cropped = ResNet50FeatureExtractor._center_crop(resized, self._center_crop_size)
        image = torch.from_numpy(cropped).permute(2, 0, 1).unsqueeze(0).to(self._device, dtype=torch.float32) / 255.0
        return (image - self._mean) / self._std


class MTRSAPTensorRTBackend:
    def __init__(
        self,
        engine_path: Path,
        class_map_path: Path,
        window_size: int,
        stride: int,
        resize_short_side: int,
        center_crop_size: int,
        model_seq_len: Optional[int] = None,
        feature_engine_path: Optional[Path] = None,
        feature_extractor_weights: str = "imagenet1k_v1",
        device: str = "cuda",
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")
        if resize_short_side <= 0 or center_crop_size <= 0:
            raise ValueError("resize_short_side and center_crop_size must be > 0")
        if center_crop_size > resize_short_side:
            raise ValueError("center_crop_size cannot exceed resize_short_side")

        ensure_torch_tensorrt_runtime()

        self._device = torch.device(device)
        self._engine_path = Path(engine_path)
        self._class_map_path = Path(class_map_path)
        if not self._engine_path.exists():
            raise FileNotFoundError(f"Activity engine not found: {self._engine_path}")
        if self._device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for MTRSAP TensorRT inference.")

        self._window_size = window_size
        self._stride = stride
        self._model_seq_len = model_seq_len
        self._module = load_trt_engine(engine_path=self._engine_path, device=self._device)
        self._feature_engine_path = Path(feature_engine_path) if feature_engine_path is not None else None
        if self._feature_engine_path is not None:
            self._feature_extractor = ResNet50TensorRTFeatureExtractor(
                engine_path=self._feature_engine_path,
                device=self._device,
                resize_short_side=resize_short_side,
                center_crop_size=center_crop_size,
            )
        else:
            self._feature_extractor = ResNet50FeatureExtractor(
                device=self._device,
                resize_short_side=resize_short_side,
                center_crop_size=center_crop_size,
                weights_name=feature_extractor_weights,
            )
        self._stream_store = ActivityStreamStore(window_size=window_size)
        self._class_names: Optional[List[str]] = None

    @property
    def model_name(self) -> str:
        return "mtrsap_tensorrt"

    def metadata(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "engine_path": str(self._engine_path),
            "class_map_path": str(self._class_map_path),
            "window_size": self._window_size,
            "stride": self._stride,
            "model_seq_len": self._model_seq_len,
            "device": str(self._device),
            "feature_extractor": {
                "name": self._feature_extractor.name,
                "output_dim": self._feature_extractor.output_dim,
                "engine_path": str(self._feature_engine_path) if self._feature_engine_path is not None else None,
            },
            "active_streams": self._stream_store.summary(),
        }

    def warmup(self, num_iters: int) -> None:
        if num_iters <= 0:
            return

        self._feature_extractor.warmup(num_iters)

        dummy = torch.zeros(
            (1, self._model_seq_len or self._window_size, self._feature_extractor.output_dim),
            dtype=torch.float32,
            device=self._device,
        )
        with torch.inference_mode():
            for _ in range(num_iters):
                run_model_once(self._module, dummy)
            if self._device.type == "cuda":
                torch.cuda.synchronize(self._device)

    def infer_stream_frame(
        self,
        stream_id: str,
        image_bgr: np.ndarray,
        frame_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> ActivityInferenceResult:
        if not stream_id:
            raise ValueError("stream_id must be provided for activity inference.")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("Expected image_bgr to have shape [H, W, 3].")

        feature_start = time.perf_counter()
        feature = self._feature_extractor.extract(image_bgr)
        feature_extraction_ms = (time.perf_counter() - feature_start) * 1000.0

        state = self._stream_store.get(stream_id)
        state.append_feature(feature)

        if state.buffer_size < self._window_size:
            return ActivityInferenceResult(
                model_name=self.model_name,
                stream_id=stream_id,
                frame_id=frame_id,
                timestamp=timestamp,
                status="buffering",
                buffer_size=state.buffer_size,
                window_size=self._window_size,
                stride=self._stride,
                feature_dim=self._feature_extractor.output_dim,
                frames_seen=state.frames_seen,
                feature_extraction_ms=feature_extraction_ms,
                inference_ms=None,
                activity=None,
            )

        if not state.should_infer(self._stride):
            return ActivityInferenceResult(
                model_name=self.model_name,
                stream_id=stream_id,
                frame_id=frame_id,
                timestamp=timestamp,
                status="stride_wait",
                buffer_size=state.buffer_size,
                window_size=self._window_size,
                stride=self._stride,
                feature_dim=self._feature_extractor.output_dim,
                frames_seen=state.frames_seen,
                feature_extraction_ms=feature_extraction_ms,
                inference_ms=None,
                activity=None,
            )

        features = state.stacked_window().unsqueeze(0).to(device=self._device, dtype=torch.float32).contiguous()
        if self._model_seq_len is not None:
            features = adjust_seq_len(features, self._model_seq_len)

        with torch.inference_mode():
            if self._device.type == "cuda":
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                logits = run_model_once(self._module, features)
                end_event.record()
                torch.cuda.synchronize(self._device)
                inference_ms = float(start_event.elapsed_time(end_event))
            else:
                infer_start = time.perf_counter()
                logits = run_model_once(self._module, features)
                inference_ms = (time.perf_counter() - infer_start) * 1000.0

        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if logits.ndim != 2:
            raise RuntimeError(f"Expected activity logits shape [B, C], got {tuple(logits.shape)}")

        probabilities = logits.softmax(dim=-1)
        score, class_id = probabilities[0].max(dim=-1)
        class_idx = int(class_id.item())
        labels = self._get_class_names(num_classes=int(probabilities.shape[1]))
        prediction = ActivityPrediction(
            label=labels[class_idx],
            class_id=class_idx,
            score=float(score.item()),
            window_size=self._window_size,
            feature_dim=self._feature_extractor.output_dim,
        )

        result = ActivityInferenceResult(
            model_name=self.model_name,
            stream_id=stream_id,
            frame_id=frame_id,
            timestamp=timestamp,
            status="ready",
            buffer_size=state.buffer_size,
            window_size=self._window_size,
            stride=self._stride,
            feature_dim=self._feature_extractor.output_dim,
            frames_seen=state.frames_seen,
            feature_extraction_ms=feature_extraction_ms,
            inference_ms=inference_ms,
            activity=prediction,
        )
        state.mark_inference(result)
        return result

    def reset_stream(self, stream_id: str) -> bool:
        if not stream_id:
            raise ValueError("stream_id must be provided for activity reset.")
        return self._stream_store.reset(stream_id)

    def _get_class_names(self, num_classes: int) -> List[str]:
        if self._class_names is None or len(self._class_names) != num_classes:
            self._class_names = load_class_names(class_map_path=self._class_map_path, num_classes=num_classes)
        return self._class_names
