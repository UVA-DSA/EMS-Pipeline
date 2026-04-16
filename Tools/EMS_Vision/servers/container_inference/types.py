from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


@dataclass
class Detection:
    label: str
    class_id: int
    score: float
    box_xyxy: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceResult:
    model_name: str
    frame_id: Optional[str]
    timestamp: Optional[str]
    image_width: int
    image_height: int
    inference_ms: float
    preprocess_ms: float
    postprocess_ms: float
    detections: List[Detection]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["detections"] = [d.to_dict() for d in self.detections]
        return data


@dataclass
class ActivityPrediction:
    label: str
    class_id: int
    score: float
    window_size: int
    feature_dim: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ActivityInferenceResult:
    model_name: str
    stream_id: str
    frame_id: Optional[str]
    timestamp: Optional[str]
    status: str
    buffer_size: int
    window_size: int
    stride: int
    feature_dim: int
    frames_seen: int
    feature_extraction_ms: float
    inference_ms: Optional[float]
    activity: Optional[ActivityPrediction]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.activity is not None:
            data["activity"] = self.activity.to_dict()
        return data
