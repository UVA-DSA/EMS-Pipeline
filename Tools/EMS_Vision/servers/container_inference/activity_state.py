from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import torch

from container_inference.types import ActivityInferenceResult


@dataclass
class ActivityStreamState:
    window_size: int
    features: Deque[torch.Tensor] = field(init=False)
    frames_seen: int = 0
    last_inference_frame_idx: Optional[int] = None
    last_result: Optional[ActivityInferenceResult] = None

    def __post_init__(self) -> None:
        self.features = deque(maxlen=self.window_size)

    @property
    def buffer_size(self) -> int:
        return len(self.features)

    def append_feature(self, feature: torch.Tensor) -> None:
        self.features.append(feature)
        self.frames_seen += 1

    def should_infer(self, stride: int) -> bool:
        if self.buffer_size < self.window_size:
            return False
        if stride <= 1:
            return True
        if self.last_inference_frame_idx is None:
            return True
        return (self.frames_seen - self.last_inference_frame_idx) >= stride

    def stacked_window(self) -> torch.Tensor:
        if self.buffer_size < self.window_size:
            raise RuntimeError("Not enough buffered features for an activity window.")
        return torch.stack(list(self.features), dim=0)

    def mark_inference(self, result: ActivityInferenceResult) -> None:
        self.last_inference_frame_idx = self.frames_seen
        self.last_result = result


class ActivityStreamStore:
    def __init__(self, window_size: int) -> None:
        self._window_size = window_size
        self._states: Dict[str, ActivityStreamState] = {}

    def get(self, stream_id: str) -> ActivityStreamState:
        state = self._states.get(stream_id)
        if state is None:
            state = ActivityStreamState(window_size=self._window_size)
            self._states[stream_id] = state
        return state

    def reset(self, stream_id: str) -> bool:
        return self._states.pop(stream_id, None) is not None

    def summary(self) -> Dict[str, Dict[str, int]]:
        return {
            stream_id: {
                "buffer_size": state.buffer_size,
                "frames_seen": state.frames_seen,
            }
            for stream_id, state in self._states.items()
        }
