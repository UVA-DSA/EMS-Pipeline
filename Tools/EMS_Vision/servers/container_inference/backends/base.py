from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np

from container_inference.types import InferenceResult


class InferenceBackend(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def infer(
        self,
        image_bgr: np.ndarray,
        frame_id: Optional[str] = None,
        timestamp: Optional[str] = None,
    ) -> InferenceResult:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        raise NotImplementedError

    def warmup(self, num_iters: int) -> None:
        if num_iters <= 0:
            return
        raise NotImplementedError(f"{self.__class__.__name__} does not implement warmup()")
