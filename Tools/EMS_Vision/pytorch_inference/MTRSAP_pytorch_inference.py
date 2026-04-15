import argparse
import csv
import json
import sys
import time
from collections import deque
from pathlib import Path
from statistics import mean
from typing import List, Optional, Tuple

import numpy as np
try:
    import cv2
except ImportError as exc:
    raise ImportError(
        "Failed to import OpenCV. In headless Docker environments, install "
        "`opencv-python-headless` instead of `opencv-python`.\n"
        "Suggested fix:\n"
        "  pip uninstall -y opencv-python opencv-contrib-python\n"
        "  pip install --no-cache-dir opencv-python-headless\n"
        "If it still fails, install OS libs:\n"
        "  apt-get update && apt-get install -y libglib2.0-0 libxcb1\n"
    ) from exc

import torch
import torch.nn as nn
import torch.nn.functional as F

from conversion.model_to_tensorRT import _load_state_dict, infer_model_config
from models.mtrsap_model import TransformerInferenceModel


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def load_pytorch_model(
    checkpoint_path: Path,
    device: torch.device,
    nhead: int,
    dropout: float,
    batch_first: bool,
):
    state = _load_state_dict(checkpoint_path)
    inferred = infer_model_config(state)

    if inferred["d_model"] % nhead != 0:
        raise ValueError(
            f"d_model={inferred['d_model']} is not divisible by nhead={nhead}. "
            "Pass the correct --nhead used in training."
        )

    model = TransformerInferenceModel(
        input_dim=inferred["input_dim"],
        d_model=inferred["d_model"],
        output_dim=inferred["output_dim"],
        kernel_size=inferred["kernel_size"],
        nhead=nhead,
        num_layers=inferred["num_layers"],
        dropout=dropout,
        max_len=inferred["max_len"],
        batch_first=batch_first,
    ).to(device).eval()

    load_result = model.load_state_dict(state, strict=False)
    return model, inferred, load_result


def build_resnet50_feature_extractor(device: torch.device) -> nn.Module:
    try:
        import torchvision.models as models
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torchvision is required for frame feature extraction.") from exc

    weights = models.ResNet50_Weights.IMAGENET1K_V1
    resnet = models.resnet50(weights=weights)
    backbone = nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()
    for param in backbone.parameters():
        param.requires_grad = False
    return backbone


def _resize_shorter_side(batch: torch.Tensor, shorter_side: int) -> torch.Tensor:
    # batch shape: [T, C, H, W]
    _, _, h, w = batch.shape
    if min(h, w) == shorter_side:
        return batch

    if h <= w:
        new_h = shorter_side
        new_w = int(round((w / h) * shorter_side))
    else:
        new_w = shorter_side
        new_h = int(round((h / w) * shorter_side))

    return F.interpolate(batch, size=(new_h, new_w), mode="bilinear", align_corners=False, antialias=True)


def _center_crop(batch: torch.Tensor, crop_size: int) -> torch.Tensor:
    # batch shape: [T, C, H, W]
    _, _, h, w = batch.shape
    if crop_size > h or crop_size > w:
        raise ValueError(f"Center crop size ({crop_size}) exceeds resized frame size ({h}, {w}).")
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return batch[:, :, top : top + crop_size, left : left + crop_size]


def preprocess_window_frames(
    frames_bgr: List,
    resize_short_side: int,
    center_crop_size: int,
    device: torch.device,
) -> torch.Tensor:
    rgb_frames = [cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) for frame_bgr in frames_bgr]
    np_batch = np.stack(rgb_frames, axis=0)  # [T, H, W, C], uint8
    batch = torch.from_numpy(np_batch).permute(0, 3, 1, 2).to(device=device, dtype=torch.float32) / 255.0

    batch = _resize_shorter_side(batch, resize_short_side)
    batch = _center_crop(batch, center_crop_size)

    mean_t = IMAGENET_MEAN.to(device)
    std_t = IMAGENET_STD.to(device)
    batch = (batch - mean_t) / std_t
    return batch


def extract_window_features(
    frames_bgr: List,
    extractor: nn.Module,
    device: torch.device,
    resize_short_side: int,
    center_crop_size: int,
    feature_batch_size: int,
) -> torch.Tensor:
    frame_tensor = preprocess_window_frames(
        frames_bgr=frames_bgr,
        resize_short_side=resize_short_side,
        center_crop_size=center_crop_size,
        device=device,
    )
    frame_count = frame_tensor.shape[0]

    features = []
    with torch.inference_mode():
        for start in range(0, frame_count, feature_batch_size):
            end = min(start + feature_batch_size, frame_count)
            out = extractor(frame_tensor[start:end])  # [n, 2048, 1, 1]
            out = out.flatten(1)  # [n, 2048]
            features.append(out)

    window_features = torch.cat(features, dim=0).unsqueeze(0).contiguous()  # [1, T, 2048]
    return window_features


def adjust_seq_len(features: torch.Tensor, target_seq_len: int) -> torch.Tensor:
    current_seq_len = int(features.shape[1])
    if current_seq_len == target_seq_len:
        return features
    if current_seq_len > target_seq_len:
        return features[:, :target_seq_len, :].contiguous()
    pad_len = target_seq_len - current_seq_len
    pad = features[:, -1:, :].repeat(1, pad_len, 1)
    return torch.cat([features, pad], dim=1).contiguous()


def timed_inference(model: nn.Module, model_input: torch.Tensor, device: torch.device):
    with torch.inference_mode():
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output = model(model_input)
            end_event.record()
            torch.cuda.synchronize()
            infer_ms = start_event.elapsed_time(end_event)
        else:
            t0 = time.perf_counter()
            output = model(model_input)
            infer_ms = (time.perf_counter() - t0) * 1000.0

    if isinstance(output, (list, tuple)):
        output = output[0]
    return output, float(infer_ms)


def load_class_names(class_map_path: Path, num_classes: int) -> List[str]:
    labels = [f"class_{idx}" for idx in range(num_classes)]
    if not class_map_path.exists():
        print(f"[warn] Class mapping file not found: {class_map_path}. Using fallback class_<id> labels.")
        return labels

    try:
        with open(class_map_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        keysteps = data.get("keysteps", {})
        if not isinstance(keysteps, dict):
            print(f"[warn] Invalid class mapping format in {class_map_path}. Using fallback labels.")
            return labels

        for label, class_id in keysteps.items():
            if isinstance(class_id, int) and 0 <= class_id < num_classes:
                labels[class_id] = label
    except Exception as exc:
        print(f"[warn] Failed to read class mapping ({class_map_path}): {exc}. Using fallback labels.")

    return labels


def load_resnet_feature_array(npy_path: Path) -> torch.Tensor:
    return _load_feature_array_2d(npy_path=npy_path, feature_name="resnet")


def _load_feature_array_2d(npy_path: Path, feature_name: str) -> torch.Tensor:
    arr = np.load(str(npy_path))
    feat = torch.from_numpy(arr)

    # Expected shapes:
    # [T, F]   -> standard
    # [1, T, F] -> squeeze batch dim
    if feat.ndim == 3 and feat.shape[0] == 1:
        feat = feat.squeeze(0)
    if feat.ndim != 2:
        raise ValueError(f"Expected {feature_name} npy shape [T, F] (or [1, T, F]), got {tuple(feat.shape)}")

    return feat.float().contiguous()


def load_i3d_fused_feature_array(rgb_npy_path: Path, flow_npy_path: Path) -> torch.Tensor:
    rgb = _load_feature_array_2d(npy_path=rgb_npy_path, feature_name="i3d rgb")
    flow = _load_feature_array_2d(npy_path=flow_npy_path, feature_name="i3d flow")

    rgb_len = int(rgb.shape[0])
    flow_len = int(flow.shape[0])
    common_len = min(rgb_len, flow_len)
    if common_len <= 0:
        raise ValueError(f"Invalid I3D feature lengths: rgb={rgb_len}, flow={flow_len}")
    if rgb_len != flow_len:
        print(
            f"[warn] I3D rgb/flow sequence length mismatch: rgb={rgb_len}, flow={flow_len}. "
            f"Truncating both to {common_len}."
        )

    rgb = rgb[:common_len]
    flow = flow[:common_len]

    # Keep fusion order consistent with training preprocess in utils.py: cat((flow, rgb), dim=-1).
    fused = torch.cat((flow, rgb), dim=-1).float().contiguous()
    return fused


def _resize_shorter_side_frame_rgb(frame_rgb: np.ndarray, shorter_side: int) -> np.ndarray:
    h, w, _ = frame_rgb.shape
    if min(h, w) == shorter_side:
        return frame_rgb
    if h <= w:
        new_h = shorter_side
        new_w = int(round((w / h) * shorter_side))
    else:
        new_w = shorter_side
        new_h = int(round((h / w) * shorter_side))
    return cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def _center_crop_tchw(batch: torch.Tensor, crop_size: int) -> torch.Tensor:
    # batch shape: [T, C, H, W]
    _, _, h, w = batch.shape
    if crop_size > h or crop_size > w:
        raise ValueError(f"Center crop size ({crop_size}) exceeds frame size ({h}, {w}).")
    top = (h - crop_size) // 2
    left = (w - crop_size) // 2
    return batch[:, :, top : top + crop_size, left : left + crop_size]


def _resolve_existing_path(path_value: Optional[Path], candidates: List[Path], label: str) -> Path:
    if path_value is not None:
        if path_value.exists():
            return path_value
        raise FileNotFoundError(f"{label} not found: {path_value}")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{label} not found. Checked: {', '.join(str(c) for c in candidates)}")


def _import_runtime_i3d_modules(i3d_runtime_root: Optional[Path]):
    if i3d_runtime_root is not None:
        sys.path.insert(0, str(i3d_runtime_root))
    try:
        from models.i3d.i3d_src.i3d_net import I3D  # type: ignore
        from models.raft.extract_raft import DATASET_to_RAFT_CKPT_PATHS  # type: ignore
        from models.raft.raft_src.raft import RAFT, InputPadder  # type: ignore
        from utils.utils import dp_state_to_normal  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Runtime I3D extraction requires your original extractor dependencies:\n"
            "  - models.i3d.i3d_src.i3d_net.I3D\n"
            "  - models.raft.extract_raft.DATASET_to_RAFT_CKPT_PATHS\n"
            "  - models.raft.raft_src.raft.RAFT/InputPadder\n"
            "  - utils.utils.dp_state_to_normal\n"
            "Provide --i3d-runtime-root pointing to that project root, or use precomputed "
            "--i3d-rgb-npy-path/--i3d-flow-npy-path."
        ) from exc
    return I3D, RAFT, InputPadder, DATASET_to_RAFT_CKPT_PATHS, dp_state_to_normal


def extract_i3d_fused_features_from_video(
    video_path: Path,
    device: torch.device,
    resize_short_side: int,
    center_crop_size: int,
    i3d_stack_size: int,
    i3d_step_size: int,
    i3d_runtime_root: Optional[Path],
    i3d_rgb_checkpoint_path: Optional[Path],
    i3d_flow_checkpoint_path: Optional[Path],
    i3d_raft_checkpoint_path: Optional[Path],
) -> Tuple[torch.Tensor, float]:
    if i3d_stack_size <= 0 or i3d_step_size <= 0:
        raise ValueError("--i3d-stack-size and --i3d-step-size must be > 0")
    if i3d_step_size > i3d_stack_size:
        raise ValueError("--i3d-step-size should be <= --i3d-stack-size")

    I3D, RAFT, InputPadder, DATASET_to_RAFT_CKPT_PATHS, dp_state_to_normal = _import_runtime_i3d_modules(i3d_runtime_root)

    runtime_root = i3d_runtime_root if i3d_runtime_root is not None else Path(".")
    repo_root = Path(__file__).resolve().parents[2]
    rgb_ckpt = _resolve_existing_path(
        path_value=i3d_rgb_checkpoint_path,
        candidates=[
            runtime_root / "models" / "i3d" / "checkpoints" / "i3d_rgb.pt",
            repo_root / "models" / "i3d" / "checkpoints" / "i3d_rgb.pt",
            runtime_root / "i3d_rgb.pt",
            repo_root / "i3d_rgb.pt",
        ],
        label="I3D RGB checkpoint",
    )
    flow_ckpt = _resolve_existing_path(
        path_value=i3d_flow_checkpoint_path,
        candidates=[
            runtime_root / "models" / "i3d" / "checkpoints" / "i3d_flow.pt",
            repo_root / "models" / "i3d" / "checkpoints" / "i3d_flow.pt",
            runtime_root / "i3d_flow.pt",
            repo_root / "i3d_flow.pt",
        ],
        label="I3D flow checkpoint",
    )
    raft_default = Path(DATASET_to_RAFT_CKPT_PATHS["sintel"]) if "sintel" in DATASET_to_RAFT_CKPT_PATHS else None
    raft_ckpt = _resolve_existing_path(
        path_value=i3d_raft_checkpoint_path,
        candidates=[p for p in [raft_default] if p is not None],
        label="RAFT checkpoint",
    )

    raft_model = RAFT().to(device).eval()
    raft_state = torch.load(str(raft_ckpt), map_location="cpu")
    raft_state = dp_state_to_normal(raft_state)
    raft_model.load_state_dict(raft_state)

    i3d_rgb_model = I3D(num_classes=400, modality="rgb").to(device).eval()
    i3d_rgb_model.load_state_dict(torch.load(str(rgb_ckpt), map_location="cpu"))

    i3d_flow_model = I3D(num_classes=400, modality="flow").to(device).eval()
    i3d_flow_model.load_state_dict(torch.load(str(flow_ckpt), map_location="cpu"))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    native_fps = float(cap.get(cv2.CAP_PROP_FPS))

    rgb_stack = []
    rgb_features = []
    flow_features = []
    padder = None

    with torch.inference_mode():
        while cap.isOpened():
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = _resize_shorter_side_frame_rgb(frame_rgb, resize_short_side)
            frame_t = torch.from_numpy(frame_rgb).permute(2, 0, 1).to(dtype=torch.float32)  # [3, H, W], 0..255
            rgb_stack.append(frame_t)

            # Need stack_size + 1 frames to build stack_size RGB/flow pairs.
            if len(rgb_stack) - 1 < i3d_stack_size:
                continue

            rgb_seq = torch.stack(rgb_stack, dim=0).to(device=device, dtype=torch.float32)  # [stack+1, 3, H, W]
            if padder is None:
                padder = InputPadder(rgb_seq.shape)
            rgb_padded = padder.pad(rgb_seq)
            flow_seq = raft_model(rgb_padded[:-1], rgb_padded[1:])
            if isinstance(flow_seq, (list, tuple)):
                flow_seq = flow_seq[0]
            if flow_seq.ndim != 4:
                raise RuntimeError(f"Unexpected RAFT output shape: {tuple(flow_seq.shape)}")

            rgb_stream_tchw = rgb_seq[:-1]  # [stack, 3, H, W]
            rgb_stream_tchw = _center_crop_tchw(rgb_stream_tchw, center_crop_size)
            rgb_stream = ((rgb_stream_tchw / 255.0) * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()

            flow_stream_tchw = _center_crop_tchw(flow_seq, center_crop_size)  # [stack, 2, H, W]
            flow_stream_tchw = torch.clamp(flow_stream_tchw, min=-20.0, max=20.0)
            flow_uint8 = torch.round((flow_stream_tchw + 20.0) * (255.0 / 40.0)).clamp(0.0, 255.0)
            flow_stream = ((flow_uint8 / 255.0) * 2.0 - 1.0).permute(1, 0, 2, 3).unsqueeze(0).contiguous()

            rgb_feat = i3d_rgb_model(rgb_stream, features=True)
            flow_feat = i3d_flow_model(flow_stream, features=True)
            if isinstance(rgb_feat, (tuple, list)):
                rgb_feat = rgb_feat[0]
            if isinstance(flow_feat, (tuple, list)):
                flow_feat = flow_feat[0]
            if rgb_feat.ndim != 2 or flow_feat.ndim != 2:
                raise RuntimeError(
                    f"Unexpected I3D feature shapes: rgb={tuple(rgb_feat.shape)}, flow={tuple(flow_feat.shape)}"
                )

            rgb_features.extend(rgb_feat.detach().cpu().tolist())
            flow_features.extend(flow_feat.detach().cpu().tolist())
            rgb_stack = rgb_stack[i3d_step_size:]

    cap.release()

    if len(rgb_features) == 0 or len(flow_features) == 0:
        raise RuntimeError(
            "Runtime I3D extraction produced no features. "
            "Increase video length or decrease --i3d-stack-size/--window-seconds."
        )

    rgb_tensor = torch.tensor(rgb_features, dtype=torch.float32)
    flow_tensor = torch.tensor(flow_features, dtype=torch.float32)
    common_len = min(int(rgb_tensor.shape[0]), int(flow_tensor.shape[0]))
    if int(rgb_tensor.shape[0]) != int(flow_tensor.shape[0]):
        print(
            f"[warn] Runtime I3D rgb/flow feature count mismatch: rgb={int(rgb_tensor.shape[0])}, "
            f"flow={int(flow_tensor.shape[0])}. Truncating both to {common_len}."
        )
    rgb_tensor = rgb_tensor[:common_len]
    flow_tensor = flow_tensor[:common_len]
    fused = torch.cat((flow_tensor, rgb_tensor), dim=-1).float().contiguous()
    return fused, native_fps


def validate_feature_dim(feature_dim: int, expected_dim: int, source_desc: str) -> None:
    if feature_dim != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: {source_desc} produces F={feature_dim}, "
            f"but checkpoint expects input_dim={expected_dim}."
        )


def build_arg_parser(default_checkpoint: Path, default_csv: Path, default_class_map: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sliding-window MTRSAP inference on video using a PyTorch checkpoint")
    parser.add_argument(
        "--feature-type",
        type=str,
        default="resnet",
        choices=["resnet", "i3d_rgb_flow"],
        help="Feature source type: resnet (video/resnet npy) or i3d_rgb_flow (fused flow+rgb npy)",
    )
    parser.add_argument("--video-path", type=Path, default=None, help="Path to input video")
    parser.add_argument(
        "--resnet-npy-path",
        type=Path,
        default=None,
        help="Path to pre-extracted ResNet feature .npy file (sanity mode, skips frame feature extraction)",
    )
    parser.add_argument("--i3d-rgb-npy-path", type=Path, default=None, help="Path to pre-extracted I3D RGB feature .npy file")
    parser.add_argument(
        "--i3d-flow-npy-path",
        type=Path,
        default=None,
        help="Path to pre-extracted I3D flow feature .npy file",
    )
    parser.add_argument(
        "--i3d-runtime-root",
        type=Path,
        default=None,
        help="Project root that contains models.i3d/models.raft/utils modules for runtime I3D extraction",
    )
    parser.add_argument("--i3d-rgb-checkpoint-path", type=Path, default=None, help="I3D RGB checkpoint path (.pt)")
    parser.add_argument("--i3d-flow-checkpoint-path", type=Path, default=None, help="I3D flow checkpoint path (.pt)")
    parser.add_argument("--i3d-raft-checkpoint-path", type=Path, default=None, help="RAFT checkpoint path (.pth/.pt)")
    parser.add_argument("--i3d-stack-size", type=int, default=64, help="I3D runtime stack size (needs stack+1 frames)")
    parser.add_argument("--i3d-step-size", type=int, default=64, help="I3D runtime step size between stacks")
    parser.add_argument("--checkpoint-path", type=Path, default=default_checkpoint, help="PyTorch checkpoint path (.pt)")
    parser.add_argument("--csv-path", type=Path, default=default_csv, help="CSV output path")
    parser.add_argument("--class-map-path", type=Path, default=default_class_map, help="Path to class_id_mappings.json")

    parser.add_argument("--window-seconds", type=float, default=1.0, help="Sliding window duration in seconds")
    parser.add_argument("--stride-seconds", type=float, default=0.5, help="Sliding window stride in seconds")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS for window math (default: video FPS)")
    parser.add_argument("--model-seq-len", type=int, default=None, help="Optional fixed T for model input (pad/truncate)")

    parser.add_argument("--resize-short-side", type=int, default=256, help="Resize shorter frame side before crop")
    parser.add_argument("--center-crop-size", type=int, default=224, help="Center crop size after resize")
    parser.add_argument("--feature-batch-size", type=int, default=32, help="Frame batch size for feature extraction")
    parser.add_argument("--max-windows", type=int, default=None, help="Optional max number of windows")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Inference device")

    parser.add_argument("--nhead", type=int, default=4, help="Transformer attention heads (must divide d_model)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used in positional encoding/transformer")
    parser.add_argument("--batch-first", action="store_true", default=True, help="Use batch-first transformer layout")
    parser.add_argument(
        "--disable-batch-first",
        action="store_false",
        dest="batch_first",
        help="Disable batch-first transformer layout",
    )
    return parser


def main() -> None:
    script_path = Path(__file__).resolve()
    egoems_root = script_path.parents[2]
    default_checkpoint = egoems_root / "Benchmarks" / "ActionRecognition" / "MTRSAP" / "checkpoints" / "val_best_model.pt"
    default_csv = script_path.parent / "mtrsap_pytorch_inference_results.csv"
    default_class_map = egoems_root / "Tools" / "annotation_generation" / "class_id_mappings.json"

    args = build_arg_parser(
        default_checkpoint=default_checkpoint,
        default_csv=default_csv,
        default_class_map=default_class_map,
    ).parse_args()

    if args.feature_type == "resnet":
        if args.i3d_rgb_npy_path is not None or args.i3d_flow_npy_path is not None:
            raise ValueError("I3D npy inputs can only be used with --feature-type i3d_rgb_flow")
        if args.video_path is None and args.resnet_npy_path is None:
            raise ValueError("For --feature-type resnet, provide at least one input source: --video-path or --resnet-npy-path")
        if args.video_path is not None and not args.video_path.exists():
            raise FileNotFoundError(f"Video not found: {args.video_path}")
        if args.resnet_npy_path is not None and not args.resnet_npy_path.exists():
            raise FileNotFoundError(f"ResNet npy file not found: {args.resnet_npy_path}")
        feature_source = "resnet_npy" if args.resnet_npy_path is not None else "video_resnet"
    else:
        if args.resnet_npy_path is not None:
            raise ValueError("--resnet-npy-path cannot be combined with --feature-type i3d_rgb_flow")
        has_i3d_npy_inputs = args.i3d_rgb_npy_path is not None or args.i3d_flow_npy_path is not None
        if has_i3d_npy_inputs:
            if args.i3d_rgb_npy_path is None or args.i3d_flow_npy_path is None:
                raise ValueError("Provide both --i3d-rgb-npy-path and --i3d-flow-npy-path, or provide neither for runtime extraction")
            if not args.i3d_rgb_npy_path.exists():
                raise FileNotFoundError(f"I3D RGB npy file not found: {args.i3d_rgb_npy_path}")
            if not args.i3d_flow_npy_path.exists():
                raise FileNotFoundError(f"I3D flow npy file not found: {args.i3d_flow_npy_path}")
            if args.video_path is not None:
                raise ValueError("When --i3d-rgb-npy-path/--i3d-flow-npy-path are provided, omit --video-path.")
            feature_source = "i3d_npy"
        else:
            if args.video_path is None:
                raise ValueError(
                    "For --feature-type i3d_rgb_flow without precomputed npy paths, provide --video-path for runtime extraction."
                )
            if not args.video_path.exists():
                raise FileNotFoundError(f"Video not found: {args.video_path}")
            feature_source = "video_i3d_runtime"

    if not args.checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")

    if args.window_seconds <= 0 or args.stride_seconds <= 0:
        raise ValueError("--window-seconds and --stride-seconds must be > 0")
    if args.feature_batch_size <= 0:
        raise ValueError("--feature-batch-size must be > 0")
    if args.resize_short_side <= 0 or args.center_crop_size <= 0:
        raise ValueError("--resize-short-side and --center-crop-size must be > 0")
    if args.center_crop_size > args.resize_short_side:
        raise ValueError("--center-crop-size should be <= --resize-short-side")
    if args.i3d_stack_size <= 0 or args.i3d_step_size <= 0:
        raise ValueError("--i3d-stack-size and --i3d-step-size must be > 0")
    if args.i3d_step_size > args.i3d_stack_size:
        raise ValueError("--i3d-step-size should be <= --i3d-stack-size")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    print("[1/4] Loading PyTorch checkpoint")
    model, inferred, load_result = load_pytorch_model(
        checkpoint_path=args.checkpoint_path,
        device=device,
        nhead=args.nhead,
        dropout=args.dropout,
        batch_first=args.batch_first,
    )
    print(f"  - checkpoint: {args.checkpoint_path}")
    print(
        "  - inferred model: "
        f"input_dim={inferred['input_dim']}, d_model={inferred['d_model']}, "
        f"output_dim={inferred['output_dim']}, kernel_size={inferred['kernel_size']}, "
        f"num_layers={inferred['num_layers']}, max_len={inferred['max_len']}"
    )
    if load_result.missing_keys:
        print(f"[warn] Missing keys while loading: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"[info] Ignored unexpected checkpoint keys: {len(load_result.unexpected_keys)}")

    if feature_source == "resnet_npy":
        print("[2/4] Loading pre-extracted ResNet features")
        extractor = None
        feature_all = load_resnet_feature_array(args.resnet_npy_path)
        validate_feature_dim(
            feature_dim=int(feature_all.shape[1]),
            expected_dim=int(inferred["input_dim"]),
            source_desc="resnet npy features",
        )
        total_source_frames = int(feature_all.shape[0])
        native_fps = None
        if args.fps is None:
            effective_fps = 30.0
            print("[warn] --fps not provided for npy mode; defaulting to 30.0 for window time reporting.")
        else:
            effective_fps = float(args.fps)
    elif feature_source == "video_resnet":
        print("[2/4] Building ResNet50 feature extractor")
        validate_feature_dim(
            feature_dim=2048,
            expected_dim=int(inferred["input_dim"]),
            source_desc="on-the-fly ResNet50 frame features",
        )
        extractor = build_resnet50_feature_extractor(device=device)
        feature_all = None
        cap = cv2.VideoCapture(str(args.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {args.video_path}")
        native_fps = float(cap.get(cv2.CAP_PROP_FPS))
        effective_fps = float(args.fps) if args.fps is not None else native_fps
        total_source_frames = None
    elif feature_source == "video_i3d_runtime":
        print("[2/4] Extracting I3D flow+rgb features from video at runtime")
        extractor = None
        feature_all, native_fps = extract_i3d_fused_features_from_video(
            video_path=args.video_path,
            device=device,
            resize_short_side=args.resize_short_side,
            center_crop_size=args.center_crop_size,
            i3d_stack_size=args.i3d_stack_size,
            i3d_step_size=args.i3d_step_size,
            i3d_runtime_root=args.i3d_runtime_root,
            i3d_rgb_checkpoint_path=args.i3d_rgb_checkpoint_path,
            i3d_flow_checkpoint_path=args.i3d_flow_checkpoint_path,
            i3d_raft_checkpoint_path=args.i3d_raft_checkpoint_path,
        )
        validate_feature_dim(
            feature_dim=int(feature_all.shape[1]),
            expected_dim=int(inferred["input_dim"]),
            source_desc="runtime fused i3d flow+rgb features",
        )
        total_source_frames = int(feature_all.shape[0])
        effective_fps = float(args.fps) if args.fps is not None else native_fps
    else:
        print("[2/4] Loading and fusing pre-extracted I3D flow+rgb features")
        extractor = None
        feature_all = load_i3d_fused_feature_array(
            rgb_npy_path=args.i3d_rgb_npy_path,
            flow_npy_path=args.i3d_flow_npy_path,
        )
        validate_feature_dim(
            feature_dim=int(feature_all.shape[1]),
            expected_dim=int(inferred["input_dim"]),
            source_desc="fused i3d flow+rgb features",
        )
        total_source_frames = int(feature_all.shape[0])
        native_fps = None
        if args.fps is None:
            effective_fps = 30.0
            print("[warn] --fps not provided for npy mode; defaulting to 30.0 for window time reporting.")
        else:
            effective_fps = float(args.fps)

    if effective_fps <= 0:
        effective_fps = 30.0

    window_frames = max(1, int(round(args.window_seconds * effective_fps)))
    stride_frames = max(1, int(round(args.stride_seconds * effective_fps)))

    print("[3/4] Running sliding-window inference")
    print(f"  - feature type: {args.feature_type}")
    if feature_source == "resnet_npy":
        print(f"  - resnet npy: {args.resnet_npy_path}")
        print(f"  - source feature shape: {tuple(feature_all.shape)}")
    elif feature_source == "i3d_npy":
        print(f"  - i3d rgb npy : {args.i3d_rgb_npy_path}")
        print(f"  - i3d flow npy: {args.i3d_flow_npy_path}")
        print(f"  - fused source feature shape: {tuple(feature_all.shape)}")
    elif feature_source == "video_i3d_runtime":
        print(f"  - video: {args.video_path}")
        print(f"  - runtime fused feature shape: {tuple(feature_all.shape)}")
        print(f"  - i3d stack/step: {args.i3d_stack_size}/{args.i3d_step_size}")
    elif feature_source == "video_resnet":
        print(f"  - video: {args.video_path}")
    print(f"  - fps used: {effective_fps:.3f}")
    if native_fps is not None:
        print(f"  - video native fps: {native_fps:.3f}")
    print(f"  - window: {window_frames} frames ({args.window_seconds:.2f}s)")
    print(f"  - stride: {stride_frames} frames ({args.stride_seconds:.2f}s)")
    print(f"  - preprocess: resize_short_side={args.resize_short_side}, center_crop={args.center_crop_size}")
    if args.model_seq_len is not None:
        print(f"  - model seq len override: {args.model_seq_len}")

    start_frame_idx = 0
    window_idx = 0
    rows = []
    feature_times_ms = []
    infer_times_ms = []
    total_times_ms = []
    class_names = None

    print("-" * 120)
    print(
        f"{'win':>5} {'start_s':>10} {'end_s':>10} {'argmax':>8} {'confidence':>12} "
        f"{'feature_ms':>12} {'infer_ms':>10} {'total_ms':>10}  label"
    )
    print("-" * 120)

    if feature_source in {"resnet_npy", "i3d_npy", "video_i3d_runtime"}:
        if total_source_frames < window_frames:
            raise RuntimeError("Feature source is shorter than one window. Increase source length or reduce --window-seconds.")

        for start_frame_idx in range(0, total_source_frames - window_frames + 1, stride_frames):
            t0 = time.perf_counter()
            window_feats = feature_all[start_frame_idx : start_frame_idx + window_frames]  # [T, F]
            features = window_feats.unsqueeze(0).to(device=device, dtype=torch.float32).contiguous()
            if args.model_seq_len is not None:
                features = adjust_seq_len(features, args.model_seq_len)
            feature_ms = (time.perf_counter() - t0) * 1000.0

            logits, infer_ms = timed_inference(model, features, device=device)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            pred_idx_i = int(pred_idx.item())
            conf_f = float(conf.item())

            if class_names is None:
                class_names = load_class_names(class_map_path=args.class_map_path, num_classes=int(logits.shape[1]))
            pred_label = class_names[pred_idx_i] if pred_idx_i < len(class_names) else f"class_{pred_idx_i}"

            start_sec = start_frame_idx / effective_fps
            end_sec = (start_frame_idx + window_frames) / effective_fps
            total_ms = feature_ms + infer_ms

            print(
                f"{window_idx:5d} {start_sec:10.3f} {end_sec:10.3f} {pred_idx_i:8d} {conf_f:12.4f} "
                f"{feature_ms:12.3f} {infer_ms:10.3f} {total_ms:10.3f}  {pred_label}"
            )

            rows.append(
                {
                    "window_idx": window_idx,
                    "start_frame": start_frame_idx,
                    "end_frame_exclusive": start_frame_idx + window_frames,
                    "start_sec": round(start_sec, 6),
                    "end_sec": round(end_sec, 6),
                    "argmax_idx": pred_idx_i,
                    "argmax_label": pred_label,
                    "confidence": conf_f,
                    "feature_ms": feature_ms,
                    "infer_ms": infer_ms,
                    "total_ms": total_ms,
                }
            )
            feature_times_ms.append(feature_ms)
            infer_times_ms.append(infer_ms)
            total_times_ms.append(total_ms)

            window_idx += 1
            if args.max_windows is not None and window_idx >= args.max_windows:
                break
    else:
        buffer = deque()
        while len(buffer) < window_frames:
            ok, frame = cap.read()
            if not ok:
                break
            buffer.append(frame)

        if len(buffer) < window_frames:
            cap.release()
            raise RuntimeError("Video is shorter than one window. Increase video length or reduce --window-seconds.")

        while len(buffer) == window_frames:
            frames = list(buffer)

            t0 = time.perf_counter()
            features = extract_window_features(
                frames_bgr=frames,
                extractor=extractor,
                device=device,
                resize_short_side=args.resize_short_side,
                center_crop_size=args.center_crop_size,
                feature_batch_size=args.feature_batch_size,
            )
            if args.model_seq_len is not None:
                features = adjust_seq_len(features, args.model_seq_len)
            feature_ms = (time.perf_counter() - t0) * 1000.0

            logits, infer_ms = timed_inference(model, features, device=device)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            pred_idx_i = int(pred_idx.item())
            conf_f = float(conf.item())

            if class_names is None:
                class_names = load_class_names(class_map_path=args.class_map_path, num_classes=int(logits.shape[1]))
            pred_label = class_names[pred_idx_i] if pred_idx_i < len(class_names) else f"class_{pred_idx_i}"

            start_sec = start_frame_idx / effective_fps
            end_sec = (start_frame_idx + window_frames) / effective_fps
            total_ms = feature_ms + infer_ms

            print(
                f"{window_idx:5d} {start_sec:10.3f} {end_sec:10.3f} {pred_idx_i:8d} {conf_f:12.4f} "
                f"{feature_ms:12.3f} {infer_ms:10.3f} {total_ms:10.3f}  {pred_label}"
            )

            rows.append(
                {
                    "window_idx": window_idx,
                    "start_frame": start_frame_idx,
                    "end_frame_exclusive": start_frame_idx + window_frames,
                    "start_sec": round(start_sec, 6),
                    "end_sec": round(end_sec, 6),
                    "argmax_idx": pred_idx_i,
                    "argmax_label": pred_label,
                    "confidence": conf_f,
                    "feature_ms": feature_ms,
                    "infer_ms": infer_ms,
                    "total_ms": total_ms,
                }
            )
            feature_times_ms.append(feature_ms)
            infer_times_ms.append(infer_ms)
            total_times_ms.append(total_ms)

            window_idx += 1
            if args.max_windows is not None and window_idx >= args.max_windows:
                break

            for _ in range(stride_frames):
                if not buffer:
                    break
                buffer.popleft()
                start_frame_idx += 1
                ok, frame = cap.read()
                if ok:
                    buffer.append(frame)

        cap.release()

    if not rows:
        raise RuntimeError("No windows were processed.")

    args.csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv_path, mode="w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 120)
    print("[4/4] Summary")
    print(f"  - windows processed : {len(rows)}")
    print(f"  - avg feature ms    : {mean(feature_times_ms):.3f}")
    print(f"  - avg infer ms      : {mean(infer_times_ms):.3f}")
    print(f"  - avg total ms      : {mean(total_times_ms):.3f}")
    print(f"  - avg windows/s     : {1000.0 / mean(total_times_ms):.3f}")
    print(f"  - csv saved         : {args.csv_path}")


if __name__ == "__main__":
    main()
