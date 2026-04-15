import argparse
import math
import statistics
from pathlib import Path
from typing import Tuple

import torch


def ensure_torch_tensorrt_runtime() -> None:
    """
    Ensure Torch-TensorRT custom classes are registered before torch.jit.load.
    Serialized TRT TorchScript modules contain types like:
    __torch__.torch.classes.tensorrt.Engine
    which are unknown unless torch_tensorrt is imported first.
    """
    try:
        import torch_tensorrt  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch_tensorrt is required to load this engine. "
            "Use an NVIDIA PyTorch container/image that includes Torch-TensorRT, "
            "or install a matching torch_tensorrt version for your current PyTorch/CUDA build."
        ) from exc


def build_parser(default_engine: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run TensorRT TorchScript model on dummy 1s video and benchmark latency.")
    parser.add_argument(
        "--engine",
        type=Path,
        default=default_engine,
        help="Path to TensorRT TorchScript module (.ts) created by model_to_tensorRT.py",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for inference.")
    parser.add_argument("--seconds", type=float, default=1.0, help="Dummy video duration in seconds.")
    parser.add_argument("--fps", type=int, default=30, help="Dummy video frames per second.")
    parser.add_argument("--height", type=int, default=224, help="Dummy video frame height.")
    parser.add_argument("--width", type=int, default=224, help="Dummy video frame width.")
    parser.add_argument("--input-dim", type=int, default=2048, help="Expected feature dimension of the TRT model input.")
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Override temporal length sent to TRT model. If omitted, uses seconds*fps.",
    )
    parser.add_argument("--warmup-iters", type=int, default=20, help="Number of warmup iterations.")
    parser.add_argument("--iters", type=int, default=100, help="Number of measured iterations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible dummy input.")
    return parser


def make_dummy_video(batch_size: int, num_frames: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    # Dummy RGB video in [0, 1], shape: [B, T, C, H, W]
    return torch.rand(batch_size, num_frames, 3, height, width, device=device, dtype=torch.float32)


def video_to_features(video: torch.Tensor, input_dim: int) -> torch.Tensor:
    """
    Convert video tensor [B, T, C, H, W] to simple per-frame features [B, T, input_dim].
    This is only for synthetic inference testing and timing.
    """
    rgb_mean = video.mean(dim=(3, 4))  # [B, T, 3]
    rgb_std = video.std(dim=(3, 4), unbiased=False)  # [B, T, 3]

    motion = torch.zeros_like(rgb_mean)
    motion[:, 1:, :] = (rgb_mean[:, 1:, :] - rgb_mean[:, :-1, :]).abs()

    base = torch.cat([rgb_mean, rgb_std, motion], dim=-1)  # [B, T, 9]
    repeat_factor = math.ceil(input_dim / base.shape[-1])
    features = base.repeat(1, 1, repeat_factor)[..., :input_dim]
    return features.contiguous()


def match_seq_len(features: torch.Tensor, target_seq_len: int) -> torch.Tensor:
    current_seq_len = features.shape[1]
    if current_seq_len == target_seq_len:
        return features

    if current_seq_len > target_seq_len:
        return features[:, :target_seq_len, :].contiguous()

    pad_frames = target_seq_len - current_seq_len
    pad = features[:, -1:, :].repeat(1, pad_frames, 1)
    return torch.cat([features, pad], dim=1).contiguous()


def is_shape_mismatch_error(exc: RuntimeError) -> bool:
    msg = str(exc)
    patterns = [
        "setInputShape",
        "Error while setting the input shape",
        "execute_engine.cpp:149",
        "ops.tensorrt.execute_engine",
    ]
    return any(pattern in msg for pattern in patterns)


def find_compatible_seq_len(
    module: torch.jit.ScriptModule,
    base_features: torch.Tensor,
    preferred_seq_len: int,
    fallback_seq_lens: list,
) -> int:
    tried = []
    candidates = []
    for seq_len in [preferred_seq_len] + fallback_seq_lens:
        if seq_len > 0 and seq_len not in candidates:
            candidates.append(seq_len)

    with torch.inference_mode():
        for seq_len in candidates:
            probe_input = match_seq_len(base_features, seq_len)
            try:
                _ = module(probe_input)
                torch.cuda.synchronize()
                return seq_len
            except RuntimeError as exc:
                if is_shape_mismatch_error(exc):
                    tried.append(seq_len)
                    continue
                raise

    tried_str = ", ".join(str(x) for x in tried) if tried else "none"
    raise RuntimeError(
        "Could not find a compatible sequence length for this TensorRT engine. "
        f"Tried: [{tried_str}]. Pass --seq-len with the exact compile-time length."
    )


def run_benchmark(module: torch.jit.ScriptModule, model_input: torch.Tensor, warmup_iters: int, iters: int) -> Tuple[torch.Tensor, list]:
    with torch.inference_mode():
        for _ in range(warmup_iters):
            _ = module(model_input)
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        times_ms = []
        output = None

        for _ in range(iters):
            start_event.record()
            output = module(model_input)
            end_event.record()
            torch.cuda.synchronize()
            times_ms.append(start_event.elapsed_time(end_event))

    if isinstance(output, (tuple, list)):
        output = output[0]

    return output, times_ms


def print_report(
    engine_path: Path,
    video: torch.Tensor,
    features: torch.Tensor,
    output: torch.Tensor,
    times_ms: list,
    seconds: float,
) -> None:
    avg_ms = statistics.mean(times_ms)
    med_ms = statistics.median(times_ms)
    min_ms = min(times_ms)
    max_ms = max(times_ms)
    p95_ms = torch.tensor(times_ms).quantile(0.95).item()

    batch_size, num_frames = int(video.shape[0]), int(video.shape[1])
    model_seq_len = int(features.shape[1])
    clips_per_sec = (1000.0 / avg_ms) * batch_size
    frames_per_sec = clips_per_sec * num_frames
    realtime_factor = clips_per_sec * seconds

    top_idx = int(torch.argmax(output[0]).item()) if output.ndim == 2 and output.shape[0] > 0 else -1

    print()
    print("=" * 80)
    print("TensorRT Inference Benchmark")
    print("=" * 80)
    print(f"Engine path            : {engine_path}")
    print(f"Device                 : {video.device}")
    print(f"Dummy video shape      : {tuple(video.shape)}  (B, T, C, H, W)")
    print(f"Model input shape      : {tuple(features.shape)}  (B, T, F)")
    print(f"Model timesteps/clip   : {model_seq_len}")
    print(f"Model output shape     : {tuple(output.shape)}")
    print(f"Predicted class (argmax): {top_idx}")
    print("-" * 80)
    print("Latency (ms)")
    print(f"  avg                  : {avg_ms:.3f}")
    print(f"  median               : {med_ms:.3f}")
    print(f"  p95                  : {p95_ms:.3f}")
    print(f"  min                  : {min_ms:.3f}")
    print(f"  max                  : {max_ms:.3f}")
    print("-" * 80)
    print("Throughput")
    print(f"  clips/s              : {clips_per_sec:.2f}")
    print(f"  frames/s             : {frames_per_sec:.2f}")
    print(f"  real-time factor     : {realtime_factor:.2f}x")
    print("=" * 80)
    print()


def main() -> None:
    script_path = Path(__file__).resolve()
    egoems_root = script_path.parents[3]
    default_engine = egoems_root / "Benchmarks" / "ActionRecognition" / "MTRSAP" / "checkpoints" / "val_best_model_trt.ts"

    args = build_parser(default_engine).parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for TensorRT inference test.")
    if not args.engine.exists():
        raise FileNotFoundError(f"TensorRT module not found: {args.engine}")
    if args.seconds <= 0:
        raise ValueError("--seconds must be > 0")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.iters <= 0:
        raise ValueError("--iters must be > 0")
    if args.warmup_iters < 0:
        raise ValueError("--warmup-iters must be >= 0")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda")
    num_frames = max(1, int(round(args.seconds * args.fps)))
    print(f"[1/4] Loading TensorRT TorchScript module: {args.engine}")
    ensure_torch_tensorrt_runtime()
    try:
        module = torch.jit.load(str(args.engine), map_location=device).eval()
    except RuntimeError as exc:
        message = str(exc)
        if "__torch__.torch.classes.tensorrt.Engine" in message:
            raise RuntimeError(
                "Failed to load TensorRT TorchScript module because TensorRT custom class "
                "registration is missing in this runtime. Ensure torch_tensorrt is installed "
                "and importable in this environment, and that engine/runtime versions are compatible."
            ) from exc
        raise

    print("[2/4] Creating dummy 1-second video and model input")
    video = make_dummy_video(
        batch_size=args.batch_size,
        num_frames=num_frames,
        height=args.height,
        width=args.width,
        device=device,
    )
    base_features = video_to_features(video=video, input_dim=args.input_dim)

    if args.seq_len is None:
        fallback_seq_lens = [150, 120, 90, 60, 32, 64, 128, 160, 180, 200, 256, 300]
        target_seq_len = find_compatible_seq_len(
            module=module,
            base_features=base_features,
            preferred_seq_len=num_frames,
            fallback_seq_lens=fallback_seq_lens,
        )
        if target_seq_len != num_frames:
            print(
                f"[info] Auto-selected compatible seq-len={target_seq_len} "
                f"(dummy video has {num_frames} frames at {args.fps} FPS)."
            )
    else:
        target_seq_len = args.seq_len

    features = match_seq_len(features=base_features, target_seq_len=target_seq_len)

    print("[3/4] Running timed inference")
    try:
        output, times_ms = run_benchmark(
            module=module,
            model_input=features,
            warmup_iters=args.warmup_iters,
            iters=args.iters,
        )
    except RuntimeError as exc:
        if is_shape_mismatch_error(exc):
            extra = "The engine likely has a static input shape."
        else:
            extra = "Runtime error from TensorRT engine."
        shape_hint = (
            f"\nInput sent to model had shape {tuple(features.shape)}. {extra} "
            "Pass exact compile-time --seq-len and --input-dim if auto-detection is not enough."
        )
        raise RuntimeError(f"Inference failed: {exc}{shape_hint}") from exc

    print("[4/4] Printing benchmark report")
    print_report(
        engine_path=args.engine,
        video=video,
        features=features,
        output=output,
        times_ms=times_ms,
        seconds=args.seconds,
    )


if __name__ == "__main__":
    main()
