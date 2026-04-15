import argparse
import pickle
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn


def load_checkpoint_compat(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu")
    except (pickle.UnpicklingError, RuntimeError) as exc:
        msg = str(exc)
        if "Weights only load failed" in msg or "Unsupported global" in msg:
            return torch.load(path, map_location="cpu", weights_only=False)
        raise


def unwrap_state_dict(ckpt) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    if any(key.startswith("module.") for key in state):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}

    if "conv1.weight" in state:
        return state

    for prefix in ("backbone.", "encoder.", "feature_extractor.", "model."):
        remapped = {}
        for key, value in state.items():
            if key.startswith(prefix):
                remapped[key.replace(prefix, "", 1)] = value
        if "conv1.weight" in remapped:
            return remapped

    raise KeyError("Could not find ResNet50 backbone keys such as 'conv1.weight' in checkpoint.")


class ResNet50FeatureWrapper(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return features.flatten(1)


def build_resnet50_feature_model(
    checkpoint_path: Optional[Path],
    weights_name: str,
    device: torch.device,
) -> nn.Module:
    try:
        import torchvision.models as models
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torchvision is required for ResNet50 conversion.") from exc

    if checkpoint_path is not None:
        weights = None
    elif weights_name == "imagenet1k_v1":
        weights = models.ResNet50_Weights.IMAGENET1K_V1
    elif weights_name == "none":
        weights = None
    else:
        raise ValueError(f"Unsupported --weights value: {weights_name}")

    resnet = models.resnet50(weights=weights)

    if checkpoint_path is not None:
        checkpoint = load_checkpoint_compat(checkpoint_path)
        state = unwrap_state_dict(checkpoint)
        load_result = resnet.load_state_dict(state, strict=False)
        ignored_missing = [key for key in load_result.missing_keys if key.startswith("fc.")]
        unexpected = [key for key in load_result.unexpected_keys if not key.startswith("fc.")]
        remaining_missing = [key for key in load_result.missing_keys if key not in ignored_missing]
        if remaining_missing:
            raise RuntimeError(f"Missing non-FC ResNet50 keys when loading checkpoint: {remaining_missing}")
        if unexpected:
            raise RuntimeError(f"Unexpected non-FC keys when loading checkpoint: {unexpected}")
        if ignored_missing:
            print(f"[info] Ignored classifier keys not used by the feature extractor: {ignored_missing}")

    backbone = nn.Sequential(*list(resnet.children())[:-1]).eval().to(device)
    return ResNet50FeatureWrapper(backbone=backbone).eval().to(device)


def run_once(module, x: torch.Tensor):
    if callable(module):
        return module(x)
    if hasattr(module, "module"):
        inner = module.module()
        return inner(x)
    raise RuntimeError(f"Unsupported compiled module type: {type(module)}")


def parse_args(default_output: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert ResNet50 feature extractor to TensorRT TorchScript engine")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional custom ResNet50 checkpoint. If omitted, torchvision weights are used based on --weights.",
    )
    parser.add_argument("--output", type=Path, default=default_output, help="Output TensorRT engine path (.ts)")
    parser.add_argument(
        "--weights",
        type=str,
        default="imagenet1k_v1",
        choices=["imagenet1k_v1", "none"],
        help="Torchvision weights to use when --checkpoint is omitted.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Compile-time batch size")
    parser.add_argument("--height", type=int, default=224, help="Compile-time input height")
    parser.add_argument("--width", type=int, default=224, help="Compile-time input width")
    parser.add_argument("--min-batch-size", type=int, default=None, help="Dynamic min batch size")
    parser.add_argument("--opt-batch-size", type=int, default=None, help="Dynamic opt batch size")
    parser.add_argument("--max-batch-size", type=int, default=None, help="Dynamic max batch size")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 kernels")
    parser.add_argument("--atol", type=float, default=1e-3, help="PT vs TRT allclose atol")
    parser.add_argument("--rtol", type=float, default=1e-3, help="PT vs TRT allclose rtol")
    return parser.parse_args()


def validate_dynamic_batch_args(args: argparse.Namespace) -> bool:
    values = [args.min_batch_size, args.opt_batch_size, args.max_batch_size]
    provided = [value is not None for value in values]
    if any(provided) and not all(provided):
        raise ValueError("Provide all of --min-batch-size, --opt-batch-size, --max-batch-size, or none of them.")
    if all(provided):
        if not (args.min_batch_size <= args.opt_batch_size <= args.max_batch_size):
            raise ValueError("Expected min_batch_size <= opt_batch_size <= max_batch_size.")
        return True
    return False


def main() -> None:
    inference_root = Path(__file__).resolve().parents[1]
    default_output = inference_root / "checkpoints" / "resnet50_feature_extractor_trt.ts"
    args = parse_args(default_output=default_output)

    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT conversion requires CUDA.")
    if args.checkpoint is not None and not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be > 0")

    dynamic_batch = validate_dynamic_batch_args(args)
    device = torch.device("cuda")

    print("[1/5] Building ResNet50 feature extractor")
    if args.checkpoint is not None:
        print(f"  - checkpoint: {args.checkpoint}")
    else:
        print(f"  - torchvision weights: {args.weights}")
    model = build_resnet50_feature_model(
        checkpoint_path=args.checkpoint,
        weights_name=args.weights,
        device=device,
    )

    batch_size = args.opt_batch_size if dynamic_batch else args.batch_size
    example_input = torch.randn(batch_size, 3, args.height, args.width, device=device, dtype=torch.float32)

    print("[2/5] Running PyTorch sanity inference")
    with torch.inference_mode():
        pt_output = model(example_input)
    print(f"  - PT output shape: {tuple(pt_output.shape)}")

    try:
        import torch_tensorrt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch_tensorrt is required for conversion.") from exc

    print("[3/5] Tracing and compiling TensorRT engine")
    traced_model = torch.jit.trace(model, example_input, check_trace=False)
    traced_model = torch.jit.freeze(traced_model.eval())

    if dynamic_batch:
        trt_input = torch_tensorrt.Input(
            min_shape=(args.min_batch_size, 3, args.height, args.width),
            opt_shape=(args.opt_batch_size, 3, args.height, args.width),
            max_shape=(args.max_batch_size, 3, args.height, args.width),
            dtype=torch.float32,
        )
    else:
        trt_input = torch_tensorrt.Input(
            shape=(args.batch_size, 3, args.height, args.width),
            dtype=torch.float32,
        )

    enabled_precisions = {torch.float32}
    if args.fp16:
        enabled_precisions.add(torch.float16)

    trt_module = torch_tensorrt.compile(
        traced_model,
        ir="ts",
        inputs=[trt_input],
        enabled_precisions=enabled_precisions,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(trt_module, str(args.output))
    print(f"  - Saved TensorRT module: {args.output}")

    print("[4/5] Running TensorRT sanity inference")
    with torch.inference_mode():
        trt_output = run_once(trt_module, example_input)
    if isinstance(trt_output, (tuple, list)):
        trt_output = trt_output[0]

    max_abs_diff = (pt_output - trt_output).abs().max().item()
    mean_abs_diff = (pt_output - trt_output).abs().mean().item()
    is_close = torch.allclose(pt_output, trt_output, atol=args.atol, rtol=args.rtol)

    print(f"  - TRT output shape: {tuple(trt_output.shape)}")
    print(f"  - max_abs_diff: {max_abs_diff:.6f}")
    print(f"  - mean_abs_diff: {mean_abs_diff:.6f}")
    print(f"  - allclose(atol={args.atol}, rtol={args.rtol}): {is_close}")

    print("[5/5] Conversion complete")
    print("  - expected input shape: [B, 3, H, W]")
    print("  - expected input preprocessing: RGB, float32, normalized with ImageNet mean/std")
    print("  - expected output shape: [B, 2048]")

    if not is_close:
        print("[warn] PT and TRT outputs are outside tolerance. Validate with real inputs before deployment.")


if __name__ == "__main__":
    main()
