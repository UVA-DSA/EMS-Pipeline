import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.fx
import torch.nn as nn


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


def load_checkpoint_compat(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu")
    except (pickle.UnpicklingError, RuntimeError) as exc:
        msg = str(exc)
        if "Weights only load failed" in msg or "Unsupported global" in msg:
            return torch.load(path, map_location="cpu", weights_only=False)
        raise


def resolve_default_checkpoint(script_dir: Path, detr_version: str) -> Path:
    if detr_version == "ems":
        return script_dir / "checkpoints" / "ems_finetuned_detr_checkpoint.pth"
    return script_dir / "checkpoints" / "detr-r50-e632da11.pth"


def build_model(num_classes: int, checkpoint_path: Path, device: torch.device) -> nn.Module:
    checkpoint = load_checkpoint_compat(checkpoint_path)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    model = torch.hub.load("facebookresearch/detr", "detr_resnet50", pretrained=False, num_classes=num_classes)
    model.load_state_dict(state_dict, strict=True)
    # Keep forward graph inference-only to avoid auxiliary IValue structures during TRT conversion.
    if hasattr(model, "aux_loss"):
        model.aux_loss = False
    if hasattr(model, "transformer") and hasattr(model.transformer, "decoder"):
        if hasattr(model.transformer.decoder, "return_intermediate"):
            model.transformer.decoder.return_intermediate = False
    model = model.eval().to(device)
    return model


class DetrTensorOutputWrapper(nn.Module):
    def __init__(self, detr_model: nn.Module) -> None:
        super().__init__()
        self.detr_model = detr_model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.detr_model(x)
        return out["pred_logits"], out["pred_boxes"]


def run_once(module, x: torch.Tensor):
    if callable(module):
        return module(x)
    if hasattr(module, "module"):
        inner = module.module()
        return inner(x)
    raise RuntimeError(f"Unsupported compiled module type: {type(module)}")


def parse_args(default_checkpoint: Path, default_output: Path) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DETR checkpoint to TensorRT TorchScript engine")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint, help="Path to DETR checkpoint (.pth)")
    parser.add_argument("--output", type=Path, default=default_output, help="Output TensorRT engine path (.ts for TS, .ep for Dynamo)")
    parser.add_argument("--detr-version", type=str, default="ems", choices=["ems", "base"], help="Class set to use")
    parser.add_argument(
        "--ir",
        type=str,
        default="auto",
        choices=["auto", "ts", "dynamo"],
        help="TensorRT compile path: ts, dynamo, or auto fallback (recommended).",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Compile-time batch size")
    parser.add_argument("--height", type=int, default=480, help="Compile-time OPT frame height")
    parser.add_argument("--width", type=int, default=640, help="Compile-time OPT frame width")
    parser.add_argument("--min-height", type=int, default=None, help="Dynamic min height")
    parser.add_argument("--opt-height", type=int, default=None, help="Dynamic opt height")
    parser.add_argument("--max-height", type=int, default=None, help="Dynamic max height")
    parser.add_argument("--min-width", type=int, default=None, help="Dynamic min width")
    parser.add_argument("--opt-width", type=int, default=None, help="Dynamic opt width")
    parser.add_argument("--max-width", type=int, default=None, help="Dynamic max width")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 kernels")
    parser.add_argument(
        "--disable-truncate-long-double",
        action="store_true",
        help="Disable TensorRT truncate_long_and_double option (enabled by default).",
    )
    parser.add_argument("--atol", type=float, default=1e-3, help="PT vs TRT allclose atol")
    parser.add_argument("--rtol", type=float, default=1e-3, help="PT vs TRT allclose rtol")
    return parser.parse_args()


def validate_dynamic_args(args: argparse.Namespace) -> bool:
    h_vals = [args.min_height, args.opt_height, args.max_height]
    w_vals = [args.min_width, args.opt_width, args.max_width]
    h_any = any(v is not None for v in h_vals)
    w_any = any(v is not None for v in w_vals)

    if h_any != w_any:
        raise ValueError("Provide both height and width dynamic shape triplets together.")
    if not h_any and not w_any:
        return False
    if not all(v is not None for v in h_vals + w_vals):
        raise ValueError("Provide all min/opt/max values for dynamic height and width.")
    if not (args.min_height <= args.opt_height <= args.max_height):
        raise ValueError("Expected min_height <= opt_height <= max_height.")
    if not (args.min_width <= args.opt_width <= args.max_width):
        raise ValueError("Expected min_width <= opt_width <= max_width.")
    return True


def main() -> None:
    inference_root = Path(__file__).resolve().parents[1]
    default_checkpoint = resolve_default_checkpoint(inference_root, detr_version="ems")
    default_output = inference_root / "checkpoints" / "ems_finetuned_detr_trt.ts"
    args = parse_args(default_checkpoint=default_checkpoint, default_output=default_output)
    if args.detr_version == "base":
        if args.checkpoint == default_checkpoint:
            args.checkpoint = resolve_default_checkpoint(inference_root, detr_version="base")
        if args.output == default_output:
            args.output = inference_root / "checkpoints" / "detr_base_trt.ts"
    if args.ir == "dynamo" and args.output.suffix.lower() == ".ts":
        args.output = args.output.with_suffix(".ep")

    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT conversion requires CUDA.")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    dynamic_shapes = validate_dynamic_args(args)
    class_names = get_class_names(args.detr_version)
    num_classes = len(class_names)
    device = torch.device("cuda")

    print(f"[1/5] Loading DETR checkpoint: {args.checkpoint}")
    model = build_model(num_classes=num_classes, checkpoint_path=args.checkpoint, device=device)
    print("[info] Set DETR inference graph with aux_loss=False and decoder.return_intermediate=False")
    wrapped = DetrTensorOutputWrapper(model).eval().to(device)

    if dynamic_shapes:
        h = args.opt_height
        w = args.opt_width
    else:
        h = args.height
        w = args.width

    example_input = torch.randn(args.batch_size, 3, h, w, device=device, dtype=torch.float32)

    print("[2/5] Running PyTorch sanity inference")
    with torch.inference_mode():
        pt_logits, pt_boxes = wrapped(example_input)
    print(f"  - PT logits shape: {tuple(pt_logits.shape)}")
    print(f"  - PT boxes shape : {tuple(pt_boxes.shape)}")

    try:
        import torch_tensorrt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("torch_tensorrt is required for conversion.") from exc

    print("[3/5] Compiling TensorRT engine")
    if dynamic_shapes:
        trt_input = torch_tensorrt.Input(
            min_shape=(args.batch_size, 3, args.min_height, args.min_width),
            opt_shape=(args.batch_size, 3, args.opt_height, args.opt_width),
            max_shape=(args.batch_size, 3, args.max_height, args.max_width),
            dtype=torch.float32,
        )
    else:
        trt_input = torch_tensorrt.Input(shape=(args.batch_size, 3, args.height, args.width), dtype=torch.float32)

    enabled_precisions = {torch.float32}
    if args.fp16:
        enabled_precisions.add(torch.float16)

    attempted_irs = []
    trt_module = None
    used_ir = None

    candidate_irs = [args.ir] if args.ir in ("ts", "dynamo") else ["ts", "dynamo"]

    for candidate_ir in candidate_irs:
        attempted_irs.append(candidate_ir)
        try:
            if candidate_ir == "ts":
                traced = torch.jit.trace(wrapped, example_input, check_trace=False)
                traced = torch.jit.freeze(traced.eval())
                trt_module = torch_tensorrt.compile(
                    traced,
                    ir="ts",
                    inputs=[trt_input],
                    enabled_precisions=enabled_precisions,
                    truncate_long_and_double=not args.disable_truncate_long_double,
                )
            else:
                dynamo_kwargs = {
                    "ir": "dynamo",
                    "inputs": [trt_input],
                    "enabled_precisions": enabled_precisions,
                    "truncate_double": True,
                }
                try:
                    trt_module = torch_tensorrt.compile(
                        wrapped,
                        **dynamo_kwargs,
                    )
                except AssertionError as exc:
                    # Newer Torch-TensorRT builds may reject enabled_precisions when explicit typing is on.
                    if "enabled_precisions should not be used when use_explicit_typing=True" not in str(exc):
                        raise
                    print(
                        "[warn] TensorRT dynamo compile rejected enabled_precisions with explicit typing; "
                        "retrying without enabled_precisions."
                    )
                    dynamo_kwargs.pop("enabled_precisions", None)
                    trt_module = torch_tensorrt.compile(
                        wrapped,
                        **dynamo_kwargs,
                    )
            used_ir = candidate_ir
            break
        except RuntimeError as exc:
            print(f"[warn] TensorRT compile failed with ir={candidate_ir}: {exc}")
            if args.ir != "auto":
                raise
        except AssertionError as exc:
            print(f"[warn] TensorRT compile assertion failed with ir={candidate_ir}: {exc}")
            if args.ir != "auto":
                raise
        except TypeError as exc:
            print(f"[warn] TensorRT compile type mismatch with ir={candidate_ir}: {exc}")
            if args.ir != "auto":
                raise

    if trt_module is None or used_ir is None:
        raise RuntimeError(f"TensorRT compile failed for all attempted IRs: {attempted_irs}")
    print(f"  - Compile succeeded with ir={used_ir}")

    print("[4/5] Saving engine")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if used_ir == "dynamo":
        if isinstance(trt_module, torch.export.ExportedProgram):
            if args.output.suffix.lower() != ".ep":
                args.output = args.output.with_suffix(".ep")
            torch.export.save(trt_module, str(args.output))
        else:
            # Some torch_tensorrt versions return an FX GraphModule for dynamo.
            # Serialize as TorchScript fallback so runtime can load it consistently.
            if args.output.suffix.lower() != ".ts":
                args.output = args.output.with_suffix(".ts")
            if isinstance(trt_module, torch.fx.GraphModule):
                scripted = torch.jit.trace(trt_module.eval(), example_input, check_trace=False)
                scripted = torch.jit.freeze(scripted.eval())
                torch.jit.save(scripted, str(args.output))
            elif hasattr(trt_module, "module"):
                inner = trt_module.module()
                torch.jit.save(inner, str(args.output))
            else:
                scripted = torch.jit.trace(trt_module, example_input, check_trace=False)
                scripted = torch.jit.freeze(scripted.eval())
                torch.jit.save(scripted, str(args.output))
    else:
        if args.output.suffix.lower() != ".ts":
            args.output = args.output.with_suffix(".ts")
        torch.jit.save(trt_module, str(args.output))
    print(f"  - Saved TensorRT module: {args.output}")

    print("[5/5] Running TensorRT sanity inference")
    with torch.inference_mode():
        trt_logits, trt_boxes = run_once(trt_module, example_input)

    if isinstance(trt_logits, (list, tuple)):
        trt_logits = trt_logits[0]
    if isinstance(trt_boxes, (list, tuple)):
        trt_boxes = trt_boxes[0]

    logits_close = torch.allclose(pt_logits, trt_logits, atol=args.atol, rtol=args.rtol)
    boxes_close = torch.allclose(pt_boxes, trt_boxes, atol=args.atol, rtol=args.rtol)
    logits_max_diff = (pt_logits - trt_logits).abs().max().item()
    boxes_max_diff = (pt_boxes - trt_boxes).abs().max().item()

    print(f"  - TRT logits shape: {tuple(trt_logits.shape)}")
    print(f"  - TRT boxes shape : {tuple(trt_boxes.shape)}")
    print(f"  - logits max_abs_diff: {logits_max_diff:.6f}")
    print(f"  - boxes  max_abs_diff: {boxes_max_diff:.6f}")
    print(f"  - logits allclose(atol={args.atol}, rtol={args.rtol}): {logits_close}")
    print(f"  - boxes  allclose(atol={args.atol}, rtol={args.rtol}): {boxes_close}")
    if not (logits_close and boxes_close):
        print("[warn] PT and TRT outputs are outside tolerance. Validate with real inputs before deployment.")


if __name__ == "__main__":
    main()
