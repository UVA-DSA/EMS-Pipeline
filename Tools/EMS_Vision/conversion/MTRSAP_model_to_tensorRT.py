import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, Iterable

import torch

INFERENCE_ROOT = Path(__file__).resolve().parents[1]
if str(INFERENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(INFERENCE_ROOT))

from models.mtrsap_model import TransformerInferenceModel


def load_checkpoint_compat(path: Path) -> Dict:
    try:
        return torch.load(path, map_location="cpu")
    except (pickle.UnpicklingError, RuntimeError) as exc:
        msg = str(exc)
        if "Weights only load failed" in msg or "Unsupported global" in msg:
            return torch.load(path, map_location="cpu", weights_only=False)
        raise


def load_state_dict(checkpoint_path: Path) -> Dict[str, torch.Tensor]:
    ckpt = load_checkpoint_compat(checkpoint_path)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    if any(key.startswith("module.") for key in state.keys()):
        state = {key.replace("module.", "", 1): value for key, value in state.items()}

    return state


def extract_num_layers(keys: Iterable[str]) -> int:
    layer_indices = []
    pattern = re.compile(r"^transformer\.layers\.(\d+)\.")
    for key in keys:
        match = pattern.match(key)
        if match:
            layer_indices.append(int(match.group(1)))
    if not layer_indices:
        raise KeyError("Could not infer transformer layer count from checkpoint.")
    return max(layer_indices) + 1


def infer_model_config(state: Dict[str, torch.Tensor]) -> Dict[str, int]:
    required_keys = [
        "encoder.encoder.0.weight",
        "encoder.encoder.6.weight",
        "out.weight",
    ]
    for key in required_keys:
        if key not in state:
            raise KeyError(f"Missing key {key} in checkpoint.")

    conv0_shape = tuple(state["encoder.encoder.0.weight"].shape)
    conv2_shape = tuple(state["encoder.encoder.6.weight"].shape)
    out_shape = tuple(state["out.weight"].shape)

    input_dim = conv0_shape[1]
    kernel_size = conv0_shape[2]
    d_model = conv2_shape[0]
    output_dim = out_shape[0]
    num_layers = extract_num_layers(state.keys())
    max_len = int(state["pe.pe"].shape[0]) if "pe.pe" in state else 5000

    if out_shape[1] != d_model:
        raise ValueError(
            f"Checkpoint mismatch: out.weight second dim ({out_shape[1]}) != inferred d_model ({d_model})."
        )

    return {
        "input_dim": input_dim,
        "kernel_size": kernel_size,
        "d_model": d_model,
        "output_dim": output_dim,
        "num_layers": num_layers,
        "max_len": max_len,
    }


def run_once(module, x: torch.Tensor):
    if callable(module):
        return module(x)
    if hasattr(module, "module"):
        inner = module.module()
        return inner(x)
    raise RuntimeError(f"Unsupported compiled module type: {type(module)}")


def build_arg_parser(default_checkpoint: Path, default_output: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert EgoEMS MTRSAP checkpoint to TensorRT TorchScript engine")
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint, help="Path to MTRSAP checkpoint (.pt)")
    parser.add_argument("--output", type=Path, default=default_output, help="Path to save TensorRT TorchScript engine (.ts)")

    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for tracing and validation")
    parser.add_argument("--seq-len", type=int, default=30, help="Static sequence length for tracing and validation")
    parser.add_argument("--min-seq-len", type=int, default=None, help="Min seq len for dynamic TensorRT shape")
    parser.add_argument("--opt-seq-len", type=int, default=None, help="Opt seq len for dynamic TensorRT shape")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Max seq len for dynamic TensorRT shape")

    parser.add_argument("--nhead", type=int, default=4, help="Transformer attention heads (must divide d_model)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used in positional encoding and transformer")
    parser.add_argument("--batch-first", action="store_true", default=True, help="Use batch-first transformer layout")
    parser.add_argument("--disable-batch-first", action="store_false", dest="batch_first", help="Disable batch-first layout")

    parser.add_argument("--fp16", action="store_true", help="Enable FP16 kernels in TensorRT compilation")
    parser.add_argument("--atol", type=float, default=1e-3, help="Absolute tolerance for PT vs TRT output check")
    parser.add_argument("--rtol", type=float, default=1e-3, help="Relative tolerance for PT vs TRT output check")
    return parser


def validate_dynamic_shape_args(args: argparse.Namespace) -> bool:
    values = [args.min_seq_len, args.opt_seq_len, args.max_seq_len]
    provided = [value is not None for value in values]
    if any(provided) and not all(provided):
        raise ValueError("Provide all of --min-seq-len, --opt-seq-len, --max-seq-len, or none of them.")
    if all(provided):
        if not (args.min_seq_len <= args.opt_seq_len <= args.max_seq_len):
            raise ValueError("Expected min_seq_len <= opt_seq_len <= max_seq_len.")
        return True
    return False


def main() -> None:
    default_checkpoint = INFERENCE_ROOT / "checkpoints" / "mtrsap_30frames_window_resnet.pt"
    default_output = INFERENCE_ROOT / "checkpoints" / "mtrsap_30frames_window_resnet_trt.ts"
    args = build_arg_parser(default_checkpoint=default_checkpoint, default_output=default_output).parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT conversion requires a CUDA-enabled PyTorch environment.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.seq_len <= 0:
        raise ValueError("--seq-len must be > 0")

    dynamic_shapes = validate_dynamic_shape_args(args)

    print(f"[1/5] Loading checkpoint: {args.checkpoint}")
    state = load_state_dict(args.checkpoint)
    inferred = infer_model_config(state)

    print("[2/5] Inferred checkpoint config:")
    for key in ("input_dim", "kernel_size", "d_model", "output_dim", "num_layers", "max_len"):
        print(f"  - {key}: {inferred[key]}")

    if inferred["d_model"] % args.nhead != 0:
        raise ValueError(
            f"d_model={inferred['d_model']} is not divisible by nhead={args.nhead}. "
            "Pass the correct --nhead used in training."
        )

    model = TransformerInferenceModel(
        input_dim=inferred["input_dim"],
        d_model=inferred["d_model"],
        output_dim=inferred["output_dim"],
        kernel_size=inferred["kernel_size"],
        nhead=args.nhead,
        num_layers=inferred["num_layers"],
        dropout=args.dropout,
        max_len=inferred["max_len"],
        batch_first=args.batch_first,
    ).eval().cuda()

    load_result = model.load_state_dict(state, strict=False)
    if load_result.missing_keys:
        print(f"[warn] Missing keys while loading: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        print(f"[warn] Unexpected keys while loading: {load_result.unexpected_keys}")

    seq_len = args.opt_seq_len if dynamic_shapes else args.seq_len
    example_input = torch.randn(args.batch_size, seq_len, inferred["input_dim"], device="cuda", dtype=torch.float32)

    print("[3/5] Running PyTorch sanity inference")
    with torch.inference_mode():
        pt_output = model(example_input)
    print(f"  - PT output shape: {tuple(pt_output.shape)}")

    try:
        import torch_tensorrt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch_tensorrt is not installed. Install it in your runtime to enable conversion."
        ) from exc

    print("[4/5] Compiling TensorRT engine")
    traced_model = torch.jit.trace(model, example_input, check_trace=False)
    traced_model = torch.jit.freeze(traced_model.eval())

    if dynamic_shapes:
        trt_input = torch_tensorrt.Input(
            min_shape=(args.batch_size, args.min_seq_len, inferred["input_dim"]),
            opt_shape=(args.batch_size, args.opt_seq_len, inferred["input_dim"]),
            max_shape=(args.batch_size, args.max_seq_len, inferred["input_dim"]),
            dtype=torch.float32,
        )
    else:
        trt_input = torch_tensorrt.Input(
            shape=(args.batch_size, args.seq_len, inferred["input_dim"]),
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

    print("[5/5] Running TensorRT sanity inference")
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
    print("  - expected input shape: [B, 30, 2048] for the default 30-frame activity window")

    if not is_close:
        print("[warn] PT and TRT outputs are outside tolerance. Validate with real inputs before deployment.")


if __name__ == "__main__":
    main()
