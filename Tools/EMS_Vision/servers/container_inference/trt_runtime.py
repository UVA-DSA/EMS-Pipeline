import ctypes
import os
import site
from pathlib import Path
from typing import List

import torch


def torch_tensorrt_runtime_dirs() -> List[Path]:
    dirs: List[Path] = []

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        dirs.append(Path(conda_prefix) / "lib")

    dirs.append(Path(torch.__file__).resolve().parent / "lib")

    for site_dir in site.getsitepackages():
        root = Path(site_dir)
        dirs.extend(
            [
                root / "tensorrt_libs",
                root / "nvidia" / "cuda_runtime" / "lib",
                root / "nvidia" / "cu13" / "lib",
                root / "nvidia" / "cu12" / "lib",
            ]
        )

    unique_dirs: List[Path] = []
    seen = set()
    for path in dirs:
        if path.exists() and path not in seen:
            unique_dirs.append(path)
            seen.add(path)
    return unique_dirs


def preload_torch_tensorrt_runtime_libs() -> List[Path]:
    loaded: List[Path] = []
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for lib_name in ("libcudart.so.13", "libnvinfer.so.10", "libnvinfer_plugin.so.10"):
        for directory in torch_tensorrt_runtime_dirs():
            candidate = directory / lib_name
            if not candidate.exists():
                continue
            try:
                ctypes.CDLL(str(candidate), mode=mode)
                loaded.append(candidate)
                break
            except OSError:
                continue
    return loaded


def ensure_torch_tensorrt_runtime() -> None:
    try:
        import torch_tensorrt  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "torch_tensorrt is required to load this TensorRT TorchScript engine."
        ) from exc
    except OSError as exc:
        loaded_libs = preload_torch_tensorrt_runtime_libs()
        if loaded_libs:
            try:
                import torch_tensorrt  # noqa: F401
                return
            except OSError as retry_exc:
                exc = retry_exc

        runtime_dirs = torch_tensorrt_runtime_dirs()
        runtime_path_hint = ":".join(str(path) for path in runtime_dirs)
        raise RuntimeError(
            "torch_tensorrt failed to load its shared libraries.\n"
            f"Original error: {exc}\n"
            f"Detected torch CUDA build: {torch.version.cuda or 'unknown'}\n"
            "If you are running from WSL/conda, export these runtime directories before launching Python:\n"
            f"export LD_LIBRARY_PATH=\"{runtime_path_hint}:$LD_LIBRARY_PATH\"\n"
            "If that still fails, reinstall a torch_tensorrt build that matches your current torch/CUDA versions."
        ) from exc


def load_trt_engine(engine_path: Path, device: torch.device):
    if engine_path.suffix.lower() == ".ep":
        exported_program = torch.export.load(str(engine_path))
        module = exported_program.module().eval()
        if hasattr(module, "to"):
            module = module.to(device)
        return module
    return torch.jit.load(str(engine_path), map_location=device).eval()


def run_model_once(module, model_input: torch.Tensor):
    if callable(module):
        return module(model_input)
    if hasattr(module, "forward"):
        return module.forward(model_input)
    if hasattr(module, "module"):
        inner = module.module()
        return inner(model_input)
    raise RuntimeError(f"Unsupported TensorRT module type: {type(module)}")
