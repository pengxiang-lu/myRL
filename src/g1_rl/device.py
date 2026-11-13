"""Explicit device selection; requested CUDA must never silently fall back."""
import torch


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(requested)
    except (RuntimeError, ValueError) as error:
        raise ValueError("device must be cpu, cuda, cuda:N or auto") from error
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device must be cpu, cuda, cuda:N or auto")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(
                f"CUDA requested but unavailable (torch={torch.__version__}, CUDA build={torch.version.cuda}). "
                "Run uv sync and check nvidia-smi / GPU access permissions. "
                "Use --device cpu only if CPU execution is intended."
            )
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} does not exist; visible count={torch.cuda.device_count()}")
        device = torch.device(f"cuda:{index}")
        # Fail early on driver/runtime problems before creating output directories/workers.
        torch.zeros(1, device=device).add_(1)
        torch.cuda.synchronize(device)
    return device


def device_details(device: torch.device) -> dict:
    result = {"device": str(device), "torch": torch.__version__, "cuda_build": torch.version.cuda}
    if device.type == "cuda":
        result["gpu_name"] = torch.cuda.get_device_name(device)
        result["total_memory_mib"] = torch.cuda.get_device_properties(device).total_memory / 2**20
    return result


def memory_stats(device: torch.device) -> dict:
    if device.type != "cuda":
        return {"device": str(device)}
    torch.cuda.synchronize(device)
    return {"device": str(device),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20}
