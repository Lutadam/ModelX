"""Choosing the compute device and fixing the random seeds for the deep learning model."""
import random
import warnings

import numpy as np
import torch

import config


def get_device(require_gpu: bool = config.REQUIRE_GPU) -> torch.device:
    """Return the GPU if PyTorch can use one.

    Args:
        require_gpu: True raises an error when there is no GPU. False falls
            back to the CPU with a warning (training is much slower).

    Returns:
        A CUDA (NVIDIA) or MPS (Apple) device, or the CPU.
    """
    apply_gpu_settings()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.version.cuda is None:
        reason = "this PyTorch is a CPU-only build. Install the CUDA build (see README, section GPU)."
    else:
        reason = "PyTorch has a CUDA build but found no GPU. Check the NVIDIA driver."
    if require_gpu:
        raise RuntimeError(f"No GPU available: {reason} Set REQUIRE_GPU = False in config.py to run on the CPU anyway.")
    warnings.warn(f"No GPU available, using the CPU: {reason}")
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    """One line saying what the model will run on."""
    if device.type == "cuda":
        cudnn = "on" if torch.backends.cudnn.enabled else "off (config.DEEP_LEARNING_USE_CUDNN)"
        return f"Deep learning runs on GPU: {torch.cuda.get_device_name(device)}, cuDNN {cudnn}"
    if device.type == "mps":
        return "Deep learning runs on the Apple GPU (MPS)"
    return "Deep learning runs on the CPU (slow)"


def apply_gpu_settings() -> None:
    """cuDNN on or off (config.DEEP_LEARNING_USE_CUDNN), and repeatable GPU results."""
    torch.backends.cudnn.enabled = config.DEEP_LEARNING_USE_CUDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_random_seeds(seed: int = config.RANDOM_SEED) -> None:
    """Fix every random source so two runs give the same numbers. Called before every training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    apply_gpu_settings()
