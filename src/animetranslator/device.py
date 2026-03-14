"""
AnimeTranslator 设备管理模块

支持设备优先级: CUDA/ROCm > MPS > CPU
自动检测可用设备并根据设备特性配置参数

注意: AMD ROCm 使用 CUDA 兼容 API，会被检测为 CUDA 设备
"""

import gc
from enum import Enum

import torch


class DeviceType(Enum):
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    CPU = "cpu"


def is_rocm() -> bool:
    """检测是否为 ROCm 环境"""
    return hasattr(torch.version, "hip") and torch.version.hip is not None


def detect_device() -> DeviceType:
    """自动检测可用设备，优先级: CUDA/ROCm > MPS > CPU"""
    if torch.cuda.is_available():
        if is_rocm():
            return DeviceType.ROCM
        return DeviceType.CUDA
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return DeviceType.MPS
    return DeviceType.CPU


def get_device_type() -> DeviceType:
    """获取设备类型，优先读取环境变量，否则自动检测"""
    from .config import get_env

    env_device = get_env("DEVICE", "auto") or "auto"
    env_device = env_device.lower().strip()

    if env_device == "cuda":
        if torch.cuda.is_available():
            if is_rocm():
                return DeviceType.ROCM
            return DeviceType.CUDA
        else:
            from .logger import log_warning

            log_warning("⚠️ DEVICE=cuda 但 CUDA 不可用，回退到自动检测")
            return detect_device()
    elif env_device == "rocm":
        if torch.cuda.is_available() and is_rocm():
            return DeviceType.ROCM
        else:
            from .logger import log_warning

            log_warning("⚠️ DEVICE=rocm 但 ROCm 不可用，回退到自动检测")
            return detect_device()
    elif env_device == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return DeviceType.MPS
        else:
            from .logger import log_warning

            log_warning("⚠️ DEVICE=mps 但 MPS 不可用，回退到自动检测")
            return detect_device()
    elif env_device == "cpu":
        return DeviceType.CPU
    else:
        return detect_device()


def get_device_string() -> str:
    """获取 PyTorch 设备字符串"""
    device = get_device_type()
    if device == DeviceType.ROCM:
        return "cuda"
    return device.value


def get_compute_type() -> str:
    """根据设备类型返回计算精度类型"""
    device = get_device_type()

    if device in (DeviceType.CUDA, DeviceType.ROCM):
        return "float16"
    elif device == DeviceType.MPS:
        return "float16"
    else:
        return "float32"


def get_recommended_whisper_model() -> str:
    """根据设备类型和显存大小返回推荐的 Whisper 模型"""
    from .config import get_env

    env_model = get_env("WHISPER_MODEL", "") or ""
    if env_model.strip():
        return env_model.strip()

    device = get_device_type()

    if device in (DeviceType.CUDA, DeviceType.ROCM):
        vram_gb = get_cuda_vram_gb()
        if vram_gb >= 10:
            return "large-v3"
        elif vram_gb >= 6:
            return "medium"
        else:
            return "small"
    elif device == DeviceType.MPS:
        return "medium"
    else:
        return "small"


def get_cuda_vram_gb() -> float:
    """获取 CUDA 设备显存大小（GB）"""
    if not torch.cuda.is_available():
        return 0.0

    try:
        props = torch.cuda.get_device_properties(0)
        return props.total_memory / (1024**3)
    except Exception:
        return 0.0


def get_device_info() -> str:
    """获取设备信息字符串，用于日志输出"""
    device = get_device_type()

    if device == DeviceType.ROCM:
        vram_gb = get_cuda_vram_gb()
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        return f"ROCm ({gpu_name}, {vram_gb:.1f}GB VRAM)"
    elif device == DeviceType.CUDA:
        vram_gb = get_cuda_vram_gb()
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        return f"CUDA ({gpu_name}, {vram_gb:.1f}GB VRAM)"
    elif device == DeviceType.MPS:
        return "MPS (Apple Silicon)"
    else:
        return "CPU (无GPU加速，速度较慢)"


def clear_device_cache():
    """清理设备缓存"""
    device = get_device_type()

    gc.collect()

    if device in (DeviceType.CUDA, DeviceType.ROCM):
        torch.cuda.empty_cache()
    elif device == DeviceType.MPS:
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


def print_device_info():
    """打印设备信息到控制台"""
    from .logger import log_info, log_warning

    device = get_device_type()
    info = get_device_info()

    log_info(f"🖥️ 设备类型: {info}")

    if device in (DeviceType.CUDA, DeviceType.ROCM):
        whisper_model = get_recommended_whisper_model()
        compute_type = get_compute_type()
        log_info(f"🎯 Whisper 模型: {whisper_model}")
        log_info(f"⚙️ 计算精度: {compute_type}")
    elif device == DeviceType.MPS:
        whisper_model = get_recommended_whisper_model()
        compute_type = get_compute_type()
        log_info(f"🎯 Whisper 模型: {whisper_model}")
        log_info(f"⚙️ 计算精度: {compute_type}")
    else:
        log_warning("⚠️ 警告: 使用 CPU 模式，处理速度将非常慢！")
        log_info("💡 提示: 如有 NVIDIA GPU，请安装 CUDA 版 PyTorch")
        log_info("💡 提示: 如有 AMD GPU，请安装 ROCm 版 PyTorch (仅 Linux)")
        log_info("💡 提示: 如有 Apple Silicon Mac，MPS 将自动启用")
        whisper_model = get_recommended_whisper_model()
        log_info(f"🎯 Whisper 模型: {whisper_model} (轻量化)")
