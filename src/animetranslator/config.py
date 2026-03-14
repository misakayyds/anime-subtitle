"""
AnimeTranslator 核心路径配置

路径检测策略:
1. 如果是开发模式安装（pip install -e .），从包位置向上找到项目根目录
2. 否则从当前工作目录向上查找包含 pyproject.toml 的目录
3. 回退到当前工作目录

用户应在项目根目录创建 env/ 虚拟环境，并在项目根目录运行命令。
"""

import os
import sys
from pathlib import Path


def find_project_root() -> Path:
    """找到项目根目录（包含 pyproject.toml 的目录）"""
    pkg_dir = Path(__file__).resolve().parent
    src_dir = pkg_dir.parent

    if src_dir.name == "src":
        project_root = src_dir.parent
        if (project_root / "pyproject.toml").exists():
            return project_root

    current = Path.cwd()
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        if current == current.parent:
            break
        current = current.parent

    return Path.cwd()


PROJECT_ROOT = find_project_root()
INPUT_DIR = PROJECT_ROOT / "Input"
OUTPUT_DIR = PROJECT_ROOT / "Output"
ENV_FILE = PROJECT_ROOT / ".env"

SUPPORTED_VIDEO_EXTENSIONS = (".mkv", ".mp4", ".avi", ".mov", ".flv", ".wmv", ".webm")


def ensure_dirs():
    """确保必要目录存在"""
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_env():
    """加载 .env 文件"""
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE, override=True)


def get_env(key: str, default=None):
    """从环境变量获取配置值"""
    load_env()
    return os.environ.get(key, default)


def get_env_float(key: str, default: float) -> float:
    """从环境变量获取浮点数"""
    try:
        return float(get_env(key, default))
    except (TypeError, ValueError):
        return default


def get_env_int(key: str, default: int) -> int:
    """从环境变量获取整数"""
    try:
        return int(get_env(key, default))
    except (TypeError, ValueError):
        return default


def validate_config() -> list:
    """验证配置项，返回警告信息列表"""
    warnings = []

    api_key = get_env("DEEPSEEK_API_KEY", "")
    if not api_key or api_key.strip() == "":
        warnings.append("❌ DEEPSEEK_API_KEY 未配置，翻译功能将无法使用")
    elif not api_key.startswith("sk-"):
        warnings.append("⚠️ DEEPSEEK_API_KEY 格式可能不正确（通常以 'sk-' 开头）")

    device = get_env("DEVICE", "auto") or "auto"
    valid_devices = ("auto", "cuda", "rocm", "mps", "cpu")
    if device.lower() not in valid_devices:
        warnings.append(f"⚠️ DEVICE='{device}' 无效，应为: {', '.join(valid_devices)}")

    whisper_model = get_env("WHISPER_MODEL", "") or ""
    valid_models = ("", "large-v3", "medium", "small")
    if whisper_model and whisper_model not in valid_models:
        warnings.append(
            f"⚠️ WHISPER_MODEL='{whisper_model}' 无效，应为: large-v3, medium, small 或留空"
        )

    max_workers = get_env_int("MAX_API_WORKERS", 3)
    if max_workers < 1 or max_workers > 10:
        warnings.append(f"⚠️ MAX_API_WORKERS={max_workers} 超出建议范围 (1-10)")

    batch_size = get_env_int("ALIGNMENT_BATCH_SIZE", 3)
    if batch_size < 1 or batch_size > 20:
        warnings.append(f"⚠️ ALIGNMENT_BATCH_SIZE={batch_size} 超出建议范围 (1-20)")

    nsp_threshold = get_env_float("NO_SPEECH_PROB_THRESHOLD", 0.7)
    if nsp_threshold < 0.0 or nsp_threshold > 1.0:
        warnings.append(f"⚠️ NO_SPEECH_PROB_THRESHOLD={nsp_threshold} 超出有效范围 (0.0-1.0)")

    cr_threshold = get_env_float("COMPRESSION_RATIO_THRESHOLD", 2.8)
    if cr_threshold < 1.0 or cr_threshold > 10.0:
        warnings.append(f"⚠️ COMPRESSION_RATIO_THRESHOLD={cr_threshold} 超出建议范围 (1.0-10.0)")

    return warnings
