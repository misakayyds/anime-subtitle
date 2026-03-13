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