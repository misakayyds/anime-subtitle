"""
AnimeTranslator 日志模块

提供统一的日志记录功能，支持:
- 控制台彩色输出
- 文件持久化 (Output/logs/)
- 保留原有 emoji 风格
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import OUTPUT_DIR


_LOG_DIR = OUTPUT_DIR / "logs"
_LOG_DIR.mkdir(parents=True, exist_ok=True)

_logger: Optional[logging.Logger] = None
_file_handler: Optional[logging.FileHandler] = None


class DynamicStreamHandler(logging.StreamHandler):
    """动态获取 sys.stdout 的 StreamHandler，支持 stdout 被替换的场景"""

    def __init__(self):
        super().__init__()

    @property
    def stream(self):
        return sys.stdout

    @stream.setter
    def stream(self, value):
        pass


class EmojiFormatter(logging.Formatter):
    """带时间戳的日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"[{timestamp}] {record.getMessage()}"


def setup_logger(level: int = logging.INFO) -> logging.Logger:
    """初始化全局日志器"""
    global _logger, _file_handler

    if _logger is not None:
        return _logger

    _logger = logging.getLogger("animetranslator")
    _logger.setLevel(level)

    console_handler = DynamicStreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(EmojiFormatter())
    _logger.addHandler(console_handler)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = _LOG_DIR / f"animetranslator_{timestamp}.log"
    _file_handler = logging.FileHandler(log_file, encoding="utf-8")
    _file_handler.setLevel(level)
    _file_handler.setFormatter(EmojiFormatter())
    _logger.addHandler(_file_handler)

    return _logger


def get_logger(name: str = "animetranslator") -> logging.Logger:
    """获取日志器实例"""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger.getChild(name) if name != "animetranslator" else _logger


def log_info(message: str):
    """记录 INFO 级别日志"""
    get_logger().info(message)


def log_warning(message: str):
    """记录 WARNING 级别日志"""
    get_logger().warning(message)


def log_error(message: str):
    """记录 ERROR 级别日志"""
    get_logger().error(message)


def log_debug(message: str):
    """记录 DEBUG 级别日志"""
    get_logger().debug(message)
