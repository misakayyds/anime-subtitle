"""
AnimeTranslator 国际化模块 (i18n)

支持:
- 环境变量检测 (LANG/LC_ALL)
- .env 配置文件
- 默认中文
- WebUI 动态切换
"""

import json
import os
from pathlib import Path
from typing import Dict, Optional

_LOCALES_DIR = Path(__file__).parent / "locales"
_CURRENT_LOCALE: str = "zh_CN"
_TRANSLATIONS: Dict[str, str] = {}


def _detect_locale() -> str:
    """检测系统语言设置"""
    env_lang = os.environ.get("LANG", "") or os.environ.get("LC_ALL", "")
    env_lang = env_lang.lower()

    if "zh" in env_lang:
        return "zh_CN"
    elif "en" in env_lang:
        return "en_US"

    try:
        from dotenv import load_dotenv

        load_dotenv()
        config_lang = os.environ.get("ANIME_TRANSLATOR_LANG", "")
        if config_lang in ("zh_CN", "en_US"):
            return config_lang
    except ImportError:
        pass

    return "zh_CN"


def _load_translations(locale: str) -> Dict[str, str]:
    """加载翻译字典"""
    locale_file = _LOCALES_DIR / f"{locale}.json"
    if not locale_file.exists():
        return {}

    with open(locale_file, "r", encoding="utf-8") as f:
        return json.load(f)


def init_i18n(locale: Optional[str] = None) -> None:
    """初始化国际化模块"""
    global _CURRENT_LOCALE, _TRANSLATIONS

    if locale:
        _CURRENT_LOCALE = locale
    else:
        _CURRENT_LOCALE = _detect_locale()

    _TRANSLATIONS = _load_translations(_CURRENT_LOCALE)


def get_locale() -> str:
    """获取当前语言"""
    return _CURRENT_LOCALE


def set_locale(locale: str) -> None:
    """设置语言（用于WebUI动态切换）"""
    global _CURRENT_LOCALE, _TRANSLATIONS

    if locale not in ("zh_CN", "en_US"):
        return

    _CURRENT_LOCALE = locale
    _TRANSLATIONS = _load_translations(locale)


def tr(key: str, **kwargs) -> str:
    """
    翻译函数

    Args:
        key: 翻译键
        **kwargs: 格式化参数

    Returns:
        翻译后的文本
    """
    global _TRANSLATIONS
    if not _TRANSLATIONS:
        init_i18n()

    text = _TRANSLATIONS.get(key, key)

    if kwargs:
        try:
            return text.format(**kwargs)
        except KeyError:
            return text

    return text


def get_available_locales() -> list:
    """获取可用语言列表"""
    return [
        {"code": "zh_CN", "name": "简体中文"},
        {"code": "en_US", "name": "English"},
    ]


init_i18n()
