"""
AnimeTranslator CLI 命令行入口

使用方式:
    animetranslator webui
    animetranslator watch
    animetranslator watch --shutdown
    python -m animetranslator webui
"""

import argparse
import sys

from .i18n import tr


def _run_config_validation():
    """运行配置验证并打印警告"""
    from animetranslator.config import validate_config
    from animetranslator.logger import setup_logger, log_warning

    setup_logger()
    warnings = validate_config()
    if warnings:
        log_warning("=" * 50)
        log_warning(tr("config.warning.title"))
        for w in warnings:
            log_warning(f"  {w}")
        log_warning("=" * 50)


def cmd_webui(args):
    """启动 WebUI"""
    _run_config_validation()
    from animetranslator.webui import run_webui

    run_webui(port=args.port, share=args.share)


def cmd_watch(args):
    """启动看门狗监听"""
    _run_config_validation()
    from animetranslator.watcher import run_watcher

    run_watcher(shutdown_on_complete=args.shutdown)


def main():
    """CLI 主入口"""
    parser = argparse.ArgumentParser(prog="animetranslator", description=tr("cli.prog"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    webui_parser = subparsers.add_parser("webui", help=tr("cli.webui.help"))
    webui_parser.add_argument("--port", type=int, default=7860, help=tr("cli.webui.port"))
    webui_parser.add_argument("--share", action="store_true", help=tr("cli.webui.share"))
    webui_parser.set_defaults(func=cmd_webui)

    watch_parser = subparsers.add_parser("watch", help=tr("cli.watch.help"))
    watch_parser.add_argument("--shutdown", action="store_true", help=tr("cli.watch.shutdown"))
    watch_parser.set_defaults(func=cmd_watch)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
