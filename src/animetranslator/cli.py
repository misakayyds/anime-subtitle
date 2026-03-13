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


def cmd_webui(args):
    """启动 WebUI"""
    from animetranslator.webui import run_webui
    run_webui(port=args.port, share=args.share)


def cmd_watch(args):
    """启动看门狗监听"""
    from animetranslator.watcher import run_watcher
    run_watcher(shutdown_on_complete=args.shutdown)


def main():
    """CLI 主入口"""
    parser = argparse.ArgumentParser(
        prog="animetranslator",
        description="动漫智能机翻/校对工具 - 音韵炼金术四阶段管线"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    webui_parser = subparsers.add_parser("webui", help="启动 WebUI 界面")
    webui_parser.add_argument("--port", type=int, default=7860, help="WebUI 端口 (默认: 7860)")
    webui_parser.add_argument("--share", action="store_true", help="生成公网分享链接")
    webui_parser.set_defaults(func=cmd_webui)

    watch_parser = subparsers.add_parser("watch", help="启动后台看门狗监听")
    watch_parser.add_argument("--shutdown", action="store_true", help="处理完成后自动关机")
    watch_parser.set_defaults(func=cmd_watch)

    args = parser.parse_args()
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())