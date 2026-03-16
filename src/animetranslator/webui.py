"""
AnimeTranslator WebUI — Gradio 前端

独立于 watcher，共享 AlignmentEngine + translation 后端引擎
"""

import os
import sys
import io
import shutil
import threading
import queue
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import gradio as gr
from dotenv import load_dotenv, set_key

from .config import (
    INPUT_DIR,
    OUTPUT_DIR,
    ENV_FILE,
    ensure_dirs,
    load_env,
    get_env,
    get_env_int,
    get_env_float,
    SUPPORTED_VIDEO_EXTENSIONS,
)
from .alignment import AlignmentEngine
from .translation import run_translation
from .i18n import tr, set_locale, get_locale

_log_queue = queue.Queue()
_cancel_flag = threading.Event()
_is_running = threading.Event()
_engine = None


class LogCapture(io.TextIOBase):
    """日志捕获器：拦截 print() 输出重定向到 Gradio 日志框"""

    def __init__(self, log_queue, original_stdout):
        super().__init__()
        self.log_queue = log_queue
        self._stdout = original_stdout

    def write(self, text):
        if self._stdout:
            self._stdout.write(text)
            self._stdout.flush()
        if text and text.strip():
            self.log_queue.put(text.rstrip("\n"))
        return len(text) if text else 0

    def flush(self):
        if self._stdout:
            self._stdout.flush()

    def fileno(self):
        return self._stdout.fileno()


def load_env_values():
    """从 .env 文件读取当前配置"""
    load_env()
    return {
        "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
        "max_workers": get_env_int("MAX_API_WORKERS", 3),
        "batch_size": get_env_int("ALIGNMENT_BATCH_SIZE", 3),
        "nsp_threshold": get_env_float("NO_SPEECH_PROB_THRESHOLD", 0.7),
        "cr_threshold": get_env_float("COMPRESSION_RATIO_THRESHOLD", 2.8),
    }


def save_env_values(api_key, max_workers, batch_size, nsp_threshold, cr_threshold):
    """保存配置到 .env 文件并热更新进程环境变量"""
    env_path = str(ENV_FILE)

    set_key(env_path, "DEEPSEEK_API_KEY", api_key)
    set_key(env_path, "MAX_API_WORKERS", str(int(max_workers)))
    set_key(env_path, "ALIGNMENT_BATCH_SIZE", str(int(batch_size)))
    set_key(env_path, "NO_SPEECH_PROB_THRESHOLD", str(round(nsp_threshold, 2)))
    set_key(env_path, "COMPRESSION_RATIO_THRESHOLD", str(round(cr_threshold, 2)))

    os.environ["DEEPSEEK_API_KEY"] = api_key
    os.environ["MAX_API_WORKERS"] = str(int(max_workers))
    os.environ["ALIGNMENT_BATCH_SIZE"] = str(int(batch_size))
    os.environ["NO_SPEECH_PROB_THRESHOLD"] = str(round(nsp_threshold, 2))
    os.environ["COMPRESSION_RATIO_THRESHOLD"] = str(round(cr_threshold, 2))

    return "✅ " + tr("webui.config.saved")


def scan_files(input_dir, output_dir):
    """扫描输入输出目录，返回文件状态列表"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    rows = []

    if not input_path.exists():
        return [["⚠️ " + tr("webui.status.input_not_found"), "", ""]]

    for root, _, files in os.walk(input_path):
        for f in sorted(files):
            if any(f.lower().endswith(ext) for ext in SUPPORTED_VIDEO_EXTENSIONS):
                video_path = Path(root) / f
                rel = video_path.relative_to(input_path)
                ass_path = output_path / rel.with_suffix(".ass")
                json_path = output_path / rel.with_suffix("").with_name(
                    f"{video_path.stem}_alignment.json"
                )

                if ass_path.exists():
                    status = "✅ " + tr("webui.status.completed")
                elif json_path.exists():
                    status = "📝 " + tr("webui.status.draft_ready")
                else:
                    status = "⏳ " + tr("webui.status.pending")

                rows.append([str(rel), status, str(ass_path) if ass_path.exists() else ""])

    if not rows:
        return [[tr("webui.status.no_files"), "", ""]]
    return rows


def get_output_files(output_dir):
    """获取输出目录中的 .ass 文件供下载"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return []
    files = sorted(output_path.rglob("*.ass"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(f) for f in files[:20]]


def handle_upload(files, input_dir):
    """将上传的文件存入输入目录"""
    input_path = Path(input_dir)
    input_path.mkdir(parents=True, exist_ok=True)

    moved = []
    for f in files:
        src = Path(f)
        dst = input_path / src.name
        shutil.copy2(str(src), str(dst))
        moved.append(src.name)

    return "✅ " + tr("webui.status.uploaded", count=len(moved), dir=input_dir)


def run_pipeline(input_dir, output_dir):
    """后台线程：执行完整的 对齐+翻译 管线"""
    global _engine
    original_stdout = sys.__stdout__

    sys.stdout = LogCapture(_log_queue, original_stdout)

    try:
        _log_queue.put("=" * 50)
        _log_queue.put("🚀 " + tr("pipeline.start", time=datetime.now().strftime("%H:%M:%S")))
        _log_queue.put("📥 " + tr("pipeline.input_dir", dir=input_dir))
        _log_queue.put("📤 " + tr("pipeline.output_dir", dir=output_dir))
        _log_queue.put("⚙️ " + tr("pipeline.workers", count=get_env("MAX_API_WORKERS", "3")))
        _log_queue.put("=" * 50)

        if _engine is None:
            _engine = AlignmentEngine()

        max_workers = get_env_int("MAX_API_WORKERS", 3)
        api_thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        active_api_tasks = set()

        def background_translation_task(json_file_path, ass_file_path, file_name):
            try:
                _log_queue.put("🚀 " + tr("translation.start", name=file_name))
                run_translation(str(json_file_path), str(ass_file_path))

                if ass_file_path.exists():
                    _log_queue.put("🎉 " + tr("translation.complete", name=file_name))
                    if json_file_path.exists():
                        json_file_path.unlink()
            except Exception as e:
                _log_queue.put("❌ " + tr("translation.error", name=file_name, error=e))
            finally:
                if str(json_file_path) in active_api_tasks:
                    active_api_tasks.remove(str(json_file_path))

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pending = []
        for root, _, files in os.walk(input_path):
            for f in sorted(files):
                if any(f.lower().endswith(ext) for ext in SUPPORTED_VIDEO_EXTENSIONS):
                    video = Path(root) / f
                    rel = video.relative_to(input_path)
                    ass = output_path / rel.with_suffix(".ass")
                    if not ass.exists():
                        pending.append((video, rel, ass))

        if not pending:
            _log_queue.put("ℹ️ " + tr("pipeline.no_files"))
            return

        _log_queue.put("📋 " + tr("pipeline.found_files", count=len(pending)))

        for idx, (video, rel, ass) in enumerate(pending):
            if _cancel_flag.is_set():
                _log_queue.put("🛑 " + tr("pipeline.user_stopped"))
                break

            _log_queue.put(f"\n{'=' * 50}")
            _log_queue.put(
                "🎯 "
                + tr("alignment.file_header", index=idx + 1, total=len(pending), name=rel.name)
            )

            ass.parent.mkdir(parents=True, exist_ok=True)
            json_name = f"{video.stem}_alignment.json"
            json_path = ass.with_name(json_name)

            if json_path.exists():
                _log_queue.put("⏩ " + tr("alignment.found_json"))
            else:
                try:
                    _engine.load_model()
                    success = _engine.perform_ultimate_alignment(
                        video,
                        str(json_path),
                        progress_callback=lambda pct, stage: _log_queue.put(
                            "   📊 " + tr("alignment.progress", pct=pct, stage=stage)
                        ),
                    )
                    if not success or not json_path.exists():
                        _log_queue.put("❌ " + tr("alignment.failed"))
                        continue
                except Exception as e:
                    _log_queue.put("❌ " + tr("alignment.error", error=e))
                    continue

            if _cancel_flag.is_set():
                break

            _log_queue.put("📦 " + tr("alignment.draft_ready", name=rel.name))
            active_api_tasks.add(str(json_path))
            api_thread_pool.submit(background_translation_task, json_path, ass, rel.name)

            if (idx + 1) % get_env_int("ALIGNMENT_BATCH_SIZE", 3) == 0:
                _log_queue.put("🔄 " + tr("alignment.vram_clear"))
                _engine.clear_vram_cache()

        _log_queue.put(f"\n{'=' * 50}")
        if _cancel_flag.is_set():
            _log_queue.put("🛑 " + tr("pipeline.gpu_stopped"))
        else:
            _log_queue.put("🎉 " + tr("pipeline.gpu_done"))

        api_thread_pool.shutdown(wait=True)

        if _engine:
            _engine.clear_vram_cache()

        _log_queue.put("\n🏁 " + tr("pipeline.complete", time=datetime.now().strftime("%H:%M:%S")))

    except Exception as e:
        _log_queue.put("💥 " + tr("pipeline.fatal_error", error=e))
    finally:
        sys.stdout = original_stdout
        _is_running.clear()


def start_processing(input_dir, output_dir):
    """启动后台处理线程"""
    if _is_running.is_set():
        return "⚠️ " + tr("webui.process.running")

    _cancel_flag.clear()
    _is_running.set()

    thread = threading.Thread(target=run_pipeline, args=(input_dir, output_dir), daemon=True)
    thread.start()
    return "🚀 " + tr("webui.process.started")


def stop_processing():
    """设置终止标志"""
    if _is_running.is_set():
        _cancel_flag.set()
        return "🛑 " + tr("webui.process.stopped")
    return "ℹ️ " + tr("webui.process.no_task")


def poll_logs(current_logs):
    """轮询日志队列，返回更新后的日志文本"""
    new_lines = []
    while not _log_queue.empty():
        try:
            new_lines.append(_log_queue.get_nowait())
        except queue.Empty:
            break

    if new_lines:
        updated = (
            current_logs + "\n".join(new_lines) + "\n"
            if current_logs
            else "\n".join(new_lines) + "\n"
        )
        return updated
    return current_logs


def change_language(locale):
    """切换语言"""
    set_locale(locale)
    return locale


def build_ui():
    """构建 Gradio 界面"""
    env = load_env_values()
    current_locale = get_locale()

    with gr.Blocks(
        title="🎬 AnimeTranslator WebUI",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
        css="""
            .log-box textarea { font-family: 'Consolas', 'Monaco', monospace !important; font-size: 13px !important; }
            footer { display: none !important; }
        """,
    ) as app:
        locale_state = gr.State(value=current_locale)

        with gr.Row():
            gr.Markdown(f"# 🎬 {tr('webui.title')}\n**{tr('webui.subtitle')}**")
            lang_dropdown = gr.Dropdown(
                choices=[("简体中文", "zh_CN"), ("English", "en_US")],
                value=current_locale,
                label=tr("common.language"),
                scale=0,
                min_width=120,
            )
            lang_dropdown.change(
                fn=change_language,
                inputs=[lang_dropdown],
                outputs=[locale_state],
            )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(f"### 📁 {tr('webui.dir_settings')}")
                input_dir = gr.Textbox(
                    label=tr("webui.input_dir"), value=str(INPUT_DIR), interactive=True
                )
                output_dir = gr.Textbox(
                    label=tr("webui.output_dir"), value=str(OUTPUT_DIR), interactive=True
                )

                gr.Markdown(f"### 📤 {tr('webui.upload.title')}")
                upload = gr.File(
                    label=tr("webui.upload.label"),
                    file_count="multiple",
                    file_types=[".mkv", ".mp4", ".avi", ".mov", ".flv", ".wmv", ".webm"],
                    type="filepath",
                )
                upload_status = gr.Textbox(
                    label=tr("webui.upload.status"), interactive=False, lines=1
                )

                gr.Markdown(f"### ⚙️ {tr('webui.config.title')}")
                api_key = gr.Textbox(
                    label=tr("webui.config.api_key"), value=env["api_key"], type="password"
                )
                max_workers = gr.Slider(
                    1, 10, step=1, value=env["max_workers"], label=tr("webui.config.max_workers")
                )
                batch_size = gr.Slider(
                    1,
                    10,
                    step=1,
                    value=env["batch_size"],
                    label=tr("webui.config.batch_size"),
                )
                nsp_threshold = gr.Slider(
                    0.1,
                    1.0,
                    step=0.05,
                    value=env["nsp_threshold"],
                    label=tr("webui.config.nsp_threshold"),
                )
                cr_threshold = gr.Slider(
                    1.0,
                    5.0,
                    step=0.1,
                    value=env["cr_threshold"],
                    label=tr("webui.config.cr_threshold"),
                )

                save_btn = gr.Button(f"💾 {tr('webui.config.save')}", variant="secondary")
                save_status = gr.Textbox(
                    label=tr("webui.config.save_status"), interactive=False, lines=1
                )

                save_btn.click(
                    fn=save_env_values,
                    inputs=[api_key, max_workers, batch_size, nsp_threshold, cr_threshold],
                    outputs=[save_status],
                )

            with gr.Column(scale=2):
                gr.Markdown(f"### 🚀 {tr('webui.process.title')}")
                with gr.Row():
                    start_btn = gr.Button(
                        f"🚀 {tr('webui.process.start')}", variant="primary", size="lg"
                    )
                    stop_btn = gr.Button(
                        f"🛑 {tr('webui.process.stop')}", variant="stop", size="lg"
                    )

                run_status = gr.Textbox(
                    label=tr("webui.process.status"), interactive=False, lines=1
                )

                start_btn.click(
                    fn=start_processing,
                    inputs=[input_dir, output_dir],
                    outputs=[run_status],
                )
                stop_btn.click(fn=stop_processing, outputs=[run_status])

                gr.Markdown(f"### 📜 {tr('webui.log.title')}")
                log_box = gr.Textbox(
                    label="",
                    interactive=False,
                    lines=22,
                    max_lines=22,
                    autoscroll=True,
                    elem_classes=["log-box"],
                )

                timer = gr.Timer(value=1)
                timer.tick(fn=poll_logs, inputs=[log_box], outputs=[log_box])

                gr.Markdown(f"### 📋 {tr('webui.file.title')}")
                file_table = gr.Dataframe(
                    headers=[
                        tr("webui.file.header.name"),
                        tr("webui.file.header.status"),
                        tr("webui.file.header.output"),
                    ],
                    col_count=(3, "fixed"),
                    interactive=False,
                    wrap=True,
                )
                refresh_btn = gr.Button(f"🔄 {tr('webui.file.refresh')}", size="sm")
                refresh_btn.click(
                    fn=scan_files,
                    inputs=[input_dir, output_dir],
                    outputs=[file_table],
                )

                upload.upload(
                    fn=handle_upload,
                    inputs=[upload, input_dir],
                    outputs=[upload_status],
                ).then(
                    fn=scan_files,
                    inputs=[input_dir, output_dir],
                    outputs=[file_table],
                )

                gr.Markdown(f"### 📦 {tr('webui.download.title')}")
                download_list = gr.File(
                    label=tr("webui.download.label"),
                    file_count="multiple",
                    interactive=False,
                )
                download_btn = gr.Button(f"🔄 {tr('webui.download.refresh')}", size="sm")
                download_btn.click(
                    fn=get_output_files,
                    inputs=[output_dir],
                    outputs=[download_list],
                )

        app.load(fn=scan_files, inputs=[input_dir, output_dir], outputs=[file_table])

    return app


def run_webui(port=7860, share=False):
    """启动 WebUI"""
    ensure_dirs()

    from .config import validate_config

    warnings = validate_config()
    if warnings:
        print("⚠️ " + tr("config.warning.title"))
        for w in warnings:
            print(f"  {w}")
        print()

    print("🎬 " + tr("webui.launch.starting"))
    print(f"📁 {tr('webui.launch.root_dir')} {INPUT_DIR.parent}")
    print(f"📥 {tr('webui.launch.input_dir')} {INPUT_DIR}")
    print(f"📤 {tr('webui.launch.output_dir')} {OUTPUT_DIR}")
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        inbrowser=True,
    )
