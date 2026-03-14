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

    return "✅ 配置已保存并热更新！"


def scan_files(input_dir, output_dir):
    """扫描输入输出目录，返回文件状态列表"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    rows = []

    if not input_path.exists():
        return [["⚠️ 输入目录不存在", "", ""]]

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
                    status = "✅ 已完成"
                elif json_path.exists():
                    status = "📝 底稿就绪（待翻译）"
                else:
                    status = "⏳ 待处理"

                rows.append([str(rel), status, str(ass_path) if ass_path.exists() else ""])

    if not rows:
        return [["（无支持的视频文件）", "", ""]]
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

    msg = f"✅ 已导入 {len(moved)} 个文件到 {input_dir}"
    return msg


def run_pipeline(input_dir, output_dir):
    """后台线程：执行完整的 对齐+翻译 管线"""
    global _engine
    original_stdout = sys.__stdout__

    sys.stdout = LogCapture(_log_queue, original_stdout)

    try:
        _log_queue.put("=" * 50)
        _log_queue.put(f"🚀 WebUI 管线启动 @ {datetime.now().strftime('%H:%M:%S')}")
        _log_queue.put(f"📥 输入目录: {input_dir}")
        _log_queue.put(f"📤 输出目录: {output_dir}")
        _log_queue.put(f"⚙️ 翻译并发数: {get_env('MAX_API_WORKERS', '3')}")
        _log_queue.put("=" * 50)

        if _engine is None:
            _engine = AlignmentEngine()

        max_workers = get_env_int("MAX_API_WORKERS", 3)
        api_thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        active_api_tasks = set()

        def background_translation_task(json_file_path, ass_file_path, file_name):
            try:
                _log_queue.put(f"🚀 [后台分发] 开始翻译: {file_name}...")
                run_translation(str(json_file_path), str(ass_file_path))

                if ass_file_path.exists():
                    _log_queue.put(f"🎉 [后台大捷] 字幕已生成: {file_name}")
                    if json_file_path.exists():
                        json_file_path.unlink()
            except Exception as e:
                _log_queue.put(f"❌ [后台崩溃] {file_name} 翻译异常: {e}")
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
            _log_queue.put("ℹ️ 没有待处理的文件，管线结束。")
            return

        _log_queue.put(f"📋 共发现 {len(pending)} 个待处理文件。")

        for idx, (video, rel, ass) in enumerate(pending):
            if _cancel_flag.is_set():
                _log_queue.put("🛑 用户终止了处理。")
                break

            _log_queue.put(f"\n{'=' * 50}")
            _log_queue.put(f"🎯 [{idx + 1}/{len(pending)}] {rel.name}")

            ass.parent.mkdir(parents=True, exist_ok=True)
            json_name = f"{video.stem}_alignment.json"
            json_path = ass.with_name(json_name)

            if json_path.exists():
                _log_queue.put(f"⏩ 发现遗留 JSON 底稿，跳过对齐阶段。")
            else:
                try:
                    _engine.load_model()
                    success = _engine.perform_ultimate_alignment(
                        video,
                        str(json_path),
                        progress_callback=lambda pct, stage: _log_queue.put(
                            f"   📊 {pct}% — {stage}"
                        ),
                    )
                    if not success or not json_path.exists():
                        _log_queue.put(f"❌ 对齐失败，跳过此文件。")
                        continue
                except Exception as e:
                    _log_queue.put(f"❌ 对齐异常: {e}")
                    continue

            if _cancel_flag.is_set():
                break

            _log_queue.put(f"📦 底稿就绪！把 {rel.name} 踢进后台 API 线程池！")
            active_api_tasks.add(str(json_path))
            api_thread_pool.submit(background_translation_task, json_path, ass, rel.name)

            if (idx + 1) % get_env_int("ALIGNMENT_BATCH_SIZE", 3) == 0:
                _log_queue.put(f"🔄 触发定期显存清理机制...")
                _engine.clear_vram_cache()

        _log_queue.put(f"\n{'=' * 50}")
        if _cancel_flag.is_set():
            _log_queue.put("🛑 GPU 前台任务已终止！等待后台现存任务收尾...")
        else:
            _log_queue.put("🎉 所有前台 GPU 提取任务已清空！正在等待后台 DeepSeek 翻译收尾...")

        api_thread_pool.shutdown(wait=True)

        if _engine:
            _engine.clear_vram_cache()

        _log_queue.put(f"\n🏁 管线彻底处理完毕！@ {datetime.now().strftime('%H:%M:%S')}")

    except Exception as e:
        _log_queue.put(f"💥 管线致命异常: {e}")
    finally:
        sys.stdout = original_stdout
        _is_running.clear()


def start_processing(input_dir, output_dir):
    """启动后台处理线程"""
    if _is_running.is_set():
        return "⚠️ 已有任务在运行中，请等待完成或终止后再试。"

    _cancel_flag.clear()
    _is_running.set()

    thread = threading.Thread(target=run_pipeline, args=(input_dir, output_dir), daemon=True)
    thread.start()
    return "🚀 管线已启动！请查看下方实时日志。"


def stop_processing():
    """设置终止标志"""
    if _is_running.is_set():
        _cancel_flag.set()
        return "🛑 终止信号已发送，将在当前文件处理完成后停止。"
    return "ℹ️ 当前没有运行中的任务。"


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


def build_ui():
    """构建 Gradio 界面"""
    env = load_env_values()

    with gr.Blocks(
        title="🎬 AnimeTranslator WebUI",
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue"),
        css="""
            .log-box textarea { font-family: 'Consolas', 'Monaco', monospace !important; font-size: 13px !important; }
            footer { display: none !important; }
        """,
    ) as app:
        gr.Markdown(
            "# 🎬 AnimeTranslator WebUI\n**音韵炼金术** — 感知共鸣 → 分离杂质 → 提取精华 → 点石成金"
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 目录设置")
                input_dir = gr.Textbox(label="输入目录", value=str(INPUT_DIR), interactive=True)
                output_dir = gr.Textbox(label="输出目录", value=str(OUTPUT_DIR), interactive=True)

                gr.Markdown("### 📤 拖拽上传文件")
                upload = gr.File(
                    label="拖入视频文件（自动存入输入目录）",
                    file_count="multiple",
                    file_types=[".mkv", ".mp4", ".avi", ".mov", ".flv", ".wmv", ".webm"],
                    type="filepath",
                )
                upload_status = gr.Textbox(label="上传状态", interactive=False, lines=1)

                gr.Markdown("### ⚙️ 参数配置")
                api_key = gr.Textbox(
                    label="DeepSeek API Key", value=env["api_key"], type="password"
                )
                max_workers = gr.Slider(
                    1, 10, step=1, value=env["max_workers"], label="API 并发数 (MAX_API_WORKERS)"
                )
                batch_size = gr.Slider(
                    1,
                    10,
                    step=1,
                    value=env["batch_size"],
                    label="显存清理周期 (ALIGNMENT_BATCH_SIZE)",
                )
                nsp_threshold = gr.Slider(
                    0.1,
                    1.0,
                    step=0.05,
                    value=env["nsp_threshold"],
                    label="质检: no_speech_prob 阈值",
                )
                cr_threshold = gr.Slider(
                    1.0,
                    5.0,
                    step=0.1,
                    value=env["cr_threshold"],
                    label="质检: compression_ratio 阈值",
                )

                save_btn = gr.Button("💾 保存配置", variant="secondary")
                save_status = gr.Textbox(label="保存状态", interactive=False, lines=1)

                save_btn.click(
                    fn=save_env_values,
                    inputs=[api_key, max_workers, batch_size, nsp_threshold, cr_threshold],
                    outputs=[save_status],
                )

            with gr.Column(scale=2):
                gr.Markdown("### 🚀 处理控制")
                with gr.Row():
                    start_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
                    stop_btn = gr.Button("🛑 终止任务", variant="stop", size="lg")

                run_status = gr.Textbox(label="运行状态", interactive=False, lines=1)

                start_btn.click(
                    fn=start_processing,
                    inputs=[input_dir, output_dir],
                    outputs=[run_status],
                )
                stop_btn.click(fn=stop_processing, outputs=[run_status])

                gr.Markdown("### 📜 实时日志")
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

                gr.Markdown("### 📋 文件队列")
                file_table = gr.Dataframe(
                    headers=["文件", "状态", "输出路径"],
                    col_count=(3, "fixed"),
                    interactive=False,
                    wrap=True,
                )
                refresh_btn = gr.Button("🔄 刷新文件列表", size="sm")
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

                gr.Markdown("### 📦 字幕下载")
                download_list = gr.File(
                    label="已生成的字幕文件（点击下载）",
                    file_count="multiple",
                    interactive=False,
                )
                download_btn = gr.Button("🔄 刷新下载列表", size="sm")
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
        print("⚠️ 配置验证警告:")
        for w in warnings:
            print(f"  {w}")
        print()

    print("🎬 AnimeTranslator WebUI 正在启动...")
    print(f"📁 项目根目录: {INPUT_DIR.parent}")
    print(f"📥 输入目录: {INPUT_DIR}")
    print(f"📤 输出目录: {OUTPUT_DIR}")
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        inbrowser=True,
    )
