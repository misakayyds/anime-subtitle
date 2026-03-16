"""
AnimeTranslator 后台看门狗

GPU 前台独占与 API 请求后台并发的主流水线程序
支持两种模式：
- 永久监听模式（默认）
- 自动关机模式（--shutdown）
"""

import os
import time
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .config import (
    INPUT_DIR,
    OUTPUT_DIR,
    ensure_dirs,
    load_env,
    get_env_int,
    SUPPORTED_VIDEO_EXTENSIONS,
)
from .alignment import AlignmentEngine
from .translation import run_translation
from .logger import log_info, log_error, log_warning


def run_watcher(shutdown_on_complete=False):
    """运行看门狗监听

    Args:
        shutdown_on_complete: 处理完成后是否自动关机
    """
    load_env()
    ensure_dirs()

    max_api_workers = get_env_int("MAX_API_WORKERS", 3)
    alignment_batch_size = get_env_int("ALIGNMENT_BATCH_SIZE", 1)

    api_thread_pool = ThreadPoolExecutor(max_workers=max_api_workers)
    active_api_tasks = set()

    alignment_engine = AlignmentEngine()
    episodes_processed_since_last_unload = 0

    def is_file_ready(filepath):
        """检测文件是否已完成拷贝"""
        try:
            size1 = os.path.getsize(filepath)
            time.sleep(3)
            size2 = os.path.getsize(filepath)
            return size1 == size2 and size1 > 0
        except Exception:
            return False

    def get_pending_tasks():
        """获取待处理任务数量"""
        pending_count = 0
        for root, _, files in os.walk(INPUT_DIR):
            for file in files:
                if any(file.lower().endswith(ext) for ext in SUPPORTED_VIDEO_EXTENSIONS):
                    input_video = Path(root) / file
                    rel_path = input_video.relative_to(INPUT_DIR)
                    expected_output_ass = OUTPUT_DIR / rel_path.with_suffix(".ass")
                    if not expected_output_ass.exists():
                        pending_count += 1
        return pending_count

    def background_translation_task(json_path, input_video, expected_output_ass):
        """后台翻译任务"""
        try:
            log_info(f"🚀 [Background] Requesting DeepSeek translation for {input_video.name}...")
            run_translation(str(json_path), str(expected_output_ass))

            if expected_output_ass.exists():
                log_info(
                    f"🎉 [Background Success] Subtitle ready for {input_video.name}: {expected_output_ass}"
                )

            if json_path.exists():
                json_path.unlink()

            vocals_dir = INPUT_DIR.parent / "separated" / "htdemucs" / input_video.stem
            if vocals_dir.exists():
                shutil.rmtree(vocals_dir)

        except Exception as e:
            log_error(f"💥 [Background Error] {input_video.name} translation failed: {e}")
        finally:
            if input_video in active_api_tasks:
                active_api_tasks.remove(input_video)

    def process_queue():
        """处理队列中的视频文件"""
        nonlocal episodes_processed_since_last_unload

        for root, _, files in os.walk(INPUT_DIR):
            for file in files:
                if any(file.lower().endswith(ext) for ext in SUPPORTED_VIDEO_EXTENSIONS):
                    input_video = Path(root) / file
                    rel_path = input_video.relative_to(INPUT_DIR)
                    expected_output_ass = OUTPUT_DIR / rel_path.with_suffix(".ass")

                    if expected_output_ass.exists():
                        continue

                    if input_video in active_api_tasks:
                        continue

                    if not is_file_ready(input_video):
                        continue

                    log_info(f"\n" + "=" * 50)
                    log_info(f"🎯 [GPU Target] {rel_path.name}")

                    expected_output_ass.parent.mkdir(parents=True, exist_ok=True)
                    json_name = f"{input_video.stem}_alignment.json"
                    json_path = expected_output_ass.with_name(json_name)

                    try:
                        if json_path.exists():
                            log_info(f"⏩ Found existing JSON draft {json_name}")
                            log_info("⏩ Skipping stage 1 (speech recognition), resuming progress!")
                        else:
                            log_info(f"--> [GPU Exclusive] Extracting alignment...")
                            try:
                                success = alignment_engine.perform_ultimate_alignment(
                                    input_video, str(json_path)
                                )
                                if not success:
                                    log_warning(f"⚠️ Model extraction returned abnormal.")
                            except Exception as e:
                                log_warning(f"⚠️ Runtime fatal exception: {e}")

                            if not json_path.exists():
                                log_error(f"❌ Fatal: JSON not generated.")
                                continue

                        log_info(
                            f"📦 Draft ready! Kicking {rel_path.name} to background API thread pool!"
                        )

                        active_api_tasks.add(input_video)
                        api_thread_pool.submit(
                            background_translation_task, json_path, input_video, expected_output_ass
                        )

                        episodes_processed_since_last_unload += 1
                        if episodes_processed_since_last_unload >= alignment_batch_size:
                            log_info(
                                f"\n🔄 Processed {episodes_processed_since_last_unload} episodes, triggering VRAM cleanup."
                            )
                            alignment_engine.clear_vram_cache()
                            episodes_processed_since_last_unload = 0

                    except Exception as e:
                        log_error(f"💥 Unknown error: {e}")

    mode_str = "auto-shutdown" if shutdown_on_complete else "permanent"
    log_info(f"👀 Async concurrent watcher started ({mode_str})!")
    log_info(f"⚙️ Max API workers: {max_api_workers}")
    log_info(f"📥 Input directory: {INPUT_DIR}")
    log_info(f"📤 Output directory: {OUTPUT_DIR}")
    log_info("--------------------------------------------------")
    log_info("Drop entire anime season folder into Input. Script can be closed anytime.")
    log_info("When reopened, it will resume from where it left off.")
    log_info("Press Ctrl+C to stop watching.")

    has_started_work = False

    try:
        while True:
            try:
                pending = get_pending_tasks()

                if pending > 0:
                    has_started_work = True
                    process_queue()
                else:
                    if shutdown_on_complete and has_started_work:
                        log_info("\n" + "=" * 50)
                        log_info("🎉 All foreground GPU tasks cleared!")
                        log_info("⏳ Waiting for background DeepSeek translation to finish...")

                        api_thread_pool.shutdown(wait=True)
                        alignment_engine.clear_vram_cache()

                        log_info("✅ All background subtitles generated! Preparing auto-shutdown!")
                        log_info("💡 [Undo] Press Win+R, type shutdown /a to cancel!")
                        log_info("=" * 50 + "\n")

                        if os.name == "nt":
                            os.system("shutdown /s /t 60")
                        else:
                            os.system("shutdown -h +1")
                        break

            except KeyboardInterrupt:
                log_info("\nWatcher stopped. Cleaning up background and VRAM...")
                alignment_engine.clear_vram_cache()
                api_thread_pool.shutdown(wait=False)
                break
            except Exception as e:
                log_error(f"Main loop exception: {e}")

            time.sleep(10 if not shutdown_on_complete else 5)
    except KeyboardInterrupt:
        alignment_engine.clear_vram_cache()
        api_thread_pool.shutdown(wait=False)
