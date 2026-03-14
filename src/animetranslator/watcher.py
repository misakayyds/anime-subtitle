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

from .config import INPUT_DIR, OUTPUT_DIR, ensure_dirs, load_env, get_env_int
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
                if file.lower().endswith(".mkv"):
                    input_mkv = Path(root) / file
                    rel_path = input_mkv.relative_to(INPUT_DIR)
                    expected_output_ass = OUTPUT_DIR / rel_path.with_suffix(".ass")
                    if not expected_output_ass.exists():
                        pending_count += 1
        return pending_count

    def background_translation_task(json_path, input_mkv, expected_output_ass):
        """后台翻译任务"""
        try:
            log_info(f"🚀 [后台分发] 正在为 {input_mkv.name} 并发请求 DeepSeek...")
            run_translation(str(json_path), str(expected_output_ass))

            if expected_output_ass.exists():
                log_info(
                    f"🎉 [后台大捷] {input_mkv.name} 的完美字幕已直接送达: {expected_output_ass}"
                )

            if json_path.exists():
                json_path.unlink()

            vocals_dir = INPUT_DIR.parent / "separated" / "htdemucs" / input_mkv.stem
            if vocals_dir.exists():
                shutil.rmtree(vocals_dir)

        except Exception as e:
            log_error(f"💥 [后台崩溃] {input_mkv.name} API 翻译失败: {e}")
        finally:
            if input_mkv in active_api_tasks:
                active_api_tasks.remove(input_mkv)

    def process_queue():
        """处理队列中的视频文件"""
        nonlocal episodes_processed_since_last_unload

        for root, _, files in os.walk(INPUT_DIR):
            for file in files:
                if file.lower().endswith(".mkv"):
                    input_mkv = Path(root) / file
                    rel_path = input_mkv.relative_to(INPUT_DIR)
                    expected_output_ass = OUTPUT_DIR / rel_path.with_suffix(".ass")

                    if expected_output_ass.exists():
                        continue

                    if input_mkv in active_api_tasks:
                        continue

                    if not is_file_ready(input_mkv):
                        continue

                    log_info(f"\n" + "=" * 50)
                    log_info(f"🎯 [前台 GPU 锁定目标] {rel_path.name}")

                    expected_output_ass.parent.mkdir(parents=True, exist_ok=True)
                    json_name = f"{input_mkv.stem}_alignment.json"
                    json_path = expected_output_ass.with_name(json_name)

                    try:
                        if json_path.exists():
                            log_info(f"⏩ 检测到遗留的 JSON 底稿 {json_name}")
                            log_info("⏩ 已自动跳过第一阶段(语音识别)，直接恢复进度！")
                        else:
                            log_info(f"--> [独占显卡] 压榨 GPU 提取对齐中...")
                            try:
                                success = alignment_engine.perform_ultimate_alignment(
                                    input_mkv, str(json_path)
                                )
                                if not success:
                                    log_warning(f"⚠️ 模型提取遭遇异常返回。")
                            except Exception as e:
                                log_warning(f"⚠️ 捕获到模型运行时致命异常: {e}")

                            if not json_path.exists():
                                log_error(f"❌ 致命错误：未生成 JSON。")
                                continue

                        log_info(f"📦 底稿就绪！把 {rel_path.name} 一脚踹进后台 API 线程池！")

                        active_api_tasks.add(input_mkv)
                        api_thread_pool.submit(
                            background_translation_task, json_path, input_mkv, expected_output_ass
                        )

                        episodes_processed_since_last_unload += 1
                        if episodes_processed_since_last_unload >= alignment_batch_size:
                            log_info(
                                f"\n🔄 已连续处理 {episodes_processed_since_last_unload} 集，触发定期显存清理机制。"
                            )
                            alignment_engine.clear_vram_cache()
                            episodes_processed_since_last_unload = 0

                    except Exception as e:
                        log_error(f"💥 发生未知错误: {e}")

    mode_str = "自动关机版" if shutdown_on_complete else "永不关机版"
    log_info(f"👀 异步并发巡逻监听已启动 ({mode_str})!")
    log_info(f"⚙️ 当前最大 API 并发数: {max_api_workers}")
    log_info(f"📥 监听目录: {INPUT_DIR}")
    log_info(f"📤 输出目录: {OUTPUT_DIR}")
    log_info("--------------------------------------------------")
    log_info("将整季番剧文件夹拖入 Input，中途可随时关闭脚本。")
    log_info("重新打开时，将自动从上次中断的地方继续。")
    log_info("按下 Ctrl+C 可停止监听。")

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
                        log_info("🎉 报告老板：所有前台 GPU 任务已清空！")
                        log_info("⏳ 正在等待后台 DeepSeek 翻译收尾工作...")

                        api_thread_pool.shutdown(wait=True)
                        alignment_engine.clear_vram_cache()

                        log_info("✅ 所有后台字幕均已成功生成！准备执行自动关机程序！")
                        log_info("💡 【后悔药】按下 Win+R，输入 shutdown /a 取消关机！")
                        log_info("=" * 50 + "\n")

                        if os.name == "nt":
                            os.system("shutdown /s /t 60")
                        else:
                            os.system("shutdown -h +1")
                        break

            except KeyboardInterrupt:
                log_info("\n监听已手动停止。正在清理后台与显存...")
                alignment_engine.clear_vram_cache()
                api_thread_pool.shutdown(wait=False)
                break
            except Exception as e:
                log_error(f"主监听循环异常: {e}")

            time.sleep(10 if not shutdown_on_complete else 5)
    except KeyboardInterrupt:
        alignment_engine.clear_vram_cache()
        api_thread_pool.shutdown(wait=False)
