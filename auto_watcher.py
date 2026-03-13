import os
import time
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from last_alignment import AlignmentEngine

# 加载环境变量
load_dotenv()

# 基础目录配置 (动态获取当前脚本所在目录)
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "Input"
OUTPUT_DIR = BASE_DIR / "Output"
ENV_PYTHON = BASE_DIR / "env" / "Scripts" / "python.exe"

# 确保输入输出文件夹存在
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ⚙️ 核心并发控制台
# 允许后台同时向 DeepSeek 发起几个视频的翻译请求？（默认 3）
MAX_API_WORKERS = int(os.environ.get("MAX_API_WORKERS", 3))

# 引擎常驻显存配置（默认 1）
ALIGNMENT_BATCH_SIZE = int(os.environ.get("ALIGNMENT_BATCH_SIZE", 1))

# 线程池与任务雷达（防止重复派发任务）
api_thread_pool = ThreadPoolExecutor(max_workers=MAX_API_WORKERS)
active_api_tasks = set()

# 全局显卡翻译引擎（保持单例加载）
alignment_engine = AlignmentEngine()
episodes_processed_since_last_unload = 0

def is_file_ready(filepath):
    """
    心跳检测：判断文件是否已经从硬盘拷贝完毕
    """
    try:
        size1 = os.path.getsize(filepath)
        time.sleep(3)
        size2 = os.path.getsize(filepath)
        return size1 == size2 and size1 > 0
    except Exception:
        return False

def get_pending_tasks():
    pending_count = 0
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(".mkv"):
                input_mkv = Path(root) / file
                rel_path = input_mkv.relative_to(INPUT_DIR)
                expected_output_ass = OUTPUT_DIR / rel_path.with_suffix('.ass')
                if not expected_output_ass.exists():
                    pending_count += 1
    return pending_count

def background_translation_task(json_path, input_mkv, expected_output_ass):
    """【后台消费者】专门负责在后台悄悄请求大模型和清理垃圾，绝对不阻塞 GPU"""
    try:
        print(f"🚀 [后台分发] 正在为 {input_mkv.name} 并发请求 DeepSeek...")
        
        # 强制传参让最终字幕直接生成在 expected_output_ass
        subprocess.run([str(ENV_PYTHON), "llm_translation.py", str(json_path), str(expected_output_ass)], cwd=BASE_DIR, check=True)
        
        if expected_output_ass.exists():
            print(f"🎉 [后台大捷] {input_mkv.name} 的完美字幕已直接送达: {expected_output_ass}")
            
        # 垃圾回收
        if json_path.exists():
            json_path.unlink()
            
    except subprocess.CalledProcessError as e:
        print(f"💥 [后台崩溃] {input_mkv.name} API 翻译失败，已降级或中止。")
    finally:
        # 任务做完（不管成功失败），必须把它的名字从雷达里抹掉
        if input_mkv in active_api_tasks:
            active_api_tasks.remove(input_mkv)

def process_queue():
    global episodes_processed_since_last_unload
    
    """【前台生产者】GPU 专属通道，只负责干力气活，干完就扔给后台"""
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(".mkv"):
                input_mkv = Path(root) / file
                rel_path = input_mkv.relative_to(INPUT_DIR)
                expected_output_ass = OUTPUT_DIR / rel_path.with_suffix('.ass')
                
                # 1. 如果成品有了，跳过
                if expected_output_ass.exists():
                    continue
                    
                # 2. 如果这集视频【正在后台被 DeepSeek 翻译中】，绝不能碰，直接跳过去找下一集！
                if input_mkv in active_api_tasks:
                    continue
                    
                if not is_file_ready(input_mkv):
                    continue
                    
                print(f"\n" + "="*50)
                print(f"🎯 [前台 GPU 锁定目标] {rel_path.name}")
                # 定义 JSON 底稿的路径（直接放置到对应的 Output 目录中！）
                expected_output_ass.parent.mkdir(parents=True, exist_ok=True)
                json_name = f"{input_mkv.stem}_alignment.json"
                json_path = expected_output_ass.with_name(json_name)

                try:
                    # 🛑 第二道安检（断点续传）：检查 JSON 是否已经存在
                    if json_path.exists():
                        print(f"⏩ 检测到遗留的 JSON 底稿 {json_name}")
                        print("⏩ 已自动跳过第一阶段(语音识别)，直接恢复进度！")
                    else:
                        print(f"--> [独占显卡] 压榨 5070 Ti 提取对齐中...")
                        try:
                            # 🚀 [优化点] 告别每次开子进程加载模型，直接调用进程内常驻引擎，极大节省加载时间
                            success = alignment_engine.perform_ultimate_alignment(input_mkv, str(json_path))
                            if not success:
                                print(f"⚠️ 模型提取遭遇异常返回。")
                        except Exception as e:
                            print(f"⚠️ 捕获到模型运行时致命异常: {e}")
                            
                        if not json_path.exists():
                            print(f"❌ 致命错误：未生成 JSON。")
                            continue
                            
                    print(f"📦 底稿就绪！把 {rel_path.name} 一脚踹进后台 API 线程池！")
                    
                    # 🌟 核心魔法：把文件加入雷达，丢进后台线程池，然后 GPU 立刻去处理下一个文件！
                    active_api_tasks.add(input_mkv)
                    api_thread_pool.submit(background_translation_task, json_path, input_mkv, expected_output_ass)
                    
                    # [优化点] 显存垃圾回收机制
                    episodes_processed_since_last_unload += 1
                    if episodes_processed_since_last_unload >= ALIGNMENT_BATCH_SIZE:
                        print(f"\n🔄 已连续处理 {episodes_processed_since_last_unload} 集，触发定期显存清理机制。")
                        alignment_engine.clear_vram_cache()
                        episodes_processed_since_last_unload = 0
                        
                except Exception as e:
                    print(f"💥 发生未知错误: {e}")

if __name__ == "__main__":
    print(f"👀 异步并发巡逻监听已启动 (并行消费者模式, 永不关机)!")
    print(f"⚙️ 当前最大 API 并发数: {MAX_API_WORKERS}")
    print(f"📥 监听目录: {INPUT_DIR}")
    print(f"📤 输出目录: {OUTPUT_DIR}")
    print("--------------------------------------------------")
    print("将整季番剧文件夹拖入 Input，中途可随时关闭脚本。")
    print("重新打开时，将自动从上次中断的地方继续。")
    print("按下 Ctrl+C 可停止监听。")
    
    while True:
        try:
            pending = get_pending_tasks()
            
            if pending > 0:
                process_queue()
            else:
                pass # 这个版本不需要全部完成后的关机大结局，保持无限监听
        except KeyboardInterrupt:
            print("\n监听已手动停止。正在清理后台与显存...")
            alignment_engine.clear_vram_cache()
            api_thread_pool.shutdown(wait=False)
            break
        except Exception as e:
            print(f"主监听循环异常: {e}")
            
        time.sleep(10)