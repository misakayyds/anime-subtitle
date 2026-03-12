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

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ⚙️ 核心并发控制台
# 允许后台同时向 DeepSeek 发起几个视频的翻译请求？（默认 3）
MAX_API_WORKERS = int(os.environ.get("MAX_API_WORKERS", 3))

# 引擎常驻显存配置（默认 3）
ALIGNMENT_BATCH_SIZE = int(os.environ.get("ALIGNMENT_BATCH_SIZE", 3))

# 线程池与任务雷达（防止重复派发任务）
api_thread_pool = ThreadPoolExecutor(max_workers=MAX_API_WORKERS)
active_api_tasks = set()

# 全局显卡翻译引擎（保持单例加载）
alignment_engine = AlignmentEngine()
episodes_processed_since_last_unload = 0

def is_file_ready(filepath):
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
        vocals_dir = BASE_DIR / "separated" / "htdemucs" / input_mkv.stem
        if vocals_dir.exists():
            shutil.rmtree(vocals_dir)
            
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

                # GPU 开始干活或直接捡漏
                if json_path.exists():
                    print(f"⏩ 发现遗留 JSON 底稿，GPU 直接跳过！")
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

if __name__ == "__main__":
    print(f"👀 异步流水线监听已启动 (并行消费者模式)!")
    print(f"⚙️ 当前最大 API 并发数: {MAX_API_WORKERS}")
    print("--------------------------------------------------")
    
    has_started_work = False
    
    while True:
        try:
            pending = get_pending_tasks()
            
            if pending > 0:
                has_started_work = True
                process_queue()
            else:
                if has_started_work:
                    print("\n" + "="*50)
                    print("🎉 报告老板：所有前台 GPU 任务已清空！")
                    print("⏳ 正在等待后台 DeepSeek 翻译收尾工作...")
                    
                    # 优雅关闭：卡在这里等所有后台 API 请求完成，绝不提前关机
                    api_thread_pool.shutdown(wait=True)
                    
                    # 彻底打扫战场
                    alignment_engine.clear_vram_cache()
                    
                    print("✅ 所有后台字幕均已成功生成！准备执行自动关机程序！")
                    print("💡 【后悔药】按下 Win+R，输入 shutdown /a 取消关机！")
                    print("="*50 + "\n")
                    
                    os.system("shutdown /s /t 60")
                    break
                    
        except KeyboardInterrupt:
            print("\n监听已手动停止。正在清理后台与显存...")
            alignment_engine.clear_vram_cache()
            api_thread_pool.shutdown(wait=False)
            break
        except Exception as e:
            print(f"主循环异常: {e}")
            
        time.sleep(5)