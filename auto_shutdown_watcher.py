import os
import time
import shutil
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# 基础目录配置
BASE_DIR = Path(r"F:\AnimeTranslator")
INPUT_DIR = BASE_DIR / "Input"
OUTPUT_DIR = BASE_DIR / "Output"
ENV_PYTHON = BASE_DIR / "env" / "Scripts" / "python.exe"

INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ⚙️ 核心并发控制台
# 允许后台同时向 DeepSeek 发起几个视频的翻译请求？（推荐 3-5，太高会被 API 官方限流报错）
MAX_API_WORKERS = 3

# 线程池与任务雷达（防止重复派发任务）
api_thread_pool = ThreadPoolExecutor(max_workers=MAX_API_WORKERS)
active_api_tasks = set()

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
        subprocess.run([str(ENV_PYTHON), "llm_translation.py", str(json_path)], cwd=BASE_DIR, check=True)
        
        expected_output_ass.parent.mkdir(parents=True, exist_ok=True)
        generated_ass = BASE_DIR / f"{input_mkv.stem}_bilingual.ass"
        
        if generated_ass.exists():
            shutil.move(str(generated_ass), str(expected_output_ass))
            print(f"🎉 [后台大捷] {input_mkv.name} 的完美字幕已送达！")
            
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
                print("="*50)
                
                json_name = f"{input_mkv.stem}_alignment.json"
                json_path = BASE_DIR / json_name

                # GPU 开始干活或直接捡漏
                if json_path.exists():
                    print(f"⏩ 发现遗留 JSON 底稿，GPU 直接跳过！")
                else:
                    print(f"--> [独占显卡] 压榨 5070 Ti 提取对齐中...")
                    try:
                        subprocess.run([str(ENV_PYTHON), "last_alignment.py", str(input_mkv)], cwd=BASE_DIR, check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"⚠️ 捕获到模型退出错误码: {e.returncode} (显存释放特性)")
                        
                    if not json_path.exists():
                        print(f"❌ 致命错误：未生成 JSON。")
                        continue
                        
                print(f"📦 底稿就绪！把 {rel_path.name} 一脚踹进后台 API 线程池！")
                
                # 🌟 核心魔法：把文件加入雷达，丢进后台线程池，然后 GPU 立刻去处理下一个文件！
                active_api_tasks.add(input_mkv)
                api_thread_pool.submit(background_translation_task, json_path, input_mkv, expected_output_ass)

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
                    
                    print("✅ 所有后台字幕均已成功生成！准备执行自动关机程序！")
                    print("💡 【后悔药】按下 Win+R，输入 shutdown /a 取消关机！")
                    print("="*50 + "\n")
                    
                    os.system("shutdown /s /t 60")
                    break
                    
        except KeyboardInterrupt:
            print("\n监听已手动停止。")
            api_thread_pool.shutdown(wait=False)
            break
        except Exception as e:
            print(f"主循环异常: {e}")
            
        time.sleep(5)