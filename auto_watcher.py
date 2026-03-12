import os
import time
import shutil
import subprocess
from pathlib import Path

# 基础目录配置
BASE_DIR = Path(r"F:\AnimeTranslator")
INPUT_DIR = BASE_DIR / "Input"
OUTPUT_DIR = BASE_DIR / "Output"
ENV_PYTHON = BASE_DIR / "env" / "Scripts" / "python.exe"

# 确保输入输出文件夹存在
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

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

def process_queue():
    # 遍历 Input 目录下的所有文件
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.lower().endswith(".mkv"):
                input_mkv = Path(root) / file
                
                # 计算输出路径
                rel_path = input_mkv.relative_to(INPUT_DIR)
                expected_output_ass = OUTPUT_DIR / rel_path.with_suffix('.ass')
                
                # 🛑 第一道安检（断点续传）：如果成品字幕已经存在，直接跳过！
                if expected_output_ass.exists():
                    continue
                    
                # 规则2：如果文件还在拷贝中，暂不处理，等下一轮
                if not is_file_ready(input_mkv):
                    continue
                    
                print(f"\n==================================================")
                print(f"🎯 锁定目标: {rel_path.name}")
                print(f"==================================================")
                
                # 定义 JSON 底稿的路径
                json_name = f"{input_mkv.stem}_alignment.json"
                json_path = BASE_DIR / json_name

                try:
                    # 🛑 第二道安检（断点续传）：检查 JSON 是否已经存在
                    if json_path.exists():
                        print(f"⏩ 检测到遗留的 JSON 底稿 {json_name}")
                        print("⏩ 已自动跳过第一阶段(语音识别)，直接恢复进度！")
                    else:
                        # 只有 JSON 不存在时，才去压榨 5070 Ti 跑提取和对齐
                        print("--> 步骤 1/2: Demucs 人声提取与时间轴对齐 (压榨 5070 Ti 中...)")
                        try:
                            subprocess.run([str(ENV_PYTHON), "last_alignment.py", str(input_mkv)], cwd=BASE_DIR, check=True)
                        except subprocess.CalledProcessError as e:
                            # 拦截 Windows 下模型显存释放时的常见崩溃
                            print(f"\n⚠️ 捕获到模型退出错误码: {e.returncode} (底层显存释放特性)")
                            print("👉 正在检查 JSON 底稿是否幸存...")
                        
                        if not json_path.exists():
                            print(f"❌ 致命错误：未找到生成的 JSON 底稿 {json_path}，已跳过该集。")
                            continue
                        print("✅ 底稿生成成功！进入翻译阶段...")

                    # ==========================================
                    # 步骤 2：调用翻译脚本
                    # ==========================================
                    print("--> 步骤 2/2: DeepSeek 注入灵魂翻译中...")
                    subprocess.run([str(ENV_PYTHON), "llm_translation.py", str(json_path)], cwd=BASE_DIR, check=True)
                    
                    # 步骤 3：整理结构与垃圾回收
                    print("--> 步骤 3/3: 还原目录结构与清理缓存空间...")
                    expected_output_ass.parent.mkdir(parents=True, exist_ok=True)
                    
                    generated_ass = BASE_DIR / f"{input_mkv.stem}_bilingual.ass"
                    if generated_ass.exists():
                        shutil.move(str(generated_ass), str(expected_output_ass))
                        print(f"🎉 大获全胜! 完美双语字幕已送达: {expected_output_ass}")
                        
                    # 【核心垃圾回收】只有走到这一步（大模型翻译成功了），才允许删除 JSON
                    if json_path.exists():
                        json_path.unlink()
                        
                    vocals_dir = BASE_DIR / "separated" / "htdemucs" / input_mkv.stem
                    if vocals_dir.exists():
                        shutil.rmtree(vocals_dir)
                        
                except subprocess.CalledProcessError as e:
                    print(f"💥 处理时发生严重错误，已跳过该集。报错信息: {e}")
                except Exception as e:
                    print(f"💥 发生未知错误: {e}")

if __name__ == "__main__":
    print(f"👀 自动化巡逻监听已启动 (已开启断点续传保护)!")
    print(f"📥 监听目录: {INPUT_DIR}")
    print(f"📤 输出目录: {OUTPUT_DIR}")
    print("--------------------------------------------------")
    print("将整季番剧文件夹拖入 Input，中途可随时关闭脚本。")
    print("重新打开时，将自动从上次中断的地方继续。")
    print("按下 Ctrl+C 可停止监听。")
    
    while True:
        try:
            process_queue()
        except KeyboardInterrupt:
            print("\n监听已手动停止。")
            break
        except Exception as e:
            print(f"主监听循环异常: {e}")
            
        time.sleep(10)