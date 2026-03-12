import os
import sys
import json
import subprocess
import gc
from pathlib import Path
import stable_whisper
import torch

class AlignmentEngine:
    def __init__(self):
        self.model = None

    def load_model(self):
        """延迟加载：只有真要用了才读取大模型进显存"""
        if self.model is None:
            print("第二阶段：加载 Stable-Whisper 外挂引擎 (Large-v3)...")
            self.model = stable_whisper.load_faster_whisper('large-v3', device='cuda', compute_type='float16')
        else:
            print("第二阶段：Stable-Whisper 引擎已在显存驻留，极速跳过加载...")

    def clear_vram_cache(self):
        """温和清理：仅丢弃当前集产生的临时数据和显存碎片，绝不销毁底层模型引擎（防止 Windows 下 C++ 核心崩溃）"""
        print("🧹 正在执行显存垃圾回收（清理当前集残留片段）...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ 当前集数据已清空，显存碎片已整理！")

    def perform_ultimate_alignment(self, video_path, expected_json_path=None):
        video_file = Path(video_path)
        video_name = video_file.stem # 自动获取视频名字，比如 "234"
        
        # 1. 动态调用 Demucs 提取纯人声 (保持 subprocess，免得污染主进程各种状态)
        print(f"第一阶段：正在使用 Demucs 提取 [{video_name}] 的纯净人声...")
        demucs_cmd = [
            sys.executable, "-m", "demucs.separate", 
            "-n", "htdemucs", 
            "--two-stems=vocals", 
            "-d", "cuda", 
            str(video_path)
        ]
        
        result = subprocess.run(demucs_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Demucs处理失败: {result.stderr}")
            return False
            
        # 动态拼接音频路径
        vocals_path = Path(f"separated/htdemucs/{video_name}/vocals.wav")
        if not vocals_path.exists():
            print(f"错误：找不到提取的人声文件: {vocals_path}")
            return False
            
        # 2. 调用 Stable-Whisper 进行听写
        self.load_model()
        
        # 极度严苛的 VAD 参数，只要停顿 0.4 秒直接物理砍断
        result_sub = self.model.transcribe_stable(
            str(vocals_path), 
            language='ja', 
            vad=True, 
            vad_threshold=0.2,  # 保持 0.2，拯救“Kimo”等微弱发音
            condition_on_previous_text=False,
            # 👈 换成这条“万能 ACG 催眠咒语”
            initial_prompt="以下はアニメのセリフです。日常会話をはじめ、「キモい」「クソ」「放せ」「待て」などの砕けた口語表現や命令形、ため息、戦闘時の叫び声などが含まれます。" 
        )
        
        # 3. 魔法切词
        print("第三阶段：正在进行物理强制切词...")
        result_sub.split_by_punctuation([('。', ' '), ('！', ' '), ('？', ' '), ('!', ' '), ('?', ' ')])
        result_sub.split_by_length(max_chars=22)
        result_sub.split_by_gap(max_gap=0.5)
        
        # 4. 动态导出 JSON
        final_subs = {}
        sub_id = 1
        
        for segment in result_sub.segments:
            # 终极物理防线：剔除超过8秒的惨叫/乱码
            if segment.end - segment.start > 8.0:
                continue
            if not segment.text.strip():
                continue
                
            final_subs[str(sub_id)] = {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "ja_text": segment.text.strip()
            }
            sub_id += 1
            
        if expected_json_path:
            output_path = expected_json_path
        else:
            output_path = f"{video_name}_alignment.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_subs, f, ensure_ascii=False, indent=4)
            
        print(f"✅ 完美的底稿已生成！保存在: {output_path}")
        return True


if __name__ == "__main__":
    # 兼容原有的命令行单文件启动模式
    if len(sys.argv) != 2:
        print("使用方法: python last_alignment.py <视频文件路径>")
        sys.exit(1)
    
    input_video = sys.argv[1]
    if not os.path.exists(input_video):
        print(f"错误: 找不到视频文件 {input_video}")
        sys.exit(1)
        
    engine = AlignmentEngine()
    success = engine.perform_ultimate_alignment(input_video)
    engine.clear_vram_cache() # 跑完清理碎片
    
    if not success:
        sys.exit(1)