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
        try:
            # 回归 Whisper 大模型最强原生态：开启前后文推断，降低阈值防吞音
            result_sub = self.model.transcribe(
                str(vocals_path), 
                language='ja', 
                beam_size=5,
                vad_filter=True, 
                # 进一步降低阈值和时间：只抛弃真正的绝对死寂，哪怕只有 0.02 秒的破音/打岔/惊呼也要留下来
                vad_parameters=dict(min_speech_duration_ms=20, threshold=0.1), 
                # 开启前后文联系：不仅防漏句，还能让 Whisper 根据上一句推测当前这句极大声却含糊不清的台词是什么
                condition_on_previous_text=True, 
                no_speech_threshold=0.6,
                initial_prompt="以下はアニメのセリフです。話し声が少しでもあれば絶対に出力してください。「あ」「えっ」「うわっ」などの短い叫び声や感嘆詞も漏らさず文字起ししてください。" 
            )
        except Exception as e:
            print(f"⚠️ 模型转写发生异常: {str(e)}")
            return False
        
        # 3. 魔法切词与时间轴精修
        print("第三阶段：正在进行物理强制切词...")
        # 遇到各种日文常见句号/感叹号才切，保持句子完整性
        result_sub.split_by_punctuation([('。', ' '), ('！', ' '), ('？', ' '), ('!', ' '), ('?', ' ')])
        # 恢复物理切分，由于关掉了不稳定的词级时间戳，这里以 Whisper 原始大段落为准进行切分防重叠
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