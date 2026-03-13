"""快速调试脚本：打印 SenseVoice generate() 的原始输出结构"""
import json
import sys
import tempfile
import subprocess
from funasr import AutoModel

# 用一个短音频测试
video_path = sys.argv[1] if len(sys.argv) > 1 else None
if not video_path:
    print("用法: python debug_sensevoice.py <视频文件>")
    sys.exit(1)

# 提取前 30 秒音频用于快速测试
tmp_audio = tempfile.mktemp(suffix=".wav")
subprocess.run([
    "ffmpeg", "-y", "-i", video_path,
    "-map", "0:a:0", "-ss", "0", "-t", "30",
    "-ar", "16000", "-ac", "1", tmp_audio
], capture_output=True)

print("=== 加载模型 ===")
model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    trust_remote_code=True,
    device="cuda",
)

print("\n=== 调用 generate ===")
results = model.generate(
    input=tmp_audio,
    cache={},
    language="ja",
    use_itn=True,
    batch_size_s=60,
)

print(f"\n=== results 类型: {type(results)} ===")
print(f"=== results 长度: {len(results)} ===")

for i, r in enumerate(results):
    print(f"\n--- result[{i}] 类型: {type(r)} ---")
    if isinstance(r, dict):
        print(f"    keys: {list(r.keys())}")
        for k, v in r.items():
            v_repr = repr(v)
            if len(v_repr) > 500:
                v_repr = v_repr[:500] + "..."
            print(f"    [{k}] ({type(v).__name__}): {v_repr}")
    elif isinstance(r, (list, tuple)):
        print(f"    长度: {len(r)}")
        for j, item in enumerate(r[:3]):
            print(f"    [{j}] ({type(item).__name__}): {repr(item)[:300]}")
    else:
        print(f"    值: {repr(r)[:500]}")

import os
os.unlink(tmp_audio)
print("\n=== 调试完成 ===")
