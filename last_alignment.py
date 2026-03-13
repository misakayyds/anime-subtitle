import os
import sys
import json
import re
import subprocess
import gc
import tempfile
from pathlib import Path
import stable_whisper
import torch
import torchaudio
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess

# Whisper 常见幻觉关键词黑名单
HALLUCINATION_PATTERNS = [
    "ご視聴ありがとう", "チャンネル登録", "おやすみなさい", "お疲れ様",
    "セリフ", "字幕", "ご覧いただき", "次回もお楽しみ", "高评価", "グッドボタン",
]

# 呼吸/喘气废话过滤器：纯由这些成分组成的句子直接丢弃
BREATHING_PATTERN = re.compile(
    r'^[はふうあえおっぁぅぇぉー！!\u2026。、\s]*$'
)

# SenseVoice 音频事件标签
TAG_BGM = "<|BGM|>"
TAG_SPEECH = "<|Speech|>"
TAG_MUSIC = "<|MUSIC|>"

# OP/ED 检测参数
OP_WINDOW_SEC = 5 * 60       # 开头 5 分钟
ED_WINDOW_SEC = 5 * 60       # 结尾 5 分钟
OP_ED_MIN_DURATION = 85      # OP/ED 最短持续时间（秒）
OP_ED_MAX_DURATION = 95      # OP/ED 最长持续时间（秒）

# 质检阈值（放宽以减少误杀真实台词）
NO_SPEECH_PROB_THRESHOLD = 0.7
COMPRESSION_RATIO_THRESHOLD = 2.8

# 张量切片 padding（秒）
SLICE_PADDING_SEC = 0.3


class AlignmentEngine:
    def __init__(self):
        self.whisper_model = None
        self.sensevoice_model = None
        self.vad_model = None

    def load_model(self):
        """加载 fsmn-vad（测距仪）、SenseVoice（雷达）和 Stable-Whisper（狙击枪）三引擎"""
        if self.vad_model is None:
            print("📏 正在加载 fsmn-vad 测距引擎...")
            self.vad_model = AutoModel(
                model="fsmn-vad",
                device="cuda",
                vad_kwargs={
                    "threshold": 0.3,              # 降低阈值 → 更敏感，不漏掉轻声台词
                    "min_speech_duration_ms": 80,   # 最短语音段 → 短促"嗯""啊"不被吞
                },
            )
            print("✅ fsmn-vad 测距引擎已就绪！")
        else:
            print("📏 fsmn-vad 测距引擎已在显存驻留，跳过加载。")

        if self.sensevoice_model is None:
            print("🛰️ 正在加载 SenseVoice 全局雷达引擎 (SenseVoiceSmall + fsmn-vad)...")
            self.sensevoice_model = AutoModel(
                model="iic/SenseVoiceSmall",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                trust_remote_code=True,
                device="cuda",
            )
            print("✅ SenseVoice 雷达引擎已就绪！")
        else:
            print("🛰️ SenseVoice 雷达引擎已在显存驻留，跳过加载。")

        if self.whisper_model is None:
            print("🎯 正在加载 Stable-Whisper 狙击引擎 (Large-v3)...")
            self.whisper_model = stable_whisper.load_faster_whisper(
                'large-v3', device='cuda', compute_type='float16'
            )
            print("✅ Stable-Whisper 狙击引擎已就绪！")
        else:
            print("🎯 Stable-Whisper 狙击引擎已在显存驻留，跳过加载。")

    def clear_vram_cache(self):
        print("🧹 正在执行显存垃圾回收（清理当前集残留片段）...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("✅ 当前集数据已清空，显存碎片已整理！")

    # ========================================================================
    # 第一阶段：SenseVoice 全局雷达粗扫
    # ========================================================================
    def _sensevoice_scan(self, audio_path):
        """
        用 fsmn-vad 获取时间戳 + SenseVoice 获取标签，合并为带标签的碎片列表。
        返回: [{"start": float_sec, "end": float_sec, "text": str, "tags": [str]}]
        """
        print("📡 第一阶段：SenseVoice 全局雷达粗扫中...")

        # --- Step 1: fsmn-vad 单独跑，拿到每个碎片的精确时间戳 ---
        print("   📏 fsmn-vad 正在测距...")
        vad_results = self.vad_model.generate(
            input=str(audio_path),
            cache={},
        )
        # vad_results 格式: [{"key": "...", "value": [[start_ms, end_ms], [start_ms, end_ms], ...]}]
        vad_segments = []
        for vad_res in vad_results:
            if isinstance(vad_res, dict) and "value" in vad_res:
                for seg in vad_res["value"]:
                    vad_segments.append((seg[0] / 1000.0, seg[1] / 1000.0))  # ms → sec
        print(f"   📏 测距完成，共发现 {len(vad_segments)} 个音频段。")

        # --- Step 2: SenseVoice 跑标签识别 ---
        print("   🛰️ SenseVoice 正在扫描标签...")
        sv_results = self.sensevoice_model.generate(
            input=str(audio_path),
            cache={},
            language="ja",
            use_itn=True,
            batch_size_s=60,
        )

        # SenseVoice 输出: [{"key": "...", "text": "<|ja|><|EMO|><|Event|><|itn|>文本 <|ja|>..."}]
        # 所有 VAD 段的结果拼在一个 text 字段中，用 <|ja|> 作为每段的起始标记
        combined_text = ""
        for sv_res in sv_results:
            if isinstance(sv_res, dict):
                combined_text += sv_res.get("text", "")

        # --- Step 3: 按 <|ja|> 分割，解析每个段的标签和文本 ---
        # 分割模式: <|ja|><|EMO_UNKNOWN|><|BGM|><|withitn|>actual text
        segment_pattern = re.compile(r'<\|ja\|>')
        raw_segments = segment_pattern.split(combined_text)
        # 第一个元素是空字符串（split 之前没有内容），跳过
        raw_segments = [s for s in raw_segments if s.strip()]

        parsed_segments = []
        for raw_seg in raw_segments:
            # 提取标签: <|EMO_UNKNOWN|>, <|BGM|>, <|Speech|>, <|MUSIC|>, <|withitn|> 等
            tags = re.findall(r'<\|(\w+)\|>', raw_seg)
            # 移除所有标签，剩余的就是纯文本
            clean_text = re.sub(r'<\|[^|]+\|>', '', raw_seg).strip()
            # 过滤掉无意义的标签（EMO_UNKNOWN, withitn, woitn 等）
            event_tags = [t for t in tags if t in ("BGM", "Speech", "MUSIC", "Laughter", "Applause")]
            parsed_segments.append({
                "text": clean_text,
                "tags": event_tags,
            })

        # --- Step 4: 将 VAD 时间戳与 SenseVoice 标签一一对应 ---
        fragments = []
        n_match = min(len(vad_segments), len(parsed_segments))
        if len(vad_segments) != len(parsed_segments):
            print(f"   ⚠️ VAD 段数({len(vad_segments)}) ≠ SenseVoice 段数({len(parsed_segments)})，"
                  f"取最小值 {n_match} 进行匹配。")

        for i in range(n_match):
            start, end = vad_segments[i]
            parsed = parsed_segments[i]
            fragments.append({
                "start": start,
                "end": end,
                "text": parsed["text"],
                "tags": parsed["tags"],
            })

        print(f"   📡 雷达粗扫完成，共探测到 {len(fragments)} 个音频碎片。")
        return fragments

    # ========================================================================
    # 第二阶段：划定"禁飞区"
    # ========================================================================
    def _filter_no_fly_zones(self, fragments, audio_duration):
        """
        过滤纯音乐片段 + 识别 OP/ED 禁飞区。
        返回通过过滤的"幸存碎片"列表。
        """
        print("🚧 第二阶段：划定 Whisper 禁飞区...")

        # --- 第一步：识别 OP/ED 禁飞区 ---
        no_fly_zones = []

        # OP 检测：视频开头 0 ~ OP_WINDOW_SEC
        op_zone = self._detect_op_ed_zone(fragments, 0, OP_WINDOW_SEC)
        if op_zone:
            no_fly_zones.append(op_zone)
            print(f"   🎵 已锁定 OP 禁飞区: {op_zone[0]:.1f}s ~ {op_zone[1]:.1f}s "
                  f"（跨度 {op_zone[1] - op_zone[0]:.1f}s）")

        # ED 检测：视频结尾 倒数 ED_WINDOW_SEC
        ed_start = max(0, audio_duration - ED_WINDOW_SEC)
        ed_zone = self._detect_op_ed_zone(fragments, ed_start, audio_duration)
        if ed_zone:
            no_fly_zones.append(ed_zone)
            print(f"   🎵 已锁定 ED 禁飞区: {ed_zone[0]:.1f}s ~ {ed_zone[1]:.1f}s "
                  f"（跨度 {ed_zone[1] - ed_zone[0]:.1f}s）")

        if not no_fly_zones:
            print("   ℹ️ 未检测到 OP/ED 禁飞区。")

        # --- 第二步：逐碎片过滤 ---
        survivors = []
        dropped_music = 0
        dropped_oped = 0

        for frag in fragments:
            # 检查是否在禁飞区内
            in_no_fly = False
            for zone_start, zone_end in no_fly_zones:
                if frag["start"] >= zone_start and frag["end"] <= zone_end:
                    in_no_fly = True
                    break
                # 碎片大部分落入禁飞区也丢弃（超过50%重叠）
                overlap_s = max(0, min(frag["end"], zone_end) - max(frag["start"], zone_start))
                frag_dur = frag["end"] - frag["start"]
                if frag_dur > 0 and overlap_s / frag_dur > 0.5:
                    in_no_fly = True
                    break

            if in_no_fly:
                dropped_oped += 1
                continue

            # 纯音乐过滤：标签含 BGM/MUSIC 且几乎无文字（放宽到 ≤2 字，避免误杀 BGM 下的短台词）
            frag_tags = frag.get("tags", [])
            is_music = any(t in ("BGM", "MUSIC") for t in frag_tags)
            has_speech_tag = "Speech" in frag_tags
            text_len = len(frag.get("text", ""))

            if is_music and not has_speech_tag and text_len <= 2:
                dropped_music += 1
                continue

            survivors.append(frag)

        print(f"   🚧 过滤完成: 丢弃 OP/ED {dropped_oped} 片, "
              f"丢弃纯音乐 {dropped_music} 片, "
              f"幸存 {len(survivors)} 片碎片。")
        return survivors

    def _detect_op_ed_zone(self, fragments, window_start, window_end):
        """
        在指定时间窗口内检测连续的音乐标签片段。
        如果总跨度在 85~95 秒之间，返回 (zone_start, zone_end)。
        """
        # 筛选窗口内含音乐标签的碎片
        music_frags = []
        for frag in fragments:
            if frag["start"] >= window_start and frag["end"] <= window_end:
                frag_tags = frag.get("tags", [])
                if any(t in ("BGM", "MUSIC") for t in frag_tags):
                    music_frags.append(frag)

        if not music_frags:
            return None

        # 按时间排序
        music_frags.sort(key=lambda x: x["start"])

        # 寻找连续的音乐片段簇（允许片段间最多 3 秒间隔）
        clusters = []
        current_cluster = [music_frags[0]]

        for i in range(1, len(music_frags)):
            gap = music_frags[i]["start"] - music_frags[i - 1]["end"]
            if gap <= 3.0:  # 3秒容差，允许音乐片段间的短暂间隔
                current_cluster.append(music_frags[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [music_frags[i]]
        clusters.append(current_cluster)

        # 检查每个簇的总跨度是否在 OP/ED 范围内
        for cluster in clusters:
            span_start = cluster[0]["start"]
            span_end = cluster[-1]["end"]
            span_duration = span_end - span_start

            if OP_ED_MIN_DURATION <= span_duration <= OP_ED_MAX_DURATION:
                return (span_start, span_end)

        return None

    # ========================================================================
    # 第三阶段：内存张量切片 + Whisper 精确狙击
    # ========================================================================
    def _whisper_snipe(self, waveform, sr, survivors):
        """
        对幸存碎片做内存张量切片，喂给 stable-whisper 精确转写。
        返回带时间轴的台词列表（含 Whisper 内部指标用于第四阶段质检）。
        """
        total = len(survivors)
        print(f"🎯 第三阶段：Whisper 精确狙击 {total} 个有效目标...")

        all_segments = []

        for i, frag in enumerate(survivors):
            frag_start = frag["start"]
            frag_end = frag["end"]

            # 进度汇报（每 20 个或最后一个）
            if (i + 1) % 20 == 0 or i == total - 1:
                print(f"   🎯 进度: {i + 1}/{total} ({(i + 1) / total * 100:.0f}%)")

            # 张量切片 + padding
            pad_start = max(0, frag_start - SLICE_PADDING_SEC)
            pad_end = min(waveform.shape[1] / sr, frag_end + SLICE_PADDING_SEC)

            start_sample = int(pad_start * sr)
            end_sample = int(pad_end * sr)
            segment_waveform = waveform[:, start_sample:end_sample]

            if segment_waveform.shape[1] == 0:
                continue

            # 写入临时文件喂给 stable-whisper（faster-whisper 不支持直接张量输入）
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            try:
                torchaudio.save(tmp_path, segment_waveform, sr)

                result_sub = self.whisper_model.transcribe(
                    tmp_path,
                    language='ja',
                    beam_size=8,        # 提高到 8，多探索路径减少听错概率
                    vad_filter=False,   # 不需要二次 VAD，SenseVoice 已经做了
                    condition_on_previous_text=False,
                    no_speech_threshold=0.8,
                    initial_prompt="以下はアニメのセリフです。「お」「う」「え」「あ」「ん」「おい」「おう」「ああ」「うん」",
                )

                # 分句处理
                result_sub.split_by_punctuation(
                    [('。', ' '), ('！', ' '), ('？', ' '), ('!', ' '), ('?', ' ')]
                )
                result_sub.split_by_punctuation([('、', ' ')])
                result_sub.split_by_length(max_chars=35)
                result_sub.split_by_gap(max_gap=0.5)

                for segment in result_sub.segments:
                    seg_duration = segment.end - segment.start
                    if seg_duration > 15.0:
                        continue

                    text = segment.text.strip()
                    if not text:
                        continue

                    if any(pattern in text for pattern in HALLUCINATION_PATTERNS):
                        continue

                    # 呼吸/喘气废话过滤
                    if BREATHING_PATTERN.match(text):
                        continue

                    # 还原绝对时间轴（加上切片偏移量）
                    abs_start = round(segment.start + pad_start, 3)
                    abs_end = round(segment.end + pad_start, 3)

                    # 收集 Whisper 内部指标用于第四阶段质检
                    no_speech_prob = getattr(segment, 'no_speech_prob', 0.0)
                    compression_ratio = getattr(segment, 'compression_ratio', 0.0)

                    all_segments.append({
                        "start": abs_start,
                        "end": abs_end,
                        "ja_text": text,
                        "_no_speech_prob": no_speech_prob,
                        "_compression_ratio": compression_ratio,
                    })

            except Exception as e:
                print(f"   ⚠️ 碎片 {i+1} 狙击失败: {str(e)}")
            finally:
                try:
                    os.unlink(tmp_path)
                except:
                    pass

        # --- 去重：不同 VAD 碎片的 padding 区域可能让 Whisper 重复识别同一句 ---
        all_segments.sort(key=lambda x: x["start"])
        deduped = []
        for seg in all_segments:
            if deduped:
                prev = deduped[-1]
                overlap_start = max(seg["start"], prev["start"])
                overlap_end = min(seg["end"], prev["end"])
                overlap = max(0, overlap_end - overlap_start)
                seg_dur = seg["end"] - seg["start"]
                # 与前一句重叠超过 50% 视为重复
                if seg_dur > 0 and overlap / seg_dur > 0.5:
                    continue
            deduped.append(seg)

        if len(all_segments) != len(deduped):
            print(f"   🔗 去重: {len(all_segments)} → {len(deduped)}（移除 {len(all_segments) - len(deduped)} 句重复）")

        print(f"   🎯 狙击完成，共命中 {len(deduped)} 句候选台词。")
        return deduped

    # ========================================================================
    # 第四阶段：终极质检员
    # ========================================================================
    def _quality_check(self, segments):
        """
        检查 no_speech_prob 和 compression_ratio，拦截漏网的 BGM 幻觉。
        """
        print("🔬 第四阶段：终极质检中...")

        passed = []
        dropped_nsp = 0
        dropped_cr = 0

        for seg in segments:
            nsp = seg.get("_no_speech_prob", 0.0)
            cr = seg.get("_compression_ratio", 0.0)

            if nsp > NO_SPEECH_PROB_THRESHOLD:
                dropped_nsp += 1
                continue

            if cr > COMPRESSION_RATIO_THRESHOLD:
                dropped_cr += 1
                continue

            # 移除内部指标，只保留输出字段
            clean_seg = {
                "start": seg["start"],
                "end": seg["end"],
                "ja_text": seg["ja_text"],
            }
            passed.append(clean_seg)

        print(f"   🔬 质检完成: 拦截 no_speech_prob 超标 {dropped_nsp} 句, "
              f"拦截 compression_ratio 超标 {dropped_cr} 句, "
              f"最终通过 {len(passed)} 句台词。")
        return passed

    # ========================================================================
    # 主入口：perform_ultimate_alignment（接口不变）
    # ========================================================================
    def perform_ultimate_alignment(self, video_path, expected_json_path=None):
        video_file = Path(video_path)
        video_name = video_file.stem

        self.load_model()

        # --- 提取音频到内存 ---
        print(f"🎬 正在从 [{video_name}] 提取原声音频到内存...")
        # 先用 ffmpeg 提取音频为临时 WAV（16kHz 单声道）
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_audio_path = tmp.name

        try:
            extract_cmd = [
                "ffmpeg", "-y", "-i", str(video_path),
                "-map", "0:a:0",
                "-ar", "16000", "-ac", "1",
                tmp_audio_path
            ]
            result = subprocess.run(
                extract_cmd, capture_output=True, text=True,
                encoding='utf-8', errors='ignore'
            )
            if result.returncode != 0:
                print(f"❌ 音频提取失败: {result.stderr}")
                return False

            # 加载波形到内存（后续张量切片复用）
            waveform, sr = torchaudio.load(tmp_audio_path)
            audio_duration = waveform.shape[1] / sr
            print(f"   ✅ 音频已加载到内存，时长 {audio_duration:.1f}s，采样率 {sr}Hz")

            # === 第一阶段：SenseVoice 全局雷达粗扫 ===
            fragments = self._sensevoice_scan(tmp_audio_path)

            # === 第二阶段：划定禁飞区 ===
            survivors = self._filter_no_fly_zones(fragments, audio_duration)

            # === 第三阶段：Whisper 精确狙击 ===
            raw_segments = self._whisper_snipe(waveform, sr, survivors)

            # === 第四阶段：终极质检 ===
            final_segments = self._quality_check(raw_segments)

        except Exception as e:
            print(f"⚠️ 管线执行异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            try:
                os.unlink(tmp_audio_path)
            except:
                pass

        # 排序并编号输出
        final_segments.sort(key=lambda x: x["start"])
        final_subs = {}
        for i, seg in enumerate(final_segments, 1):
            final_subs[str(i)] = seg

        if expected_json_path:
            output_path = expected_json_path
        else:
            output_path = f"{video_name}_alignment.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_subs, f, ensure_ascii=False, indent=4)

        print(f"✅ 完美的底稿已生成！共 {len(final_subs)} 句台词，保存在: {output_path}")
        return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("使用方法: python last_alignment.py <视频文件路径>")
        sys.exit(1)

    input_video = sys.argv[1]
    if not os.path.exists(input_video):
        print(f"错误: 找不到视频文件 {input_video}")
        sys.exit(1)

    engine = AlignmentEngine()
    success = engine.perform_ultimate_alignment(input_video)
    engine.clear_vram_cache()

    if not success:
        sys.exit(1)