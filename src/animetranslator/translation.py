"""
AnimeTranslator 翻译模块

读取 JSON 底稿 → DeepSeek API 翻译 → 生成双语 ASS 字幕
"""
import os
import sys
import json
import asyncio
from pathlib import Path
from datetime import timedelta
from openai import AsyncOpenAI

from .config import ENV_FILE, get_env, get_env_int


def seconds_to_ass_time(seconds):
    """秒数转换为 ASS 时间格式"""
    td = timedelta(seconds=seconds)
    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60
    whole_secs = int(secs)
    centiseconds = int((secs - whole_secs) * 100)
    return f"{hours}:{minutes:02d}:{whole_secs:02d}.{centiseconds:02d}"


def chunk_list(lst, chunk_size):
    """列表分块"""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def generate_ass_file(translated_data, output_path):
    """生成双语 ASS 字幕文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("[Script Info]\nTitle: Bilingual Subtitles\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n\n")
        f.write("[V4+ Styles]\nFormat: Name, Fontname, fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write("Style: Default,微软雅黑,50,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,0,2,20,20,30,1\n\n")
        f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")

        items = list(translated_data.items())

        for i in range(len(items)):
            item_id, item_data = items[i]

            ja_text = item_data.get('ja_corrected', '').strip()
            zh_text = item_data.get('zh_translated', '').strip()

            if not ja_text and not zh_text:
                continue
            if not ja_text:
                ja_text = item_data.get('ja_text', '').strip()

            start_sec = item_data['start']
            end_sec = item_data['end']

            end_sec += 0.4

            if i + 1 < len(items):
                _, next_item_data = items[i+1]
                next_start_sec = next_item_data['start']

                if end_sec > next_start_sec:
                    ideal_end = next_start_sec - 0.05
                    if ideal_end <= start_sec:
                        end_sec = start_sec + 0.5
                    else:
                        end_sec = ideal_end

            if (end_sec - start_sec) < 1.0:
                if i + 1 < len(items) and (start_sec + 1.0) <= items[i+1][1]['start']:
                    end_sec = start_sec + 1.0
                elif i + 1 == len(items):
                    end_sec = start_sec + 1.0

            duration = end_sec - start_sec

            if duration > 4.5 and len(ja_text) <= 3:
                continue
            if duration > 7.0:
                end_sec = start_sec + 4.0

            start_time = seconds_to_ass_time(start_sec)
            end_time = seconds_to_ass_time(end_sec)

            ass_text = f"{zh_text}\\N{{\\fs45}}{ja_text}"
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{ass_text}\n")


async def translate_json(input_json_path, expected_output_ass=None):
    """异步翻译 JSON 底稿"""
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE, override=True)

    json_file = Path(input_json_path)

    if expected_output_ass:
        output_ass_path = str(expected_output_ass)
    else:
        base_name = json_file.stem.replace("_alignment", "")
        output_ass_path = str(json_file.with_name(f"{base_name}_bilingual.ass"))

    print(f"正在读取原轴数据: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        alignment_data = json.load(f)

    sorted_items = sorted(alignment_data.items(), key=lambda x: int(x[0]))
    chunks = list(chunk_list(sorted_items, 40))

    api_key = get_env("DEEPSEEK_API_KEY")

    if not api_key or api_key.strip() == "":
        print("\n❌ 致命错误：未找到有效的 DeepSeek API Key！")
        print(f"💡 请确保项目根目录下存在 `.env` 文件，并正确填写了 DEEPSEEK_API_KEY。")
        print(f"   项目根目录: {ENV_FILE.parent}")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    current_file_name = Path(input_json_path).stem

    system_prompt = f"""你是一个精通日本ACG文化、拥有极高语言造诣的顶级字幕组校对兼翻译。
我将提供由AI语音识别生成的动漫台词（JSON格式）。

【核心自适应任务】
正在处理的视频文件名是：【{current_file_name}】
请你推断动漫作品，强制使用官方译名和专有术语。

【🚨神级纠错法则 1：同音错字反推】
ASR经常把角色名写成发音相同的汉字。请将其"罗马音化（拼音化）"，替换为正确的片假名角色名。

【🚨神级纠错法则 2：视觉梗与双关语的绝对保留】
绝对不准吞掉带有双关语或模仿动物的卖萌词汇（如 もうもううし、にゃ等），必须保留并生动意译！

【🚨神级纠错法则 3：中二病、生造词与"写A读B（当て字）"的降维打击】
遇到《命运石之门》、《Fate》、《咒术回战》等包含庞大世界观的作品时，ASR极易将"咒语"、"特有名词"识别成毫无关联的日常词汇。
一旦你通过文件名锁定了作品，请**强制调取该作品的官方词典**！
例如：ASR若识别出类似"エルプ..."的发音，你必须将其修正为"エル・プサイ・コングルゥ (El Psy Kongroo)"。
若遇到"写做A读作B"的词（如 读音为 シュタインズ・ゲート），请在日文中修正为官方的汉字写法（運命石の扉），并在中文里使用公认的圈内译名！

【🚨神级纠错法则 4：Whisper 幻觉滤除（斩鬼！）】
ASR在遇到纯BGM或静音时会产生"幻觉"，凭空捏造出诸如"ご視聴ありがとうございました(感谢您的收看)"、"おやすみなさい(晚安)"、"チャンネル登録(请订阅)"等YouTuber常用语。
一旦你发现这些与动漫当前残酷/紧张剧情毫无关联的突兀问候语，请立刻将其判定为幻觉，**直接将日文和中文都修正为 ""（留空）**，绝对不要让它们出现在字幕里！

【🚨神级纠错法则 5：拟声词、口癖与生理反应的意译（拒绝音译！）】
遇到角色呕吐、打嗝、惨叫的拟声词（如"ゲロー/Gero"，即呕吐声），**绝对不允许音译成"格罗"这种不知所云的名字**！
必须结合二次元语境，翻译成对应的中文拟声词或动作描述（例如将"ゲロー"翻译成"呕——"或"吐了"），并保留日文原词。

【🚨最高级别警告：绝对拒绝机器味与翻译腔！】
1. 贴合语境与情绪：人大声说话、吐槽、发酒疯时，大胆调整句式，增加地道的中文语气词。
2. 破除语法束缚：把生硬的书面语改成极度自然的口语！

你的任务：
1. 日文纠错：修复同音字、神级还原世界观专属生造词与咒语。
2. 中文神级精翻：彻底消除机器味！确保专有名词100%原汁原味！

【上下文参考 (仅供参考连贯性，**绝对不要**把它包含在你的翻译返回结果中)】
[CONTEXT_PLACEHOLDER]

必须以严格的 JSON 格式返回，键名为传入的绝对 ID。
返回示例: {{"1": {{"ja_corrected": "修正后的日文", "zh_translated": "中文翻译"}}}}"""

    all_translated = {}

    max_workers = get_env_int("MAX_API_WORKERS", 3)

    print(f"🚀 开始超高速异步并发翻译（并发限制: {max_workers}）...")

    semaphore = asyncio.Semaphore(max_workers)

    async def process_chunk_async(idx, chunk, context_str):
        print(f"⏳ [队列中] 区块 {idx+1}/{len(chunks)} 准备就绪...")
        async with semaphore:
            print(f"🚀 [处理中] 正在猛烈请求区块 {idx+1}/{len(chunks)}...")
            chunk_dict = {item_id: item_data for item_id, item_data in chunk}
            input_text = "\n".join([f"[ID: {item_id}] {item_data['ja_text']}" for item_id, item_data in chunk])
            current_system_prompt = system_prompt.replace("[CONTEXT_PLACEHOLDER]", context_str)

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = await client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": current_system_prompt},
                            {"role": "user", "content": input_text}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.8,
                        timeout=120.0
                    )

                    content = response.choices[0].message.content
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]

                    api_result = json.loads(content)

                    if not api_result:
                        raise ValueError("API 返回了空字典的错误格式")

                    print(f"✅ [成功] 区块 {idx+1}/{len(chunks)} 翻译完成！")

                    for item_id, item_data in chunk_dict.items():
                        if item_id in api_result:
                            ja = str(api_result[item_id].get('ja_corrected') or '').strip()
                            zh = str(api_result[item_id].get('zh_translated') or '').strip()

                            if not zh and ja:
                                item_data['ja_corrected'] = ja
                                item_data['zh_translated'] = ja
                            else:
                                item_data['ja_corrected'] = ja
                                item_data['zh_translated'] = zh
                        else:
                            item_data['ja_corrected'] = item_data['ja_text']
                            item_data['zh_translated'] = "【翻译漏句】"
                    return chunk_dict

                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"⚠️ [警告] 区块 {idx+1}/{len(chunks)} 处理异常 ({str(e)})，正在进行第 {attempt+1} 次重试...")
                        await asyncio.sleep(2 ** attempt)
                    else:
                        print(f"❌ [崩溃降级] 区块 {idx+1}/{len(chunks)} 连续 {max_retries} 次处理失败: {str(e)}，降级保留原文。")
                        for item_id, item_data in chunk_dict.items():
                            item_data['ja_corrected'] = item_data['ja_text']
                            item_data['zh_translated'] = "【API失败暂缺】"
                        return chunk_dict

    tasks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            context_str = "（这是第一段，无前文）"
        else:
            previous_chunk = chunks[i-1]
            last_4_items = previous_chunk[-4:]
            context_lines = []
            for item_id, item_data in last_4_items:
                context_lines.append(f"[{item_id}] 刚刚讲过的话: {item_data['ja_text']}")
            context_str = "\n".join(context_lines)

        tasks.append(process_chunk_async(i, chunk, context_str))

    results = await asyncio.gather(*tasks)

    for chunk_res in results:
        all_translated.update(chunk_res)

    print(f"生成 ASS 完美双语字幕文件中...")
    generate_ass_file(all_translated, output_ass_path)
    print(f"✅ 大功告成！双语字幕已保存为: {output_ass_path}")


def run_translation(input_json_path, expected_output_ass=None):
    """同步入口：翻译 JSON 底稿"""
    asyncio.run(translate_json(input_json_path, expected_output_ass))