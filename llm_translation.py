import os
import sys
import json
import time
from pathlib import Path
from datetime import timedelta
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

def seconds_to_ass_time(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = td.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60
    whole_secs = int(secs)
    centiseconds = int((secs - whole_secs) * 100)
    return f"{hours}:{minutes:02d}:{whole_secs:02d}.{centiseconds:02d}"

def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def generate_ass_file(translated_data, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("[Script Info]\nTitle: Bilingual Subtitles\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n\n")
        f.write("[V4+ Styles]\nFormat: Name, Fontname, fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write("Style: Default,微软雅黑,50,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,3,0,2,20,20,30,1\n\n")
        f.write("[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        # 将字典转为列表，方便我们“偷看”下一句
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
            
            # 1. 强制延长消失时间（视觉残留）
            end_sec += 0.4 
            
            # 2. 最低阅读时间保障（保底 1.5 秒）
            if (end_sec - start_sec) < 1.5:
                end_sec = start_sec + 1.5
                
            # 🚀 3. 终极防碰撞雷达（解决你的担忧）
            if i + 1 < len(items):
                # 偷看下一句的数据
                _, next_item_data = items[i+1]
                next_start_sec = next_item_data['start']
                
                # 如果我延长的结束时间，侵犯到了下一句的开始时间
                if end_sec > next_start_sec:
                    # 强行把我的结束时间“砍”到下一句开始前 0.05 秒（留出极其微小的闪烁间隔，防止黏连）
                    end_sec = next_start_sec - 0.05
            
            # 重新计算最终的 duration 用于异常拦截
            duration = end_sec - start_sec
            
            # 异常超长轴物理截断
            if duration > 4.5 and len(ja_text) <= 3:
                continue
            if duration > 6.0:
                end_sec = start_sec + 4.0
            
            # 为了防止上面砍得太狠导致 end_sec 小于 start_sec 的极端情况保底
            if end_sec <= start_sec:
                end_sec = start_sec + 0.1

            start_time = seconds_to_ass_time(start_sec)
            end_time = seconds_to_ass_time(end_sec)
            
            ass_text = f"{zh_text}\\N{{\\fs45}}{ja_text}"
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{ass_text}\n")

def main(input_json_path, expected_output_ass=None):
    json_file = Path(input_json_path)
    
    if expected_output_ass:
        output_ass_path = str(expected_output_ass)
    else:
        # 动态生成输出文件名：把 "234_alignment.json" 变成 "234_bilingual.ass"
        base_name = json_file.stem.replace("_alignment", "")
        output_ass_path = str(json_file.with_name(f"{base_name}_bilingual.ass"))

    print(f"正在读取原轴数据: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        alignment_data = json.load(f)
    
    sorted_items = sorted(alignment_data.items(), key=lambda x: int(x[0]))
    chunks = list(chunk_list(sorted_items, 40))
    
    # ⚠️ 请在这里填入你【重新生成】的 API Key
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    # 防呆机制：如果用户没建 .env 文件或者忘记填了，直接报错退出，而不是浪费算力去请求
    if not api_key or api_key.strip() == "" or api_key == "sk-在这里填入你新生成的真实密钥":
        print("\n❌ 致命错误：未找到有效的 DeepSeek API Key！")
        print("💡 请确保项目根目录下存在 `.env` 文件，并正确填写了 DEEPSEEK_API_KEY。")
        sys.exit(1)
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    # 动态提取当前处理的文件名
    current_file_name = Path(input_json_path).stem 

    system_prompt = f"""你是一个精通日本ACG文化、拥有极高语言造诣的顶级字幕组校对兼翻译。
我将提供由AI语音识别生成的动漫台词（JSON格式）。

【核心自适应任务】
正在处理的视频文件名是：【{current_file_name}】
请你推断动漫作品，强制使用官方译名和专有术语。

【🚨神级纠错法则 1：同音错字反推】
ASR经常把角色名写成发音相同的汉字。请将其“罗马音化（拼音化）”，替换为正确的片假名角色名。

【🚨神级纠错法则 2：视觉梗与双关语的绝对保留】
绝对不准吞掉带有双关语或模仿动物的卖萌词汇（如 もうもううし、にゃ等），必须保留并生动意译！

【🚨神级纠错法则 3：中二病、生造词与“写A读B（当て字）”的降维打击】
遇到《命运石之门》、《Fate》、《咒术回战》等包含庞大世界观的作品时，ASR极易将“咒语”、“特有名词”识别成毫无关联的日常词汇。
一旦你通过文件名锁定了作品，请**强制调取该作品的官方词典**！
例如：ASR若识别出类似“エルプ...”的发音，你必须将其修正为“エル・プサイ・コングルゥ (El Psy Kongroo)”。
若遇到“写做A读作B”的词（如 读音为 シュタインズ・ゲート），请在日文中修正为官方的汉字写法（運命石の扉），并在中文里使用公认的圈内译名！

【🚨神级纠错法则 4：Whisper 幻觉滤除（斩鬼！）】
ASR在遇到纯BGM或静音时会产生“幻觉”，凭空捏造出诸如“ご視聴ありがとうございました(感谢您的收看)”、“おやすみなさい(晚安)”、“チャンネル登録(请订阅)”等YouTuber常用语。
一旦你发现这些与动漫当前残酷/紧张剧情毫无关联的突兀问候语，请立刻将其判定为幻觉，**直接将日文和中文都修正为 ""（留空）**，绝对不要让它们出现在字幕里！

【🚨神级纠错法则 5：拟声词、口癖与生理反应的意译（拒绝音译！）】
遇到角色呕吐、打嗝、惨叫的拟声词（如“ゲロー/Gero”，即呕吐声），**绝对不允许音译成“格罗”这种不知所云的名字**！
必须结合二次元语境，翻译成对应的中文拟声词或动作描述（例如将“ゲロー”翻译成“呕——”或“吐了”），并保留日文原词。

【🚨最高级别警告：绝对拒绝机器味与翻译腔！】
1. 贴合语境与情绪：人大声说话、吐槽、发酒疯时，大胆调整句式，增加地道的中文语气词。
2. 破除语法束缚：把生硬的书面语改成极度自然的口语！

你的任务：
1. 日文纠错：修复同音字、神级还原世界观专属生造词与咒语。
2. 中文神级精翻：彻底消除机器味！确保专有名词100%原汁原味！

必须以严格的 JSON 格式返回，键名为传入的绝对 ID。
返回示例: {{"1": {{"ja_corrected": "修正后的日文", "zh_translated": "中文翻译"}}}}"""

    all_translated = {}
    print(f"开始注入灵魂，共 {len(chunks)} 个区块...")
    
    for i, chunk in enumerate(chunks):
        print(f"[{i+1}/{len(chunks)}] 正在请求 DeepSeek 翻译与纠错，请耐心等待...")
        chunk_dict = {item_id: item_data for item_id, item_data in chunk}
        
        input_text = "\n".join([f"[ID: {item_id}] {item_data['ja_text']}" for item_id, item_data in chunk])
        
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                response_format={"type": "json_object"},
                temperature=0.8
            )
            
            api_result = json.loads(response.choices[0].message.content)
            
            for item_id, item_data in chunk_dict.items():
                if item_id in api_result:
                    item_data['ja_corrected'] = api_result[item_id].get('ja_corrected', '')
                    item_data['zh_translated'] = api_result[item_id].get('zh_translated', '')
                else:
                    item_data['ja_corrected'] = item_data['ja_text']
                    item_data['zh_translated'] = "【翻译漏句】"
                
                all_translated[item_id] = item_data
                
        except Exception as e:
            print(f"第 {i+1} 块处理崩溃: {str(e)}，已降级保留原文。")
            for item_id, item_data in chunk_dict.items():
                item_data['ja_corrected'] = item_data['ja_text']
                item_data['zh_translated'] = "【API失败暂缺】"
                all_translated[item_id] = item_data
                
        time.sleep(1) 
        
    print(f"生成 ASS 完美双语字幕文件中...")
    generate_ass_file(all_translated, output_ass_path)
    print(f"✅ 大功告成！双语字幕已保存为: {output_ass_path}")

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("使用方法: python llm_translation.py <JSON文件路径> [可选: 输出ASS文件路径]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not os.path.exists(input_file):
        print(f"错误: 找不到文件 {input_file}")
        sys.exit(1)
        
    if len(sys.argv) == 3:
        main(input_file, sys.argv[2])
    else:
        main(input_file)