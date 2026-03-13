# AnimeTranslator 动漫智能机翻/校对工具

## 简介
这是一个高自动化、低人工干预的动漫双语字幕（ASS）生成工具。项目整合了先进的语音活动检测（fsmn-vad）、音频事件分类（SenseVoice）、语音识别（Stable-Whisper）与大模型翻译纠错（DeepSeek API）技术，通过一套**"探照灯与禁飞区"**四阶段管线，将原始视频文件批量、极速地转化为带样式的高质量中日双语字幕。

## 核心架构：四阶段管线

```
原声视频 MKV
    │
    ▼
📡 第一阶段：SenseVoice 全局雷达粗扫
    │  fsmn-vad 精确测距 → 每个音频碎片的起止时间
    │  SenseVoice 标签识别 → BGM / Speech / MUSIC 分类
    │
    ▼
🚧 第二阶段：划定 Whisper 禁飞区
    │  OP/ED 连续性检测（85~95s 连续音乐标签 → 整块丢弃）
    │  纯音乐过滤（BGM/MUSIC 标签 + 无文字 → 丢弃）
    │
    ▼
🎯 第三阶段：Whisper 精确狙击
    │  张量切片（内存 Tensor Slicing，零磁盘 IO）
    │  逐碎片喂给 Stable-Whisper Large-v3
    │  智能断句 + 去重 + 呼吸废话过滤
    │
    ▼
🔬 第四阶段：终极质检员
    │  no_speech_prob > 0.7 → 丢弃（环境音伪装）
    │  compression_ratio > 2.8 → 丢弃（复读机幻觉）
    │
    ▼
✅ 输出 _alignment.json 底稿 → DeepSeek 翻译 → 双语 ASS 字幕
```

## 核心特性
1. **SenseVoice 智能标签分类**：取代传统的 Demucs 人声分离，直接在原声上用 SenseVoice 打标签区分 BGM/Speech/MUSIC，保留完整语境，GPU 耗时从分钟级降至秒级。
2. **OP/ED 禁飞区自动识别**：番剧特化功能，自动检测视频开头 5 分钟和结尾 5 分钟内的连续音乐段（85~95 秒），整块丢弃，彻底杜绝 Whisper 听译歌词。
3. **内存张量切片零 IO**：音频波形一次加载到显存，后续所有碎片用 Tensor Slicing 直接切取，前后各加 0.3 秒 Padding 防止吞音，避免反复 ffmpeg 编解码。
4. **四重过滤防幻觉**：SenseVoice 标签 → OP/ED 禁飞区 → Whisper no_speech_prob / compression_ratio 质检 → 呼吸废话正则过滤，层层拦截。
5. **DeepSeek 上下文神级翻译**：
    - 根据发音推测并修正同音字、角色名、中二生造词与咒语等专有名词。
    - 结合动作、情绪甚至生理反应语境意译拟声词，保留剧本视觉梗，彻底消除生硬的"机器味"。
    - 硬核判断并滤除 Whisper 常见的"感谢收看"、"请订阅"等幻觉输出。
6. **智能硬件与缓存调度**：
   - 支持**动态位置挂靠**，无需修改由于挪走根目录而报红的代码参数。
   - **三引擎常驻显存**：fsmn-vad + SenseVoice + Stable-Whisper 大模型无需逢剧必载，结合 `ALIGNMENT_BATCH_SIZE` 的定期显存清理策略，完美避开 OOM 幽灵 Bug。
   - 全程产生的文件**100% 镜像复刻 Input 中的多层子文件夹**，保持项目根目录终极洁癖。
7. **两套独立的前端（共享底层管线）**：
   - **前端 A: 交互式 WebUI (`webui.py`)**：纯小白向操作，支持浏览器拖拽上传、可视化调整所有 `.env` 参数并热重载、实时查阅运行日志和字幕结果下载。
   - **前端 B: 后台看门狗 (`auto_watcher.py` / `auto_shutdown_watcher.py`)**：适合自动化挂机，前线 GPU 无限处理切词，后方 API 线程池高并发翻译，支持断点续传。处理完后自动关机。

## 环境配置与依赖
*   本工具依赖一块 NVIDIA GPU（推荐 RTX 5060 Ti 或同级以上）以支撑 SenseVoice / Stable-Whisper 大模型的本地运算（默认 `cuda` + `float16` 精度）。
*   项目需将虚拟环境放置于工程根目录的 `env/` 下。
*   要求在工程根目录下建有 `.env` 文件，并正确配置 DeepSeek API 令牌。
*   首次运行时，FunASR 会自动从 ModelScope 下载 SenseVoice-Small 和 fsmn-vad 模型（约 1-2GB）。

```bash
# 核心依赖环境参考 (建议配合 Python 3.12 虚拟环境)
pip install -r requirements.txt
```

## 目录结构
*   `Input/`：输入目录，待处理的动漫视频（如 `.mkv` 格式）拷贝至此。
*   `Output/`：输出目录，最终生成的中日双语字幕（`.ass` 格式）保存在此处。
*   `webui.py`: **[新] 全新构建的浏览器 WebUI 主入口**（支持参数可视化调节和手动队列管理）。
*   `auto_watcher.py`：支持 GPU 前台独占与 API 请求后台并发的主流水线程序（经典异步监控版）。
*   `auto_shutdown_watcher.py`：带自动关机的无人值守挂机版。
*   `last_alignment.py`：底稿/原轴生成核心逻辑（SenseVoice 雷达粗扫 → 禁飞区过滤 → Whisper 精确狙击 → 质检 → 导出 `_alignment.json` 底稿）。
*   `llm_translation.py`：大模型翻译核心逻辑（读取 JSON 分块合并上下文 → DeepSeek API → 样式注入 `.ass` 文件）。

## 使用流程
1. 确保项目根目录下存在 `.env` 文件：
   ```env
   # DeepSeek API 密钥配置
   DEEPSEEK_API_KEY=sk-xxxxxx

   # 异步并发大模型翻译任务数（建议 3-5）
   MAX_API_WORKERS=3

   # 引擎常驻显存配置（设置几集清理一次模型显存，建议 1-5。设为 1 即每集清理）
   ALIGNMENT_BATCH_SIZE=3
   ```
2. **启动方式（二选一即可，GPU 同一时间只能被一个霸占）：**

   **方式 A：启动可视化 WebUI（推荐）**
   ```bash
   env\Scripts\python webui.py
   # 启动后在浏览器打开 http://127.0.0.1:7860
   ```

   **方式 B：启动经典后台看门狗**
   ```bash
   env\Scripts\python auto_watcher.py
   # 自动监控 Input/ 文件夹并静默处理
   ```

3. 将 `.mkv` 文件拖入 WebUI，或手动拷贝至 `Input/` 文件夹。
4. 喝杯咖啡。翻译完毕后，中日双语 `.ass` 字幕会出现在 `Output/` 或在 WebUI 直接下载。
