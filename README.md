# AnimeTranslator 动漫智能机翻/校对工具

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

这是一个高自动化、低人工干预的动漫双语字幕（ASS）生成工具。项目整合了先进的语音活动检测（fsmn-vad）、音频事件分类（SenseVoice）、语音识别（Stable-Whisper）与大模型翻译纠错（DeepSeek API）技术，通过「音韵炼金术」四阶段管线，将原始视频文件批量、极速地转化为带样式的高质量中日双语字幕。

## 安装

### 系统要求

- Python 3.10+
- **GPU（推荐）**：
  - NVIDIA GPU（推荐 RTX 3060 Ti 或同级以上）+ CUDA 11.8+
  - AMD GPU（RX 6000/7000 系列）+ ROCm 6.x（仅 Linux）
  - Apple Silicon Mac (M1/M2/M3 系列)
- **CPU 模式**：支持无显卡运行，但速度较慢（约 10-20 倍耗时）
- **ffmpeg**（必须，用于音频提取）

### 资源消耗

| 设备类型 | 内存 (RAM) | 显存 (VRAM) | Whisper 模型 |
|----------|-----------|-------------|--------------|
| NVIDIA GPU (10GB+) | 3 GB | 10 GB+ | large-v3 |
| NVIDIA GPU (6-10GB) | 3 GB | 6-10 GB | medium |
| AMD GPU (ROCm) | 3 GB | 8 GB+ | medium |
| Apple Silicon | 8 GB+ | 共享内存 | medium |
| CPU | 8 GB+ | - | small |

> 实际消耗会根据视频长度和并发设置有所波动。

### 安装步骤

```bash
# 1. Clone 项目
git clone https://github.com/misakayyds/AnimeTranslator.git
cd AnimeTranslator

# 2. 创建虚拟环境
python -m venv env

# 3. 激活虚拟环境
# Windows:
env\Scripts\activate
# Linux/Mac:
source env/bin/activate

# 4. 安装 PyTorch（根据您的设备选择）
# NVIDIA GPU (CUDA 12.8):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
# NVIDIA GPU (CUDA 12.1):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
# NVIDIA GPU (CUDA 11.8):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# AMD GPU (ROCm 6.2, 仅 Linux):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
# Apple Silicon Mac (MPS):
pip install torch torchaudio
# CPU only (无显卡):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 5. 安装 AnimeTranslator
pip install -e .

# 6. 配置 API Key
copy .env.example .env
# 编辑 .env 文件，填入您的 DeepSeek API Key
```

### 安装 ffmpeg

本项目依赖 ffmpeg 进行音频提取，请确保系统已安装。

**Windows:**
```bash
# 方式一：使用 winget
winget install ffmpeg

# 方式二：使用 Chocolatey
choco install ffmpeg

# 方式三：手动下载
# 从 https://www.gyan.dev/ffmpeg/builds/ 下载，解压后将 bin 目录添加到 PATH
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**验证安装:**
```bash
ffmpeg -version
```

### 获取 DeepSeek API Key

1. 访问 [DeepSeek 开放平台](https://platform.deepseek.com/)
2. 注册并登录
3. 在 API Keys 页面创建新的 API Key
4. 将 API Key 填入 `.env` 文件的 `DEEPSEEK_API_KEY`

## 使用方法

### 方式一：WebUI（推荐）

```bash
# 确保在项目目录下，虚拟环境已激活
animetranslator webui

# 或使用 Python 模块方式
python -m animetranslator webui

# 自定义端口
animetranslator webui --port 8080

# 生成公网分享链接
animetranslator webui --share
```

启动后在浏览器打开 http://127.0.0.1:7860

### 方式二：后台看门狗

```bash
# 永久监听模式
animetranslator watch

# 自动关机模式（处理完成后 60 秒关机）
animetranslator watch --shutdown
```

### 工作流程

1. 将视频文件放入 `Input/` 目录（支持多层子文件夹）
   - 支持格式：`.mkv`, `.mp4`, `.avi`, `.mov`, `.flv`, `.wmv`, `.webm`
2. 运行 `animetranslator webui` 或 `animetranslator watch`
3. 等待处理完成
4. 在 `Output/` 目录获取双语 `.ass` 字幕文件

## 核心架构：音韵炼金术

```
原声视频文件
    │
    ▼
📡 一阶·感知共鸣 (SenseVoice)
    │  fsmn-vad 精确测距 → 每个音频碎片的起止时间
    │  SenseVoice 标签识别 → BGM / Speech / MUSIC 分类
    │
    ▼
🚧 二阶·分离杂质 (过滤)
    │  OP/ED 连续性检测（85~95s 连续音乐标签 → 整块丢弃）
    │  纯音乐过滤（BGM/MUSIC 标签 + 无文字 → 丢弃）
    │
    ▼
🎯 三阶·提取精华 (Whisper)
    │  张量切片（内存 Tensor Slicing，零磁盘 IO）
    │  逐碎片喂给 Stable-Whisper Large-v3
    │  智能断句 + 去重 + 呼吸废话过滤
    │
    ▼
🔬 四阶·点石成金 (质检 + 翻译)
    │  no_speech_prob > 0.7 → 丢弃（环境音伪装）
    │  compression_ratio > 2.8 → 丢弃（复读机幻觉）
    │
    ▼
✅ 输出 _alignment.json 底稿 → DeepSeek 翻译 → 双语 ASS 字幕
```

## 核心特性

1. **SenseVoice 智能标签分类**：取代传统的 Demucs 人声分离，直接在原声上用 SenseVoice 打标签区分 BGM/Speech/MUSIC，保留完整语境，GPU 耗时从分钟级降至秒级。

2. **OP/ED 杂质自动识别**：番剧特化功能，自动检测视频开头 5 分钟和结尾 5 分钟内的连续音乐段（85~95 秒），整块丢弃，彻底杜绝 Whisper 听译歌词。

3. **内存张量切片零 IO**：音频波形一次加载到显存，后续所有碎片用 Tensor Slicing 直接切取，前后各加 0.3 秒 Padding 防止吞音，避免反复 ffmpeg 编解码。

4. **四重过滤防幻觉**：SenseVoice 标签 → OP/ED 杂质过滤 → Whisper no_speech_prob / compression_ratio 质检 → 呼吸废话正则过滤，层层拦截。

5. **DeepSeek 上下文神级翻译**：
   - 根据发音推测并修正同音字、角色名、中二生造词与咒语等专有名词
   - 结合动作、情绪甚至生理反应语境意译拟声词，保留剧本视觉梗
   - 硬核判断并滤除 Whisper 常见的"感谢收看"、"请订阅"等幻觉输出

6. **智能硬件与缓存调度**：
   - 三引擎常驻显存：fsmn-vad + SenseVoice + Stable-Whisper 大模型无需逢剧必载
   - 定期显存清理策略，完美避开 OOM 幽灵 Bug

7. **结构化日志系统**：
   - 统一的日志记录，支持控制台和文件双输出
   - 日志文件保存在 `Output/logs/` 目录，格式为 `animetranslator_YYYYMMDD_HHMMSS.log`
   - 带时间戳的日志格式，便于问题排查

8. **配置自动验证**：
   - 启动时自动检查配置项有效性
   - API Key 格式验证、参数范围检查
   - 问题提示清晰，降低配置错误率

## 配置说明

编辑项目根目录下的 `.env` 文件：

```env
# DeepSeek API 密钥配置（必填）
DEEPSEEK_API_KEY=sk-xxxxxx

# 设备配置（可选: auto/cuda/mps/cpu，默认 auto）
# auto - 自动检测，优先级: CUDA > MPS > CPU
# cuda - 强制使用 NVIDIA GPU
# mps  - 强制使用 Apple Silicon GPU
# cpu  - 强制使用 CPU（速度较慢）
DEVICE=auto

# Whisper 模型大小（可选，默认根据设备自动选择）
# large-v3 / medium / small
# WHISPER_MODEL=large-v3

# 异步并发大模型翻译任务数（建议 3-5）
MAX_API_WORKERS=3

# 引擎常驻显存配置（设置几集清理一次模型显存，建议 1-5）
ALIGNMENT_BATCH_SIZE=3

# 质检阈值（通常不需要修改）
NO_SPEECH_PROB_THRESHOLD=0.7
COMPRESSION_RATIO_THRESHOLD=2.8
```

## 目录结构

```
AnimeTranslator/
├── pyproject.toml           # 打包配置
├── README.md
├── LICENSE
├── .env.example             # 配置模板
├── .gitignore
├── requirements.txt         # 依赖参考
├── src/
│   └── animetranslator/     # 核心代码
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py           # CLI 入口
│       ├── config.py        # 配置管理
│       ├── logger.py        # 日志模块
│       ├── device.py        # 设备管理
│       ├── alignment.py     # 对齐引擎
│       ├── translation.py   # 翻译模块
│       ├── watcher.py       # 后台看门狗
│       └── webui.py         # WebUI 界面
├── tests/
├── Input/                   # 输入目录（待处理视频）
├── Output/                  # 输出目录（生成的字幕 + 日志）
│   └── logs/                # 日志文件目录
└── env/                     # 虚拟环境（用户创建）
```

## 常见问题

### Q: 如何取消自动关机？

在自动关机倒计时内，按下 `Win+R`，输入 `shutdown /a` 即可取消。

### Q: 首次运行很慢？

首次运行时，FunASR 会自动从 ModelScope 下载 SenseVoice-Small 和 fsmn-vad 模型（约 1-2GB），请耐心等待。

### Q: 支持哪些视频格式？

支持多种常见视频格式：`.mkv`, `.mp4`, `.avi`, `.mov`, `.flv`, `.wmv`, `.webm`

### Q: 如何查看处理进度？

- WebUI 模式：在浏览器中查看实时日志
- Watch 模式：在终端查看输出日志

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest tests/

# 代码格式化
black src/

# 代码检查
ruff check src/
```

## 后续计划

详细开发路线图请查看 [docs/ROADMAP.md](docs/ROADMAP.md)

### 1. 更多模型适配

- [ ] **轻量级模型选项**：支持 Whisper Medium/Small，降低显存需求至 4GB
- [ ] **其他 ASR 引擎**：集成 WhisperX、Moonshine 等备选方案
- [ ] **多语言支持**：扩展到韩语、英语等其他语言的字幕生成

### 2. 更多翻译 API 支持

- [ ] **OpenAI GPT**：支持 GPT-4o、GPT-4-turbo 等模型
- [ ] **Claude**：支持 Anthropic Claude 系列
- [ ] **Google Gemini**：支持 Google 最新大模型
- [ ] **通义千问**：支持阿里云 Qwen API
- [ ] **智谱 GLM**：支持智谱 ChatGLM API
- [ ] **OpenAI 兼容接口**：支持任意 OpenAI 兼容的 API 服务

### 3. 本地 LLM 支持

- [ ] **Ollama 集成**：支持通过 Ollama 运行本地模型
- [ ] **llama.cpp**：支持 GGUF 格式的量化模型
- [ ] **vLLM**：支持高性能本地推理
- [ ] **推荐模型**：Qwen2.5、Llama3、GLM-4 等开源模型适配
- [ ] **离线模式**：完全脱离网络运行的本地化方案

### 4. 多设备支持

- [x] **NVIDIA GPU (CUDA)**：RTX 系列显卡，CUDA 11.8+
- [x] **AMD GPU (ROCm)**：RX 6000/7000 系列，仅 Linux，需安装 ROCm 驱动和 PyTorch ROCm 版
- [x] **Apple Silicon (MPS)**：M1/M2/M3 系列 GPU 加速
- [x] **纯 CPU 模式**：支持无显卡用户使用，速度换可用性
- [x] **自动设备检测**：根据硬件自动选择最优设备和模型

### 5. 功能增强

- [ ] WebUI 国际化（i18n）
- [ ] Docker 一键部署
- [x] 更多视频格式支持（MP4、AVI、MOV、FLV 等）
- [ ] 字幕编辑器：实时预览和手动修正
- [ ] 批量处理进度追踪和断点续传优化

欢迎提交 Issue 或 PR 参与贡献！

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 致谢

- [FunASR](https://github.com/modelscope/FunASR) - SenseVoice 语音识别
- [Stable-Whisper](https://github.com/jianfch/stable-ts) - Whisper 稳定版
- [DeepSeek](https://www.deepseek.com/) - 大模型翻译
- [Gradio](https://www.gradio.app/) - WebUI 框架