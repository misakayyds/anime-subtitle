# AnimeTranslator - Anime Smart Translation/Proofreading Tool

**English** | [简体中文](README_CN.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A highly automated, low-manual-intervention tool for generating bilingual anime subtitles (ASS). This project integrates advanced Voice Activity Detection (fsmn-vad), Audio Event Classification (SenseVoice), Speech Recognition (Stable-Whisper), and LLM Translation with Correction (DeepSeek API). Through a four-stage "Phonetic Alchemy" pipeline, it batch-converts raw video files into styled, high-quality Chinese-Japanese bilingual subtitles at high speed.

## Installation

### System Requirements

- Python 3.10+
- **GPU (Recommended)**:
  - NVIDIA GPU (RTX 3060 Ti or equivalent recommended) + CUDA 11.8+
  - AMD GPU (RX 6000/7000 series) + ROCm 6.x (Linux only)
  - Apple Silicon Mac (M1/M2/M3 series)
- **CPU Mode**: Supported without GPU, but slower (~10-20x longer)
- **ffmpeg** (Required for audio extraction)

### Resource Consumption

| Device Type | RAM | VRAM | Whisper Model |
|-------------|-----|------|---------------|
| NVIDIA GPU (10GB+) | 3 GB | 10 GB+ | large-v3 |
| NVIDIA GPU (6-10GB) | 3 GB | 6-10 GB | medium |
| AMD GPU (ROCm) | 3 GB | 8 GB+ | medium |
| Apple Silicon | 8 GB+ | Shared | medium |
| CPU | 8 GB+ | - | small |

> Actual consumption may vary based on video length and concurrency settings.

### Installation Steps

```bash
# 1. Clone the project
git clone https://github.com/misakayyds/AnimeTranslator.git
cd AnimeTranslator

# 2. Create virtual environment
python -m venv env

# 3. Activate virtual environment
# Windows:
env\Scripts\activate
# Linux/Mac:
source env/bin/activate

# 4. Install PyTorch (Choose based on your device)
# NVIDIA GPU (CUDA 12.8):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
# NVIDIA GPU (CUDA 12.1):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
# NVIDIA GPU (CUDA 11.8):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
# AMD GPU (ROCm 6.2, Linux only):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
# Apple Silicon Mac (MPS):
pip install torch torchaudio
# CPU only (no GPU):
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# 5. Install AnimeTranslator
pip install -e .

# 6. Configure API Key
copy .env.example .env
# Edit .env file and fill in your DeepSeek API Key
```

### Install ffmpeg

This project requires ffmpeg for audio extraction.

**Windows:**
```bash
# Option 1: Using winget
winget install ffmpeg

# Option 2: Using Chocolatey
choco install ffmpeg

# Option 3: Manual download
# Download from https://www.gyan.dev/ffmpeg/builds/, extract and add bin directory to PATH
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update && sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Verify installation:**
```bash
ffmpeg -version
```

### Get DeepSeek API Key

1. Visit [DeepSeek Platform](https://platform.deepseek.com/)
2. Register and login
3. Create a new API Key on the API Keys page
4. Fill the API Key in `DEEPSEEK_API_KEY` in the `.env` file

## Usage

### Option 1: WebUI (Recommended)

```bash
# Ensure you're in the project directory with virtual environment activated
animetranslator webui

# Or using Python module
python -m animetranslator webui

# Custom port
animetranslator webui --port 8080

# Generate public share link
animetranslator webui --share
```

After startup, open http://127.0.0.1:7860 in your browser

### Option 2: Background Watcher

```bash
# Permanent monitoring mode
animetranslator watch

# Auto-shutdown mode (shutdown 60 seconds after completion)
animetranslator watch --shutdown
```

### Workflow

1. Place video files in the `Input/` directory (supports nested subfolders)
   - Supported formats: `.mkv`, `.mp4`, `.avi`, `.mov`, `.flv`, `.wmv`, `.webm`
2. Run `animetranslator webui` or `animetranslator watch`
3. Wait for processing to complete
4. Get bilingual `.ass` subtitle files in the `Output/` directory

## Core Architecture: Phonetic Alchemy

```
Raw Video File
    │
    ▼
📡 Stage 1: Resonant Perception (SenseVoice)
    │  fsmn-vad precision ranging → start/end time for each audio segment
    │  SenseVoice label recognition → BGM / Speech / MUSIC classification
    │
    ▼
🚧 Stage 2: Impurity Separation (Filtering)
    │  OP/ED continuity detection (85~95s continuous music → discard whole block)
    │  Pure music filtering (BGM/MUSIC label + no text → discard)
    │
    ▼
🎯 Stage 3: Essence Extraction (Whisper)
    │  Tensor slicing (in-memory, zero disk IO)
    │  Feed each segment to Stable-Whisper Large-v3
    │  Smart sentence breaking + deduplication + breath filler filtering
    │
    ▼
🔬 Stage 4: Transmutation (Quality Check + Translation)
    │  no_speech_prob > 0.7 → discard (ambient noise disguise)
    │  compression_ratio > 2.8 → discard (repetition hallucination)
    │
    ▼
✅ Output _alignment.json draft → DeepSeek translation → bilingual ASS subtitle
```

## Key Features

1. **SenseVoice Smart Label Classification**: Replaces traditional Demucs vocal separation, directly labeling BGM/Speech/MUSIC on original audio, preserving full context. GPU time reduced from minutes to seconds.

2. **OP/ED Impurity Auto-Detection**: Anime-specialized feature that automatically detects continuous music segments (85~95 seconds) within the first and last 5 minutes, discarding entire blocks to prevent Whisper from transcribing lyrics.

3. **In-Memory Tensor Slicing Zero IO**: Audio waveform loaded to VRAM once, all subsequent segments extracted via Tensor Slicing with 0.3s padding on both ends to prevent audio clipping, avoiding repeated ffmpeg encoding/decoding.

4. **Four-Layer Hallucination Prevention**: SenseVoice labels → OP/ED filtering → Whisper no_speech_prob / compression_ratio quality check → breath filler regex filtering, intercepting at every layer.

5. **DeepSeek Context-Aware Translation**:
   - Infers and corrects homophones, character names, chuunibyou coined words and spells based on pronunciation
   - Contextually translates onomatopoeia based on actions, emotions, and even physiological reactions, preserving visual gags
   - Intelligently filters out common Whisper hallucinations like "Thanks for watching" and "Please subscribe"

6. **Smart Hardware & Cache Management**:
   - Three engines resident in VRAM: fsmn-vad + SenseVoice + Stable-Whisper large model, no need to reload per episode
   - Periodic VRAM cleanup strategy, avoiding OOM ghost bugs

7. **Structured Logging System**:
   - Unified logging with both console and file output
   - Log files saved in `Output/logs/` directory, format: `animetranslator_YYYYMMDD_HHMMSS.log`
   - Timestamped log format for easier troubleshooting

8. **Automatic Configuration Validation**:
   - Automatic validity check of configuration items at startup
   - API Key format validation, parameter range checking
   - Clear problem prompts, reducing configuration errors

## Configuration

Edit the `.env` file in the project root:

```env
# DeepSeek API key configuration (required)
DEEPSEEK_API_KEY=sk-xxxxxx

# Device configuration (optional: auto/cuda/mps/cpu, default auto)
# auto - Auto-detect, priority: CUDA > MPS > CPU
# cuda - Force NVIDIA GPU
# mps  - Force Apple Silicon GPU
# cpu  - Force CPU (slower)
DEVICE=auto

# Whisper model size (optional, auto-selected based on device)
# large-v3 / medium / small
# WHISPER_MODEL=large-v3

# Async concurrent LLM translation tasks (recommended 3-5)
MAX_API_WORKERS=3

# Engine VRAM residency config (set how many episodes before clearing model VRAM, recommended 1-5)
ALIGNMENT_BATCH_SIZE=3

# Quality check thresholds (usually no need to modify)
NO_SPEECH_PROB_THRESHOLD=0.7
COMPRESSION_RATIO_THRESHOLD=2.8
```

## Directory Structure

```
AnimeTranslator/
├── pyproject.toml           # Package config
├── README.md
├── LICENSE
├── .env.example             # Config template
├── .gitignore
├── requirements.txt         # Dependencies reference
├── src/
│   └── animetranslator/     # Core code
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py           # CLI entry
│       ├── config.py        # Config management
│       ├── logger.py        # Logging module
│       ├── device.py        # Device management
│       ├── alignment.py     # Alignment engine
│       ├── translation.py   # Translation module
│       ├── watcher.py       # Background watcher
│       └── webui.py         # WebUI interface
├── tests/
├── Input/                   # Input directory (videos to process)
├── Output/                  # Output directory (generated subtitles + logs)
│   └── logs/                # Log files directory
└── env/                     # Virtual environment (user created)
```

## FAQ

### Q: How to cancel auto-shutdown?

During the auto-shutdown countdown, press `Win+R` and enter `shutdown /a` to cancel.

### Q: First run is slow?

On first run, FunASR automatically downloads SenseVoice-Small and fsmn-vad models from ModelScope (~1-2GB), please wait patiently.

### Q: What video formats are supported?

Multiple common video formats are supported: `.mkv`, `.mp4`, `.avi`, `.mov`, `.flv`, `.wmv`, `.webm`

### Q: How to check processing progress?

- WebUI mode: View real-time logs in browser
- Watch mode: View output logs in terminal

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
black src/

# Code linting
ruff check src/
```

## Roadmap

See [docs/ROADMAP.md](docs/ROADMAP.md) for detailed development roadmap

### 1. More Model Support

- [ ] **Lightweight model options**: Support Whisper Medium/Small, reducing VRAM requirement to 4GB
- [ ] **Alternative ASR engines**: Integrate WhisperX, Moonshine and other alternatives
- [ ] **Multi-language support**: Extend to Korean, English and other language subtitle generation

### 2. More Translation API Support

- [ ] **OpenAI GPT**: Support GPT-4o, GPT-4-turbo models
- [ ] **Claude**: Support Anthropic Claude series
- [ ] **Google Gemini**: Support Google's latest LLM
- [ ] **Qwen**: Support Alibaba Cloud Qwen API
- [ ] **GLM**: Support Zhipu ChatGLM API
- [ ] **OpenAI-compatible interfaces**: Support any OpenAI-compatible API service

### 3. Local LLM Support

- [ ] **Ollama integration**: Support running local models via Ollama
- [ ] **llama.cpp**: Support GGUF format quantized models
- [ ] **vLLM**: Support high-performance local inference
- [ ] **Recommended models**: Qwen2.5, Llama3, GLM-4 and other open-source model adaptations
- [ ] **Offline mode**: Complete offline operation solution

### 4. Multi-Device Support

- [x] **NVIDIA GPU (CUDA)**: RTX series, CUDA 11.8+
- [x] **AMD GPU (ROCm)**: RX 6000/7000 series, Linux only, requires ROCm driver and PyTorch ROCm version
- [x] **Apple Silicon (MPS)**: M1/M2/M3 series GPU acceleration
- [x] **CPU-only mode**: Support for users without GPU, trading speed for accessibility
- [x] **Auto device detection**: Automatically select optimal device and model based on hardware

### 5. Feature Enhancements

- [x] WebUI internationalization (i18n)
- [ ] Docker one-click deployment
- [x] More video format support (MP4, AVI, MOV, FLV, etc.)
- [ ] Subtitle editor: Real-time preview and manual correction
- [ ] Batch processing progress tracking and resume optimization

Contributions via Issues or PRs are welcome!

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FunASR](https://github.com/modelscope/FunASR) - SenseVoice speech recognition
- [Stable-Whisper](https://github.com/jianfch/stable-ts) - Stable Whisper
- [DeepSeek](https://www.deepseek.com/) - LLM translation
- [Gradio](https://www.gradio.app/) - WebUI framework