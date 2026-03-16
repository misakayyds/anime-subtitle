# Changelog

All notable changes to this project will be documented in this file.

## [1.2.2] - 2026-03-16

### Added

- **Internationalization (i18n) support**: Full multi-language support for global users
  - New `i18n.py` module with translation function `tr()`
  - Locale files: `locales/zh_CN.json` (Chinese) and `locales/en_US.json` (English)
  - Language detection: Environment variable `LANG`/`LC_ALL` → `.env` config → default Chinese
  - WebUI language switcher: Dropdown menu for real-time language switching
  - CLI help text translations
  - Terminal output translations for device detection, config validation, and watcher

- **Bilingual README**: Documentation now available in both languages
  - `README.md` - English version (default)
  - `README_CN.md` - Chinese version
  - Language switch links at the top of each file

- **New configuration option**: `ANIME_TRANSLATOR_LANG` in `.env`
  - Supported values: `zh_CN` (Chinese), `en_US` (English)

### Changed

- WebUI now uses `tr()` function for all UI text, labels, and messages
- CLI help text now supports internationalization
- Device detection and config validation messages now in English (technical output)
- Watcher mode messages now in English (technical output)

## [1.2.1] - 2026-03-14

### Added

- **Multi-format video support**: Extended video format support beyond MKV
  - Supported formats: `.mkv`, `.mp4`, `.avi`, `.mov`, `.flv`, `.wmv`, `.webm`
  - New configuration constant `SUPPORTED_VIDEO_EXTENSIONS` in `config.py`
  - WebUI upload component now accepts all supported formats
  - File scanner automatically detects all supported video formats

### Changed

- Updated file scanning logic in `watcher.py` and `webui.py` to use centralized format list
- Improved file status display messages in WebUI

## [1.2.0] - 2026-03-14

### Added

- **Structured logging system**: New `logger.py` module with unified logging
  - Dual output: console + file persistence
  - Log files saved to `Output/logs/animetranslator_YYYYMMDD_HHMMSS.log`
  - Timestamp format: `[HH:MM:SS] message`

- **Configuration validation**: New `validate_config()` function in `config.py`
  - Validates `DEEPSEEK_API_KEY` presence and format
  - Validates `DEVICE` value (auto/cuda/rocm/mps/cpu)
  - Validates `WHISPER_MODEL` value (large-v3/medium/small)
  - Validates numeric ranges for `MAX_API_WORKERS`, `ALIGNMENT_BATCH_SIZE`, thresholds
  - Warnings displayed at startup without blocking execution

### Changed

- Replaced all `print()` calls with structured logging functions (`log_info`, `log_warning`, `log_error`)
- Unified version number across `__init__.py` and `pyproject.toml`
- Fixed placeholder GitHub URLs in `pyproject.toml` (`your-username` → `misakayyds`)

## [1.1.1] - 2026-03-14

### Fixed

- **WebUI freeze during VRAM cleanup**: Removed `torch.cuda.synchronize()` that could cause deadlock in multi-threaded environment when processing multiple episodes

## [1.1.0] - 2026-03-14

### Added

- **Multi-device support**: Extended GPU compatibility beyond NVIDIA
  - **NVIDIA GPU (CUDA)**: Auto-select Whisper model based on VRAM size
  - **AMD GPU (ROCm)**: Support for RX 6000/7000 series (Linux only)
  - **Apple Silicon (MPS)**: Support for M1/M2/M3 series GPU acceleration
  - **CPU mode**: Fallback option for systems without GPU (slower)

- **Smart device detection**: Automatic device detection with priority: CUDA/ROCm > MPS > CPU

- **Adaptive model selection**: Auto-recommend Whisper model based on device capabilities
  - 10GB+ VRAM → large-v3
  - 6-10GB VRAM → medium
  - MPS/CPU → medium/small

- **New configuration options**:
  - `DEVICE`: Specify device type (auto/cuda/rocm/mps/cpu)
  - `WHISPER_MODEL`: Optional manual override for Whisper model size

- **New module**: `device.py` - Centralized device management


### Changed

- Updated `.env.example` with new device configuration options
- Updated documentation with multi-device installation instructions
- Improved memory cleanup to support all device types

### Installation

**AMD GPU users (Linux only)**:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

**Apple Silicon users**:
```bash
pip install torch torchaudio
```

## [1.0.0] - Initial Release

### Features

- Four-stage pipeline: SenseVoice → Filter → Whisper → Translate
- SenseVoice smart audio classification (BGM/Speech/MUSIC)
- OP/ED automatic detection and filtering
- Stable-Whisper Large-v3 for accurate transcription
- DeepSeek API for high-quality translation
- WebUI and watch mode for batch processing
- Hallucination prevention with multiple quality checks