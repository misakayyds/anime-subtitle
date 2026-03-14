# Changelog

All notable changes to this project will be documented in this file.

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