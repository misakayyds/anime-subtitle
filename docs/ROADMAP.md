# AnimeTranslator 开发路线图

> 记录项目后续开发计划，按模块分类，标注优先级和状态。

---

## 📊 优先级说明

| 标记 | 含义 |
|------|------|
| P0 | 核心功能，近期重点开发 |
| P1 | 重要功能，计划中 |
| P2 | 增强功能，待定 |
| ✅ | 已完成 |

---

## 🌐 翻译模块 (Translation)

### P0 - 多服务商支持

**目标**：支持多个翻译 API 服务商，避免单点故障

**计划支持的服务商**：
- [x] DeepSeek（当前支持）
- [ ] OpenAI GPT-4o / GPT-4-turbo
- [ ] Anthropic Claude
- [ ] Google Gemini
- [ ] 阿里云通义千问 (Qwen)
- [ ] 智谱 ChatGLM
- [ ] OpenAI 兼容接口（支持任意自建服务）

**配置设计**：
```env
TRANSLATION_PROVIDER=deepseek
TRANSLATION_FALLBACK=openai,claude

DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_MODEL=deepseek-chat

OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o

CLAUDE_API_KEY=sk-xxx
CLAUDE_MODEL=claude-3-5-sonnet
```

---

### P0 - 改进崩溃降级逻辑

**当前问题**：
- 3次重试失败后直接返回原文，标注 `【API失败暂缺】`
- 无备用服务切换机制

**改进方案**：
```
主服务失败
  → 重试 3 次
  → 切换备用服务 1（如 OpenAI）
  → 重试 2 次
  → 切换备用服务 2（如 Claude）
  → 最终降级返回原文
```

**配置设计**：
```env
TRANSLATION_MAX_RETRIES=3
TRANSLATION_FALLBACK_ENABLED=true
TRANSLATION_FALLBACK_ORDER=openai,claude
TRANSLATION_TIMEOUT=120
```

---

### P1 - 服务端校对模式

**目标**：主服务翻译 → 校对服务润色，提升翻译质量

**使用场景**：
- DeepSeek 翻译（性价比高）→ GPT-4 校对（质量高）
- 适合对翻译质量要求较高的用户

**配置设计**：
```env
PROOFREAD_ENABLED=true
PROOFREAD_PROVIDER=openai
PROOFREAD_MODEL=gpt-4o
```

**流程**：
```
DeepSeek 翻译 → GPT-4 校对润色 → 输出最终字幕
```

---

### P2 - 多服务端同步调用

**目标**：同时调用多个 API，智能选择最佳结果

**模式**：
| 模式 | 说明 |
|------|------|
| `race` | 竞速模式，谁先返回用谁 |
| `vote` | 投票模式，多数结果一致则采纳 |
| `best` | 评分模式，用另一个 API 评估质量选最优 |

**配置设计**：
```env
TRANSLATION_PARALLEL_MODE=race
TRANSLATION_PARALLEL_PROVIDERS=deepseek,openai
```

---

## 🎙️ 语音识别模块 (ASR)

### P1 - 轻量级模型选项

- [ ] 支持 Whisper Medium/Small
- [ ] 降低显存需求至 4GB
- [ ] 配置项：`WHISPER_MODEL=medium|small`

---

### P2 - 多 ASR 引擎支持

- [ ] WhisperX（更快的时间对齐）
- [ ] Moonshine（轻量级替代方案）
- [ ] FunASR Paraformer（阿里开源模型）

---

### P2 - 多语言支持

- [ ] 韩语字幕生成
- [ ] 英语字幕生成
- [ ] 配置项：`SOURCE_LANGUAGE=ja|ko|en`

---

## 🖥️ WebUI 模块

### ✅ 国际化 (i18n)

- [x] 英文界面支持
- [x] 自动检测系统语言
- [x] WebUI 语言切换下拉框
- [x] CLI 帮助文本国际化
- [x] 双语文档 (README)

---

### P2 - 字幕编辑器

- [ ] 实时预览字幕效果
- [ ] 手动修正翻译结果
- [ ] 时间轴调整
- [ ] 保存修改后的字幕

---

### P2 - Docker 部署

- [ ] Dockerfile
- [ ] docker-compose.yml
- [ ] 一键部署脚本

---

## 📁 文件格式支持

### ✅ 已完成

- [x] MKV
- [x] MP4
- [x] AVI
- [x] MOV
- [x] FLV
- [x] WMV
- [x] WebM

### P2 - 字幕格式支持

- [ ] SRT 输出格式
- [ ] SRT 输入（直接翻译现有字幕）
- [ ] VTT 输出格式

---

## 🔧 配置与日志

### ✅ 已完成

- [x] 结构化日志系统
- [x] 配置自动验证
- [x] 日志文件持久化

### P2 - 配置增强

- [ ] WebUI 配置热更新
- [ ] 配置预设（快速切换不同服务商组合）

---

## 🏗️ 架构优化

### P1 - 模块化重构

**目标**：解耦各模块，支持整合包和功能扩展

**当前问题**：
- `alignment.py` 508行，四阶段逻辑全部堆在一个类
- `webui.py` UI与业务逻辑耦合
- `translation.py` DeepSeek 硬编码
- 无统一 Pipeline 抽象

**优化方向**：
- [ ] 抽取 Provider 基类（ASR/翻译/VAD）
- [ ] 拆分 alignment.py 为独立 stages/
- [ ] 添加 Pipeline 编排器 + 事件系统
- [ ] 分离 UI/业务逻辑

---

### P2 - 插件系统

**目标**：支持第三方插件扩展

- [ ] 插件加载机制
- [ ] 插件 API 接口
- [ ] 插件配置管理

---

### P2 - 整合包支持

**目标**：支持配置预设，整合包无需改代码

- [ ] YAML 配置文件支持
- [ ] 配置预设系统（default/quality/fast/offline）
- [ ] 整合包打包脚本

---

## 📋 版本规划

| 版本 | 重点功能 | 状态 |
|------|----------|------|
| v1.2.x | 多格式视频支持、日志系统、国际化 (i18n) | ✅ 已发布 |
| v1.3.0 | 多服务商支持、降级逻辑改进 | 计划中 |
| v1.4.0 | Provider 抽象、校对模式、轻量级模型 | 计划中 |
| v1.5.0 | Pipeline 编排器、多 ASR 引擎、多语言支持 | 计划中 |
| v1.6.0 | 插件系统、整合包配置预设 | 计划中 |
| v2.0.0 | 字幕编辑器、Docker 部署 | 远期规划 |

---

## 🤝 贡献指南

欢迎参与开发！请查看以下方式：

1. 提交 Issue 报告 Bug 或建议新功能
2. 提交 PR 贡献代码
3. 参与文档翻译和改进

---

*最后更新：2026-03-16*