# Qwen3-ASR 智能语音识别工具 (AIBL 整合 GUI 封装版)

一键图形化客户端, 底层使用阿里通义千问开源的 **Qwen3-ASR-1.7B** 高精度语音识别大模型
与配套的 **Qwen3-ForcedAligner-0.6B** 时间戳对齐模型. 全程本地离线运行, 不联网调用云端
API, 不限次数, 不限时长, 音频和模型都在本机硬盘处理, 隐私性强.

## 环境要求

- macOS (Apple Silicon 推荐, 也支持 Intel CPU)
- Python 3.10+ (脚本会在首次运行时自动创建本地 venv)
- ffmpeg (用于解码 MP4/MP3/.strm 等容器, `brew install ffmpeg`)

## 快速使用

1. 把音频/视频文件或 `.strm` 拖到「批量转录」标签页的文件列表.
2. 在「设置」标签页点击「立即下载所选 ASR 模型」和「下载对齐器」(首次约 5GB), 然后
   「加载 / 重新加载模型」.
3. 回到「批量转录」, 点「开始转录」, 每条识别结果会实时显示在底部日志面板.

## 核心功能

- **文件批量转录**: 支持 MP4/MP3/.strm/网址等输入, 自动生成带精准时间轴的 SRT. 可设置
  单行字数、单段时长, 关闭对齐器则在低端/CPU 设备上提速.
- **实时语音听写**: 麦克风实时收音转文字, 内置能量端点检测 (灵敏度 1-5 级, 停顿/截断
  均可调), 适合会议、直播字幕.
- **可调模型参数**:
  - 模型切换 1.7B 高精度 / 0.6B 轻量版
  - 设备 自动 / Apple GPU (MPS) / CUDA / CPU
  - 精度 自动 (MPS→float16, CUDA→bfloat16) / float16 / bfloat16 / float32
  - 语种自动识别 (auto), 支持中英日韩、粤语、方言等 30 语言 + 22 中文方言
  - 是否开启时间戳对齐 (字符/单词级, 保证字幕时间轴精准匹配人声)

## 项目结构

```
Qwen3-ASR-Tool/
  start.command          # macOS 双击启动 (自动建 venv + 装依赖)
  requirements.txt
  src/
    app.py               # 入口
    config.py            # 持久化设置 (config.json)
    audio.py             # ffmpeg 解码 / .strm 解析
    model_manager.py     # 模型加载、设备/精度选择、本地下载
    segment.py           # 批量分段 (能量边界优化)
    srt.py               # SRT 生成与按字数/时长重切
    transcriber.py       # 批量转录 worker (QThread, 实时日志信号)
    realtime.py          # 实时听写 worker (PyAudio + 能量端点)
    gui/
      main_window.py     # 顶层窗口 + 深色主题 + 状态栏
      batch_widget.py    # 批量转录 UI
      realtime_widget.py# 实时听写 UI
      settings_widget.py# 设置 / 下载 / 加载 UI
      log_panel.py       # 实时日志面板
  scripts/fetch.py       # 命令行模型下载工具
  models/                # 自动下载的模型权重 (约 5GB)
  output/                # 生成的 SRT 字幕
```

## 命令行用法

```bash
# 1) 首次设置 (如已用 start.command 则省略)
source venv/bin/activate
pip install -r requirements.txt

# 2) 手动预下载模型 (避免首次 GUI 内下载)
python scripts/fetch.py --source modelscope \
  --models Qwen/Qwen3-ASR-1.7B Qwen/Qwen3-ForcedAligner-0.6B Qwen/Qwen3-ASR-0.6B

# 3) 启动 GUI
python -m src.app
```

## 说明

- 实时听写在 Apple Silicon/CPU 上采用「端点→离线整段识别」策略 (Qwen3-ASR 原生流式
  仅 vLLM+CUDA 支持), 延迟为「一次发音时长 + 模型解码秒数」, 仍为近实时且准确.
- 所有字幕均为标准 SRT, 直接适配 Jellyfin / EMBY / 剪辑软件.
- 默认设备在 Apple Silicon 上选 MPS, 精度 float16 (MPS 对 bfloat16 支持不完整).
  如遇 MPS 报错请在「设置」切回 CPU.
