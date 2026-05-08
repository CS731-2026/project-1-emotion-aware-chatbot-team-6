# 智能语音活动检测 (VAD) - 方案 2

## 问题
之前固定 5 秒录音会导致：
- **短句子被拉长** - 说完"停止"，还要等 4 秒
- **长句子被截断** - 复杂问题还没说完就停止了

## 解决方案
使用 **智能语音活动检测 (Voice Activity Detection)**：
- 检测用户停止说话（1+ 秒静音）后**立即停止**，无需等待最大时间
- 保留最小时长（0.5 秒）以避免误触发

## 工作原理

```
User:    "Hi"
System:  [录音开始]
         [0-0.5s] 用户说话 → 重置"静音计时器"
         [0.5-1.5s] 用户说话 → 重置"静音计时器"
         [1.5-2.5s] 沉默（无声音）→ 静音计时器 = 1.0s
         [触发停止] ✓ 检测到 1 秒静音
         [总耗时] 2.5 秒（不是 5 秒！）

User:    "Can you explain how autonomous vehicles work?"
System:  [录音开始]
         [0-3s] 用户说话 → 不断重置"静音计时器"
         [3-4s] 沉默 → 静音计时器 = 1.0s
         [触发停止] ✓ 检测到 1 秒静音
         [总耗时] 4 秒（不是 5 秒的浪费！）
```

## 技术实现

### 1. 基于能量的 VAD（无依赖库）
```python
# 计算音频块的 RMS 能量
rms_energy = np.sqrt(np.mean(chunk**2))

# 如果能量 > 阈值，判定为"有语音"
if rms_energy > 0.02:  # 阈值 (可调)
    last_voice_time = now()  # 重置静音计时器
else:
    silence_duration = now() - last_voice_time
    if silence_duration >= 1.0s:  # 静音超过 1 秒
        stop_recording()
```

### 2. 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vad_enabled` | True | 启用/禁用 VAD |
| `min_duration_seconds` | 0.5 | 最小录音时长（防止过早停止） |
| `silence_duration_seconds` | 1.0 | 检测多少秒静音后停止 |
| `max_duration_seconds` | 8.0 | 绝对最大时长（安全保障） |
| `vad_threshold` | 0.02 | 音频能量阈值（RMS） |

## 命令行使用

```bash
# 默认：8 秒最大，1 秒静音即停止
python -m drivesense

# 定制：10 秒最大，0.5 秒静音即停止（快速反应）
python -m drivesense --max-recording-duration 10.0 --silence-threshold 0.5

# 宽松：15 秒最大，2 秒静音才停止（等更长时间）
python -m drivesense --max-recording-duration 15.0 --silence-threshold 2.0
```

## 配置示例

### 场景 1：快速对话（出租车司机）
```bash
python -m drivesense \
  --max-recording-duration 6.0 \
  --silence-threshold 0.5
```
→ 快速停止，无延迟，适合短命令

### 场景 2：详细问题（需要思考）
```bash
python -m drivesense \
  --max-recording-duration 12.0 \
  --silence-threshold 1.5
```
→ 给用户更多思考时间

### 场景 3：复杂对话（多轮讨论）
```bash
python -m drivesense \
  --max-recording-duration 15.0 \
  --silence-threshold 2.0
```
→ 最宽松，等待用户停顿

## 代码实现位置

### speech.py - 核心 VAD 逻辑
[drivesense/backend/speech.py](drivesense/backend/speech.py#L25) - `record_microphone_audio()` 函数
- 计算音频能量
- 追踪最后一次语音检测时间
- 条件停止

### gui.py - GUI 集成
[drivesense/frontend/gui.py](drivesense/frontend/gui.py#L920) - `start_recording()` 方法
- 从命令行参数读取配置
- 传递给 SpeechWorker

### wake_word.py - 禁用 VAD
[drivesense/backend/wake_word.py](drivesense/backend/wake_word.py#L133) - WakeWordListener & ContinuedConversationListener
- 禁用 VAD（`vad_enabled=False`）
- 因为这些是短的 1 秒监控块，不是用户输入

## 消息输出示例

```
[VAD] Detected 1.5s of silence; stopping early at 3.2s
转录：User input was successfully captured after 3.2 seconds
```

## 故障排查

| 问题 | 原因 | 解决方案 |
|------|------|--------|
| 录音立即停止 | 阈值太高，检测不到声音 | 增加 `--silence-threshold 2.0` |
| 录音太长 | 背景噪声被当作语音 | 确保环境安静，或调整 VAD 阈值 |
| 句子被中断 | 用户有停顿（如"嗯..") | 增加 `--silence-threshold 1.5` |
| 总是录满 8 秒 | VAD 未启用或阈值太低 | 检查 `vad_enabled=True` |

## 性能

- **CPU 开销** - 极小（每 50ms 计算一次 RMS）
- **延迟** - 无（无额外库加载）
- **依赖** - numpy（已有）

## 与其他组件的互动

✅ **WakeWordListener** - 仍用 1 秒固定块  
✅ **ContinuedConversationListener** - 仍用 1 秒固定块  
✅ **SpeechWorker** - 使用智能 VAD (**新**)  
✅ **LLM 输入** - 自动送出（无需等待）  

## 未来改进

- [ ] 实现更精确的 VAD 库（webrtcvad, silero-vad）
- [ ] 学习用户的说话节奏，自动调整静音阈值
- [ ] 频谱分析而不只是 RMS（更准确）
- [ ] 环境自适应（初始化时学习背景噪声水平）
