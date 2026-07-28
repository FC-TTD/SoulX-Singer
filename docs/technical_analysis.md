# SoulX-Singer 项目功能分析与 Prompt 最佳实践

## 核心架构概览

SoulX-Singer 包含两个独立的推理体系：

```
SoulX-Singer
├── SVS (歌声合成) — model.pt
│   ├── 输入：Prompt音频 + Prompt元数据 + Target元数据
│   └── 控制模式：melody-controlled / score-controlled
│
└── SVC (歌声转换) — model-svc.pt
    ├── 输入：Prompt音频 + Target音频（直接波形）
    └── 无需歌词/MIDI转录
```

---

## 一、SVS — 歌声合成（Singing Voice Synthesis）

### 功能定位
从**歌词+旋律信息**生成目标歌声，同时"克隆"Prompt音频的音色。

### 工作流程

```
Prompt音频 → 预处理 → prompt metadata.json ┐
Target音频 → 预处理 → target metadata.json ┼→ 模型推理 → 合成歌声
Prompt WAV（参考音色）                      ┘
```

### 两种控制模式

| 模式 | 输入依赖 | 原理 | 适用场景 |
|------|----------|------|----------|
| **melody-controlled** | F0 轨迹（连续音高曲线） | 直接用目标音频的实际演唱曲线驱动 | 保留原唱者的情感波动、颤音等细节 |
| **score-controlled** | MIDI note_pitch（离调音符） | 按乐谱音高严格生成 | 需要精确音高；MIDI编辑后使用 |

### 模型架构（`soulxsinger.py`）

```
Phoneme Embedding   ─┐
Note Pitch Embedding ─┼→ ConvNeXtV2 PreFlow → 帧级特征扩展 → CFM Flow Matching Decoder → Vocoder → 音频
Note Type Embedding  ─┤                ↑
F0 Embedding ────────┘                Mel Spectrogram (提取自Prompt WAV)
```

- **Flow Matching Decoder**：条件流匹配扩散（CFM），`n_steps=32`，`cfg=3`
- **Vocoder**：将梅尔频谱转为 24kHz 波形
- **自动变调（auto_shift）**：比较 Prompt 和 Target 的中位音高后自动计算半音偏移

---

## 二、SVC — 歌声转换（Singing Voice Conversion）

### 功能定位
将目标歌声（含旋律/歌词）转换为Prompt音色，**无需转录**。

### 工作流程

```
Prompt音频 → 预处理（仅F0/人声分离）→ prompt.wav + prompt_f0.npy ┐
Target音频 → 预处理（仅F0/人声分离）→ target.wav + target_f0.npy ┼→ SVC推理 → 转换歌声
```

### SVC模型架构（`soulxsinger_svc.py`）

```
Whisper Encoder（提取内容特征）─┐
F0 Embedding ──────────────────┼→ CFM Decoder → Vocoder
Prompt Mel Spectrogram ────────┘
```

**关键差异**：SVC 用 **Whisper Encoder** 提取语义内容特征（语言无关），而非音素标注。

### 长音频分段推理
当目标超过 30 秒时，自动按 F0 零点（静音段）分割为 15–30 秒的片段，带 overlap 平滑拼接。

---

## 三、预处理流水线（Preprocess Pipeline）

### 完整步骤

```
原始音频
    ↓
1. 人声分离（Lead Vocal Separation，mel-band-roformer-karaoke）
    ↓
2. 去混响（Vocal Dereverberation，dereverb_mel_band_roformer）
    ↓
3. F0提取（RMVPE）+ VAD（声音活动检测）
    ↓
4. 歌词转录（中文：Paraformer；英文：Parakeet-tdt-0.6b-v2）
    ↓
5. 音符转录（ROSVOT）
    ↓
metadata.json（SVS）/ vocal.wav + vocal_f0.npy（SVC）
```

### Metadata JSON 格式详解

每个段落（segment）是一个 JSON 对象，字段含义如下：

```json
{
  "index": "vocal_320_10687",     // 片段标识（起止时间ms）
  "language": "Mandarin",          // 语种
  "time": [320, 10687],            // 在原音频中的起止时间（ms）
  "duration": "0.23 0.34 ...",     // 每个音符的时长（秒），空格分隔
  "text": "<SP> 除 了 想 你 ...",   // 歌词文字，<SP>代表静音/停顿
  "phoneme": "<SP> zh_chu2 ...",   // 音素序列（带声调数字后缀）
  "note_pitch": "0 62 65 67 ...",  // MIDI 音高（0=静音，MIDI标准）
  "note_type": "1 2 2 2 3 ...",    // 音符类型（1=onset，2=sustain，3=end）
  "f0": "0.0 0.0 294.1 ..."        // F0 轨迹（Hz），每 20ms 一帧，0代表无声
}
```

---

## 四、Prompt 最佳实践

### 4.1 Prompt 音频选择要点

> Prompt 音频是音色克隆的核心，质量直接决定输出音色的相似度。

| 要素 | 建议 | 原因 |
|------|------|------|
| **时长** | 5–30秒（默认限制30秒） | 过短则音色信息不足；过长会截断 |
| **干净程度** | 纯人声（无伴奏）最佳 | 伴奏会污染音色提取；若有伴奏需开启 `vocal_sep` |
| **去混响** | 干声（无回响）更好 | 混响会影响音色辨别度 |
| **音域选择** | 选取目标曲目音域范围内的演唱段落 | 使自动变调更精准 |
| **内容稳定性** | 选取稳定演唱段落，避免假声/气声转换处 | 过渡段音色不稳定 |
| **语言** | 无限制（跨语言）| 模型通过音色解耦支持跨语言合成 |

### 4.2 Prompt 元数据优化

```
自动转录质量链：歌声质量 → ASR准确率 → 音符对齐 → 合成质量
```

**强烈建议手动校正 Prompt Metadata**：
1. 运行 `preprocess.sh` 生成 `metadata.json` 和 `vocal.mid`
2. 将 MIDI 导入 [SoulX-Singer-Midi-Editor](https://huggingface.co/spaces/Soul-AILab/SoulX-Singer-Midi-Editor)
3. 校正：歌词错误、音高偏移、时长对齐
4. 导出校正后的 MIDI → 转回 `metadata.json`

### 4.3 Target 元数据优化

Target 决定"唱什么"，其元数据质量更关键：

**score-controlled 模式**（推荐精细控制时使用）：
- MIDI 音高需精确对应乐谱
- 歌词需与音符时间对齐
- 建议使用 Midi-Editor 手动标注后再生成 metadata

**melody-controlled 模式**（适合快速转换）：
- F0 轨迹自动提取，质量取决于原始音频
- Target 音频建议使用纯人声（开启 `target_vocal_sep=yes`）

### 4.4 关键推理参数

| 参数 | 默认值 | 调节建议 |
|------|--------|----------|
| `n_steps` | 32 | 增大到 50–100 可提升质量，代价是速度 |
| `cfg` | 3 | **SVS**：增大→音色相似度↑，但易失真；建议 2–4<br>**SVC**：建议 1–3 |
| `auto_shift` | True | 建议保持开启；Prompt和Target音域差距大时尤为重要 |
| `pitch_shift` | 0 | 若auto_shift效果不佳，手动指定半音数（±36） |
| `seed` | 12306/42 | 固定seed保证可复现；换seed多采样挑选 |
| `use_fp16` | False | 用`--fp16`开启，减少显存约一半，适合拮据显存场景 |

### 4.5 SVC 特有参数

| 参数 | 说明 |
|------|------|
| `prompt_vocal_sep` | Prompt 有伴奏时必须开启 |
| `target_vocal_sep` | Target 有伴奏时必须开启（SVC默认True） |
| `auto_mix_acc` | 自动将分离出的伴奏与转换结果混合，便于直出成品 |

---

## 五、常见问题与调优策略

### 问题1：合成声音音色相似度低
**原因**：Prompt 音频质量差（有噪音/混响/伴奏）
**解决**：
- 开启 `prompt_vocal_sep=yes` + Prompt 音频预先用专业工具降噪
- 使用更长且稳定的 Prompt 片段

### 问题2：合成歌声音高偏离
**原因**：Prompt 和 Target 音域差距过大，auto_shift 不够精确
**解决**：
- 手动设置 `pitch_shift`（正数=升调，负数=降调）
- score-controlled 模式优先（音高确定）

### 问题3：歌词吐字不清晰
**原因**：Target Metadata 中 phoneme 与 duration 对齐不准确
**解决**：
- 使用 Midi-Editor 手动调整音符起止时间
- 重新生成 metadata

### 问题4：合成结果失真/电音感
**原因**：cfg 值过高，Flow Matching 过度条件化
**解决**：降低 cfg（1.5–2.5），或增加 n_steps

### 问题5：长音频（SVC）中段音色不稳定
**原因**：分段推理的 overlap 拼接缝隙
**解决**：分段单独转换（手动切割为 30 秒以内的片段）

---

## 六、CLI 快速参考

### SVS (歌声合成)
```bash
python -m cli.inference \
    --device cuda \
    --model_path pretrained_models/SoulX-Singer/model.pt \
    --config soulxsinger/config/soulxsinger.yaml \
    --prompt_wav_path <prompt.wav> \
    --prompt_metadata_path <prompt_metadata.json> \
    --target_metadata_path <target_metadata.json> \
    --phoneset_path soulxsinger/utils/phoneme/phone_set.json \
    --save_dir outputs/ \
    --control score \   # 或 melody
    --auto_shift \
    --pitch_shift 0 \
    --fp16
```

### SVC (歌声转换)
```bash
python -m cli.inference_svc \
    --device cuda \
    --model_path pretrained_models/SoulX-Singer/model-svc.pt \
    --config soulxsinger/config/soulxsinger.yaml \
    --prompt_wav_path <prompt.wav> \
    --target_wav_path <target.wav> \
    --prompt_f0_path <prompt_f0.npy> \
    --target_f0_path <target_f0.npy> \
    --save_dir outputs/ \
    --auto_shift \
    --pitch_shift 0 \
    --fp16
```

### 元数据转 MIDI（用于 Midi-Editor 校正）
```bash
python -m preprocess.tools.midi_parser \
    --meta2midi \
    --meta metadata.json \
    --midi vocal.mid

# 编辑后转回
python -m preprocess.tools.midi_parser \
    --midi2meta \
    --midi vocal_edited.mid \
    --meta edit_metadata.json \
    --vocal vocal.wav
```
