> Offline example: all model responses and scores below are synthetic test outputs. Audio files contain test pulses, not speech.

# Listen2Serve 评测报告

## 运行溯源

| 溯源项 | 值 |
|---|---|
| run_id | offline |
| 代码版本 | 57633a48（含未提交改动） |
| 数据版本 | v6.5 |
| 被测模型 | dashscope/offline-single |
| 演绎模型 | openai/offline-simulator |
| 裁判模型 | openai/offline-judge（voice: openai/offline-voice） |
| TTS | dashscope/qwen-audio-3.0-tts-plus |
| 生成时间 | 2026-10-17T12:06:05+0800 |

总样本数：1
完整通话：1；端点截断、关键轮可评：0。
KeyTurnPass / VoicePass 包含已完整取得关键轮的截断通话；FlowRate / TaskScore 仅汇总完整通话，截断通话的全程指标为 N/A。
有效分母：KeyTurnPass=1，VoicePass=1，FlowRate=1，TaskScore=1。

## 总体指标

**主指标**

| 指标 | 回答的问题 | 值 |
|---|---|---|
| KeyTurnPass（关键轮应对通过率，0-1） | 关键轮策略是否回应用户状态 | 100.0% |
| VoicePass（声音通过率，0-1） | 三维均分 ≥3.5 且各维 ≥3 | 100.0% |
| KeyTurnScore（弦外之音量规均分，MOS 1-5） | 对用户真实状态的识别与承接质量 | 4.00 |
| FlowRate（过程清单命中比例，部分计 0.5，0-1） | 全对话必需动作的完成程度 | 0.00 |
| TaskScore（终局结果，0-2 归一 0/0.5/1） | 对话任务的达成程度 | 0.00 |
| VoiceMean（三维均分，MOS 1-5） | 角色匹配、状态适配和自然度的平均分 | 4.00 |
| Voice 角色声音匹配 / 状态声音适配 / 自然度 | — | 4.00 / 4.00 / 4.00 |

**质量监控量（不算被测成绩）**

| 指标 | 值 |
|---|---|
| VocabViolationRate（用户话术词表，全轮） | 0.0% |

## 分层结果（角色 × 提示条件 × 动态）

| 角色 | 提示条件 | 措辞 | 动态 | N | 完整/截断 | Flow/Task N | KeyTurnPass | VoicePass | KeyTurnScore | FlowRate | TaskScore | VoiceMean | VocabViolation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hotline | N/E | neutral | all_positive | 1 | 1/0 | 1/1 | 100.0% | 100.0% | 4.00 | 0.00 | 0.00 | 4.00 | 0.0% |