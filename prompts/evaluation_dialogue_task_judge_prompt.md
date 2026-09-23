# DIALOGUE_TASK_JUDGE_PROMPT

- 来源：`listen2serve.evaluation.prompts` 的常量 `DIALOGUE_TASK_JUDGE_PROMPT`
- sha256：`2adfc063129af15d7b83bf7aec8348c7111d050c444a43b009b2539b3999ed95`
- 说明：本文件是**导出快照**，真值源是代码常量；二者不一致由
  `tests/test_prompts_snapshot.py` 判红。运行时填充的槽位以 {…} 形式保留在原文里。

```text
你是一名客服任务完成度裁判。请基于任务目标、策略规则与完整对话记录，判定客服任务完成度（0-2 分），并输出 JSON。

## 评分标准
- 2：任务目标完全达成（如：促成还款承诺/完成产品推介/解决用户问题），且无违规
- 1：任务部分推进但未完全达成（如：用户同意考虑但未给承诺），无违规
- 0：任务未推进，或触发禁止动作，或对话无效循环

## 任务目标
{task_goal}

## 策略规则
{policy_rules}

## 完整对话记录
{history}

## 输出格式
只输出 JSON：
{{
  "score": 0|1|2,
  "reason": "简要理由（中文，50字内）"
}}

```
