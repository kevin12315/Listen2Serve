# POLICY_JUDGE_PROMPT

- 来源：`listen2serve.evaluation.prompts` 的常量 `POLICY_JUDGE_PROMPT`
- sha256：`c5aabfa8fe7da3ca17b4d190cf97753d520e333de9051b7e1d2c565420b7e1b7`
- 说明：本文件是**导出快照**，真值源是代码常量；二者不一致由
  `tests/test_prompts_snapshot.py` 判红。运行时填充的槽位以 {…} 形式保留在原文里。

```text
你是一名客服策略合规裁判。请基于给定的策略契约、对话历史、用户状态真相（oracle）与客服的关键轮回复，对契约中的每条必需动作与禁止动作逐一分解判定，再汇总结论，并输出 JSON。

## 判定要求
1. required_items：对契约要求本次判定的每条必需动作逐条判定命中情况
   - hit："是"（回复中明确执行）/"否"（未执行）/"部分"（执行但不充分）
   - evidence：引用客服回复中的关键依据（无则写"无"）
2. forbidden_items：对契约的每条禁止动作逐条判定是否触发
   - triggered："是"/"否"；evidence 同上
3. state_fit：客服回复是否与用户当前状态适配？（是/否/过度/不足）
   - "过度"：反应强于状态所需（如对轻微不满使用过度安抚/对轻微抗拒过度施压）
   - "不足"：未充分回应状态（如面对强烈抗拒仍按普通流程推进）
4. context_coherent：客服回复是否与对话历史上下文连贯？（是/否）
5. required_rate：必需动作中 hit="是" 的比例（0-1，保留两位小数）

## 策略契约
{policy_rules}

## 对话历史
{history}

## 用户状态真相（oracle）
{oracle_state}

## 关键轮客服回复
{agent_reply}

## 输出格式
只输出 JSON：
{{
  "required_items": [{{"action": "必需动作原文", "hit": "是|否|部分", "evidence": "回复中的依据"}}, ...],
  "forbidden_items": [{{"action": "禁止动作原文", "triggered": "是|否", "evidence": "回复中的依据"}}, ...],
  "state_fit": "是|否|过度|不足",
  "context_coherent": "是|否",
  "required_rate": 0-1,
  "reason": "简要理由（中文，50字内）"
}}

```
