# LEAKAGE_PROBE_PROMPT

- 来源：`listen2serve.evaluation.prompts` 的常量 `LEAKAGE_PROBE_PROMPT`
- sha256：`2302188684f37d7521793801902af6bc629114bc50692dd2eb63236e3fef1fd3`
- 说明：本文件是**导出快照**，真值源是代码常量；二者不一致由
  `tests/test_prompts_snapshot.py` 判红。运行时填充的槽位以 {…} 形式保留在原文里。

```text
你是一名只能看到文字记录的分析员。你听不到录音，拿不到说话人的语气、语速、音量，也不知道他真实的意图。请仅凭下面这两句文字作判断。

## 文字记录
客服：{agent_text}
用户：{user_text}

## 判定要求
1. surface_score：只读用户这句文字（不听录音），他的情绪或立场在字面上表露到什么程度？
   - 1＝完全看不出，就是一句中性的业务话（问细节、问流程、说事实），既无情绪词也无态度结论
   - 2＝只有一点口风（如“那个…”“能不能…”这类进退词、半句话），具体态度仍看不准
   - 3＝能看出有保留或有倒向，但态度没说完、留了口子
   - 4＝态度结论已经写在字面上（如明确说不接受/不需要/等不起/愿意配合），但没带情绪词
   - 5＝态度结论和情绪词都写在字面上（如“烦”“气”“怨”“感谢”“这事儿真叫人…”）
2. state_guess：仅凭文字猜用户此刻的状态，只能选一个：
   neutral（中性）/ cooperative（配合）/ displeased（不满）/ urgent（着急）/ doubtful（质疑）/ hushed（压低、回避）
   看不出来就选 neutral，不要猜。
3. cue：文字里让你这么判的词或短语（直接摘引原文）；若根本无依据就写“无”。

不要联想语气、不要推测弦外之音。你只评“字面上写了多少”。

## 输出格式
只输出 JSON：
{{
  "surface_score": 1|2|3|4|5,
  "state_guess": "neutral|cooperative|displeased|urgent|doubtful|hushed",
  "cue": "摘引原文或“无”"
}}

```
